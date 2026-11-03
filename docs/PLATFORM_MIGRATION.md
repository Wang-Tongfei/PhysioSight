# Migrating PhysioVision to the `physiosight` package

This guide describes how PhysioVision should stop embedding the engine as a
Git submodule and depend on the package instead. Nothing in PhysioVision was
changed by this refactor.

## Why it must change

`backend/app/services/monitor_engine.py` puts the submodule on `sys.path` and
imports eight top-level modules by name (`config`, `camera`, `pose_estimator`,
`hand_estimator`, `exercise_analysis`, `clip_recorder`, `notifier`, `hud`).
Those modules now live in `legacy/` and will be deleted once the platform has
migrated. Until then the platform keeps working because both the submodule and
`backend/Dockerfile` (`PHYSIO_AI_BOT_COMMIT`) pin an old engine commit. **Do not
bump the submodule to this branch without doing the migration.**

Interim option, if a bump is unavoidable: change `BOT_ROOT` to
`PROJECT_ROOT / "Physio_AI_Bot" / "legacy"` and, after creating `cfg`, point
`cfg.model_path` / `cfg.hand_model_path` at `Physio_AI_Bot/models/` (the legacy
`config.py` resolves models relative to its own folder). The legacy modules
are otherwise unchanged, so the existing import code works.

## Files to change

| File | Change |
|---|---|
| `.gitmodules`, `Physio_AI_Bot/` | Remove the submodule (`git submodule deinit -f Physio_AI_Bot && git rm -f Physio_AI_Bot`) |
| `backend/requirements.txt` | Add `physiosight[mediapipe,notify] @ git+https://github.com/Wang-Tongfei/PhysioSight@<tag>`; drop `opencv-python`, `mediapipe` (now transitive) |
| `backend/Dockerfile`, root `Dockerfile` | Remove the clone/`COPY Physio_AI_Bot/` steps; install the requirement above. Copy the two model files (`models/*.task`) from the engine repository or download them at build time, and pass their paths explicitly |
| `backend/app/services/monitor_engine.py` | Replace `_load_bot_modules()` and the analysis part of `_run()` (see below). Keep job management, MJPEG streaming, ownership checks and persistence |
| `backend/app/schemas/session.py` | Add an upload schema that wraps `physiosight.session.Session` |
| `backend/app/models/session.py` | Store the Session JSON (new `session_json` JSON column or a `session_records` table) next to the summary columns |
| `backend/tests/test_monitor_engine.py` | Use `physiosight.testing.synth` + `physiosight.pipeline.analyze_sequence` to test without video, or keep the upload test with a small generated clip |
| `README.md`, `docs/ARCHITECTURE.md`, `docs/API_SPEC.md` | Replace references to `Physio_AI_Bot` |

## New calling convention

```python
from physiosight.capture.sources import VideoFileSource
from physiosight.exercises.spec import load_spec
from physiosight.perception.mediapipe_backend import MediaPipePoseBackend
from physiosight.pipeline import Analysis
from physiosight.render.overlay import draw_hud, draw_pose
from physiosight.session.models import SourceInfo
from physiosight.events import FaultDetected
from physiosight import i18n

spec, sha = load_spec(exercise, side=track_arm if exercise == "bicep_curl" else None)
backend = MediaPipePoseBackend(str(POSE_MODEL))
analysis = Analysis(
    spec, target=spec.default_target,
    source=SourceInfo(kind="video", width=w, height=h, nominal_fps=fps),
    pose_backend=backend.name, pose_backend_version=backend.version, spec_sha256=sha,
    params=patient_calibration,          # dict or None - see "thresholds" below
    smoother=None,                       # LandmarkSmoother() is opt-in, see DECISIONS D12
)
analysis.bus.subscribe(FaultDetected, on_fault)   # clips / alerts

for t, frame in source.frames():                  # t = seconds (container time, or client time)
    pose = backend.process(frame, t)
    result = analysis.step(t, pose)               # FrameResult: progress, form_ok, hint_key, metric_value, events
    draw_pose(frame, pose); draw_hud(frame, result, i18n.text(spec.name_key), analysis.analyzer.target,
                                     "sec" if spec.kind == "hold" else "reps")
    status["coach_message"] = i18n.text(result.hint_key, "zh", progress=result.progress,
                                        target=analysis.analyzer.target)
    if analysis.done:
        break

session = analysis.finish()                       # physiosight.session.Session
```

Notes:

* **Browser frames** (`BrowserFrameSource`): pass a timestamp per frame. Prefer
  the client capture time sent with each JPEG; otherwise use server receive
  time. Because the engine is time-based, 10 fps and 30 fps give the same
  decisions (BC-2).
* **Do not flip** uploaded or browser frames before `backend.process` (BC-4).
  Mirror only the image you send back to the browser.
* **Never call the engine with wall-clock time for uploads.** Use the
  container timestamps from `VideoFileSource.frames()` (BC-5). Upload
  processing no longer needs the `frame_interval` sleep for correctness, only
  for live-like streaming.
* **Hands** are display-only: `physiosight.perception.hands.MediaPipeHandBackend`.

## Data mapping

| Platform field (today) | From the Session |
|---|---|
| `total_reps` | `session.summary.achieved` (holds: credited seconds) |
| `rom_achieved_deg` | from `session.reps[*].angles[...]` – e.g. median of `max_deg` of `knee_flexion_left` over valid reps. Today it is max − min of the per-frame metric across the whole session, which mixes the set-up phase into ROM |
| `rom_target_deg` | the spec parameter it corresponds to (`session.engine.params`), not a separate hard-coded `ROM_TARGETS` table |
| `compensation_detected` | `any(session.faults)` or, better, the specific codes |
| `movement_quality_score` | platform heuristic; compute from `reps` (valid / attempts, per-fault counts) and label it as a heuristic |
| `fatigue_index` | today this is the ratio of frames with `form_ok == False`; it is not fatigue. Rename (e.g. `fault_frame_ratio`) or derive a real trend (e.g. rep duration or peak-angle change over the set) from `reps` |
| `data_status` | `insufficient_data` when `summary.attempts == 0` or `tracking.reliable_ratio` is low |
| new | `tracking.frames_view_ok`, `reps[*].view_ok` – show a warning when angles were measured from an unsuitable viewpoint |

Validate uploads with the model itself:

```python
from physiosight.session.models import Session
session = Session.model_validate_json(body)       # rejects inconsistent sessions
```

The frontend can generate TypeScript types from `schema/session.schema.json`
(e.g. `json-schema-to-typescript`).

## Thresholds and calibration

Per-patient thresholds are a dict of spec parameters, validated against the
spec's bounds (`physiosight.exercises.spec.resolve_params`). Store them per
patient and exercise in the platform; the effective values end up in
`session.engine.params`, so every report is reproducible. List the parameters
with `physiosight specs` and the YAML files in `src/physiosight/exercises/specs/`.

## Target architecture (later)

Run inference on the patient's device (the `physiosight` CLI locally, or
MediaPipe in the browser) and upload only the Session JSON and, when needed,
an anonymised clip. The platform then becomes "store Sessions → therapist
reports and ROM trends" and no longer needs OpenCV or MediaPipe on the server.
