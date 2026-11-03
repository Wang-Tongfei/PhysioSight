# PhysioSight refactoring plan

Status: executed on branch `refactor/physiosight-engine`. Decisions taken during
execution are in [DECISIONS.md](DECISIONS.md); every intentional behaviour
difference is in [BEHAVIOR_CHANGES.md](BEHAVIOR_CHANGES.md).

## 1. Assessment of the existing code (verified)

| Finding | Evidence | Consequence |
|---|---|---|
| One class per exercise; curl, squat and push-up repeat the same ready → start → target → return logic | `exercise_analysis.py` (618 lines), `BicepCurlAnalyzer`, `SquatAnalyzer`, `PushUpAnalyzer` | Adding an exercise means copying ~120 lines; fixes must be applied 3× |
| Thresholds hard-coded in `update()` (160/145/100, asymmetry 20, alignment 160, drift 35, 0.6/0.8 s, support offsets 0.035/0.055) | same file | No per-patient calibration; no record of which thresholds produced a result |
| `config.py` covers only the curl; `partial_up_angle` is stored but never read | `exercise_analysis.py:148-153`, `config.py` | Dead configuration |
| `form_analysis.RepCounter` is unused | only `calculate_angle` is imported | Dead code |
| Stability is counted in **frames** (`READY_STABLE_FRAMES=5`, `TARGET=3`, `RETURN=4`) | `exercise_analysis.py:16-18` | Behaviour depends on fps: on the synthetic suite the legacy engine counts 0 of 3 correct squats at 10 fps that it counts at 30 fps (`BC2-squat-10fps`) |
| Angles computed on **normalised** (x, y) | `form_analysis.calculate_angle` on `PoseResult` coordinates | Angles are distorted when width ≠ height: on 16:9 frames the legacy engine counts 0 of 3 correct squats (`BC1-squat-16x9`) |
| `PoseEstimator` stamps frames with `time.monotonic()` | `pose_estimator.py:71` | Offline results depend on processing speed |
| Frames are mirrored **before** pose estimation, for camera *and* video | `physio_form_monitor.py:73` | BlazePose labels left/right anatomically on the unmirrored image, so "right arm" tracked the patient's left arm (see BC-4) |
| Frames without a detected person are skipped entirely | `physio_form_monitor.py:95` | A repetition can "survive" any period out of view |
| Hint text written inside `update()` | all analysers | No localisation; UI and logic coupled |
| 3 tests, none for squat / push-up / plank counting | `test_safety_guards.py` | No safety net for refactoring |
| Platform imports 8 top-level modules through `sys.path` + `importlib` | PhysioVision `backend/app/services/monitor_engine.py:87-114` | Any file move breaks the platform; see [PLATFORM_MIGRATION.md](PLATFORM_MIGRATION.md) |

## 2. Options considered

**A. Incremental cleanup** – extract a base class, move thresholds into
`config.py`, convert frame counts to seconds. Low risk and fast, but adding an
exercise still requires code, there is no data contract, and the result is
still a script rather than a library.

**B. Declarative exercise specs + two generic state machines (chosen).**
Exercises are YAML files that declare geometric features, phase conditions,
checks (fault codes), recorded angles and parameter bounds. A small,
non-`eval` expression language evaluates conditions. One repetition machine
and one hold machine execute every spec. New exercises need no engine code;
thresholds become data that can be layered (spec defaults → patient
calibration → session overrides) and are written into every Session.
Risk: a DSL is code to maintain – mitigated by keeping it tiny (arithmetic,
comparisons, `and/or/not`, `abs/min/max/mean`), validating every name at load
time, and testing it with malformed/unsafe inputs.

**C. Python classes registered through entry points.** Fully typed and
debuggable, but each exercise is code; non-programmers (therapists, the
research side) cannot add or tune exercises, and thresholds stay scattered.

**D. Learned models (e.g. PoseRAC-style counting, skeleton-based quality
assessment) instead of rules.** These need labelled rehabilitation video
that does not exist yet, and rules remain the auditable baseline that a
learned model has to beat. Deferred; the design keeps it possible: analysers
consume backend-independent `Observation`s and emit the same Session, so a
learned counter is a third analyser type, not a rewrite.

Data contract: Protocol Buffers were considered; **Pydantic v2 models with an
exported, committed JSON Schema** were chosen because both consumers
(FastAPI backend, Python research code) are Python, FastAPI validates
Pydantic natively, and the JSON Schema lets the Next.js frontend generate
TypeScript types.

## 3. Final architecture

```
capture/      FrameSource: VideoFileSource (container timestamps), CameraSource (backend fallback)
perception/   PoseBackend protocol, PoseFrame (pixels, named skeletons), MediaPipe pose + hands,
              LandmarkSequence (.npz cache)
signal/       One Euro filter (Casiez et al. 2012), confidence-aware LandmarkSmoother
kinematics/   joint_angle / segment_incline / vertical_offset / distance_ratio features
exercises/    expr (safe DSL), spec (YAML models, layered params), machines (repetition, hold),
              specs/*.yaml
session/      Session contract (Pydantic) + builder
events.py     FaultDetected, RepStarted/Finished, HoldSegmentClosed, Tracking*, TargetReached
plugins/      ClipRecorder, TelegramNotifier (event subscribers)
render/       skeleton, hands, anonymised head, HUD
live.py       interactive monitor (successor of physio_form_monitor.py)
pipeline.py   Analysis (step-wise, live and offline), analyze_sequence, extract_landmarks
eval/         ICC, Bland-Altman, counting MAE/OBO, per-rep fault P/R/F1
cli.py        physiosight specs | schema | extract | analyze | run | notify-test
tools/        baseline (legacy replay + fixtures), validate (your own videos), experiments
legacy/       the original engine, unmodified, kept for baseline regeneration
```

Key properties:

* All stability requirements are in **seconds** from frame timestamps.
* All geometry is in **pixels** (or metric world landmarks), never normalised coordinates.
* Specs refer to keypoints **by name**, so the same spec runs on BlazePose-33
  and COCO-17 (RTMPose) skeletons – tested.
* Pose extraction is separated from analysis by a landmark cache, so state
  machine changes are evaluated on identical keypoints and CI needs no
  MediaPipe, camera or video.

## 4. Scope of this round

In scope: installable package; spec-driven engine reproducing the legacy
behaviour; Session contract; live monitor with all legacy features (camera
and video, body + hands, anonymous head, 4 exercises, key controls, fault
clips, Telegram); baseline + parity tests; property tests; CI; validation
scripts for local videos; one new exercise as YAML (shoulder flexion);
documentation including platform migration.

Explicitly **not** in scope: modifying PhysioVision; training any model;
an RTMPose/ONNX backend (the interface and COCO-17 support exist, but without
real video it could not be validated, so it is not shipped); monocular 3D
lifting; browser-side inference; offline gap interpolation; clinical
validation (requires participants, a reference system and ethics approval).

## 5. Acceptance criteria

1. `uv pip install -e ".[dev]"` and `pytest` pass on a machine with no camera,
   no video and no MediaPipe; with MediaPipe installed the legacy job also passes.
2. On the 28-case synthetic baseline suite the new engine reproduces the
   legacy engine frame by frame (count after every frame and every fault event).
3. Every intentional difference has an id in BEHAVIOR_CHANGES.md and a test
   showing the legacy result and the corrected result.
4. Counting is invariant to frame rate (10/30/60 fps) and aspect ratio, never
   advances on low-confidence input and never counts without reaching depth
   (Hypothesis property tests).
5. `physiosight run` offers every function of `physio_form_monitor.py`.
6. ruff and mypy are clean; GitHub Actions runs both jobs.
7. A newcomer can install, run the tests and analyse a video in ten minutes
   following the README.
