# Behaviour changes versus the legacy engine

Baseline: the legacy engine (now `legacy/`) replayed on the synthetic suite in
`tools/baseline/cases.py`, recorded in `tests/fixtures/legacy_baseline.json`.
For the 28 cases marked `identical` the new engine reproduces the legacy
count after **every frame** and **every fault event** (`tests/test_legacy_parity.py`).
Everything that differs on purpose is listed here.

| Id | Change | Why | Who is affected | Evidence |
|---|---|---|---|---|
| BC-1 | Angles and offsets are computed in **pixels** instead of normalised (x, y) | Normalised coordinates stretch one axis whenever width ≠ height, distorting every angle | Any non-square input (all common cameras, 16:9 / 4:3) | Fixture cases `BC1-squat-16x9` (legacy 0/3, new 3/3) and `BC1-pushup-16x9` (legacy 0/3, new 3/3); property test `test_count_does_not_depend_on_aspect_ratio` |
| BC-2 | Stability requirements are **seconds**, not frame counts | Frame counts make behaviour depend on fps (platform ~10 fps vs local ~30 fps) | Inputs not at 30 fps; at 30 fps behaviour is identical | Case `BC2-squat-10fps` (legacy 0/3 at 10 fps but 3/3 at 30 fps; new 3/3 at both); `test_squat_count_is_frame_rate_invariant` (10/30/60 fps) |
| BC-3 | Frames **without a detected person** count as tracking loss: after `tracking_grace_s` the attempt is aborted and recorded as `aborted` | The legacy loop skipped those frames, so a repetition could be completed after any time out of view | Patients leaving the frame mid-repetition | `test_frames_without_a_person_abort_the_repetition` |
| BC-4 | Frames are **not mirrored before pose estimation**; only the live camera *view* is mirrored. Video files are not mirrored at all | BlazePose labels left/right on the image it sees; mirroring first swapped the patient's sides, so `track_arm=right` followed the left arm. The legacy CLI also mirrored video files | Curl side selection; any left/right-specific result; platform uploads were already unmirrored | Mirroring of display coordinates: `test_mirror_pose_flips_x_only`. Side labelling is a property of the pose model and is not covered by synthetic tests – verify once with a real video (VALIDATION.md §3) |
| BC-5 | Analysis time comes from **frame timestamps** (video container time, or camera time) instead of `time.monotonic()` at processing time | For video files the legacy "movement too fast" and hold checks measured processing speed, not movement speed | Offline video analysis on fast or slow machines | `VideoFileSource` / `MediaPipePoseBackend` use `t`; all analysis tests pass explicit timestamps |
| BC-6 | Switching side (`a`), resetting (`r`) or switching exercise (`1`-`5`) ends the current **Session** and starts a new one | A Session must describe one exercise on one side with one set of thresholds. Legacy kept the count when switching arm | Live users who switched arm mid-set | `live.py` (`_end_segment`) |
| BC-7 | Fault clips are recorded from the **anonymised rendered view** (head covered by a filled oval) and limited to **one per fault code per 15 s** | Legacy clips contained the raw face (its README warned about this) and one clip + Telegram upload was sent per fault event | Clip recipients; `--show-face` restores outline-only rendering | `test_head_is_covered_by_default`; `ClipRecorder(cooldown_s=15)` |
| BC-8 | User-facing text comes from locale files by key (`en`, `zh`); `"Squat reps: n/m"` / `"Push-up reps: n/m"` became `"Reps: n/m"`; the one-frame `"Good rep! (n/m)"` banner is now the `RepFinished` event printed to the console | Text separated from logic; localisation | Anyone matching on status strings (the platform displayed them) | `i18n/en.yaml`, `i18n/zh.yaml` |
| BC-9 | Duration and hold comparisons use a 1e-6 s tolerance | Legacy rejected e.g. a 0.6 s repetition against a 0.6 s minimum because `1.6333 - 1.0333 < 0.6` in floating point | Movements exactly at a threshold | `EPS` in `exercises/machines.py` |
| BC-10 | Telegram credentials are read from `PHYSIOSIGHT_TG_TOKEN` / `PHYSIOSIGHT_TG_CHAT`; the legacy `PHYSIO_TG_*` names are still accepted. Start / complete messages say "PhysioSight" | Consistent naming | Existing installations keep working | `plugins/telegram.py` |
| BC-11 | Removed: `config.py` (thresholds are spec parameters, overridable with `--set key=value` / `--params file.json`), `partial_up_angle` (was never used), `form_analysis.RepCounter` (unused), `requirements.txt` (use the `pyproject.toml` extras), `send_test_alert.py` / `send_test_video.py` (now `physiosight notify-test [--video clip.mp4]`) | One configuration mechanism; dead code | Scripts that imported these modules (the platform – see PLATFORM_MIGRATION.md) | `legacy/` keeps the originals |

Additions (no legacy equivalent): Session JSON output (`--sessions-dir`,
`analyze -o`), per-repetition angle extrema and `view_ok`, the
`shoulder_flexion` exercise (key `5`), layered threshold calibration,
landmark caching (`extract`), JSON Schema export.

Unchanged on purpose: no keypoint smoothing by default (`--smooth` enables One Euro, see D12), thresholds and their defaults, fault codes, the hint
precedence, the plank's reset after 0.75 s of broken form, the 0.25 s cap on
credited time per frame, targets (10 reps, 30 s plank), the confidence gates
(visibility ≥ 0.55, presence ≥ 0.50), and stopping analysis at the target.
