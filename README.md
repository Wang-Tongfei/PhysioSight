# PhysioSight

Single-camera, real-time, markerless movement analysis for rehabilitation
exercises. PhysioSight tracks the body with a pose model (MediaPipe BlazePose
by default), counts repetitions and holds, checks form against declarative,
per-patient-calibratable rules, and writes a structured **Session** record per
exercise set.

> **Not a medical device.** PhysioSight is a development, demonstration and
> research tool. It does not diagnose, and its thresholds and angles are not
> clinically validated. Angles from a single 2D camera are projections and
> depend on the viewpoint (see [docs/VALIDATION.md](docs/VALIDATION.md)).

## Quick start (about 10 minutes)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
git clone https://github.com/Wang-Tongfei/PhysioSight && cd PhysioSight
uv venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[mediapipe,notify,dev]"
pytest                                        # ~1 min, no camera or video needed
```

Live camera (key controls below), or a video file:

```bash
physiosight run                     # camera 0, bicep curl
physiosight run 1 -e squat          # camera 1
physiosight run path/to/video.mp4 -e pushup --sessions-dir sessions/
```

Offline, reproducible analysis (pose extraction once, analysis many times):

```bash
physiosight extract video.mp4 -o video.npz                 # MediaPipe -> landmark cache
physiosight analyze video.npz -e squat -o session.json --lang zh
physiosight analyze video.npz -e squat --set depth_angle=110   # calibrated threshold
```

Other commands: `physiosight specs` (list exercises), `physiosight schema`
(Session JSON Schema), `physiosight notify-test [--video clip.mp4]` (Telegram check).

Linux needs the EGL/GL runtime for MediaPipe: `sudo apt-get install libegl1 libgles2 libgl1`.

### Key controls (`physiosight run`)

Click the video window first.

| Key | Action |
|---|---|
| `1` `2` `3` `4` `5` | Bicep curl, squat, plank, push-up, shoulder flexion |
| `a` | Switch side (sided exercises) |
| `r` | Reset (starts a new Session) |
| `q` | Quit |

## Exercises

| Id | Kind | Recommended view | Faults |
|---|---|---|---|
| `bicep_curl` | repetition, left/right | front or three-quarter (counting); side for elbow angle | `partial_curl`, `upper_arm_drift`, `movement_too_fast` |
| `squat` | repetition | front or front three-quarter (counting, symmetry); side for knee angle | `shallow_squat`, `uneven_squat`, `movement_too_fast` |
| `pushup` | repetition | full side view | `shallow_pushup`, `pushup_body_alignment`, `movement_too_fast` |
| `plank` | hold | full side view | `plank_body_alignment` |
| `shoulder_flexion` | repetition, left/right | side view | `insufficient_range`, `elbow_bent`, `movement_too_fast` |

General guidance: keep the whole body in frame, use good light, avoid
clothing that hides the joints, move slowly. A repetition only counts if it
starts from the ready position, reaches the target, passes all checks and
returns. The plank timer resets after about 0.75 s of continuously broken form.

### Adding an exercise

Write a YAML file – no engine code. `shoulder_flexion` was added this way:
see [src/physiosight/exercises/specs/shoulder_flexion.yaml](src/physiosight/exercises/specs/shoulder_flexion.yaml).
A spec declares parameters with bounds, required keypoints, geometric
features (`joint_angle`, `segment_incline`, `vertical_offset`,
`distance_ratio`), phase conditions with hold times in seconds, checks that
raise fault codes, angles to record per repetition, and hint keys. Conditions
use a small safe expression language (`abs(knee_l - knee_r) > p.max_asymmetry`).
Use it with `-e path/to/my_exercise.yaml`; put user-facing text in
`src/physiosight/i18n/{en,zh}.yaml`.

### Thresholds and calibration

Every threshold is a spec parameter with a default and allowed range. Values
are resolved in layers – spec default → patient calibration (`--params
patient.json`) → session override (`--set key=value`) – and the effective
values are written into each Session for reproducibility.

## Session output

One JSON document per exercise set ([schema](schema/session.schema.json),
model in `physiosight.session.models`): exercise and side, source (size, fps,
mirroring), engine (version, pose backend, spec version and hash, resolved
parameters), tracking quality, summary, every repetition attempt (timing,
outcome `valid`/`invalid`/`aborted`, fault codes, per-angle min/max, whether
the viewpoint suited the angle), hold segments and fault events. It contains
no images, landmarks or patient identity.

## Architecture

```
FrameSource ─► PoseBackend ─► LandmarkSmoother ─► Analyzer (spec-driven) ─► events ─► plugins (clips, Telegram)
 camera/video   MediaPipe        One Euro           repetition | hold         │
                (named skeleton)                                              └─► SessionBuilder ─► Session JSON
```

Details, options considered and scope: [docs/PLAN.md](docs/PLAN.md).
Decisions: [docs/DECISIONS.md](docs/DECISIONS.md).
Differences from the original engine: [docs/BEHAVIOR_CHANGES.md](docs/BEHAVIOR_CHANGES.md).
Platform integration: [docs/PLATFORM_MIGRATION.md](docs/PLATFORM_MIGRATION.md).

```
src/physiosight/   capture, perception, signal, kinematics, exercises (+ specs/), session,
                   plugins, render, eval, live.py, pipeline.py, cli.py
tests/             geometry, expressions, specs, state machines, properties (Hypothesis),
                   legacy parity (fixture), CLI, rendering, validation tools
tools/baseline/    legacy replay, baseline suite, fixture recorder, real-video comparison
tools/validate/    counting and angle-agreement reports for your own recordings
tools/experiments/ synthetic robustness study
legacy/            the original engine, unchanged (baseline reference; removed after the platform migrates)
models/            MediaPipe pose and hand models
```

## Testing

```bash
pytest                         # all tests; legacy-replay test runs only if mediapipe is installed
ruff check src tests tools && mypy src/physiosight
```

The test suite needs no camera, video or GPU. The analysis layer is tested on
synthetic landmark sequences with exact ground truth, including frame-by-frame
parity with the original engine on 28 recorded cases and property tests
(counts identical at 10/30/60 fps and any aspect ratio; never counting
without reaching depth; low-confidence input never advancing). See
[docs/VALIDATION.md](docs/VALIDATION.md) for validating on real recordings.

## Telegram alerts (optional)

Disabled unless both variables are set (the old `PHYSIO_TG_*` names also work):

```powershell
[Environment]::SetEnvironmentVariable("PHYSIOSIGHT_TG_TOKEN", "your-bot-token", "User")
[Environment]::SetEnvironmentVariable("PHYSIOSIGHT_TG_CHAT", "your-chat-id", "User")
```

Then `physiosight notify-test` (and `--video clips/x.mp4`). Never commit tokens;
`.env` and `*.token` are git-ignored.

## Fault clips

When a fault is detected, the last ~4 s of the rendered view are saved to
`clips/` (at most one clip per fault code per 15 s) and, if configured, sent
via Telegram. `--clips ''` disables recording.

## Privacy

- Pose landmarks and video are not stored unless you run `extract` or record clips.
- Sessions contain no images, landmarks or identity.
- The head is covered by a filled oval in the view and in clips, placed from
  ears and shoulders only (eye/nose/mouth keypoints are not used).
  `--show-face` turns this off. If no person is detected in a frame, nothing
  is covered – clips can still contain faces or bystanders; review before sharing.
- `clips/`, `*.mp4`, `*.avi`, `*.npz` and `cache/` are git-ignored.

## References

- V. Bazarevsky et al., *BlazePose: On-device Real-time Body Pose Tracking*, CVPR Workshops 2020, arXiv:2006.10204.
- T. Jiang et al., *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*, arXiv:2303.07399, 2023.
- G. Casiez, N. Roussel, D. Vogel, *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems*, CHI 2012, doi:10.1145/2207676.2208639.
- A. Wang et al., *A webcam-based machine learning approach for three-dimensional range of motion evaluation*, PLOS ONE 2023, e0293178 (arXiv:2310.07322).
- H. Hu et al., *TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting*, CVPR 2022 (RepCount).
- Z. Yao, X. Cheng, Y. Zou, *PoseRAC: Pose Saliency Transformer for Repetitive Action Counting*, arXiv:2303.08450, 2023; H. Chen, M. C. Leu, *Joint-based PoseRAC*, arXiv:2308.08632, 2023.
- A. Ismail-Fawaz et al., *Deep Learning for Skeleton Based Human Motion Rehabilitation Assessment: A Benchmark* (Rehab-Pile), arXiv:2507.21018, 2025.
- M. Capecci et al., *The KIMORE Dataset*, IEEE TNSRE 2019; A. Vakanski et al., *A Data Set of Human Body Movements for Physical Rehabilitation Exercises* (UI-PRMD), Data 2018.
- W. Zhu et al., *MotionBERT*, ICCV 2023; S. Mehraban et al., *MotionAGFormer*, WACV 2024; M. Medrano-Paredes et al., arXiv:2510.02264, 2025 (monocular 3D vs IMU, healthy participants).
- D. Pagnon et al., *Pose2Sim*, JOSS 2022; *Sports2D*, JOSS 2024; S. Uhlrich et al., *OpenCap*, PLOS Comput. Biol. 2023. FreeMoCap (AGPL) was consulted as prior art only; no code was copied.
- P. Shrout, J. Fleiss, *Intraclass correlations*, Psychol. Bull. 1979; K. McGraw, S. Wong, Psychol. Methods 1996; T. Koo, M. Li, J. Chiropr. Med. 2016; J. M. Bland, D. G. Altman, Lancet 1986.
