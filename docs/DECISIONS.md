# Decision log

Short records of decisions taken during the refactor: context → decision →
reason (→ consequence). Newest last.

### D1 – One name, defined once
`physiosight` is the distribution, import package and console command. The name
and version live only in `pyproject.toml`; code reads them through
`importlib.metadata` (`physiosight.NAME`, `physiosight.__version__`), and the
Session records them from there.

### D2 – Declarative YAML specs + a safe expression language (plan option B)
See PLAN §2. The language has no attribute access, indexing, strings or
calls other than `abs/min/max/mean`, rejects `__` in names, and every name is
validated against the spec's features, derived values and parameters at load
time. NaN makes every comparison false, so a missing keypoint can never
satisfy a condition.

### D3 – Exactly two state machines
Repetition (ready → start → target → return, checks, minimum duration) and
hold (credit time while valid). Home rehabilitation programmes are almost
entirely made of these two shapes (the exercises in KIMORE and UI-PRMD are
repetitions). Alternating-side exercises are handled as one spec per side
for now; a third machine type would be added only when a real exercise needs it.

### D4 – Seconds, not frames
Holds are "continuously true for `hold_s` seconds" using frame timestamps.
Defaults were chosen so that at 30 fps they equal the legacy frame counts
(5/3/4 frames → 0.12/0.06/0.09 s; 2-frame tracking guard → 0.03 s), which is
what makes frame-exact parity possible at 30 fps.

### D5 – Pixel-space geometry
Normalised coordinates distort angles for non-square frames. All features use
pixels; `vertical_offset` keeps the legacy "fraction of image height" unit
for parity and also offers body-scale normalisation (`scale: segment`).

### D6 – Never mirror before inference
Mirroring is a display concern. Analysis runs on the original frame, so
left/right are anatomical; the live view is mirrored for cameras only.

### D7 – Viewpoint is recorded, not enforced, for squat and curl
2D joint angles are projections. Webcam BlazePose agreement with optical
motion capture is poor for shoulder flexion (ICC ≈ 0.18) and elbow flexion
(≈ 0.53) (Wang et al., PLOS ONE 2023). Push-up, plank and shoulder flexion
declare `view: side` and a `view_check`; each repetition records `view_ok`
and the Session counts `frames_view_ok`. For squat and curl the legacy README
recommends a front view, and the squat's left/right symmetry check needs
both legs visible, so the view is not constrained there and their recorded
angles must be read as projections. Counting is never blocked by the view
check – it annotates data validity for therapists and research.

### D8 – The Session is patient-agnostic and image-free
The engine does not know patients. The platform associates a Session with a
patient on upload. No landmarks, frames or identities are in the Session.
`schema_version` is pinned (`"0.1.0"`); any change bumps it and is noted in
the migration guide. `schema/session.schema.json` is committed and a test
fails if it drifts from the model.

### D9 – Baseline as fixtures, recorded before the refactor
`tools/baseline/record_fixtures.py` replays the unmodified legacy engine on a
deterministic synthetic suite and stores per-frame progress changes and fault
events (`tests/fixtures/legacy_baseline.json`, 11 kB). The parity tests compare
the new engine with the fixture, so they run without the legacy code or
MediaPipe; a separate test re-derives the fixture from the legacy code when
MediaPipe is available.

### D10 – Synthetic landmarks for the analysis layer, real video for the pose layer
A 2D stick figure posed by forward kinematics gives exact ground truth for
angles and counts. This validates geometry and state machines (and frame-rate,
aspect-ratio and noise robustness) but says nothing about pose-model
accuracy; that requires the validation scripts in `tools/validate/` with real
recordings and a reference system (VALIDATION.md).

### D11 – Legacy code moves to `legacy/`, unchanged
It is still needed to regenerate the baseline and as a behavioural reference.
It is excluded from the package and from linting. The platform is not
affected immediately because its Dockerfile pins an engine commit; it should
migrate to the package (PLATFORM_MIGRATION.md), after which `legacy/` can be
deleted.

### D12 – One Euro smoothing is available but off by default
Speed-adaptive low-pass filtering (Casiez et al., CHI 2012) is implemented,
timestamp-based and confidence-aware, and was initially planned as the
default. The synthetic robustness study reversed that: with the default
parameters (min_cutoff 1 Hz, beta 0.01) it *lowered* exact-count accuracy at
every noise level (0.981 → 0.967 without noise, 0.958 → 0.933 at 4 px) because
its lag shaves the peaks that the depth/flexion thresholds test, while the
seconds-based hold timers already absorb jitter. It stays opt-in (`--smooth`)
until it is tuned and evaluated on real pose noise, which differs from the
synthetic Gaussian noise. This also keeps the default behaviour equal to the
legacy raw signal.

### D13 – Fault clips are anonymised and rate-limited
Clips are recorded from the rendered view, with the head covered by a filled
oval placed from ears and shoulders (eye/nose/mouth keypoints are not used).
One clip per fault code per 15 s, as the platform already did. `--show-face`
restores the outline-only overlay.

### D14 – No research repository in this round; evaluation metrics live in `physiosight.eval`
The brief limits this round to PhysioSight. The metrics needed to validate
the engine on real data (ICC(2,1) with CI, Bland-Altman, counting MAE/OBO,
fault P/R/F1) are small, tested and needed by `tools/validate/`, so they ship
as an optional extra (`physiosight[eval]`, adds SciPy). Training code,
datasets and model comparisons belong in a separate research repository later.

### D15 – No RTMPose backend yet
The `PoseBackend` protocol, named skeletons and the COCO-17 test show that a
COCO-17 backend plugs in unchanged. Without real video in this environment it
could not be validated, so shipping it would be shipping untested code.

### D16 – Documentation in English
The existing README is English and the target audience includes reviewers of
an open-source project. Locale files provide Chinese user-facing text.

### D17 – On-frame text stays English
OpenCV's Hershey fonts cannot draw CJK glyphs. Rendering Chinese would add a
font dependency (Pillow + a CJK font file). Localised strings are available
to any UI through `i18n.text(key, "zh")`; the CLI prints them.

### D18 – Corrections to the reference material
* The PoseRAC paper (Yao et al., 2023) reports OBO 0.56 and MAE 0.236 on RepCount;
  the 0.211 MAE with joint angles comes from a follow-up (Chen & Leu, 2023), not
  from PoseRAC itself. Counting reports here include both MAE and OBO.
* The MotionAGFormer 9.27° RMSE (Medrano-Paredes et al., 2025) is against IMU-based
  OpenSim angles in healthy participants on daily-living activities; it is
  not a rehabilitation-population result.
