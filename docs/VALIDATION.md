# Validating PhysioSight on real recordings

Everything in CI uses synthetic landmarks: it checks the geometry and the
state machines, not how accurate the pose model is on real people. This page
describes how to measure that locally. None of these numbers exist yet. Do
not quote any accuracy figure that you have not measured yourself.

## 1. Baseline: legacy vs new on your videos

```bash
uv pip install -e ".[mediapipe]"
python -m tools.baseline.make_baseline videos/squat_*.mp4 -e squat -o baseline/
```

For each video this caches landmarks once (`baseline/<name>.npz`), writes the
legacy per-frame trace (`<name>.legacy.jsonl`) and the new Session
(`<name>.session.json`), and prints the count and fault totals side by side.
Differences should be explainable by [BEHAVIOR_CHANGES.md](BEHAVIOR_CHANGES.md)
(aspect ratio BC-1 and frame rate BC-2 are the usual ones). Re-run with
`baseline/*.npz` instead of videos to skip pose extraction.

## 2. Counting and fault detection against your labels

Label each video (`labels.csv`):

```csv
video,exercise,side,true_reps,true_faults
squat_01.mp4,squat,,8,
squat_02.mp4,squat,,0,shallow_squat
curl_01.mp4,bicep_curl,right,10,upper_arm_drift
```

```bash
uv pip install -e ".[mediapipe,eval]"
python tools/validate/evaluate.py counts labels.csv --cache cache/ --sessions-out sessions/ -o report_counts.md
```

Reports exact-count rate, off-by-one accuracy (OBO), absolute and normalised
MAE (the RepCount metrics used by TransRAC and PoseRAC), and session-level
precision/recall of fault codes.

## 3. Left/right check (BC-4)

Record yourself side-on raising only your **right** arm forward several times, then:

```bash
physiosight extract right_arm.mp4 -o right.npz
physiosight analyze right.npz -e shoulder_flexion --side right
physiosight analyze right.npz -e shoulder_flexion --side left
```

The `right` run should count the repetitions; the `left` run should not.

## 4. Joint-angle agreement against a reference

Record the same repetitions with PhysioSight and a reference system:
optical motion capture (gold standard), a multi-camera / IMU pipeline such as
OpenCap (Uhlrich et al., 2023) or Pose2Sim (Pagnon et al., 2022), or at least a
goniometer at the end position. Then fill `reference.csv`:

```csv
session_file,rep_index,angle,reference_deg,stat
squat_01.json,1,knee_flexion_left,104.0,max_deg
squat_01.json,2,knee_flexion_left,101.5,max_deg
```

```bash
python tools/validate/evaluate.py angles reference.csv --sessions sessions/ -o report_angles.md
```

The report gives ICC(2,1) (two-way random, absolute agreement, single
measurement) with its 95% CI and Koo & Li (2016) band, Bland-Altman bias and
95% limits of agreement (Bland & Altman, 1986), RMSE and MAE per angle.

Study design notes:

* Film side-on for sagittal-plane angles (`view_ok` in the Session tells you
  whether that held). 2D angles from other viewpoints are projections.
* Split data **by participant** when thresholds are tuned on part of it.
* For published results, healthy-volunteer data does not transfer to patients
  (both cited 3D benchmarks used healthy participants only). Studies with
  people require informed consent and usually ethics approval.
* Known prior result to compare against: webcam BlazePose vs OptiTrack in 25
  healthy adults (Wang et al., PLOS ONE 2023) found good agreement for some
  shoulder motions (ICC ≈ 0.89–0.92) but poor agreement for shoulder flexion
  (≈ 0.18) and elbow flexion (≈ 0.53).

## 5. Synthetic robustness (reproducible, no data needed)

```bash
python -m tools.experiments.synthetic_robustness --n 30 -o docs/results/synthetic_robustness
```

The committed result is [results/synthetic_robustness.md](results/synthetic_robustness.md).
