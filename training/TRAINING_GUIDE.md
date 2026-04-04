# Segmentation Model Training Guide

## Overview

The AV stack uses a lightweight segmentation model (`SimpleSegNet`) for lane detection. It classifies each pixel into 3 classes:
- **0** = background
- **1** = left lane line (yellow center line)
- **2** = right lane line (white right edge)

Training uses **pseudo-labels** from the CV-based `SimpleLaneDetector` on ground-truth recordings. The GT follower drives perfectly centered using Unity's track path, producing clean frames where the CV detector can reliably label lane lines.

### Label semantics

The road has three painted lines: white left edge, yellow center dashes, white right edge. The model only needs to track the two boundaries of the ego lane:
- **Class 1** = yellow center line (left boundary of ego lane)
- **Class 2** = white right edge line (right boundary of ego lane)

The **left white edge line** (oncoming lane boundary) is labeled as background (class 0). This is handled automatically during dataset preparation — white pixels to the left of the yellow center line are filtered out.

## Quick Start — Overnight Training

Run this before bed. It collects GT data across all tracks, generates pseudo-labels, and trains the model:

```bash
# Full pipeline (~3 hours: 108 GT runs + dataset prep + 20-epoch training):
./tools/segmentation/train_pipeline.sh

# Quick test (~15 min: 2 tracks, 2 speeds, 3 lighting = 12 runs):
./tools/segmentation/train_pipeline.sh --quick

# Already have recordings? Skip to dataset prep + training:
./tools/segmentation/train_pipeline.sh --skip-collect --epochs 25
```

**Prerequisites:**
- Unity must be open or the built player must exist (`unity/AVSimulation/mybuild.app`)
- The first run builds the player automatically; subsequent runs reuse it

**Output:**
- Recordings: `data/recordings/seg_training/`
- Dataset: `data/segmentation_dataset/`
- Checkpoint: `data/segmentation_dataset/checkpoints/segnet_best.pt`
- Log: `data/segmentation_dataset/training_pipeline.log`

**Morning checklist:**
1. Check the log: `tail -20 data/segmentation_dataset/training_pipeline.log`
2. Validate model output on a recording (see Step 4 below)
3. Run E2E test (see Step 5 below)

---

## Pipeline Details

```
1. Collect GT recordings    -->  tools/segmentation/collect_training_data.sh
2. Generate dataset          -->  training/scripts/prepare_segmentation_dataset.py
3. Train model               -->  training/scripts/train_segmentation.py
4. Validate on recordings    -->  tools/segmentation/preview_segmentation_masks.py
5. E2E test                  -->  start_av_stack.sh on target track
```

---

## Step 1: Collect Ground Truth Recordings

The batch collection script runs the GT follower across all tracks, speeds, and lighting conditions.

### Commands
```bash
# Preview what will run (no actual runs):
./tools/segmentation/collect_training_data.sh --dry-run

# Collect all tracks (9 tracks x 4 speeds x 3 lighting = 108 runs):
./tools/segmentation/collect_training_data.sh

# Subset for quick iteration:
./tools/segmentation/collect_training_data.sh --tracks s_loop hill_highway --speeds 8 12
```

### What it does
- Runs `start_ground_truth.sh` for each combination of track, speed, and lighting
- Automatically caps duration to 95% of one lap (prevents recording the track-end wrap seam)
- Saves recordings to `data/recordings/seg_training/` with descriptive names
- Skips existing recordings (use `--no-skip` to force re-collection)
- Logs results to `collection_log.txt`

### Tracks (9 total)
| Track | Length | Key features |
|-------|--------|-------------|
| oval | ~314m | Simple baseline, no grade |
| s_loop | ~452m | S-curves, mixed radii |
| circle | ~314m | Constant curvature |
| hairpin_15 | ~390m | Tight R15 turns |
| highway_65 | ~3100m | Long straights, gentle curves |
| mixed_radius | ~500m | R25-R200 variety |
| sweeping_highway | ~2200m | High-speed sweepers |
| hill_highway | ~983m | Grades up to 5%, elevation changes |
| autobahn_30 | ~1800m | High-speed, wide lanes |

### Speeds (4 default)
6, 8, 10, 12 m/s — covers slow hairpin speeds to highway cruising.

### Lighting presets (3 default)
| Preset | Sun altitude | Sun azimuth | Effect |
|--------|-------------|-------------|--------|
| noon | 80 deg | 180 deg | Bright, overhead, minimal shadows |
| afternoon | 45 deg | 220 deg | Default-like, moderate shadows |
| dawn | 15 deg | 90 deg | Low sun, longer shadows, darker road |

Custom lighting for a single run:
```bash
./start_ground_truth.sh --track-yaml tracks/s_loop.yml --speed 8 \
    --sun-altitude 10 --sun-azimuth 270 --duration 50
```

### Camera perspective
The GT car's camera height matches the physics-driven AV car's ride height (0.30m suspension offset applied in GT mode in `CarController.cs`). This ensures training data has the same perspective as actual E2E runs. Verified: horizon line differs by 1 pixel between GT and normal mode.

### Track-end safety
Tracks are loops but may have a geometric seam at the wrap point. The collection script computes track length from the YAML (`tools/segmentation/track_length.py`) and caps recording duration to 95% of one lap so the car stops before reaching the seam.

---

## Step 2: Generate Training Dataset

```bash
python training/scripts/prepare_segmentation_dataset.py \
    --recordings data/recordings/seg_training/ \
    --output-dir data/segmentation_dataset/
```

This:
1. Iterates all `.h5` files in the recordings directory
2. Runs `SimpleLaneDetector` (CV) on each frame to produce pseudo-label masks
3. **Filters left-side white pixels**: white pixels to the left of the yellow center line are set to background, preventing the oncoming lane's edge line from being mislabeled as the right lane boundary
4. Filters out bad frames (sparse lane pixels, off-center ground truth)
5. Saves `images/`, `masks/`, and optional `overlays/` to the output directory
6. Generates `metadata.csv` with per-frame stats

### Filtering criteria
- `min_total_pixels=400` — skip frames with very few lane pixels
- `min_lane_pixels=150` — each lane class needs enough pixels
- `max_lane_center_abs=1.5m` — skip frames where car is far off-center
- `stride=2` — sample every 2nd frame (reduces redundancy)

### Checking pseudo-label quality
```bash
# Preview CV detector masks on a specific recording:
python tools/segmentation/preview_segmentation_masks.py \
    --recording data/recordings/seg_training/<file>.h5 \
    --frames 50,150,300 --save_overlay

# Check output in tmp/segmentation_preview/
```

---

## Step 3: Train the Model

```bash
python training/scripts/train_segmentation.py \
    --data-dir data/segmentation_dataset/ \
    --epochs 20
```

### Model architecture
`SimpleSegNet`: lightweight encoder-decoder
- Input: 320x640 RGB
- Output: 320x640 with 3 classes
- Loss: CrossEntropyLoss
- Size: ~187KB checkpoint

### Training parameters
| Parameter | Default | Recommended |
|-----------|---------|-------------|
| batch_size | 8 | 8 (increase if GPU memory allows) |
| epochs | 5 | 20 for multi-track dataset |
| learning_rate | 1e-3 | 1e-3 |
| val_split | 0.2 | 0.2 (80/20 train/val) |

### How much data is needed?
For our synthetic Unity environment (clean lines, consistent textures):
- **Minimum viable**: 3 tracks x 2 speeds = ~3K frames after filtering
- **Good**: 9 tracks x 4 speeds x 1 lighting = ~13K frames
- **Full**: 9 tracks x 4 speeds x 3 lighting = ~40K frames

Geometric diversity (different curvatures, grades, road widths) matters more than raw volume.

### Checkpoint
Saved to: `data/segmentation_dataset/checkpoints/segnet_best.pt`

---

## Step 4: Validate

### On recording frames
```bash
# Run trained model on recording frames and save overlays:
python tools/segmentation/preview_segmentation_masks.py \
    --recording data/recordings/<file>.h5 \
    --frames 50-100 --segmentation --save_overlay

# Output: tmp/segmentation_preview/frame_NNNNNN_seg_overlay.png
# Yellow overlay = left lane (class 1), White overlay = right lane (class 2)
```

### On all frames (check coverage stats)
```bash
python tools/segmentation/preview_segmentation_masks.py \
    --recording data/recordings/<file>.h5 \
    --all --segmentation --output_dir tmp/seg_full
```

Per-frame output shows pixel counts and coverage percentage.

### What to look for
- Left lane pixels (yellow) should track the center dashed line
- Right lane pixels (white) should track the right road edge
- Coverage should be 5-15% (not 0%, not 50%+)
- No large blobs on the car hood or sky (indicates training data issue)
- No white-class pixels on the left side of the road (label contamination)

---

## Step 5: E2E Test

```bash
# Run on the target track:
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean \
    --track-yaml tracks/hill_highway.yml --duration 60

# Analyze:
python tools/analyze/analyze_drive_overall.py --latest
```

Check that:
- Perception layer score is not penalized for blind perception
- `blind_perception_rate` is close to 0% (not 98%+)
- Trajectory layer >= 80 (no late turn-in from missing detections)
- Overall score improved vs pre-training baseline

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Model labels car hood as lane | Training data camera height mismatch | Fixed in CarController.cs (GT ride height offset). Re-collect and retrain. |
| 0% coverage on all frames | Model outputs only background | Check training data has enough lane pixels. Increase epochs. |
| Good on straights, bad on curves | Not enough curve training data | Add more tracks with tight radii (hairpin, mixed_radius, s_loop) |
| White lead car confused with lane | Training on ACC scenario recordings | Only use base track GT recordings (no lead vehicles) |
| Frames look different in GT vs AV | Camera height mismatch | Compare frames at same index. Horizon should be within 2px. |
| Left white edge labeled as class 2 | Label filtering bug | Check `build_label_mask()` in `training/utils/segmentation_dataset.py` |
| Recording wraps/jumps at end | Duration exceeds one lap | Collection script auto-caps; or use `track_length.py` to compute safe duration |

---

## Key files

| File | Purpose |
|------|---------|
| `tools/segmentation/train_pipeline.sh` | End-to-end overnight pipeline |
| `tools/segmentation/collect_training_data.sh` | Batch GT data collection |
| `tools/segmentation/track_length.py` | Compute track length and safe duration |
| `tools/segmentation/preview_segmentation_masks.py` | Visualize model output or CV masks on recordings |
| `training/scripts/prepare_segmentation_dataset.py` | Generate pseudo-labeled dataset from recordings |
| `training/scripts/train_segmentation.py` | Train SimpleSegNet model |
| `training/utils/segmentation_dataset.py` | Label mask builder (includes left-white filtering) |
| `perception/models/segmentation_model.py` | Model definition and checkpoint loader |

---

## Important notes

- **Never train on ACC scenario recordings** (`tracks/scenarios/`) — white lead vehicles will confuse the model
- **Only use base tracks** (`tracks/*.yml`) for GT data collection — no lead vehicles present
- **Pseudo-labels come from CV detector** — if CV detector fails on a track, those frames get filtered out
- **Track-end safety**: Collection script auto-caps duration to 95% of one lap
- **Re-training after Unity changes**: If road materials, lighting, or camera FOV change, re-collect GT data and retrain
- **Left white line filtering**: The dataset builder automatically filters white pixels on the wrong side of the yellow center line — do not remove this logic
