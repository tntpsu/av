# Circular Drive Issue Analysis

## Problem
Car is going in a small circle to the right instead of following the oval track.

## Root Cause Analysis

### Most Likely Causes (in order):

#### 1. **Path Curvature is Consistently Positive** (Most Likely)
**Symptom:** Path curvature from Unity is always positive (right turn)

**Why this causes circles:**
- Positive curvature → positive steering → car turns right
- If curvature is always positive, car will always turn right → circular motion

**How to verify:**
- Check `ground_truth/path_curvature` in recording
- If average curvature > 0.01 1/m, this is the issue

**Possible reasons:**
- Unity's `GroundTruthReporter.GetPathCurvature()` is calculating curvature incorrectly
- Curvature sign is inverted (should be negative for left turns, positive for right)
- Unity is calculating curvature at wrong point on path

#### 2. **Ground Truth Mode Not Active**
**Symptom:** Unity feedback shows `ground_truth_mode_active = false`

**Why this causes issues:**
- Without ground truth mode, Unity uses physics-based control
- Physics mode is less precise and may not follow exact steering commands
- Could cause drift or incorrect behavior

**How to verify:**
- Check `unity_feedback/ground_truth_mode_active` in recording
- If all values are `false`, ground truth mode was never enabled

#### 3. **Steering Sign Error**
**Symptom:** Steering calculation produces positive steering when it should be negative (or vice versa)

**Why this causes circles:**
- If steering sign is wrong, car steers away from path
- This creates a feedback loop: error increases → wrong steering → error increases more

**How to verify:**
- Check `control/steering` vs `vehicle/steering_angle` in recording
- Check if steering direction matches expected based on lane center

#### 4. **Path Curvature Not Being Calculated**
**Symptom:** All path curvature values are zero

**Why this causes issues:**
- If curvature is always zero, steering calculation falls back to lateral correction only
- Without path curvature, car can't follow curved paths correctly
- May cause oscillations or incorrect steering

**How to verify:**
- Check `ground_truth/path_curvature` in recording
- If all values are near zero, path curvature is not being calculated

## Why Our Tests Didn't Catch This

### What Our Tests Verify:
✅ **Steering calculation is mathematically correct**
- Tests verify: `steering_angle = atan(wheelbase * curvature)`
- Tests verify coordinate system interpretation
- Tests verify lateral correction works

✅ **Unit-level correctness**
- Tests verify individual components work correctly
- Tests verify edge cases (straight path, sharp turns, etc.)

### What Our Tests DON'T Verify:
❌ **Integration with Unity path geometry**
- Tests use synthetic/simplified paths
- Don't test with actual Unity oval track geometry
- Don't verify Unity is sending correct path curvature

❌ **Path curvature sign correctness**
- Tests assume curvature sign is correct
- Don't verify positive curvature = right turn, negative = left turn
- Don't test what happens when curvature is consistently positive

❌ **Ground truth mode activation**
- Tests don't verify Unity actually enables ground truth mode
- Don't test what happens if ground truth mode is not active

❌ **End-to-end path following**
- Tests don't simulate multi-frame path following
- Don't verify car actually follows the path over time
- Don't catch issues that only appear after many frames

❌ **Unity data correctness**
- Tests don't verify Unity is calculating/sending correct data
- Don't test what happens if Unity sends wrong curvature values

## Recommended Fixes

### Immediate Actions:
1. **Check path curvature in recording**
   - If consistently positive → fix Unity's curvature calculation
   - If all zeros → fix Unity's curvature calculation

2. **Check ground truth mode**
   - If never active → fix Unity ground truth mode activation
   - Verify `ground_truth_mode` is being sent correctly

3. **Add integration test**
   - Test with actual recording data
   - Verify steering matches expected path
   - Test multi-frame path following

### Long-term Improvements:
1. **Add integration tests using recording data**
   - Replay recordings and verify steering is correct
   - Test with problematic scenarios from recordings

2. **Add Unity data validation**
   - Verify path curvature sign is correct
   - Verify ground truth mode is active
   - Verify all required data is present

3. **Add end-to-end simulation test**
   - Simulate car following a path over multiple frames
   - Verify car stays on path
   - Catch circular motion issues

## How to Analyze Recording

Run this command (when h5py is available):
```bash
python3 tools/analyze_circular_drive.py data/recordings/latest.h5
```

Or manually check in Python:
```python
import h5py
import numpy as np

with h5py.File('data/recordings/latest.h5', 'r') as f:
    # Check path curvature
    curvatures = f['ground_truth/path_curvature'][:]
    print(f"Average curvature: {np.mean(curvatures):.6f} 1/m")
    
    # Check ground truth mode
    gt_mode = f['unity_feedback/ground_truth_mode_active'][:]
    print(f"GT mode active: {np.sum(gt_mode)} / {len(gt_mode)}")
    
    # Check steering
    steering = f['vehicle/steering_angle'][:]
    print(f"Average steering: {np.mean(steering):.2f} deg")
```

## Expected Findings

Based on the circular motion to the right, I expect to find:

1. **Path curvature is consistently positive** (most likely)
   - Unity is calculating positive curvature for the entire track
   - This causes constant right steering → circular motion

2. **OR ground truth mode is not active**
   - Unity is using physics mode instead of direct velocity control
   - Physics mode may not follow steering commands correctly

3. **OR steering sign is inverted**
   - Steering calculation produces wrong sign
   - Car steers away from path instead of toward it

