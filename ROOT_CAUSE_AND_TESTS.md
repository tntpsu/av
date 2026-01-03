# Root Cause Analysis & Pre-Unity Tests

## Summary

I've created comprehensive tests and analysis tools to verify steering calculation **before running Unity**. Here's what's been done:

## ✅ Created Tests

### 1. `tests/test_exact_steering_calculation.py`
**Pure Python test** (no Unity needed) that verifies:
- ✅ Steering calculation is mathematically correct
- ✅ Coordinate system interpretation is correct  
- ✅ Path curvature to steering conversion works
- ✅ Lateral correction works correctly
- ✅ Edge cases handled (straight path, sharp turns, etc.)

**Run:** `python3 tests/test_exact_steering_calculation.py`

### 2. `tests/test_steering_from_recording.py`
**Uses actual recording data** to verify:
- ✅ Recalculates steering from recorded inputs
- ✅ Compares with recorded calculated steering
- ✅ Identifies discrepancies

**Run:** `python3 tests/test_steering_from_recording.py [recording_file.h5]`

### 3. `tools/analyze_recording.py`
**Analyzes recordings** to identify issues:
- ✅ Checks if ground truth data is recorded
- ✅ Verifies steering calculation correctness
- ✅ Identifies position jumps and other issues

**Run:** `python3 tools/analyze_recording.py [recording_file.h5]`

## 🔍 Root Cause Analysis Framework

### Potential Issues (in order of likelihood):

#### 1. **Missing Ground Truth Data** (Most Likely)
**Symptom:** `ground_truth_path_curvature` is all zeros in recording

**Root Cause:** Unity not sending path curvature data

**Fix:**
- Check `GroundTruthReporter.GetPathCurvature()` is being called
- Check `AVBridge` is reading and sending curvature
- Verify bridge server is receiving data

**Test:** Run `analyze_recording.py` - it will flag if curvature is all zeros

#### 2. **Steering Sign Wrong**
**Symptom:** Car steers away from path instead of toward it

**Root Cause:** Wrong sign in steering calculation or Unity application

**Fix:**
- Run `test_exact_steering_calculation.py` to verify calculation
- Check Unity steering direction
- Verify coordinate system interpretation

**Test:** `test_exact_steering_calculation.py` tests coordinate system

#### 3. **Path Curvature Calculation Wrong**
**Symptom:** Steering doesn't match expected for curved paths

**Root Cause:** Wrong curvature calculation in Unity

**Fix:**
- Check `GroundTruthReporter.GetPathCurvature()` implementation
- Verify curvature sign (positive = right turn, negative = left turn)
- Check curvature magnitude (should be 1/radius in meters)

**Test:** `test_steering_from_recording.py` verifies curvature interpretation

#### 4. **Ground Truth Mode Not Active**
**Symptom:** Car behaves erratically, not following path

**Root Cause:** Unity not in ground truth mode

**Fix:**
- Check Unity console for "Ground Truth Mode ENABLED" logs
- Verify `groundTruthMode` is being set in control commands
- Check `CarController.ApplyGroundTruthControls()` is being called

## 📊 Analysis Checklist

When analyzing a recording, check:

### Data Recording
- [ ] `ground_truth_path_curvature` exists and has non-zero values
- [ ] `ground_truth_lane_center_x` exists and varies
- [ ] `control/calculated_steering_angle_deg` is recorded
- [ ] `control/raw_steering` is recorded
- [ ] `control/path_curvature_input` is recorded

### Calculation Correctness
- [ ] Calculated steering matches: `atan(wheelbase * curvature) / maxSteerAngle`
- [ ] Lateral correction is: `gain * lane_center`
- [ ] Steering is clipped to [-1.0, 1.0]

### Coordinate System
- [ ] `lane_center > 0` → steering > 0 (car left → steer right)
- [ ] `lane_center < 0` → steering < 0 (car right → steer left)
- [ ] `curvature > 0` → steering > 0 (right turn)
- [ ] `curvature < 0` → steering < 0 (left turn)

### Unity Application
- [ ] Actual steering angle matches calculated
- [ ] Car position changes smoothly (no large jumps)
- [ ] Lateral error decreases over time (car moving toward center)

## 🧪 Test Workflow

### Before Unity Run:
```bash
# 1. Run calculation tests (should all pass)
python3 tests/test_exact_steering_calculation.py

# 2. If tests fail, fix calculation logic
# 3. Re-run tests until all pass
```

### After Unity Run:
```bash
# 1. Analyze the recording
python3 tools/analyze_recording.py data/recordings/latest.h5

# 2. Test steering from recording
python3 tests/test_steering_from_recording.py data/recordings/latest.h5

# 3. If issues found, add regression tests
# 4. Fix issues and re-test
```

## 🔧 Tweaking Tests Based on Data

### Example: Add Test Case from Problematic Recording

If you find a specific scenario that fails:

```python
def test_regression_from_recording():
    """Test case based on actual problematic recording data."""
    calc = ExactSteeringCalculator()
    
    # Use actual values from recording
    curvature = 0.015  # From recording frame 123
    lane_center = 1.2  # From recording frame 123
    
    steering, metadata = calc.calculate_steering(curvature, lane_center)
    
    # Verify expected behavior
    assert steering > 0.0, "Should steer right for this scenario"
    assert abs(steering - expected_value) < 0.01
```

### Example: Update Parameters Based on Data

If you discover better parameter values:

```python
# In test file, update default parameters:
calc = ExactSteeringCalculator(
    lateral_correction_gain=0.3,  # Updated based on recording analysis
    steering_deadband=0.01  # Updated based on recording analysis
)
```

## 📝 Next Steps

1. **Run pre-Unity tests** to verify calculation is correct
2. **Analyze latest recording** to identify specific issues
3. **Fix identified issues** based on analysis
4. **Add regression tests** for any bugs found
5. **Re-run tests** to verify fixes

## 🎯 Expected Results

### If Tests Pass:
- ✅ Steering calculation is mathematically correct
- ✅ Coordinate system is correct
- ✅ Path curvature conversion works

### If Tests Fail:
- ❌ Calculation logic needs fixing
- ❌ Coordinate system interpretation wrong
- ❌ Mathematical formula incorrect

### If Recording Analysis Shows Issues:
- Check Unity is sending ground truth data
- Verify Unity is applying steering correctly
- Check for timing or synchronization issues

