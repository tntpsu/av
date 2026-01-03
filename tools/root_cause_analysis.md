# Root Cause Analysis: Car Position vs Ground Truth

## Analysis Framework

### Potential Root Causes

1. **Steering Calculation Issues**
   - Wrong sign in steering calculation
   - Incorrect path curvature interpretation
   - Wrong coordinate system interpretation
   - Mathematical error in bicycle model

2. **Unity Application Issues**
   - Steering not being applied correctly
   - Ground truth mode not active
   - Wrong steering direction in Unity
   - Timing issues (steering applied too late/early)

3. **Data Recording Issues**
   - Ground truth data not being recorded
   - Path curvature not being sent from Unity
   - Coordinate system mismatch between Unity and Python

4. **Path Geometry Issues**
   - Wrong path curvature calculation in Unity
   - Incorrect lane center calculation
   - Coordinate system mismatch

## Test Plan (Pre-Unity)

### Test 1: Steering Calculation Correctness
**Purpose:** Verify steering calculation is mathematically correct

**Test:** `test_exact_steering_calculation.py`
- Tests all steering scenarios
- Verifies coordinate system interpretation
- Validates mathematical correctness

**Run:** `python3 tests/test_exact_steering_calculation.py`

### Test 2: Coordinate System Verification
**Purpose:** Verify coordinate system interpretation matches Unity

**Test:** Verify lane_center sign interpretation
- `lane_center > 0` → Car LEFT of center → Steer RIGHT (positive)
- `lane_center < 0` → Car RIGHT of center → Steer LEFT (negative)

### Test 3: Path Curvature to Steering Conversion
**Purpose:** Verify bicycle model calculation

**Formula:** `steering_angle = atan(wheelbase * curvature)`
**Test:** Verify for various curvature values

### Test 4: Lateral Correction
**Purpose:** Verify lateral error correction

**Formula:** `lateral_correction = gain * lane_center`
**Test:** Verify correction direction and magnitude

## Analysis Checklist

When analyzing a recording:

1. **Check Ground Truth Data**
   - [ ] Is `ground_truth_lane_center_x` being recorded?
   - [ ] Is `ground_truth_path_curvature` being recorded?
   - [ ] Are values all zeros? (indicates missing data)

2. **Check Steering Calculation**
   - [ ] Is `calculated_steering_angle_deg` being recorded?
   - [ ] Does it match expected: `atan(wheelbase * curvature)`?
   - [ ] Is `raw_steering` correct?

3. **Check Actual Steering**
   - [ ] Is `vehicle_state.steering_angle` matching calculated?
   - [ ] Is steering being applied in Unity?

4. **Check Position Trajectory**
   - [ ] Is car position changing?
   - [ ] Are there large position jumps? (indicates issues)
   - [ ] Is car following expected path?

5. **Check Lateral Error**
   - [ ] Is `lane_center` decreasing over time? (good)
   - [ ] Is `lane_center` increasing over time? (bad - wrong steering)

## Common Issues and Fixes

### Issue 1: All Ground Truth Values are Zero
**Symptom:** `ground_truth_lane_center_x` and `ground_truth_path_curvature` are all 0.0

**Root Cause:** Unity not sending ground truth data

**Fix:**
- Check `GroundTruthReporter` is enabled in Unity
- Check `AVBridge` is reading ground truth data
- Check bridge server is receiving data

### Issue 2: Steering Calculation Wrong
**Symptom:** Calculated steering doesn't match expected formula

**Root Cause:** Mathematical error in calculation

**Fix:**
- Run `test_exact_steering_calculation.py` to verify
- Check coordinate system interpretation
- Verify bicycle model formula

### Issue 3: Car Not Following Path
**Symptom:** Car position diverges from expected path

**Root Cause:** Steering direction wrong or not being applied

**Fix:**
- Check steering sign in Unity
- Verify ground truth mode is active
- Check steering is being sent to Unity

### Issue 4: Large Position Jumps
**Symptom:** Car position changes by >1m per frame

**Root Cause:** Physics issues or teleportation

**Fix:**
- Check Unity physics settings
- Verify ground truth mode is using direct velocity control
- Check for teleportation in Unity

## Next Steps

1. Run pre-Unity tests to verify calculation correctness
2. Analyze latest recording to identify specific issues
3. Create targeted tests based on recorded data
4. Fix identified issues
5. Re-run tests and verify fixes

