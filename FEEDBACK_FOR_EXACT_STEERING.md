# Feedback Needed to Verify Exact Steering Calculation

## Key Insight

Since we use **EXACT calculation** (no tuning, no guessing), we don't need to "correct" anything. But we **DO need to verify** it's working correctly!

## Essential Feedback to Record

### 1. **INPUTS to Our Calculation** (to verify we have the right data)

**Currently Recorded:**
- ✅ `ground_truth_lane_center_x` (lateral error)
- ✅ Car position, heading, speed

**MISSING (Critical!):**
- ❌ **`groundTruthPathCurvature`** - This is the KEY input to our exact calculation!
  - Formula: `steering_angle = atan(wheelbase * curvature)`
  - Without this, we can't verify our calculation is correct!

**Optional (for debugging):**
- `groundTruthDesiredHeading` - Useful to verify heading tracking

### 2. **OUTPUTS from Our Calculation** (to verify we computed correctly)

**Currently Recorded:**
- ✅ `control_command.steering` (what we sent to Unity)
- ✅ `vehicle_state.steering_angle` (what Unity actually applied)

**Missing (for verification):**
- Calculated steering angle in degrees: `steerInput * maxSteerAngle`
- Raw steering before smoothing: `raw_steering`
- Lateral correction term: `lateral_correction_gain * lane_center`

### 3. **RESULTS** (to verify it worked)

**Currently Recorded:**
- ✅ Car position over time (trajectory)
- ✅ Lateral error (`ground_truth_lane_center_x`)

**Can Compute (if we record inputs):**
- Lateral error over time (should stay near 0)
- Heading error over time (if we record desired heading)
- Path following accuracy (distance from path)

## What We Should Add to Recording

### VehicleState (add fields):
```python
ground_truth_path_curvature: float = 0.0  # Path curvature (1/meters) - CRITICAL!
ground_truth_desired_heading: float = 0.0  # Desired heading (degrees) - Optional
```

### ControlCommand (add fields for exact steering):
```python
calculated_steering_angle_deg: Optional[float] = None  # Steering angle in degrees
raw_steering: Optional[float] = None  # Before smoothing
lateral_correction: Optional[float] = None  # Lateral correction term
path_curvature_input: Optional[float] = None  # Path curvature used in calculation
```

## Why This Matters

### Without Path Curvature:
- ❌ Can't verify our calculation is correct
- ❌ Can't debug if steering is wrong
- ❌ Can't replay and analyze what happened

### With Path Curvature:
- ✅ Can verify: `steering_angle = atan(wheelbase * curvature)`
- ✅ Can debug: Compare calculated vs actual steering
- ✅ Can analyze: Why steering might be wrong (wrong curvature? wrong calculation?)

## Example Verification

If we record path curvature, we can verify:

```python
# Recorded data:
path_curvature = 0.02  # 1/meters (curved path)
calculated_steering = 0.5  # What we computed
actual_steering = 0.48  # What Unity applied

# Verification:
expected_steering_angle = atan(2.5 * 0.02) = 0.0499 rad = 2.86°
expected_steerInput = 2.86° / 30° = 0.095

# If calculated_steering ≈ expected_steerInput, calculation is correct!
# If not, we can debug why (wrong curvature? wrong formula? Unity issue?)
```

## Summary

**Critical Missing Data:**
1. **Path curvature** (`groundTruthPathCurvature`) - MUST record this!
2. **Desired heading** (`groundTruthDesiredHeading`) - Nice to have

**Nice to Have:**
- Calculated steering angle breakdown (raw, smoothed, lateral correction)
- This helps debug if something is wrong

**Already Have:**
- Car position, heading, speed
- Lateral error
- Control commands sent
- Actual steering applied

## Action Items

1. **Add `groundTruthPathCurvature` to VehicleState recording** (CRITICAL)
2. **Add `groundTruthDesiredHeading` to VehicleState recording** (Optional)
3. **Add calculated steering breakdown to ControlCommand** (Optional, for debugging)

With this data, we can fully verify our exact steering calculation is working correctly!

