# Steering Calculation Tests

## Overview

These tests verify the exact steering calculation **before running Unity**, giving confidence that the math is correct.

## Test Files

### 1. `test_exact_steering_calculation.py`
**Purpose:** Test steering calculation logic without Unity

**What it tests:**
- Straight path steering (lateral error correction)
- Curved path steering (path curvature conversion)
- Coordinate system interpretation
- Mathematical correctness
- Steering clipping

**Run:**
```bash
python3 tests/test_exact_steering_calculation.py
```

**Expected:** All tests pass

### 2. `test_steering_from_recording.py`
**Purpose:** Test steering calculation using data from actual Unity recordings

**What it tests:**
- Recalculates steering from recorded inputs
- Compares with recorded calculated steering
- Verifies path curvature interpretation
- Verifies coordinate system

**Run:**
```bash
python3 tests/test_steering_from_recording.py [recording_file.h5]
```

**If no file provided:** Uses latest recording

**Expected:** Recalculated steering matches recorded steering

## Test Workflow

### Before Unity Run:
1. Run `test_exact_steering_calculation.py` - should all pass
2. If tests fail, fix the calculation logic

### After Unity Run:
1. Run `test_steering_from_recording.py` with the recording
2. Check if recalculated steering matches recorded
3. If not, investigate differences

### Tweaking Tests Based on Data:

If you find issues in recordings, you can:

1. **Add test cases** based on problematic scenarios:
   ```python
   def test_specific_issue_from_recording():
       # Test case based on actual recording data
       calc = ExactSteeringCalculator()
       steering, _ = calc.calculate_steering(
           path_curvature=0.02,  # From recording
           lane_center=1.5  # From recording
       )
       # Verify expected behavior
   ```

2. **Update test parameters** if you discover better values:
   ```python
   # If you find lateral_correction_gain should be different:
   calc = ExactSteeringCalculator(lateral_correction_gain=0.3)  # Updated value
   ```

3. **Add regression tests** for specific bugs:
   ```python
   def test_regression_bug_xyz():
       # Test that reproduces a specific bug
       # This ensures the bug doesn't come back
   ```

## What These Tests Verify

### ✅ Mathematical Correctness
- Steering calculation formula: `steering_angle = atan(wheelbase * curvature)`
- Lateral correction: `lateral_correction = gain * lane_center`
- Steering clipping: `steering ∈ [-1.0, 1.0]`

### ✅ Coordinate System
- `lane_center > 0` → Car LEFT of center → Steer RIGHT (positive)
- `lane_center < 0` → Car RIGHT of center → Steer LEFT (negative)
- `curvature > 0` → Right turn → Positive steering
- `curvature < 0` → Left turn → Negative steering

### ✅ Edge Cases
- Straight path (curvature = 0)
- Car at center (lane_center = 0)
- Very sharp turns (high curvature)
- Large lateral errors

## Integration with CI/CD

These tests can be run in CI/CD pipelines:

```bash
# Run all steering tests
python3 tests/test_exact_steering_calculation.py
python3 tests/test_steering_from_recording.py

# Or with pytest (if available)
pytest tests/test_exact_steering_calculation.py -v
```

## Troubleshooting

### Tests Fail
1. Check calculation logic in `ground_truth_follower.py`
2. Verify coordinate system interpretation
3. Check mathematical formulas

### Recording Test Shows Differences
1. Check if recording has the required data (path_curvature, lane_center)
2. Verify data was recorded correctly
3. Check for timing issues (steering applied at wrong time)

### All Tests Pass But Unity Still Wrong
1. Check Unity is applying steering correctly
2. Verify ground truth mode is active
3. Check coordinate system in Unity matches Python
4. Verify path curvature is being calculated correctly in Unity

