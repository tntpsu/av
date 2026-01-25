# Test Suite Documentation

Comprehensive test suite for the AV Stack covering control, trajectory, perception, and integration scenarios.

## Quick Start

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_control.py -v

# Run with coverage
pytest tests/ --cov=perception --cov=trajectory --cov=control --cov-report=term-missing

# Run tests and drop into debugger on failure
pytest tests/ --pdb
```

## Test Categories

### Control System Tests

#### `test_control.py`
**Purpose:** Core control system functionality

**What it tests:**
- PID controller basic functionality
- Steering command calculation
- Error handling
- Control command limits

**Run:**
```bash
pytest tests/test_control.py -v
```

#### `test_control_lateral_error_performance.py` ⭐ **NEW**
**Purpose:** Control system performance validation (catches tuning issues before Unity testing)

**What it tests:**
- Lateral error RMSE on straight roads (< 0.4m acceptable, < 0.2m good)
- Lateral error RMSE on curved roads (< 0.4m acceptable)
- Step response convergence (< 2 seconds, < 0.2m final error)

**Why it's important:**
- Uses same RMSE thresholds as Path Tracking metrics in debug visualizer
- Catches control tuning issues **before** Unity testing
- Would have caught the 0.565m RMSE issue that showed up in Unity

**Run:**
```bash
pytest tests/test_control_lateral_error_performance.py -v
```

**If test fails:**
- See `docs/CONTROL_TUNING_RECOMMENDATIONS.md` for tuning guidance
- Typically indicates need to increase `kp` (proportional gain)

#### `test_steering_controller.py`
**Purpose:** Steering controller specific tests

**What it tests:**
- Steering direction correctness
- Steering rate limiting
- Steering saturation
- Feedforward vs feedback balance

**Run:**
```bash
pytest tests/test_steering_controller.py -v
```

#### `test_control_straight_road.py`
**Purpose:** Control behavior on straight roads

**What it tests:**
- Lateral error correction
- Heading error correction
- Deadband behavior
- Convergence on straight paths

**Run:**
```bash
pytest tests/test_control_straight_road.py -v
```

#### `test_constant_steering_on_curves.py`
**Purpose:** Verify constant steering is maintained on curves

**What it tests:**
- Steering doesn't reset during curves
- Integral is preserved on curves
- Feedforward provides base steering
- Smooth transitions

**Run:**
```bash
pytest tests/test_constant_steering_on_curves.py -v
```

#### `test_pid_integral_accumulation.py` / `test_pid_integral_accumulation_realistic.py`
**Purpose:** Test PID integral windup prevention

**What it tests:**
- Integral accumulation over time
- Integral reset mechanisms
- Integral decay
- Windup prevention

**Run:**
```bash
pytest tests/test_pid_integral_accumulation.py -v
pytest tests/test_pid_integral_accumulation_realistic.py -v
```

#### `test_oscillation_detection.py`
**Purpose:** Detect and analyze oscillation patterns

**What it tests:**
- Oscillation frequency detection
- Error sign changes
- Steering oscillation
- Root cause identification

**Run:**
```bash
pytest tests/test_oscillation_detection.py -v
```

#### `test_heading_lateral_conflicts.py`
**Purpose:** Test handling of heading vs lateral error conflicts

**What it tests:**
- Conflict detection
- Conflict resolution (prioritize lateral error)
- Steering direction correctness during conflicts
- Error weighting

**Run:**
```bash
pytest tests/test_heading_lateral_conflicts.py -v
```

#### `test_error_clip_limitation.py`
**Purpose:** Test error clipping to prevent overreaction

**What it tests:**
- Error clipping limits
- Large error handling
- Clipping preserves sign
- Clipping doesn't affect small errors

**Run:**
```bash
pytest tests/test_error_clip_limitation.py -v
```

#### `test_exact_steering_calculation.py`
**Purpose:** Test steering calculation logic without Unity

**What it tests:**
- Straight path steering (lateral error correction)
- Curved path steering (path curvature conversion)
- Coordinate system interpretation
- Mathematical correctness
- Steering clipping

**Run:**
```bash
pytest tests/test_exact_steering_calculation.py -v
```

#### `test_steering_from_recording.py`
**Purpose:** Test steering calculation using data from actual Unity recordings

**What it tests:**
- Recalculates steering from recorded inputs
- Compares with recorded calculated steering
- Verifies path curvature interpretation
- Verifies coordinate system

**Run:**
```bash
pytest tests/test_steering_from_recording.py [recording_file.h5]
```

#### `test_steering_logic_simple.py`
**Purpose:** Simple steering logic validation

**What it tests:**
- Basic steering direction
- Steering magnitude
- Edge cases

**Run:**
```bash
pytest tests/test_steering_logic_simple.py -v
```

### Trajectory Planning Tests

#### `test_trajectory.py`
**Purpose:** Core trajectory planning functionality

**What it tests:**
- Trajectory generation
- Reference point calculation
- Lane center calculation
- Trajectory smoothing

**Run:**
```bash
pytest tests/test_trajectory.py -v
```

#### `test_trajectory_straight_road.py`
**Purpose:** Trajectory behavior on straight roads

**What it tests:**
- Reference point on straight paths
- Lane centering
- Smoothing behavior
- Bias correction

**Run:**
```bash
pytest tests/test_trajectory_straight_road.py -v
```

#### `test_trajectory_edge_cases.py`
**Purpose:** Edge case handling in trajectory planning

**What it tests:**
- Single lane detection
- No lane detection
- Invalid lane data
- Extreme lane positions

**Run:**
```bash
pytest tests/test_trajectory_edge_cases.py -v
```

#### `test_trajectory_reference_point_offset.py`
**Purpose:** Test reference point offset calculations

**What it tests:**
- Reference point accuracy
- Offset from lane center
- Lookahead distance
- Coordinate conversion

**Run:**
```bash
pytest tests/test_trajectory_reference_point_offset.py -v
```

#### `test_trajectory_heading_fix_with_real_data.py`
**Purpose:** Test trajectory heading fix using real recording data

**What it tests:**
- Heading calculation correctness
- Path curvature interpretation
- Real-world scenario validation

**Run:**
```bash
pytest tests/test_trajectory_heading_fix_with_real_data.py -v
```

#### `test_reference_point_stability.py`
**Purpose:** Test reference point stability over time

**What it tests:**
- Reference point smoothing
- Stability during lane detection failures
- Temporal consistency
- Noise reduction

**Run:**
```bash
pytest tests/test_reference_point_stability.py -v
```

#### `test_reference_point_freeze_prevention.py`
**Purpose:** Prevent reference point from freezing

**What it tests:**
- Freeze detection
- Recovery mechanisms
- State reset on failures
- Temporal filtering

**Run:**
```bash
pytest tests/test_reference_point_freeze_prevention.py -v
```

#### `test_trajectory_state_reuse.py`
**Purpose:** Test trajectory state reuse and persistence

**What it tests:**
- State preservation
- State reset conditions
- State consistency
- Memory management

**Run:**
```bash
pytest tests/test_trajectory_state_reuse.py -v
```

#### `test_adaptive_bias_correction.py`
**Purpose:** Test adaptive bias correction in trajectory planning

**What it tests:**
- Bias detection
- Bias correction application
- Correction persistence
- Correction thresholds

**Run:**
```bash
pytest tests/test_adaptive_bias_correction.py -v
```

#### `test_smoothing_reset_on_failures.py`
**Purpose:** Test smoothing reset when lane detection fails

**What it tests:**
- Smoothing state reset
- Failure detection
- Recovery behavior
- State consistency

**Run:**
```bash
pytest tests/test_smoothing_reset_on_failures.py -v
```

### Perception Tests

#### `test_perception_straight_road.py`
**Purpose:** Perception behavior on straight roads

**What it tests:**
- Lane detection accuracy
- Lane position calculation
- Coordinate conversion
- Detection consistency

**Run:**
```bash
pytest tests/test_perception_straight_road.py -v
```

#### `test_perception_ground_truth.py`
**Purpose:** Compare perception output with ground truth

**What it tests:**
- Detection accuracy vs ground truth
- Position errors
- Width calculation
- Coordinate system alignment

**Run:**
```bash
pytest tests/test_perception_ground_truth.py -v
```

### Coordinate System Tests

#### `test_coordinate_system.py`
**Purpose:** Core coordinate system validation

**What it tests:**
- Image to vehicle coordinate conversion
- Coordinate system consistency
- Sign conventions
- Unit conversions

**Run:**
```bash
pytest tests/test_coordinate_system.py -v
```

#### `test_coordinate_system_validation.py`
**Purpose:** Validate coordinate system assumptions

**What it tests:**
- Coordinate transformation correctness
- Frame consistency
- Unit consistency
- Sign conventions

**Run:**
```bash
pytest tests/test_coordinate_system_validation.py -v
```

#### `test_coordinate_conversion_assumptions.py`
**Purpose:** Test assumptions in coordinate conversion

**What it tests:**
- Conversion formulas
- Distance calculations
- FOV assumptions
- Camera parameters

**Run:**
```bash
pytest tests/test_coordinate_conversion_assumptions.py -v
```

#### `test_camera_offset.py`
**Purpose:** Test camera offset correction

**What it tests:**
- Lateral offset correction
- Longitudinal offset correction
- Offset impact on detection
- Calibration

**Run:**
```bash
pytest tests/test_camera_offset.py -v
```

### Ground Truth Tests

#### `test_ground_truth_steering.py`
**Purpose:** Test ground truth steering calculation

**What it tests:**
- Ground truth path following
- Steering from ground truth
- Coordinate system alignment
- Path curvature calculation

**Run:**
```bash
pytest tests/test_ground_truth_steering.py -v
```

#### `test_ground_truth_steering_synthetic.py`
**Purpose:** Synthetic ground truth steering tests

**What it tests:**
- Steering calculation from synthetic paths
- Mathematical correctness
- Edge cases
- Path following accuracy

**Run:**
```bash
pytest tests/test_ground_truth_steering_synthetic.py -v
```

#### `test_ground_truth_coordinate_system_fix.py`
**Purpose:** Test ground truth coordinate system fixes

**What it tests:**
- Coordinate system alignment
- Ground truth accuracy
- Frame transformations
- Validation

**Run:**
```bash
pytest tests/test_ground_truth_coordinate_system_fix.py -v
```

#### `test_ground_truth_lane_calculation.py`
**Purpose:** Test ground truth lane position calculation

**What it tests:**
- Lane center calculation
- Lane width calculation
- Position accuracy
- Coordinate conversion

**Run:**
```bash
pytest tests/test_ground_truth_lane_calculation.py -v
```

#### `test_replay_ground_truth.py`
**Purpose:** Test replaying ground truth data

**What it tests:**
- Data replay functionality
- Ground truth data loading
- Data consistency
- Replay accuracy

**Run:**
```bash
pytest tests/test_replay_ground_truth.py -v
```

### Integration Tests

#### `test_integration.py`
**Purpose:** Basic integration tests

**What it tests:**
- Component integration
- Data flow
- Error propagation
- System stability

**Run:**
```bash
pytest tests/test_integration.py -v
```

#### `test_av_stack_integration.py`
**Purpose:** Full AV stack integration tests

**What it tests:**
- End-to-end pipeline
- Component interactions
- Data recording
- System behavior

**Run:**
```bash
pytest tests/test_av_stack_integration.py -v
```

#### `test_av_stack_startup.py`
**Purpose:** Test AV stack startup and initialization

**What it tests:**
- Component initialization
- Configuration loading
- Bridge connection
- Error handling

**Run:**
```bash
pytest tests/test_av_stack_startup.py -v
```

#### `test_integration_scenarios.py`
**Purpose:** Real-world integration scenarios

**What it tests:**
- Common driving scenarios
- Edge cases
- Failure modes
- Recovery behavior

**Run:**
```bash
pytest tests/test_integration_scenarios.py -v
```

#### `test_system_stability.py`
**Purpose:** Test system stability over time

**What it tests:**
- Long-term stability
- Memory leaks
- State consistency
- Performance degradation

**Run:**
```bash
pytest tests/test_system_stability.py -v
```

### System Tests

#### `test_safety_mechanisms.py`
**Purpose:** Test safety mechanisms

**What it tests:**
- Emergency stops
- Bounds checking
- Error thresholds
- Safety limits

**Run:**
```bash
pytest tests/test_safety_mechanisms.py -v
```

#### `test_divergence_prevention.py`
**Purpose:** Test divergence prevention mechanisms

**What it tests:**
- Error divergence detection
- Divergence prevention
- Recovery mechanisms
- Stability

**Run:**
```bash
pytest tests/test_divergence_prevention.py -v
```

#### `test_regression.py`
**Purpose:** Regression tests for fixed bugs

**What it tests:**
- Previously fixed bugs
- Known issues
- Bug reproduction
- Fix validation

**Run:**
```bash
pytest tests/test_regression.py -v
```

#### `test_real_data_validation.py`
**Purpose:** Validate against real recorded data

**What it tests:**
- Real data compatibility
- Data format validation
- Data consistency
- Recording accuracy

**Run:**
```bash
pytest tests/test_real_data_validation.py -v
```

### Specialized Tests

#### `test_heading_effect.py`
**Purpose:** Test heading effect on perception and control

**What it tests:**
- Heading impact on detection
- Heading error calculation
- Coordinate transformation
- Control response

**Run:**
```bash
pytest tests/test_heading_effect.py -v
```

#### `test_oval_path_calculation.py`
**Purpose:** Test path calculation on oval tracks

**What it tests:**
- Curved path handling
- Path curvature calculation
- Steering on curves
- Trajectory on curves

**Run:**
```bash
pytest tests/test_oval_path_calculation.py -v
```

#### `test_visualizer_line_alignment.py`
**Purpose:** Test visualizer line alignment

**What it tests:**
- Visualization accuracy
- Line rendering
- Overlay alignment
- Coordinate conversion

**Run:**
```bash
pytest tests/test_visualizer_line_alignment.py -v
```

#### `test_reproduce_red_green_misalignment.py`
**Purpose:** Reproduce and test red/green line misalignment bug

**What it tests:**
- Bug reproduction
- Fix validation
- Edge case handling
- Regression prevention

**Run:**
```bash
pytest tests/test_reproduce_red_green_misalignment.py -v
```

#### `test_config_integration.py`
**Purpose:** Test configuration system integration

**What it tests:**
- Config loading
- Config validation
- Config application
- Default values

**Run:**
```bash
pytest tests/test_config_integration.py -v
```

## Test Coverage

### Current Coverage Areas

✅ **Well Covered:**
- Control system (PID, steering, integral management)
- Trajectory planning (reference point, smoothing, bias correction)
- Coordinate system conversions
- Ground truth following
- Integration scenarios

⚠️ **Partially Covered:**
- Perception edge cases (some scenarios missing)
- Error recovery mechanisms
- Performance under load
- Multi-scenario testing

❌ **Testing Gaps:**

1. **Perception:**
   - No lane detection (both lanes fail)
   - Extreme lighting conditions
   - Occluded lanes
   - Lane marking quality variations
   - Camera calibration errors

2. **Trajectory:**
   - Rapid lane changes
   - Merging scenarios
   - Construction zones
   - Multiple lane roads

3. **Control:**
   - High-speed scenarios (>15 m/s)
   - Emergency maneuvers
   - Slippery road conditions
   - Wind disturbances

4. **Integration:**
   - Long-duration runs (>5 minutes)
   - Multiple consecutive runs
   - System recovery after failures
   - Network interruption handling

5. **Performance:**
   - Memory usage over time
   - CPU usage profiling
   - Frame rate consistency
   - Latency measurements

## Running Tests

### All Tests
```bash
pytest tests/
```

### By Category
```bash
# Control tests
pytest tests/test_control*.py tests/test_steering*.py tests/test_pid*.py

# Trajectory tests
pytest tests/test_trajectory*.py tests/test_reference*.py

# Perception tests
pytest tests/test_perception*.py

# Integration tests
pytest tests/test_integration*.py tests/test_av_stack*.py
```

### With Coverage
```bash
pytest tests/ --cov=perception --cov=trajectory --cov=control --cov-report=html
```

### Specific Test
```bash
pytest tests/test_control.py::test_steering_direction -v
```

## Writing New Tests

### Test Structure
```python
import pytest
import numpy as np
from control.pid_controller import LateralController

def test_feature_name():
    """Test description."""
    # Arrange
    controller = LateralController()
    
    # Act
    result = controller.compute_steering(...)
    
    # Assert
    assert result == expected_value
```

### Test Requirements
- **Name**: `test_*.py` files, `test_*` functions
- **Isolation**: Tests should be independent
- **Deterministic**: No random behavior (use fixed seeds)
- **Fast**: Unit tests should run quickly
- **Clear**: Test names should describe what they test

### Test Data
- Use synthetic data for unit tests
- Use real recordings for integration tests
- Mock external dependencies (Unity, file I/O)

## Continuous Integration

Tests should be run:
- Before committing code
- In CI/CD pipeline
- Before merging PRs
- After major refactoring

## Troubleshooting

### Tests Fail
1. Check test output for error messages
2. Run with `-v` for verbose output
3. Run with `--pdb` to debug
4. Check test data and assumptions

### Flaky Tests
1. Identify source of non-determinism
2. Use fixed random seeds
3. Mock time-dependent behavior
4. Isolate test dependencies

### Slow Tests
1. Profile tests to find bottlenecks
2. Use fixtures for expensive setup
3. Mock slow operations
4. Consider splitting into unit/integration tests

