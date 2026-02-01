# AV Stack Configuration Guide

## Overview

All tunable parameters for the AV stack are now configurable via `config/av_stack_config.yaml`. This allows you to tune the system without modifying code.

## Why Does the Car Steer When Centered?

**Short answer:** If the car is perfectly centered and straight, steering **should be 0**. Small steering corrections are normal due to:

1. **Perception noise** - Lane detection isn't perfect, small errors accumulate
2. **Coordinate conversion** - Small errors in pixel-to-meter conversion
3. **PID controller** - Responds to tiny errors (but deadband prevents over-correction)
4. **Reference point updates** - Reference point changes slightly each frame

**What's normal:**
- Steering varies slightly (±0.05) - **OK** (responding to small errors)
- Steering is 0 most of the time - **GOOD** (deadband working)
- Steering oscillates constantly - **BAD** (gains too high)
- Steering is constant (not 0) - **BAD** (systematic bias or PID not adapting)

## Configuration File

Location: `config/av_stack_config.yaml`

### Control Parameters

#### Lateral (Steering) Control

```yaml
control:
  lateral:
    kp: 1.0              # Proportional gain (tuned to prevent overcompensation)
    ki: 0.002            # Integral gain (tuned via parameter sweep)
    kd: 0.5              # Derivative gain (provides damping)
    max_steering: 1.0    # Maximum steering angle (-1.0 to 1.0, full authority)
    deadband: 0.01       # Deadband in radians (~0.57°)
    heading_weight: 0.6  # Weight for heading error (prioritizes heading correction)
    lateral_weight: 0.4  # Weight for lateral error
    error_clip: 1.57     # Maximum error in radians (π/2 = 90 degrees)
    integral_limit: 0.10 # Maximum integral term (prevents windup)
    control_mode: pid    # Lateral controller: pid|stanley (pid default)
    stanley_k: 1.5  # Stanley cross-track gain
    stanley_soft_speed: 2.0  # Soft speed term to avoid high gain at low speed
    stanley_heading_weight: 1.0  # Heading error weight in Stanley mode
    curve_feedforward_gain: 1.0  # Base feedforward steering gain
    curve_feedforward_threshold: 0.012  # Curvature threshold for feedforward (1/m)
    curve_feedforward_gain_min: 1.05  # Gain at low curvature
    curve_feedforward_gain_max: 1.3  # Gain at high curvature
    curve_feedforward_curvature_min: 0.003  # Start scheduling (1/m)
    curve_feedforward_curvature_max: 0.015  # End scheduling (1/m)
    curve_feedforward_curvature_clamp: 0.025  # Clamp curvature input (1/m)
    steering_rate_curvature_min: 0.003  # Begin rate scaling (1/m)
    steering_rate_curvature_max: 0.015  # Max rate scaling (1/m)
    steering_rate_scale_min: 0.9  # Minimum rate scale on sharp curves
    speed_gain_min_speed: 4.0  # Speed (m/s) below which no extra gain is applied
    speed_gain_max_speed: 10.0  # Speed (m/s) where max gain is reached
    speed_gain_min: 1.0  # Gain at/below min speed
    speed_gain_max: 1.6  # Gain at/above max speed
    speed_gain_curvature_min: 0.002  # Full speed gain below this curvature (1/m)
    speed_gain_curvature_max: 0.008  # Fade speed gain out by this curvature (1/m)
    feedback_gain_min: 1.1  # Feedback gain on higher curvature
    feedback_gain_max: 1.4  # Feedback gain on gentle curves
    feedback_gain_curvature_min: 0.003  # Full boost below this curvature (1/m)
    feedback_gain_curvature_max: 0.03  # No boost above this curvature (1/m)
```

**Tuning Guide:**
- **kp too high**: Oscillation, overcorrection
- **kp too low**: Slow response, drifts
- **deadband too high**: Car drifts before correcting
- **deadband too low**: Constant tiny corrections (wasteful)
- **lateral_weight**: Higher = more responsive to lateral offset

#### Longitudinal (Speed) Control

```yaml
control:
  longitudinal:
    kp: 0.25             # Proportional gain (faster speed tracking)
    ki: 0.02             # Integral gain (steady-state error)
    kd: 0.01             # Derivative gain (damping)
    max_accel: 2.5       # Max accel (m/s^2), aligned with speed planner
    max_decel: 3.0       # Max decel (m/s^2), aligned with speed planner
    max_jerk: 2.0        # Max jerk (m/s^3), aligned with speed planner
    accel_feedforward_gain: 1.0  # Scale planner accel feedforward contribution
    decel_feedforward_gain: 1.0  # Scale planner decel feedforward contribution
    target_speed: 8.0    # Target speed in m/s
    max_speed: 10.0      # Maximum speed in m/s
    speed_smoothing: 0.3 # Smoothing factor (0-1)
    speed_deadband: 0.1  # Speed error deadband in m/s
    throttle_rate_limit: 0.12 # Max throttle change per frame
    brake_rate_limit: 0.15    # Max brake change per frame
    throttle_smoothing_alpha: 0.6 # Low-pass filter on throttle command
    min_throttle_when_accel: 0.1 # Floor throttle when below target
    brake_aggression: 2.0 # Brake PID gain multiplier
```

### Trajectory Planning

```yaml
trajectory:
  lookahead_distance: 20.0  # How far ahead to plan in meters
  point_spacing: 1.0        # Spacing between trajectory points in meters
  target_speed: 8.0         # Target speed for trajectory in m/s
  reference_lookahead: 8.0  # Lookahead distance for reference point in meters
  dynamic_reference_lookahead: true  # Scale reference lookahead by speed/curvature
  reference_lookahead_min: 3.0  # Minimum lookahead (meters)
  reference_lookahead_scale_min: 0.55  # Minimum scale when shortening
  reference_lookahead_speed_min: 4.0  # Start shortening above this speed (m/s)
  reference_lookahead_speed_max: 10.0  # Max shortening at/above this speed (m/s)
  reference_lookahead_curvature_min: 0.002  # Full shortening below this curvature (1/m)
  reference_lookahead_curvature_max: 0.015  # No shortening above this curvature (1/m)
  use_vehicle_frame_lookahead_ref: true  # Use Unity vehicle-frame lookahead offset for ref_x
  vehicle_frame_lookahead_scale: 1.0  # Scale ref_x when curvature sign is unknown
  vehicle_frame_lookahead_scale_left: 1.0  # Scale ref_x on left turns (curv > 0)
  vehicle_frame_lookahead_scale_right: 1.05  # Scale ref_x on right turns (curv < 0)
  max_lateral_accel: 1.3  # Curve speed cap (m/s^2)
  min_curve_speed: 3.2  # Minimum allowed speed on curves (m/s)
  speed_limit_slew_rate: 1.0  # Max change in speed limit per second (m/s^2)
  speed_limit_preview_distance: 25.0  # Preview distance ahead for upcoming limits (m)
  speed_limit_preview_decel: 1.0  # Comfortable decel used for preview cap (m/s^2)
  speed_planner:
    enabled: true
    max_accel: 2.5          # Max acceleration (m/s^2)
    max_decel: 3.0          # Max deceleration (m/s^2)
    max_jerk: 2.0           # Max jerk (m/s^3)
    min_speed: 0.0
    launch_speed_floor: 2.0
    launch_speed_floor_threshold: 1.0
    reset_gap_seconds: 0.5
    sync_speed_threshold: 50.0
  image_width: 640         # Camera image width in pixels
  image_height: 480        # Camera image height in pixels
  camera_fov: 110.0        # Camera field of view in degrees (horizontal FOV)
  camera_height: 1.2       # Camera height above ground in meters
  camera_offset_x: 0.0     # Lateral offset of camera from vehicle center (meters)
  distance_scaling_factor: 0.875  # Distance scaling to account for camera pitch
  bias_correction_threshold: 10.0  # Pixels - auto-correct bias
  reference_smoothing: 0.80  # Exponential smoothing for reference point (0.0-1.0)
  lane_smoothing_alpha: 0.8  # Exponential smoothing for lane coefficients (0.0-1.0)
```

### Safety Limits

```yaml
safety:
  max_speed: 10.0           # Hard speed limit in m/s
  emergency_brake_threshold: 1.5  # Emergency brake if speed > max_speed * this
  speed_prevention_threshold: 0.70  # Start preventing acceleration at this fraction of max_speed
  speed_prevention_brake_threshold: 0.85  # Apply light brake at this fraction of max_speed
  speed_prevention_brake_amount: 0.4  # Light brake amount when preventing speed
  lane_width: 7.0           # Lane width in meters
  car_width: 1.85           # Car width in meters
  allowed_outside_lane: 1.0  # Allowed distance outside lane before emergency stop (meters)
```

## Usage

### Default Configuration

The system automatically loads `config/av_stack_config.yaml` if it exists. If not found, uses hardcoded defaults.

### Custom Configuration

```bash
python3 av_stack.py --config /path/to/custom_config.yaml
```

### Example: Reduce Steering Sensitivity

Edit `config/av_stack_config.yaml`:

```yaml
control:
  lateral:
    kp: 0.8              # Reduced from 1.0
    deadband: 0.015      # Increased from 0.01 (less sensitive)
```

### Example: Increase Lane Centering Response

```yaml
control:
  lateral:
    lateral_weight: 0.6  # Increased from 0.5
    heading_weight: 0.4  # Decreased from 0.5
```

## Parameter Tuning Workflow

1. **Identify issue**: Car drifts, oscillates, too slow to respond, etc.
2. **Check logs**: Look at `LatErr` values in logs to see lateral error
3. **Adjust config**: Modify relevant parameters in YAML file
4. **Test**: Run test and check metrics
5. **Iterate**: Fine-tune until behavior is acceptable

## Common Tuning Scenarios

### Car Drifts to One Side
- **Increase** `lateral_weight` (more responsive to offset)
- **Decrease** `deadband` (correct smaller errors)
- **Check** `bias_correction_threshold` (may need adjustment)

### Car Oscillates (Sways Left/Right)
- **Decrease** `kp` (less aggressive)
- **Increase** `kd` (more damping)
- **Increase** `deadband` (ignore tiny errors)

### Car Too Slow to Correct
- **Increase** `kp` (more responsive)
- **Decrease** `deadband` (respond to smaller errors)
- **Increase** `lateral_weight` (prioritize centering)

### Speed Spikes
- **Decrease** `target_speed`
- **Decrease** `speed_prevention_threshold` (start preventing earlier)
- **Increase** `speed_prevention_brake_amount` (more aggressive prevention)

## Best Practices

1. **Start conservative**: Lower gains, higher deadband
2. **Tune incrementally**: Change one parameter at a time
3. **Test thoroughly**: Run multiple tests to verify changes
4. **Document changes**: Note what you changed and why
5. **Version control**: Commit config changes with test results

## Default Values

All parameters have sensible defaults. If a parameter is missing from the config file, the default value is used. See the config file for all default values.

