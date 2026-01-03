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
    kp: 0.3              # Proportional gain
    ki: 0.0              # Integral gain (0 = disabled)
    kd: 0.1              # Derivative gain
    max_steering: 0.5    # Maximum steering angle
    deadband: 0.02       # Deadband in radians (~1.1°)
    heading_weight: 0.5  # Weight for heading error
    lateral_weight: 0.5  # Weight for lateral error
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
    kp: 0.3
    ki: 0.05
    kd: 0.02
    target_speed: 8.0    # Target speed in m/s
    max_speed: 10.0      # Maximum speed in m/s
    speed_smoothing: 0.7 # Smoothing factor (0-1)
```

### Trajectory Planning

```yaml
trajectory:
  lookahead_distance: 20.0  # How far ahead to plan
  reference_lookahead: 8.0  # Lookahead for reference point
  bias_correction_threshold: 10.0  # Pixels - auto-correct bias
```

### Safety Limits

```yaml
safety:
  max_speed: 10.0
  emergency_brake_threshold: 2.0  # Emergency brake if speed > max * this
  speed_prevention_threshold: 0.85  # Start preventing at 85% of max
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
    kp: 0.2              # Reduced from 0.3
    deadband: 0.03       # Increased from 0.02 (less sensitive)
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

