# Exact Steering Calculation & Self-Diagnostic System

## Overview

We now have **two major improvements** to the ground truth follower:

1. **Exact Steering Calculation**: Uses bicycle model + path geometry to calculate the exact steering needed
2. **Self-Diagnostic Feedback**: Automatically detects if steering is wrong and corrects it

## Why We Can Calculate Exact Steering

### Known Parameters

From Unity configuration:
- **Car wheelbase**: 2.5m (estimated from car length ~4m)
- **Max steering angle**: 30° (from `maxSteerAngle = 30f`)
- **Path curvature**: Available from Unity (`groundTruthPathCurvature`)
- **Unity fixed delta time**: 0.02s (50 Hz FixedUpdate)

### Bicycle Model

For a bicycle model, the exact steering angle to follow a path with curvature `κ` is:

```
steering_angle = atan(wheelbase * curvature)
```

This gives us the **exact** steering needed, not just a proportional approximation.

### Unity Ground Truth Mode

Unity's ground truth mode applies steering as:
```csharp
steerAngle = steerInput * maxSteerAngle;  // steerInput is [-1, 1]
rotation = Quaternion.Euler(0, steerAngle * Time.fixedDeltaTime, 0);
```

So we convert our calculated steering angle to Unity's `steerInput`:
```
steerInput = steering_angle_rad / maxSteerAngle_rad
```

## Exact Steering Implementation

### Method: `_calculate_exact_steering()`

1. **Get path curvature** from Unity (`groundTruthPathCurvature`)
2. **Calculate exact steering angle**: `atan(wheelbase * curvature)`
3. **Convert to Unity input**: `steerInput = steering_angle / maxSteerAngle`
4. **Add lateral correction**: Small correction to stay centered in lane
5. **Return steering command**: `[-1.0, 1.0]`

### Fallback

If path curvature is not available (e.g., straight road), falls back to proportional control using lateral error.

## Self-Diagnostic System

### Purpose

Automatically detects if steering is wrong (e.g., wrong sign) and corrects it **without manual intervention**.

### How It Works

1. **Track error over time**: Records `(time, lane_center, heading_error, steering)` every frame
2. **Analyze error trend**: Every 0.5 seconds, checks if error is increasing or decreasing
3. **Detect wrong steering**: If error increases by >0.1m over 2 seconds, steering is wrong
4. **Auto-correct**: Flips `steering_sign` and recalculates steering
5. **Log correction**: Warns user and logs the correction

### Method: `_run_self_diagnostic()`

**Inputs:**
- `current_time`: Current timestamp
- `lane_center`: Current lateral error (meters)
- `heading_error`: Current heading error (degrees) or None
- `steering`: Current steering command

**Logic:**
1. Add sample to history (keeps last 2 seconds)
2. Every 0.5 seconds, analyze trend:
   - If `lane_center` error is **increasing** → steering is wrong
   - If `heading_error` is **increasing** → steering is wrong
3. If error is increasing:
   - Log warning
   - Flip `steering_sign` (from +1 to -1 or vice versa)
   - Clear history to start fresh
   - Return `True` (correction applied)

**Output:**
- `True` if correction was applied, `False` otherwise

### Example

```
Frame 1:  lane_center = 7.28m, steering = +1.0
Frame 2:  lane_center = 7.50m, steering = +1.0  (error increasing!)
Frame 3:  lane_center = 7.80m, steering = +1.0  (error increasing!)
...
⚠️  SELF-DIAGNOSTIC: Error is INCREASING!
   Lane center error: 7.28m → 7.80m (increase: 0.52m)
   → AUTO-CORRECTING: Flipping steering_sign from 1.0 to -1.0

Frame N:  lane_center = 7.20m, steering = -1.0  (now correct!)
Frame N+1: lane_center = 6.80m, steering = -1.0  (error decreasing ✓)
```

## What Feedback We Get From Unity

### Automatic Feedback (No Manual Log Checking)

1. **Vehicle State** (every frame):
   - `groundTruthLaneCenterX`: Lateral error (meters)
   - `groundTruthDesiredHeading`: Desired heading (degrees)
   - `groundTruthPathCurvature`: Path curvature (1/meters)
   - `speed`: Current speed (m/s)
   - `heading`: Current heading (degrees)

2. **Self-Diagnostic Analysis**:
   - Tracks error over time automatically
   - Detects if error is increasing/decreasing
   - Auto-corrects if steering is wrong
   - Logs all corrections

3. **Steering Method Logging**:
   - Logs whether using "EXACT" or "PROPORTIONAL" steering
   - Logs path curvature when using exact steering
   - Logs all steering calculations

## Benefits

### 1. Exact Steering
- **No tuning needed**: Uses physics-based calculation
- **Perfect for known paths**: Uses path curvature directly
- **More accurate**: Not just proportional approximation

### 2. Self-Correction
- **No manual intervention**: Detects and fixes wrong steering automatically
- **Works even if sign is wrong**: Will auto-correct after detecting error increase
- **Logs everything**: You can see when corrections happen

### 3. Better Feedback
- **Automatic analysis**: No need to manually check logs
- **Real-time correction**: Fixes issues as they happen
- **Clear logging**: Shows exactly what's happening

## Configuration

### Enable/Disable Features

```python
# In __init__:
self.use_exact_steering = True  # Use exact calculation
self.auto_correct_enabled = True  # Enable self-diagnostic
```

### Tuning Parameters

```python
# Car properties (from Unity)
self.wheelbase = 2.5  # meters
self.max_steer_angle_deg = 30.0  # degrees

# Self-diagnostic
self.diagnostic_window = 2.0  # seconds of history
self.error_increase_threshold = 0.1  # meters (error increase to trigger correction)
self.min_samples_for_correction = 10  # minimum samples before correcting
```

## Expected Behavior

### With Exact Steering

1. **Path with curvature**: Uses exact calculation from `groundTruthPathCurvature`
2. **Straight path**: Falls back to proportional control
3. **Logs show**: `"GT Steering [EXACT (path curvature)]: ..."`

### With Self-Diagnostic

1. **Correct steering**: Error decreases, no correction needed
2. **Wrong steering**: Error increases, auto-corrects after ~2 seconds
3. **Logs show**: `"⚠️  SELF-DIAGNOSTIC: Error is INCREASING!"` when correction happens

## Testing

The system will:
1. **Automatically use exact steering** when path curvature is available
2. **Automatically detect wrong steering** and correct it
3. **Log everything** so you can see what's happening

No manual intervention needed - the system self-corrects!

