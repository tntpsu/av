# Ground Truth Following Drive Evaluation

## Performance Summary

### Critical Findings

1. **Ground Truth Mode: ✅ ENABLED** (confirmed by user)
2. **Performance: ❌ GETTING WORSE, NOT BETTER**
   - Initial lateral error: -24.93m
   - Final lateral error: -45.61m
   - **Error increased by 20.68m** (car moving further from center)

### Root Cause Identified

**CRITICAL ISSUE: Steering Command Race Condition**

The ground truth follower computes:
- `steering = -1.0` (maximum left steering) ✓ CORRECT
- `lane_center = -13.50m` (lane center is LEFT of car)
- `p_term = -30.386` (huge proportional term, clipped to -1.0)

**BUT:** The AV stack ALSO computes and sends its own steering:
- `steering = -0.339` (from trajectory planning)
- This is logged in AV stack logs

**The Problem:**
1. Ground truth follower calls `av_stack._process_frame()` which computes AV stack control
2. AV stack sends its control command (steering=-0.339) to Unity
3. Ground truth follower then sends its control command (steering=-1.0) to Unity
4. **Whichever arrives last wins** - but Unity might be using the AV stack's command!

**Evidence:**
- Ground truth follower logs: `raw_steering=-1.000, smoothed=-1.000` ✓
- AV stack logs: `Steering=-0.339, -0.261` (different values!)
- Error getting worse suggests Unity is using the WRONG steering (AV stack's -0.339 instead of GT's -1.0)

## Data Analysis

### Ground Truth Data
- `groundTruthLaneCenterX = -13.50m` (NEGATIVE)
- Interpretation: Lane center is **13.50m to the LEFT** of car
- This means: Car is **13.50m to the RIGHT** of lane center
- **Expected steering: LEFT (negative, maximum -1.0)** to correct ✓

### Ground Truth Follower Steering
- Computed: `steering = -1.0` (maximum left) ✓ CORRECT
- `p_term = -30.386` (clipped to -1.0) ✓ CORRECT
- This is being sent to Unity ✓

### AV Stack Steering (WRONG - shouldn't be used!)
- Computed: `steering = -0.339` (from trajectory planning)
- This is ALSO being sent to Unity ❌
- **This is the problem!**

## Solution

### Fix 1: Prevent AV Stack from Sending Control (RECOMMENDED)

Modify `av_stack._process_frame()` to NOT send control when ground truth mode is active:

```python
# In av_stack.py, _process_frame()
# Check if ground truth follower is active
if not hasattr(self, '_gt_control_override') or self._gt_control_override is None:
    # Only send control if NOT in ground truth mode
    self.bridge.set_control_command(
        steering=control_command['steering'],
        throttle=control_command['throttle'],
        brake=control_command['brake']
    )
```

### Fix 2: Verify Unity is Using Ground Truth Command

Add logging in Unity to show which steering value is actually being used:
- Log in `AVBridge.RequestControlCommands()` what steering is received
- Log in `CarController.ApplyControls()` what steering is applied

### Fix 3: Coordinate System Verification

Even if steering is correct, verify coordinate system:
- If `groundTruthLaneCenterX = -13.50m` means lane center is LEFT of car
- And we steer LEFT (negative), error should DECREASE
- But error is INCREASING, suggesting either:
  1. Unity is using wrong steering (AV stack's -0.339 instead of GT's -1.0)
  2. Coordinate system interpretation is wrong
  3. Steering sign is wrong

## Recommendations

### Immediate Action

1. **Prevent AV stack from sending control in ground truth mode**
   - This is the most likely cause of the problem
   - Ground truth follower should be the ONLY source of control

2. **Add Unity logging to verify which steering is used**
   - Log received steering in AVBridge
   - Log applied steering in CarController

3. **Verify ground truth mode is actually using direct velocity**
   - Check Unity Inspector: `CarController.groundTruthMode = true`
   - Check Unity console for mode confirmation logs

### Testing

1. After fix, verify:
   - Ground truth follower sends steering=-1.0
   - Unity receives steering=-1.0
   - Unity applies steering=-1.0
   - Error starts decreasing

2. If error still increases:
   - Flip steering sign (`steering_sign = -1.0` → `steering_sign = 1.0`)
   - Or verify coordinate system interpretation

## Next Steps

1. **Implement Fix 1** (prevent AV stack from sending control)
2. **Add Unity logging** to verify which steering is used
3. **Re-run test** and check if error decreases
4. **If still failing, verify coordinate system** with manual test
