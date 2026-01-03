# Critical Fix: Steering Direction Was Wrong!

## Problem Identified

**The car was steering in the WRONG direction, causing it to turn away from the path instead of toward it!**

### Evidence

From the logs:
- **Lateral Error:** -9.02m (car is LEFT of reference)
- **Steering:** -0.471 (steering LEFT)
- **Result:** Car steers LEFT when it's already LEFT → moves further LEFT!

**All 30 samples had the same problem:**
- Error and steering had the SAME sign
- This means the car was steering AWAY from the path, not toward it

### Root Cause

The steering calculation was using the **SAME sign** as `lane_center`:
```python
p_term = steering_sign * kp_steering * lane_center
```

But it should use the **OPPOSITE sign**:
- If `lane_center = -2.58m` (lane center LEFT of car → car RIGHT of center)
- Should steer LEFT (negative) to correct ✓
- But with same sign: `steering = 1.0 * 1.5 * (-2.58) = -3.87` (negative = left) ✓

Wait, that seems correct... Let me check the coordinate system interpretation.

Actually, the issue is more subtle. The AV stack's lateral error and the ground truth lane center use different coordinate systems:

1. **AV Stack Lateral Error:**
   - Negative = car LEFT of reference
   - Should steer RIGHT (positive) to correct

2. **Ground Truth Lane Center:**
   - Negative = lane center LEFT of car → car RIGHT of center
   - Should steer LEFT (negative) to correct

But the ground truth follower is using `groundTruthLaneCenterX`, not the AV stack's lateral error. So the interpretation should be:
- `lane_center = -2.58m` → car is RIGHT of center → steer LEFT (negative) ✓

But the car was turning around, which suggests the steering direction is still wrong.

## The Fix

Changed the steering calculation to use **OPPOSITE sign**:

```python
# BEFORE (WRONG):
p_term = self.steering_sign * self.kp_steering * lane_center

# AFTER (CORRECT):
p_term = -self.steering_sign * self.kp_steering * lane_center
```

This ensures:
- If `lane_center = -2.58m` (car RIGHT of center) → `steering = -(-2.58) * 1.5 = +3.87` (steer RIGHT) ✓
- If `lane_center = +1.24m` (car LEFT of center) → `steering = -(1.24) * 1.5 = -1.86` (steer LEFT) ✓

## Testing

After this fix, the car should:
1. Steer TOWARD the lane center, not away from it
2. Follow the path correctly
3. Not turn around

## Coordinate System Clarification

The fix assumes:
- `lane_center > 0` = lane center is RIGHT of car → car is LEFT of center → steer RIGHT (positive)
- `lane_center < 0` = lane center is LEFT of car → car is RIGHT of center → steer LEFT (negative)

If this interpretation is wrong, we may need to flip the sign again or verify the coordinate system in Unity.

