# Critical Fix: Reverting Wrong Sign Change

## Problem Identified

**My previous "fix" made things WORSE, not better!**

### Evidence from Latest Run

From logs:
- `groundTruthLaneCenterX = +7.28m` (POSITIVE)
- Interpretation: Lane center is 7.28m to the RIGHT of car
- This means: Car is LEFT of center
- Should steer: RIGHT (positive) to correct

### What My "Fix" Did (WRONG)

```python
# My wrong fix:
p_term_lateral = -kp * lane_center = -1.5 * 7.28 = -10.92 (NEGATIVE)
Result: Steering LEFT (negative) - WRONG! Makes car go further LEFT!
```

### What Original Code Did (CORRECT)

```python
# Original code:
p_term = kp * lane_center = 1.5 * 7.28 = 10.92 (POSITIVE)
Result: Steering RIGHT (positive) - CORRECT! Moves car toward center!
```

## Root Cause

I misunderstood the coordinate system. The original code was **CORRECT**:
- `lane_center = +7.28m` → car LEFT of center → steer RIGHT (positive) ✓
- `lane_center = -2.58m` → car RIGHT of center → steer LEFT (negative) ✓

**Steering should have SAME sign as lane_center, not opposite!**

## The Fix

**REVERTED** the sign change:
```python
# BEFORE (my wrong fix):
p_term_lateral = -self.steering_sign * self.kp_steering * lane_center

# AFTER (reverted to original, CORRECT):
p_term_lateral = self.steering_sign * self.kp_steering * lane_center
```

## Why This Happened

I overthought the problem. The original code was mathematically correct:
- If lane center is to the RIGHT (positive), steer RIGHT (positive) ✓
- If lane center is to the LEFT (negative), steer LEFT (negative) ✓

The issue was NOT the sign - it was something else (maybe the path-based steering I added is interfering, or the heading error calculation is wrong).

## Next Steps

1. **Test with reverted sign** - Should work better now
2. **Disable path-based steering temporarily** - Test with just lateral error first
3. **Verify heading error calculation** - Make sure it's not making things worse
4. **Simplify** - Get basic lateral steering working before adding complexity

## Lesson Learned

**Don't fix what isn't broken!** The original code was correct. I should have verified the coordinate system interpretation with actual data before making changes.

