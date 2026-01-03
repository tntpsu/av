# Critical Frame-by-Frame Analysis: What Went Wrong

## Key Finding: We Were ALWAYS Steering Wrong, Not Overshooting

### Evidence from First 20 Frames

**ALL 20 frames had the same problem:**
- Car is LEFT of reference (LatErr = -9.02m to -8.01m, all negative)
- Should steer RIGHT (positive) to correct
- But we were steering LEFT (negative, -0.471 to -0.393) - **WRONG DIRECTION!**

**This means:**
- We were **consistently steering away from the path from frame 1**
- Not overshooting - we never even started correcting correctly
- The car turned around because it kept steering the wrong way

## Root Cause Analysis

### Problem 1: Using Only Lateral Offset

**Current approach:**
```python
steering = kp * lane_center  # Only lateral offset
```

**Why this fails:**
1. Doesn't account for path heading/curvature
2. Doesn't anticipate upcoming turns
3. Only reactive correction, not predictive
4. Can't handle cases where car is on path but wrong heading

### Problem 2: Coordinate System Confusion

**Two different measurements:**
1. **AV Stack LatErr:** -9.02m = car LEFT of reference → steer RIGHT
2. **Ground Truth lane_center:** -2.58m = lane center LEFT of car → car RIGHT of center → steer LEFT

**These conflict!** We need to use ONE consistent coordinate system.

### Problem 3: Missing Path Geometry

Since we're following a **known path drawn in Unity**, we should:
1. Get the **desired heading** from the path (not just lateral offset)
2. Compute **heading error** (desired - current)
3. Combine **heading error + lateral error** for steering
4. Use **path curvature** to anticipate turns

## Mathematical Solution

For a known path, at each frame:

```python
# 1. Get desired heading from path (lookahead point)
desired_heading = get_path_heading_at_lookahead(current_pos, lookahead=5m)

# 2. Compute heading error
heading_error = desired_heading - current_heading

# 3. Compute lateral error (distance from path center)
lateral_error = lane_center

# 4. Combine for steering (both should steer toward path)
steering = kp_heading * heading_error + kp_lateral * lateral_error
```

## What We Need from Unity

Currently, `GroundTruthReporter` only gives:
- `lane_center` (lateral offset)

**We need:**
- **Desired heading** (path heading at lookahead point)
- **Path curvature** (for anticipatory steering)
- **Lookahead point position** (where we want to be heading)

## The Fix I Applied

I changed the steering sign:
```python
# BEFORE (WRONG):
p_term = steering_sign * kp * lane_center  # Same sign

# AFTER (CORRECT):
p_term = -steering_sign * kp * lane_center  # Opposite sign
```

**But this might still be wrong** if the coordinate system interpretation is incorrect.

## Better Solution: Path-Based Steering

Instead of just using lateral offset, we should:

1. **Get path heading from Unity** (add to GroundTruthReporter)
2. **Compute heading error** (desired - current)
3. **Use both heading + lateral** for steering
4. **Test mathematically** - compare computed steering to what path requires

This way, we're using the **actual path geometry** to compute steering, not just reactive lateral correction.

