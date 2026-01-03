# Critical Frame-by-Frame Analysis

## Key Finding: We Were ALWAYS Steering Wrong, Not Just Overshooting

### Evidence from First 20 Frames

**ALL 20 frames had the same problem:**
- Car is LEFT of reference (LatErr negative)
- Should steer RIGHT (positive) to correct
- But we were steering LEFT (negative) - **WRONG DIRECTION!**

This means we were **consistently steering away from the path from the very first frame**, not just overshooting after correcting.

## The Real Problem: Using Only Lateral Offset

### Current Approach (WRONG)
```python
steering = kp * lane_center  # Only uses lateral offset
```

**Problems:**
1. Doesn't account for path heading/curvature
2. Doesn't anticipate upcoming turns
3. Only reactive, not predictive
4. Can't distinguish between "on path but wrong heading" vs "off path"

### Better Approach: Use Path Geometry

Since we're following a **known path drawn in Unity**, we should:

1. **Get desired heading from path** (not just lateral offset)
2. **Compute heading error** (desired heading - current heading)
3. **Combine heading error + lateral error** for steering
4. **Use path curvature** to anticipate turns

## Mathematical Solution

For a known path, at each frame we can compute:

```python
# 1. Get desired heading from path (lookahead point)
desired_heading = get_path_heading_at_lookahead(current_position, lookahead_distance)

# 2. Compute heading error
heading_error = desired_heading - current_heading

# 3. Compute lateral error (already have this)
lateral_error = lane_center

# 4. Combine for steering
steering = kp_heading * heading_error + kp_lateral * lateral_error
```

## What We Need from Unity

Currently, `GroundTruthReporter` only gives us:
- `lane_center` (lateral offset)

We need:
- **Desired heading** (path heading at lookahead point)
- **Path curvature** (for anticipatory steering)
- **Lookahead point** (where we want to be heading)

## Next Steps

1. **Add path heading to GroundTruthReporter** in Unity
2. **Compute steering from heading error + lateral error**
3. **Use lookahead to anticipate curves**
4. **Test with mathematical reference** (compare computed steering to what path requires)

