# Perception Investigation - Ground Truth Follower Test

## Summary

The ground truth follower is working correctly. The issue is that **perception is detecting lanes incorrectly**, which causes the trajectory planner to compute wrong reference headings.

## Findings

### Ground Truth (Correct)
- **Path Curvature**: 0.250669 (1/m) - **POSITIVE** = right turn
- **Lane Width**: ~7m (left: -4.0m, right: +3.0m)
- **Lane Center**: -0.5m (car is LEFT of center)
- **Steering**: 0.291 to 0.962 - **POSITIVE** = right turn ✓

### Perception (Incorrect)
- **Lane Width**: ~10m (left: -5.0m, right: +5.0m) - **TOO WIDE**
- **Lane Center**: 0.0m (centered) - **WRONG** (should be -0.5m)
- **Reference Heading**: -12.67° to -50.93° - **NEGATIVE** = left turn ✗

### The Problem

1. **Perception is detecting wrong lanes**:
   - Detecting 10m wide lane (should be 7m)
   - Detecting centered lane (car is actually 0.5m left of center)
   - This suggests perception is detecting road edges or other features, not lane lines

2. **Trajectory planner uses perception**:
   - Computes reference heading from perception lanes
   - Since perception thinks lanes curve left, reference heading is negative
   - But ground truth shows path curves right (positive curvature)

3. **Ground truth follower ignores perception**:
   - Uses ground truth path curvature directly
   - Steers correctly (right) based on ground truth
   - This is correct behavior!

## What's Being Recorded

The recording captures:
- ✅ **Ground truth data** (correct path, lanes, curvature)
- ✅ **Perception output** (incorrect - for comparison)
- ✅ **Trajectory planning** (based on perception - shows the error)
- ✅ **Control commands** (based on ground truth - correct)

This is actually **good** - we're recording both ground truth and perception so we can:
1. See how well perception performs vs ground truth
2. Identify perception errors
3. Improve perception based on ground truth comparison

## Root Cause

Perception is using CV (computer vision) fallback and detecting:
- Wrong lines (possibly road edges instead of lane lines)
- Incorrect lane width (10m vs 7m)
- Wrong lane center (0.0m vs -0.5m)

This could be due to:
1. **Lane detection threshold issues** - detecting road edges instead of lane lines
2. **Coordinate conversion errors** - image to vehicle coordinate conversion
3. **Lane identification errors** - swapping left/right or detecting wrong features

## Next Steps

1. **Fix perception** to correctly detect lane lines:
   - Adjust CV detection thresholds
   - Verify coordinate conversion
   - Check lane identification logic

2. **Verify ground truth follower** is working correctly:
   - ✅ Already confirmed - steering matches ground truth curvature
   - ✅ Recording all data correctly (after recorder bug fix)

3. **Use recordings to improve perception**:
   - Compare perception output to ground truth
   - Identify systematic errors
   - Train/improve perception model

## Conclusion

The ground truth follower test is working as intended:
- ✅ Following ground truth path correctly
- ✅ Recording all data (ground truth + perception)
- ✅ Ready for perception comparison

The perception errors are being captured and can be used to improve the perception system.

