#!/usr/bin/env python3
"""
Analyze lane detection failures from recorded data.

This script analyzes HDF5 recordings to identify:
- Which lane fails more often (left vs right)
- When failures occur (vehicle position, speed, etc.)
- Patterns in detection failures
- Validation failure reasons
"""

import sys
from pathlib import Path
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("WARNING: h5py not available. Install with: pip install h5py")
    print("Attempting to use alternative methods...")
    sys.exit(1)


def analyze_detection_failures(recording_file: Path):
    """Analyze lane detection failures from recording."""
    
    print("="*80)
    print("LANE DETECTION FAILURE ANALYSIS")
    print("="*80)
    print(f"Recording: {recording_file.name}")
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Get all available keys
        print("Available datasets:")
        def print_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_keys)
        print()
        
        # Load perception data
        if 'perception/num_lanes_detected' not in f:
            print("ERROR: No perception data found in recording")
            return
        
        num_lanes = np.array(f['perception/num_lanes_detected'])
        num_frames = len(num_lanes)
        
        # Try to get left/right lane positions
        left_lane_x = None
        right_lane_x = None
        if 'perception/left_lane_x' in f:
            left_lane_x = np.array(f['perception/left_lane_x'])
        if 'perception/right_lane_x' in f:
            right_lane_x = np.array(f['perception/right_lane_x'])
        
        # Vehicle state
        vehicle_position = None
        vehicle_speed = None
        lateral_error = None
        
        if 'vehicle/position' in f:
            vehicle_position = np.array(f['vehicle/position'])
        if 'vehicle/speed' in f:
            vehicle_speed = np.array(f['vehicle/speed'])
        if 'control/lateral_error' in f:
            lateral_error = np.array(f['control/lateral_error'])
        
        print("1. DETECTION OVERVIEW")
        print("-" * 80)
        
        dual_lane = num_lanes == 2
        single_lane = num_lanes == 1
        no_lanes = num_lanes == 0
        
        print(f"Total frames: {num_frames}")
        print(f"  Both lanes: {np.sum(dual_lane)} ({np.sum(dual_lane)/num_frames*100:.1f}%)")
        print(f"  Single lane: {np.sum(single_lane)} ({np.sum(single_lane)/num_frames*100:.1f}%)")
        print(f"  No lanes: {np.sum(no_lanes)} ({np.sum(no_lanes)/num_frames*100:.1f}%)")
        print()
        
        # Which lane fails when single lane detected?
        print("2. SINGLE-LANE ANALYSIS")
        print("-" * 80)
        
        if left_lane_x is not None and right_lane_x is not None:
            left_detected = ~np.isnan(left_lane_x)
            right_detected = ~np.isnan(right_lane_x)
            
            single_left_only = single_lane & left_detected & ~right_detected
            single_right_only = single_lane & right_detected & ~left_detected
            single_both = single_lane & left_detected & right_detected  # Shouldn't happen, but check
            
            print(f"When single lane detected ({np.sum(single_lane)} frames):")
            print(f"  Left lane only: {np.sum(single_left_only)} ({np.sum(single_left_only)/np.sum(single_lane)*100:.1f}%)")
            print(f"  Right lane only: {np.sum(single_right_only)} ({np.sum(single_right_only)/np.sum(single_lane)*100:.1f}%)")
            if np.sum(single_both) > 0:
                print(f"  Both detected (data inconsistency): {np.sum(single_both)}")
            print()
            
            # Overall failure rates
            left_failures = np.sum(~left_detected)
            right_failures = np.sum(~right_detected)
            
            print(f"Overall lane failure rates:")
            print(f"  Left lane failures: {left_failures} ({left_failures/num_frames*100:.1f}%)")
            print(f"  Right lane failures: {right_failures} ({right_failures/num_frames*100:.1f}%)")
            
            if left_failures > right_failures * 1.2:
                print(f"  ⚠️  Left lane fails {left_failures/right_failures:.1f}x more often!")
            elif right_failures > left_failures * 1.2:
                print(f"  ⚠️  Right lane fails {right_failures/left_failures:.1f}x more often!")
            else:
                print(f"  ✓ Failure rates are similar")
            print()
        else:
            print("Left/right lane position data not available")
            print()
        
        # Correlate with vehicle position
        print("3. CORRELATION WITH VEHICLE POSITION")
        print("-" * 80)
        
        if vehicle_position is not None:
            car_pos_x = vehicle_position[:, 0]
            
            # Group by position
            left_side = car_pos_x < -0.3
            center = (car_pos_x >= -0.3) & (car_pos_x <= 0.3)
            right_side = car_pos_x > 0.3
            
            print("Detection success by car position:")
            for label, mask in [('Left side (<-0.3m)', left_side),
                                ('Center (-0.3 to 0.3m)', center),
                                ('Right side (>0.3m)', right_side)]:
                if np.sum(mask) > 0:
                    dual = np.sum(mask & dual_lane)
                    single = np.sum(mask & single_lane)
                    none = np.sum(mask & no_lanes)
                    total = np.sum(mask)
                    print(f"{label:25s}: 2={dual:3d} ({dual/total*100:5.1f}%), "
                          f"1={single:2d} ({single/total*100:5.1f}%), "
                          f"0={none:2d} ({none/total*100:5.1f}%)")
            
            # Which lane fails at each position?
            if left_lane_x is not None and right_lane_x is not None:
                left_detected = ~np.isnan(left_lane_x)
                right_detected = ~np.isnan(right_lane_x)
                
                print()
                print("Lane-specific failures by position:")
                for label, mask in [('Left side', left_side),
                                    ('Center', center),
                                    ('Right side', right_side)]:
                    if np.sum(mask) > 0:
                        left_fail = np.sum(mask & ~left_detected)
                        right_fail = np.sum(mask & ~right_detected)
                        total = np.sum(mask)
                        print(f"{label:15s}: Left fails={left_fail:3d} ({left_fail/total*100:5.1f}%), "
                              f"Right fails={right_fail:3d} ({right_fail/total*100:5.1f}%)")
            print()
        else:
            print("Vehicle position data not available")
            print()
        
        # Analyze failure sequences
        print("4. FAILURE SEQUENCES")
        print("-" * 80)
        
        # Find transitions
        transitions = []
        for i in range(1, num_frames):
            prev = num_lanes[i-1]
            curr = num_lanes[i]
            if prev != curr:
                transitions.append((i, int(prev), int(curr)))
        
        print(f"Total transitions: {len(transitions)}")
        
        # Count transition types
        transition_counts = {}
        for frame, prev, curr in transitions:
            key = (prev, curr)
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        print("Transition patterns:")
        for (prev, curr), count in sorted(transition_counts.items()):
            print(f"  {prev} → {curr} lanes: {count} times")
        print()
        
        # Find early failures (first 100 frames)
        print("5. EARLY FAILURES (First 100 frames)")
        print("-" * 80)
        
        early_frames = np.arange(num_frames) < min(100, num_frames)
        early_dual = np.sum(early_frames & dual_lane)
        early_single = np.sum(early_frames & single_lane)
        early_none = np.sum(early_frames & no_lanes)
        early_total = np.sum(early_frames)
        
        print(f"First {early_total} frames:")
        print(f"  Both lanes: {early_dual} ({early_dual/early_total*100:.1f}%)")
        print(f"  Single lane: {early_single} ({early_single/early_total*100:.1f}%)")
        print(f"  No lanes: {early_none} ({early_none/early_total*100:.1f}%)")
        
        # Find first failure
        first_failure = None
        for i in range(1, min(100, num_frames)):
            if num_lanes[i] < num_lanes[i-1]:
                first_failure = i
                break
        
        if first_failure is not None:
            print(f"  First failure at frame {first_failure} (transition from {num_lanes[first_failure-1]} to {num_lanes[first_failure]} lanes)")
        print()
        
        # Correlate with speed
        print("6. CORRELATION WITH SPEED")
        print("-" * 80)
        
        if vehicle_speed is not None:
            slow = vehicle_speed < 2.0
            medium = (vehicle_speed >= 2.0) & (vehicle_speed < 6.0)
            fast = vehicle_speed >= 6.0
            
            for label, mask in [('Slow (<2 m/s)', slow),
                                ('Medium (2-6 m/s)', medium),
                                ('Fast (>6 m/s)', fast)]:
                if np.sum(mask) > 0:
                    dual = np.sum(mask & dual_lane)
                    single = np.sum(mask & single_lane)
                    total = np.sum(mask)
                    print(f"{label:20s}: 2={dual:3d} ({dual/total*100:5.1f}%), "
                          f"1={single:2d} ({single/total*100:5.1f}%)")
            print()
        else:
            print("Speed data not available")
            print()
        
        # Find problematic frames
        print("7. PROBLEMATIC FRAMES")
        print("-" * 80)
        
        # Frames where detection degrades
        problematic = []
        for i in range(1, min(200, num_frames)):
            if num_lanes[i-1] == 2 and num_lanes[i] < 2:
                problematic.append(i)
        
        if problematic:
            print(f"Frames where detection fails after success: {len(problematic)}")
            print(f"  First 20: {problematic[:20]}")
            if len(problematic) > 20:
                print(f"  ... and {len(problematic)-20} more")
            
            # Analyze which lane fails
            if left_lane_x is not None and right_lane_x is not None:
                left_detected = ~np.isnan(left_lane_x)
                right_detected = ~np.isnan(right_lane_x)
                
                left_fails = [i for i in problematic if not left_detected[i]]
                right_fails = [i for i in problematic if not right_detected[i]]
                
                print()
                print(f"  Left lane fails in {len(left_fails)} of these frames")
                print(f"  Right lane fails in {len(right_fails)} of these frames")
        else:
            print("No problematic frames found in first 200 frames")
        print()
        
        # Summary and recommendations
        print("8. SUMMARY AND RECOMMENDATIONS")
        print("-" * 80)
        
        dual_rate = np.sum(dual_lane) / num_frames * 100
        
        print(f"Overall dual-lane detection rate: {dual_rate:.1f}%")
        print()
        
        if dual_rate < 80:
            print("⚠️  LOW DUAL-LANE DETECTION RATE")
            print("   Recommendations:")
            print("   1. Relax validation thresholds")
            print("   2. Improve detection robustness")
            print("   3. Add temporal smoothing")
        
        if left_lane_x is not None and right_lane_x is not None:
            left_detected = ~np.isnan(left_lane_x)
            right_detected = ~np.isnan(right_lane_x)
            left_fail_rate = np.sum(~left_detected) / num_frames * 100
            right_fail_rate = np.sum(~right_detected) / num_frames * 100
            
            if left_fail_rate > right_fail_rate * 1.2:
                print()
                print(f"⚠️  LEFT LANE FAILS MORE ({left_fail_rate:.1f}% vs {right_fail_rate:.1f}%)")
                print("   Recommendations:")
                print("   1. Check if left lane is cut off by ROI")
                print("   2. Relax validation for left lane position")
                print("   3. Improve detection on left side of image")
                print("   4. Check if left lane has fewer line segments")
        
        print()
        print("="*80)


def main():
    """Main function."""
    
    # Find latest recording
    recordings_dir = Path('data/recordings')
    if not recordings_dir.exists():
        print("ERROR: Recordings directory not found")
        return 1
    
    recordings = sorted(recordings_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not recordings:
        print("ERROR: No recordings found")
        return 1
    
    recording_file = recordings[0]
    
    try:
        analyze_detection_failures(recording_file)
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

