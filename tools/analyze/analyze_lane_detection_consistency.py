#!/usr/bin/env python3
"""
Analyze lane detection consistency using recorded data and debug visualizations.

This script:
1. Loads HDF5 recordings
2. Analyzes detection patterns (both lanes vs single lane)
3. Correlates with debug visualizations
4. Identifies patterns in failures
5. Suggests fixes
"""

import h5py
import numpy as np
from pathlib import Path
import cv2
import json
from collections import defaultdict
import sys


def analyze_detection_consistency(recording_file: Path, debug_dir: Path = None):
    """Analyze lane detection consistency from recorded data."""
    
    print("="*80)
    print("LANE DETECTION CONSISTENCY ANALYSIS")
    print("="*80)
    print(f"Recording: {recording_file.name}")
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Get data
        num_lanes = np.array(f['perception/num_lanes_detected'])
        left_lane_x = np.array(f['perception/left_lane_x'])
        right_lane_x = np.array(f['perception/right_lane_x'])
        vehicle_position = np.array(f['vehicle/position'])
        vehicle_rotation = np.array(f['vehicle/rotation'])
        vehicle_speed = np.array(f['vehicle/speed'])
        lateral_error = np.array(f['control/lateral_error'])
        
        num_frames = len(num_lanes)
        
        # Analyze detection patterns
        print("1. DETECTION PATTERNS")
        print("-" * 80)
        
        dual_lane = num_lanes == 2
        single_lane = num_lanes == 1
        no_lanes = num_lanes == 0
        
        print(f"Total frames: {num_frames}")
        print(f"  Both lanes detected: {np.sum(dual_lane)} ({np.sum(dual_lane)/num_frames*100:.1f}%)")
        print(f"  Single lane detected: {np.sum(single_lane)} ({np.sum(single_lane)/num_frames*100:.1f}%)")
        print(f"  No lanes detected: {np.sum(no_lanes)} ({np.sum(no_lanes)/num_frames*100:.1f}%)")
        print()
        
        # Which lane fails when single lane detected?
        print("2. SINGLE-LANE ANALYSIS")
        print("-" * 80)
        
        left_detected = ~np.isnan(left_lane_x)
        right_detected = ~np.isnan(right_lane_x)
        
        single_left = single_lane & left_detected & ~right_detected
        single_right = single_lane & right_detected & ~left_detected
        
        print(f"When single lane detected:")
        print(f"  Left lane only: {np.sum(single_left)} frames ({np.sum(single_left)/np.sum(single_lane)*100:.1f}% of single-lane frames)")
        print(f"  Right lane only: {np.sum(single_right)} frames ({np.sum(single_right)/np.sum(single_lane)*100:.1f}% of single-lane frames)")
        print()
        
        # Correlate with vehicle position
        print("3. CORRELATION WITH VEHICLE POSITION")
        print("-" * 80)
        
        car_pos_x = vehicle_position[:, 0]
        
        # Group by position
        left_side = car_pos_x < -0.3
        center = (car_pos_x >= -0.3) & (car_pos_x <= 0.3)
        right_side = car_pos_x > 0.3
        
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
                transitions.append((i, prev, curr))
        
        print(f"Total transitions: {len(transitions)}")
        
        # Count transition types
        transition_counts = defaultdict(int)
        for frame, prev, curr in transitions:
            transition_counts[(int(prev), int(curr))] += 1
        
        print("Transition patterns:")
        for (prev, curr), count in sorted(transition_counts.items()):
            print(f"  {prev} → {curr} lanes: {count} times")
        print()
        
        # Find failure streaks
        print("5. FAILURE STREAKS")
        print("-" * 80)
        
        streaks = []
        current_streak = 0
        streak_start = None
        
        for i in range(num_frames):
            if num_lanes[i] < 2:
                if current_streak == 0:
                    streak_start = i
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append((streak_start, current_streak))
                    current_streak = 0
        
        if current_streak > 0:
            streaks.append((streak_start, current_streak))
        
        if streaks:
            max_streak = max(streaks, key=lambda x: x[1])
            avg_streak = np.mean([s[1] for s in streaks])
            print(f"Failure streaks: {len(streaks)}")
            print(f"  Max streak: {max_streak[1]} frames (starting at frame {max_streak[0]})")
            print(f"  Average streak: {avg_streak:.1f} frames")
        else:
            print("No failure streaks found")
        print()
        
        # Analyze early failures
        print("6. EARLY FAILURES (First 100 frames)")
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
        print()
        
        # Correlate with speed
        print("7. CORRELATION WITH SPEED")
        print("-" * 80)
        
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
        
        # Find problematic frames
        print("8. PROBLEMATIC FRAMES")
        print("-" * 80)
        
        # Frames where detection fails after success
        problematic = []
        for i in range(1, min(100, num_frames)):
            if num_lanes[i-1] == 2 and num_lanes[i] < 2:
                problematic.append(i)
        
        if problematic:
            print(f"Frames where detection fails after success: {len(problematic)}")
            print(f"  First 10: {problematic[:10]}")
            if len(problematic) > 10:
                print(f"  ... and {len(problematic)-10} more")
        else:
            print("No problematic frames found in first 100 frames")
        print()
        
        # Summary and recommendations
        print("9. SUMMARY AND RECOMMENDATIONS")
        print("-" * 80)
        
        dual_rate = np.sum(dual_lane) / num_frames * 100
        single_rate = np.sum(single_lane) / num_frames * 100
        
        print(f"Overall dual-lane detection rate: {dual_rate:.1f}%")
        print(f"Single-lane detection rate: {single_rate:.1f}%")
        print()
        
        if dual_rate < 80:
            print("⚠️  LOW DUAL-LANE DETECTION RATE")
            print("   Recommendations:")
            print("   1. Relax validation thresholds (y_range, distance_from_center)")
            print("   2. Improve left lane detection (gray dashed lines)")
            print("   3. Add temporal smoothing to prevent single-frame failures")
            print("   4. Use previous frame's lanes when current detection fails")
        
        if np.sum(single_left) > np.sum(single_right) * 1.5:
            print()
            print("⚠️  LEFT LANE FAILS MORE OFTEN")
            print("   Recommendations:")
            print("   1. Improve gray lane detection (lower saturation threshold)")
            print("   2. Adjust ROI to better capture left lane")
            print("   3. Relax left lane validation (it's farther from center)")
        
        if len(problematic) > 10:
            print()
            print("⚠️  MANY FAILURES AFTER SUCCESS")
            print("   Recommendations:")
            print("   1. Add temporal filtering to prevent sudden failures")
            print("   2. Use exponential smoothing on lane coefficients")
            print("   3. Validate against previous frame before rejecting")
        
        print()
        print("="*80)
        
        # Return analysis results
        return {
            'dual_rate': dual_rate,
            'single_rate': single_rate,
            'left_failures': np.sum(single_right),
            'right_failures': np.sum(single_left),
            'problematic_frames': problematic[:20],  # First 20
            'max_streak': max_streak[1] if streaks else 0,
        }


def analyze_debug_visualizations(debug_dir: Path, problematic_frames: list):
    """Analyze debug visualizations for problematic frames."""
    
    if not debug_dir or not debug_dir.exists():
        print("Debug visualizations directory not found, skipping image analysis")
        return
    
    print()
    print("="*80)
    print("DEBUG VISUALIZATION ANALYSIS")
    print("="*80)
    
    # Find debug images
    frame_images = sorted(debug_dir.glob('frame_*.png'))
    
    if not frame_images:
        print("No debug images found")
        return
    
    print(f"Found {len(frame_images)} debug images")
    print()
    
    # Analyze problematic frames
    if problematic_frames:
        print("Analyzing problematic frames:")
        for frame_idx in problematic_frames[:5]:  # First 5
            # Find corresponding debug image
            frame_num = frame_idx * 30  # Debug images are every 30 frames
            image_path = debug_dir / f"frame_{frame_num:06d}.png"
            
            if image_path.exists():
                print(f"  Frame {frame_idx}: {image_path.name} exists")
                # Could load and analyze image here
            else:
                print(f"  Frame {frame_idx}: No debug image found")
    
    print()
    print("="*80)


def main():
    """Main analysis function."""
    
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
    
    # Check for debug visualizations
    debug_dir = Path('tmp/debug_visualizations')
    
    # Run analysis
    try:
        results = analyze_detection_consistency(recording_file, debug_dir)
        analyze_debug_visualizations(debug_dir, results.get('problematic_frames', []))
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

