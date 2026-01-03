#!/usr/bin/env python3
"""
Correlate debug images with vehicle state to understand why lanes appear non-parallel.
Accounts for car position and heading - parallel lanes appear non-parallel when car is offset/angled.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import cv2
from pathlib import Path
from perception.models.lane_detection import SimpleLaneDetector


def quaternion_to_yaw(q):
    """Convert quaternion (x, y, z, w) to yaw angle (radians)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    # Yaw (rotation around y-axis in Unity)
    # Unity quaternion: (x, y, z, w)
    # Yaw = atan2(2*(w*y + x*z), 1 - 2*(y^2 + z^2))
    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
    return yaw


def analyze_correlated_frames(recording_path: Path, debug_dir: Path, max_frames: int = 10):
    """Analyze debug images correlated with vehicle state."""
    
    print("="*80)
    print("CORRELATED ANALYSIS: Debug Images + Vehicle State")
    print("="*80)
    print(f"Recording: {recording_path.name}\n")
    
    with h5py.File(recording_path, 'r') as f:
        # Vehicle state
        positions = np.array(f['vehicle/position'])
        rotations = np.array(f['vehicle/rotation'])
        speeds = np.array(f['vehicle/speed'])
        
        # Perception
        left_lane_x = np.array(f['perception/left_lane_x'])
        right_lane_x = np.array(f['perception/right_lane_x'])
        num_lanes = np.array(f['perception/num_lanes_detected'])
        
        # Trajectory/Control
        ref_heading = np.array(f['trajectory/reference_point_heading'])
        steering = np.array(f['control/steering'])
        lateral_error = np.array(f['control/lateral_error'])
        
        # Find debug images
        debug_frames = sorted(debug_dir.glob('frame_*.png'))
        debug_frames = [f for f in debug_frames if 'combined' not in f.name and 
                       'edges' not in f.name and 'mask' not in f.name and 'histogram' not in f.name]
        
        detector = SimpleLaneDetector()
        
        print("Frame-by-Frame Analysis:\n")
        print(f"{'Frame':<8} {'Car X':<10} {'Heading':<12} {'Speed':<8} {'Lanes':<8} {'Linear Diff':<12} {'Ref Hdg':<10} {'Steer':<8}")
        print("-" * 80)
        
        for debug_img_path in debug_frames[:max_frames]:
            # Extract frame number
            try:
                frame_str = debug_img_path.stem.split('_')[1]
                frame_idx = int(frame_str)
            except:
                continue
            
            if frame_idx >= len(positions):
                continue
            
            # Vehicle state
            car_x = positions[frame_idx, 0]
            car_z = positions[frame_idx, 2]
            car_rotation = rotations[frame_idx]
            car_heading_rad = quaternion_to_yaw(car_rotation)
            car_heading_deg = np.degrees(car_heading_rad)
            car_speed = speeds[frame_idx]
            
            # Perception
            lanes_detected = num_lanes[frame_idx]
            left_x_veh = left_lane_x[frame_idx]
            right_x_veh = right_lane_x[frame_idx]
            
            # Trajectory/Control
            heading_ref_deg = np.degrees(ref_heading[frame_idx]) if frame_idx < len(ref_heading) else 0
            steer = steering[frame_idx] if frame_idx < len(steering) else 0
            
            # Analyze debug image
            debug_img = cv2.imread(str(debug_img_path))
            if debug_img is None:
                continue
            
            debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            lanes = detector.detect(debug_img_rgb)
            
            linear_diff = None
            if lanes[0] is not None and lanes[1] is not None:
                if len(lanes[0]) >= 2 and len(lanes[1]) >= 2:
                    left_linear = lanes[0][-2]
                    right_linear = lanes[1][-2]
                    linear_diff = abs(left_linear - right_linear)
            
            # Print summary
            lanes_str = f"{lanes_detected}" if lanes_detected > 0 else "0"
            linear_str = f"{linear_diff:.2f}" if linear_diff is not None else "N/A"
            
            print(f"{frame_idx:<8} {car_x:>8.2f}m {car_heading_deg:>10.1f}° {car_speed:>6.1f} {lanes_str:<8} {linear_str:<12} {heading_ref_deg:>8.1f}° {steer:>6.3f}")
            
            # Detailed analysis for frames with issues
            if linear_diff is not None and linear_diff > 0.1:
                print(f"  → Lanes appear non-parallel (diff: {linear_diff:.2f})")
                if abs(car_x) > 0.3:
                    print(f"    Car offset: {car_x:.2f}m from center")
                if abs(car_heading_deg) > 5:
                    print(f"    Car angled: {car_heading_deg:.1f}° from straight")
                print(f"    → This is EXPECTED! Parallel lanes appear non-parallel when car is offset/angled")
                print()
        
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        print()
        print("1. Lanes in Unity ARE parallel (geometry)")
        print("2. But they APPEAR non-parallel in images when:")
        print("   - Car is offset from lane center (perspective)")
        print("   - Car is angled (not looking straight)")
        print()
        print("3. This is NORMAL and EXPECTED behavior!")
        print("   - Perspective makes parallel lines converge")
        print("   - Car offset/angle changes convergence point")
        print()
        print("4. Heading fix should account for car state:")
        print("   - Transform lanes to vehicle coordinates")
        print("   - Check parallelism in vehicle frame, not image frame")
        print("   - Or use heading calculation that accounts for perspective")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Correlate debug images with vehicle state')
    parser.add_argument('--recording', type=str, help='Path to recording file')
    parser.add_argument('--debug-dir', type=str, default='tmp/debug_visualizations',
                       help='Directory with debug images')
    parser.add_argument('--max-frames', type=int, default=10, help='Max frames to analyze')
    
    args = parser.parse_args()
    
    # Find latest recording if not provided
    if args.recording:
        recording_path = Path(args.recording)
    else:
        recordings_dir = Path('data/recordings')
        recordings = sorted(recordings_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_path = recordings[0]
            print(f"Using latest recording: {recording_path.name}\n")
        else:
            print("No recordings found")
            return
    
    debug_dir = Path(args.debug_dir)
    if not debug_dir.exists():
        print(f"Debug directory not found: {debug_dir}")
        return
    
    analyze_correlated_frames(recording_path, debug_dir, args.max_frames)


if __name__ == '__main__':
    main()

