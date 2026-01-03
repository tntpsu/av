#!/usr/bin/env python3
"""
Test script to verify if ground truth reference point is correct.
Checks what reference point Unity should be using vs what it's actually using.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse

def analyze_reference_point(recording_path: Path, frame_idx: int = 0):
    """
    Analyze ground truth reference point by comparing expected vs actual values.
    """
    print("=" * 70)
    print("GROUND TRUTH REFERENCE POINT VERIFICATION")
    print("=" * 70)
    print()
    
    with h5py.File(recording_path, 'r') as f:
        # Get vehicle state data
        vehicle_pos = f['vehicle/position'][frame_idx]  # [x, y, z]
        vehicle_rot = f['vehicle/rotation'][frame_idx]  # [x, y, z, w] quaternion
        
        # Get camera data
        camera_pos_x = f['vehicle/camera_pos_x'][frame_idx] if 'vehicle/camera_pos_x' in f else 0.0
        camera_pos_y = f['vehicle/camera_pos_y'][frame_idx] if 'vehicle/camera_pos_y' in f else 1.2
        camera_pos_z = f['vehicle/camera_pos_z'][frame_idx] if 'vehicle/camera_pos_z' in f else 0.0
        camera_forward_x = f['vehicle/camera_forward_x'][frame_idx] if 'vehicle/camera_forward_x' in f else 0.0
        camera_forward_y = f['vehicle/camera_forward_y'][frame_idx] if 'vehicle/camera_forward_y' in f else 0.0
        camera_forward_z = f['vehicle/camera_forward_z'][frame_idx] if 'vehicle/camera_forward_z' in f else 1.0
        
        # Get ground truth and perception data
        gt_left = f['ground_truth/left_lane_line_x'][frame_idx]
        gt_right = f['ground_truth/right_lane_line_x'][frame_idx]
        gt_center = f['ground_truth/lane_center_x'][frame_idx]
        
        p_left = f['perception/left_lane_line_x'][frame_idx]
        p_right = f['perception/right_lane_line_x'][frame_idx]
        p_center = (p_left + p_right) / 2.0
        
        print(f"Frame {frame_idx} Data:")
        print()
        print("Vehicle Position (car center):")
        print(f"  X={vehicle_pos[0]:.3f}m, Y={vehicle_pos[1]:.3f}m, Z={vehicle_pos[2]:.3f}m")
        print()
        print("Camera Position (from Unity):")
        print(f"  X={camera_pos_x:.3f}m, Y={camera_pos_y:.3f}m (height above ground), Z={camera_pos_z:.3f}m")
        print(f"  Forward: ({camera_forward_x:.3f}, {camera_forward_y:.3f}, {camera_forward_z:.3f})")
        print()
        
        # Calculate camera world position
        # Camera local position relative to car: typically (0, 1.2, 1.0)
        # Car center is at vehicle_pos, so camera world position would be:
        # camera_world = vehicle_pos + camera_local (rotated by car rotation)
        # But we have camera_pos_x/z which should be world coordinates
        # camera_pos_y is height above ground (local Y)
        
        # For now, assume camera is at:
        # World X = camera_pos_x (from Unity)
        # World Y = vehicle_pos[1] + camera_pos_y (car Y + camera height)
        # World Z = camera_pos_z (from Unity)
        camera_world_y = vehicle_pos[1] + camera_pos_y  # Car Y + camera height above ground
        
        print("Camera World Position (estimated):")
        print(f"  X={camera_pos_x:.3f}m, Y={camera_world_y:.3f}m, Z={camera_pos_z:.3f}m")
        print()
        
        # Calculate offset between car and camera
        car_camera_offset_x = camera_pos_x - vehicle_pos[0]
        car_camera_offset_z = camera_pos_z - vehicle_pos[2]
        car_camera_distance = np.sqrt(car_camera_offset_x**2 + car_camera_offset_z**2)
        
        print("Car-Camera Offset:")
        print(f"  X offset: {car_camera_offset_x:.3f}m")
        print(f"  Z offset: {car_camera_offset_z:.3f}m")
        print(f"  Distance: {car_camera_distance:.3f}m")
        print()
        
        print("Ground Truth (from Unity):")
        print(f"  Left: {gt_left:.3f}m, Right: {gt_right:.3f}m, Center: {gt_center:.3f}m")
        print(f"  Width: {gt_right - gt_left:.3f}m")
        print()
        
        print("Perception (detected):")
        print(f"  Left: {p_left:.3f}m, Right: {p_right:.3f}m, Center: {p_center:.3f}m")
        print(f"  Width: {p_right - p_left:.3f}m")
        print()
        
        # Calculate what ground truth SHOULD be if using different reference points
        print("=" * 70)
        print("REFERENCE POINT ANALYSIS")
        print("=" * 70)
        print()
        
        # If ground truth is using car position as reference:
        # The values should be relative to car center (0, 0, 0 in vehicle coords)
        # If ground truth is using camera position as reference:
        # The values should account for camera offset
        
        # Calculate expected offset if using camera vs car
        # If camera is 1.0m forward, and we're looking 8m ahead:
        # - Car reference: lanes are at 8m ahead from car
        # - Camera reference: lanes are at 8m ahead from camera (which is 1m forward)
        # So camera reference should see lanes 1m closer (7m ahead from car perspective)
        
        # But wait - the coordinate conversion is lateral (X), not longitudinal (Z)
        # The camera offset in X (lateral) would affect the coordinate conversion
        
        # If camera is offset laterally, the coordinate conversion would be different
        # But typically camera is centered (offset_x = 0)
        
        # The key question: Is ground truth calculating lanes relative to car or camera?
        # If relative to car: values should match perception if perception also uses car
        # If relative to camera: values should account for camera position
        
        # Check if there's a systematic offset
        center_offset = gt_center - p_center
        left_offset = gt_left - p_left
        right_offset = gt_right - p_right
        
        print("Offset Analysis (GT - Perception):")
        print(f"  Center offset: {center_offset:.3f}m")
        print(f"  Left offset: {left_offset:.3f}m")
        print(f"  Right offset: {right_offset:.3f}m")
        print()
        
        # If ground truth is using wrong reference, we'd see:
        # 1. Systematic offset in center (if camera is offset forward/backward)
        # 2. Different width (if camera offset affects calculation)
        # 3. Consistent offset across frames
        
        # Check if offset is consistent (suggests reference point issue)
        if abs(center_offset) > 0.1:
            print("⚠️  SIGNIFICANT CENTER OFFSET DETECTED")
            print(f"   This suggests ground truth might be using wrong reference point")
            if center_offset > 0:
                print(f"   GT center is {center_offset:.3f}m to the RIGHT of perception")
                print(f"   This could mean GT is using car position but should use camera")
            else:
                print(f"   GT center is {abs(center_offset):.3f}m to the LEFT of perception")
                print(f"   This could mean GT is using camera position but should use car")
        else:
            print("✅ Center offset is small - reference point might be correct")
        print()
        
        # Width difference
        width_diff = (gt_right - gt_left) - (p_right - p_left)
        print(f"Width Difference (GT - Perception): {width_diff:.3f}m")
        if abs(width_diff) > 0.5:
            print("⚠️  SIGNIFICANT WIDTH DIFFERENCE")
            print(f"   This suggests:")
            print(f"   1. Road width in Unity is wrong ({gt_right - gt_left:.3f}m)")
            print(f"   2. OR: Coordinate conversion scaling is wrong")
            print(f"   3. OR: Perception is detecting wrong features")
        else:
            print("✅ Width difference is small")
        print()
        
        # Check if offsets are consistent (suggests systematic issue)
        print("Systematic Offset Check:")
        if abs(left_offset - right_offset) < 0.1:
            print(f"  Left and right offsets are similar ({left_offset:.3f}m vs {right_offset:.3f}m)")
            print(f"  This suggests a CENTER offset (reference point issue)")
        else:
            print(f"  Left and right offsets differ ({left_offset:.3f}m vs {right_offset:.3f}m)")
            print(f"  This suggests a WIDTH issue (scaling or road width)")
        print()
        
        # Calculate what the offset should be if camera is offset
        # If camera is 1.0m forward, and we're looking 8m ahead:
        # The lateral offset at 8m would be minimal (camera is centered)
        # But if camera is offset laterally, that would affect the conversion
        
        print("Expected Behavior:")
        print("  - If GT uses CAR position: values should match perception (if perception uses car)")
        print("  - If GT uses CAMERA position: values should account for camera offset")
        print("  - Camera offset X: {:.3f}m (should be ~0 if centered)".format(car_camera_offset_x))
        print("  - Camera offset Z: {:.3f}m (forward offset)".format(car_camera_offset_z))
        print()
        
        if abs(car_camera_offset_x) > 0.1:
            print("⚠️  Camera has significant lateral offset!")
            print(f"   This would affect coordinate conversion if not accounted for")
        else:
            print("✅ Camera is centered laterally (offset X ≈ 0)")
        
        if abs(car_camera_offset_z) > 0.5:
            print(f"⚠️  Camera has significant forward offset ({car_camera_offset_z:.3f}m)!")
            print(f"   This might affect coordinate conversion if not accounted for")
        else:
            print(f"✅ Camera forward offset is small ({car_camera_offset_z:.3f}m)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ground truth reference point")
    parser.add_argument("--recording", type=str, required=True, help="Path to recording file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to test (default: 0)")
    
    args = parser.parse_args()
    
    recording_path = Path(args.recording)
    if not recording_path.exists():
        print(f"ERROR: Recording not found: {recording_path}")
        exit(1)
    
    analyze_reference_point(recording_path, args.frame)

