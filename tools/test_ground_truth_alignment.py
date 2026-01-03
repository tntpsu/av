#!/usr/bin/env python3
"""
Test script to find correct parameters for ground truth alignment.
Varies different parameters to see which combination makes green lines match red lines.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, Dict, List

def vehicle_to_image_coords(
    x_vehicle: float,
    y_vehicle: float,
    image_width: int = 640,
    image_height: int = 480,
    camera_fov: float = 110.0,
    camera_height: float = 1.2,
    horizontal_fov: float = None,
    y_lookahead: float = None
) -> Tuple[float, float]:
    """
    Convert vehicle coordinates to image coordinates.
    This is the inverse of what perception does.
    """
    # Use horizontal FOV if provided, otherwise calculate from vertical FOV
    if horizontal_fov is not None:
        fov_rad = np.radians(horizontal_fov)
    else:
        fov_rad = np.radians(camera_fov)
        # Calculate horizontal FOV from vertical
        aspect = image_width / image_height
        fov_rad = 2.0 * np.arctan(np.tan(fov_rad / 2.0) * aspect)
    
    # Calculate y position (vertical) - use provided y_lookahead if available
    if y_lookahead is not None:
        y_image = y_lookahead
    else:
        # Estimate from distance (simplified model)
        base_distance = 1.5  # meters at bottom
        if y_vehicle < base_distance:
            y_image = image_height
        else:
            y_normalized = base_distance / y_vehicle
            y_from_bottom = y_normalized * image_height
            y_image = image_height - y_from_bottom
    
    # Calculate x position (lateral) using horizontal FOV
    width_at_distance = 2.0 * y_vehicle * np.tan(fov_rad / 2.0)
    pixel_to_meter = width_at_distance / image_width
    
    # Convert x (lateral) - center of image is x=0 in vehicle coords
    x_center = image_width / 2
    x_image = x_center + (x_vehicle / pixel_to_meter)
    
    return x_image, y_image

def test_parameter_combinations(
    recording_path: Path,
    frame_idx: int = 0,
    max_combinations: int = 100
) -> List[Dict]:
    """
    Test different parameter combinations to find what makes green lines align with red lines.
    """
    print("=" * 70)
    print("GROUND TRUTH ALIGNMENT TEST")
    print("=" * 70)
    print()
    
    with h5py.File(recording_path, 'r') as f:
        # Get frame data
        if 'camera/images' not in f:
            print("ERROR: No camera images in recording")
            return []
        
        image_height, image_width = f['camera/images'].shape[1:3]
        
        # Get perception data (red lines - these are CORRECT)
        p_left = f['perception/left_lane_line_x'][frame_idx]
        p_right = f['perception/right_lane_line_x'][frame_idx]
        p_width = p_right - p_left
        
        # Get ground truth data (green lines - these are WRONG)
        gt_left = f['ground_truth/left_lane_line_x'][frame_idx]
        gt_right = f['ground_truth/right_lane_line_x'][frame_idx]
        gt_width = gt_right - gt_left
        
        # Get camera calibration data
        camera_8m_screen_y = f['vehicle/camera_8m_screen_y'][frame_idx] if 'vehicle/camera_8m_screen_y' in f else -1.0
        camera_horizontal_fov = f['vehicle/camera_horizontal_fov'][frame_idx] if 'vehicle/camera_horizontal_fov' in f else 110.0
        camera_pos_y = f['vehicle/camera_pos_y'][frame_idx] if 'vehicle/camera_pos_y' in f else 1.2
        
        print(f"Frame {frame_idx} Data:")
        print(f"  Perception (RED - CORRECT):")
        print(f"    left={p_left:.3f}m, right={p_right:.3f}m, width={p_width:.3f}m")
        print(f"  Ground Truth (GREEN - WRONG):")
        print(f"    left={gt_left:.3f}m, right={gt_right:.3f}m, width={gt_width:.3f}m")
        print(f"  Camera Calibration:")
        print(f"    camera_8m_screen_y={camera_8m_screen_y:.1f}px")
        print(f"    camera_horizontal_fov={camera_horizontal_fov:.2f}°")
        print(f"    camera_pos_y={camera_pos_y:.3f}m")
        print()
        
        # Convert perception to image coords (this is what red lines show)
        p_left_img, p_y = vehicle_to_image_coords(
            p_left, 8.0, image_width, image_height,
            camera_fov=110.0, camera_height=1.2,
            horizontal_fov=camera_horizontal_fov if camera_horizontal_fov > 0 else None,
            y_lookahead=camera_8m_screen_y if camera_8m_screen_y > 0 else None
        )
        p_right_img, _ = vehicle_to_image_coords(
            p_right, 8.0, image_width, image_height,
            camera_fov=110.0, camera_height=1.2,
            horizontal_fov=camera_horizontal_fov if camera_horizontal_fov > 0 else None,
            y_lookahead=camera_8m_screen_y if camera_8m_screen_y > 0 else None
        )
        
        print(f"Perception in image coords (RED lines):")
        print(f"  left={p_left_img:.1f}px, right={p_right_img:.1f}px, y={p_y:.1f}px")
        print()
        
        # Test different parameter combinations
        print("Testing parameter combinations...")
        print()
        
        # Parameter ranges to test
        camera_heights = [1.2]  # Focus on correct height
        # Test a wider range of FOV values to find the right one
        horizontal_fovs = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
        if camera_horizontal_fov > 0:
            horizontal_fovs.append(camera_horizontal_fov)
        horizontal_fovs = sorted(set(horizontal_fovs))
        y_lookaheads = [295.7, camera_8m_screen_y if camera_8m_screen_y > 0 else 295.7]
        # Test finer distance adjustments
        distance_adjustments = [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]
        
        results = []
        test_count = 0
        
        for cam_height in camera_heights:
            for h_fov in horizontal_fovs:
                for y_look in y_lookaheads:
                    for dist_adj in distance_adjustments:
                        if test_count >= max_combinations:
                            break
                        
                        # Convert ground truth to image coords with these parameters
                        gt_left_img, gt_y = vehicle_to_image_coords(
                            gt_left, 8.0 + dist_adj, image_width, image_height,
                            camera_fov=110.0, camera_height=cam_height,
                            horizontal_fov=h_fov,
                            y_lookahead=y_look
                        )
                        gt_right_img, _ = vehicle_to_image_coords(
                            gt_right, 8.0 + dist_adj, image_width, image_height,
                            camera_fov=110.0, camera_height=cam_height,
                            horizontal_fov=h_fov,
                            y_lookahead=y_look
                        )
                        
                        # Calculate error
                        left_error = abs(gt_left_img - p_left_img)
                        right_error = abs(gt_right_img - p_right_img)
                        total_error = left_error + right_error
                        
                        results.append({
                            'camera_height': cam_height,
                            'horizontal_fov': h_fov,
                            'y_lookahead': y_look,
                            'distance_adjustment': dist_adj,
                            'gt_left_img': gt_left_img,
                            'gt_right_img': gt_right_img,
                            'left_error': left_error,
                            'right_error': right_error,
                            'total_error': total_error
                        })
                        
                        test_count += 1
                    
                    if test_count >= max_combinations:
                        break
                if test_count >= max_combinations:
                    break
            if test_count >= max_combinations:
                break
        
        # Sort by total error
        results.sort(key=lambda x: x['total_error'])
        
        # Print top 10 results
        print("Top 10 Parameter Combinations (sorted by alignment error):")
        print()
        print(f"{'Camera Height':<15} {'H-FOV':<10} {'Y-Look':<10} {'Dist Adj':<10} {'Left Err':<10} {'Right Err':<10} {'Total Err':<10}")
        print("-" * 85)
        
        for i, result in enumerate(results[:10]):
            print(f"{result['camera_height']:<15.2f} "
                  f"{result['horizontal_fov']:<10.2f} "
                  f"{result['y_lookahead']:<10.1f} "
                  f"{result['distance_adjustment']:<10.2f} "
                  f"{result['left_error']:<10.2f} "
                  f"{result['right_error']:<10.2f} "
                  f"{result['total_error']:<10.2f}")
        
        print()
        print("Best combination:")
        best = results[0]
        print(f"  Camera Height: {best['camera_height']:.2f}m")
        print(f"  Horizontal FOV: {best['horizontal_fov']:.2f}°")
        print(f"  Y Lookahead: {best['y_lookahead']:.1f}px")
        print(f"  Distance Adjustment: {best['distance_adjustment']:.2f}m")
        print(f"  Total Error: {best['total_error']:.2f}px")
        print()
        print(f"  Ground Truth in image coords with best params:")
        print(f"    left={best['gt_left_img']:.1f}px, right={best['gt_right_img']:.1f}px")
        print(f"  Perception in image coords (target):")
        print(f"    left={p_left_img:.1f}px, right={p_right_img:.1f}px")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ground truth alignment parameters")
    parser.add_argument("--recording", type=str, required=True, help="Path to recording file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to test (default: 0)")
    parser.add_argument("--max-combinations", type=int, default=100, help="Max parameter combinations to test")
    
    args = parser.parse_args()
    
    recording_path = Path(args.recording)
    if not recording_path.exists():
        print(f"ERROR: Recording not found: {recording_path}")
        exit(1)
    
    test_parameter_combinations(recording_path, args.frame, args.max_combinations)

