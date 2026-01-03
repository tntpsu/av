#!/usr/bin/env python3
"""
Tool to analyze differences between Unity conditions and test conditions.
Helps align tests with actual Unity scene setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import cv2
from perception.models.lane_detection import SimpleLaneDetector


def analyze_unity_frame(recording_path: Path, frame_idx: int = 0):
    """Analyze a Unity frame to understand actual conditions."""
    print("="*80)
    print("UNITY FRAME ANALYSIS")
    print("="*80)
    
    with h5py.File(recording_path, 'r') as f:
        image = np.array(f['camera/images'][frame_idx])
        h, w = image.shape[:2]
        
        print(f"\nImage Properties:")
        print(f"  Shape: {image.shape}")
        print(f"  Dtype: {image.dtype}")
        print(f"  Value range: [{image.min()}, {image.max()}]")
        
        # Analyze colors
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Yellow detection
        yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
        yellow_pixels = np.sum(yellow_mask > 0)
        print(f"\nColor Analysis:")
        print(f"  Yellow pixels: {yellow_pixels} ({yellow_pixels/image.size*100:.2f}%)")
        
        # White detection
        white_mask = cv2.inRange(hsv, (0, 0, 220), (180, 15, 255))
        white_pixels = np.sum(white_mask > 0)
        print(f"  White pixels: {white_pixels} ({white_pixels/image.size*100:.2f}%)")
        
        # Gray detection
        gray_mask = cv2.inRange(hsv, (0, 0, 120), (180, 40, 180))
        gray_pixels = np.sum(gray_mask > 0)
        print(f"  Gray pixels: {gray_pixels} ({gray_pixels/image.size*100:.2f}%)")
        
        # Detect lanes
        detector = SimpleLaneDetector()
        lanes = detector.detect(image)
        
        print(f"\nLane Detection:")
        if lanes[0] is not None and lanes[1] is not None:
            # Evaluate at bottom of image
            left_x_bottom = np.polyval(lanes[0], h)
            right_x_bottom = np.polyval(lanes[1], h)
            
            print(f"  Left lane X (bottom): {left_x_bottom:.1f} pixels")
            print(f"  Right lane X (bottom): {right_x_bottom:.1f} pixels")
            print(f"  Lane width: {right_x_bottom - left_x_bottom:.1f} pixels")
            print(f"  Image center: {w/2:.1f} pixels")
            print(f"  Lane center: {(left_x_bottom + right_x_bottom)/2:.1f} pixels")
            
            # Check coefficients
            print(f"\nLane Coefficients:")
            print(f"  Left: {lanes[0]}")
            print(f"  Right: {lanes[1]}")
            
            if len(lanes[0]) >= 2 and len(lanes[1]) >= 2:
                left_linear = lanes[0][-2]
                right_linear = lanes[1][-2]
                linear_diff = abs(left_linear - right_linear)
                print(f"\nParallelism:")
                print(f"  Left linear: {left_linear:.4f}")
                print(f"  Right linear: {right_linear:.4f}")
                print(f"  Difference: {linear_diff:.4f}")
                print(f"  Parallel? {linear_diff < 0.1}")
        else:
            print(f"  ⚠ Lanes not detected!")
            print(f"  Left: {lanes[0] is not None}")
            print(f"  Right: {lanes[1] is not None}")
        
        # Check perception data
        if 'perception/left_lane_x' in f and 'perception/right_lane_x' in f:
            left_x = np.array(f['perception/left_lane_x'])[frame_idx]
            right_x = np.array(f['perception/right_lane_x'])[frame_idx]
            print(f"\nPerception Data (vehicle coordinates):")
            print(f"  Left lane X: {left_x:.3f} m")
            print(f"  Right lane X: {right_x:.3f} m")
            print(f"  Lane width: {right_x - left_x:.3f} m")


def compare_test_image():
    """Compare test image with Unity conditions."""
    print("\n" + "="*80)
    print("TEST IMAGE COMPARISON")
    print("="*80)
    
    from tests.test_perception_straight_road import create_straight_road_image
    
    test_image = create_straight_road_image()
    h, w = test_image.shape[:2]
    
    print(f"\nTest Image Properties:")
    print(f"  Shape: {test_image.shape}")
    
    # Analyze colors
    hsv = cv2.cvtColor(test_image, cv2.COLOR_RGB2HSV)
    
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 220), (180, 15, 255))
    gray_mask = cv2.inRange(hsv, (0, 0, 120), (180, 40, 180))
    
    print(f"\nColor Analysis:")
    print(f"  Yellow pixels: {np.sum(yellow_mask > 0)} ({np.sum(yellow_mask > 0)/test_image.size*100:.2f}%)")
    print(f"  White pixels: {np.sum(white_mask > 0)} ({np.sum(white_mask > 0)/test_image.size*100:.2f}%)")
    print(f"  Gray pixels: {np.sum(gray_mask > 0)} ({np.sum(gray_mask > 0)/test_image.size*100:.2f}%)")
    
    # Detect lanes
    detector = SimpleLaneDetector()
    lanes = detector.detect(test_image)
    
    print(f"\nLane Detection:")
    if lanes[0] is not None and lanes[1] is not None:
        left_x_bottom = np.polyval(lanes[0], h)
        right_x_bottom = np.polyval(lanes[1], h)
        
        print(f"  Left lane X (bottom): {left_x_bottom:.1f} pixels")
        print(f"  Right lane X (bottom): {right_x_bottom:.1f} pixels")
        print(f"  Lane width: {right_x_bottom - left_x_bottom:.1f} pixels")
    else:
        print(f"  ⚠ Lanes not detected!")
        print(f"  Left: {lanes[0] is not None}")
        print(f"  Right: {lanes[1] is not None}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Unity vs test conditions')
    parser.add_argument('--recording', type=str, help='Path to recording file')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to analyze')
    
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
    
    if not recording_path.exists():
        print(f"Recording not found: {recording_path}")
        return
    
    # Analyze Unity frame
    analyze_unity_frame(recording_path, args.frame)
    
    # Compare with test image
    compare_test_image()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\nTo align tests with Unity:")
    print("  1. Use correct colors: Yellow=(255,255,0) in RGB, not (0,255,255)")
    print("  2. Match lane positions based on actual Unity detection")
    print("  3. Use dashed pattern for left lane (white)")
    print("  4. Use solid line for right lane (yellow)")
    print("  5. Match road surface color (dark gray)")


if __name__ == '__main__':
    main()

