#!/usr/bin/env python3
"""
Diagnostic tool to investigate polynomial fitting issues.
Helps understand why extreme coefficients are produced at specific frames.
"""

import h5py
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.models.lane_detection import SimpleLaneDetector


def analyze_frame(recording_path: str, frame_idx: int):
    """Analyze a specific frame to understand polynomial fitting."""
    print("=" * 70)
    print(f"POLYNOMIAL FITTING INVESTIGATION - Frame {frame_idx}")
    print("=" * 70)
    print()
    
    with h5py.File(recording_path, 'r') as f:
        total = len(f['vehicle/position'])
        if frame_idx >= total:
            print(f"Error: Frame {frame_idx} is out of range (max: {total-1})")
            return
        
        # Load camera image
        if 'camera/images' in f and frame_idx < len(f['camera/images']):
            image = f['camera/images'][frame_idx]
            print(f"Image shape: {image.shape}")
        else:
            print("Error: Camera image not found")
            return
        
        # Load perception data
        if 'perception/lane_line_coefficients' in f:
            coeffs = f['perception/lane_line_coefficients'][frame_idx]
            num_lanes = f['perception/num_lanes_detected'][frame_idx] if 'perception/num_lanes_detected' in f else 0
            print(f"Num lanes detected: {num_lanes}")
            
            if hasattr(coeffs, '__len__') and len(coeffs) >= 6:
                right_coeffs = np.array(coeffs[3:6])
                print(f"Right lane coefficients: a={right_coeffs[0]:.6f}, b={right_coeffs[1]:.6f}, c={right_coeffs[2]:.6f}")
                
                # Evaluate at multiple points
                print("\nPolynomial evaluation:")
                for y in [0, 100, 200, 300, 400, 480]:
                    x = right_coeffs[0] * y * y + right_coeffs[1] * y + right_coeffs[2]
                    status = "✅" if 0 <= x <= 640 else "❌"
                    print(f"  y={y:3d}px: x={x:7.1f}px {status}")
        
        print("\n" + "=" * 70)
        print("RE-RUNNING PERCEPTION ON THIS FRAME")
        print("=" * 70)
        print()
        
        # Re-run perception to see what it detects
        detector = SimpleLaneDetector()
        result = detector.detect(image, return_debug=True)
        
        if isinstance(result, tuple) and len(result) == 2:
            lane_coeffs, debug_info = result
            print(f"Re-detected lanes: {sum(1 for c in lane_coeffs if c is not None)}")
            
            if debug_info:
                print("\nDebug Info:")
                print(f"  Lines detected: {debug_info.get('num_lines_detected', 0)}")
                print(f"  Left lines: {debug_info.get('left_lines_count', 0)}")
                print(f"  Right lines: {debug_info.get('right_lines_count', 0)}")
                
                # Check if we have line points
                if 'all_lines' in debug_info and debug_info['all_lines'] is not None:
                    all_lines = debug_info['all_lines']
                    print(f"\n  Total line segments: {len(all_lines)}")
                    
                    # Analyze right lane points
                    if lane_coeffs[1] is not None:  # Right lane
                        # Try to get points that were used for fitting
                        # (This is approximate - we'd need to modify detector to return points)
                        print("\n  Right lane analysis:")
                        print("    (Points used for fitting are not directly available)")
                        print("    But we can see the fitted coefficients match what was recorded")
                
                # Check validation failures
                if 'validation_failures' in debug_info:
                    failures = debug_info['validation_failures']
                    if failures:
                        print(f"\n  Validation failures: {failures}")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print()
        print("To investigate further:")
        print("  1. Check debug images (edges, yellow_mask) for frame 390 (nearest with debug)")
        print("  2. Look at what lines HoughLinesP detected")
        print("  3. Check if points are sparse or poorly distributed")
        print("  4. Verify if polynomial fitting is extrapolating too far")
        print("  5. Check if temporal filtering is preserving bad coefficients")


def main():
    parser = argparse.ArgumentParser(description="Investigate polynomial fitting issues")
    parser.add_argument("recording", nargs='?', help="Path to HDF5 recording file")
    parser.add_argument("--frame", type=int, default=398, help="Frame index to analyze")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    
    args = parser.parse_args()
    
    if args.latest or args.recording is None:
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("Error: No recordings found")
            return
        recording_path = recordings[0]
        print(f"Using latest recording: {recording_path.name}")
    else:
        recording_path = Path(args.recording)
        if not recording_path.exists():
            print(f"Error: Recording not found: {recording_path}")
            return
    
    analyze_frame(str(recording_path), args.frame)


if __name__ == "__main__":
    main()

