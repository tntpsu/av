"""
Frame-by-frame analysis tool for perception accuracy.

Compares detection results with ground truth for each frame to identify issues.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Camera parameters (from config)
IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 480.0
CAMERA_FOV = 85.0  # degrees
CAMERA_HEIGHT = 1.0  # meters
CAMERA_OFFSET_X = 0.0411  # meters
LOOKAHEAD_DISTANCE = 8.0  # meters


def convert_pixel_to_vehicle_coords(x_pixels: float, y_pixels: float, 
                                     lookahead_distance: float = LOOKAHEAD_DISTANCE) -> Tuple[float, float]:
    """
    Convert image pixel coordinates to vehicle coordinates.
    
    Args:
        x_pixels: X coordinate in image (0=left, width=right)
        y_pixels: Y coordinate in image (0=top, height=bottom)
        lookahead_distance: Reference distance ahead in meters
    
    Returns:
        (x_vehicle, y_vehicle) in meters
    """
    fov_rad = np.radians(CAMERA_FOV)
    
    # Calculate pixel-to-meter conversion at lookahead distance
    width_at_lookahead = 2.0 * lookahead_distance * np.tan(fov_rad / 2.0)
    pixel_to_meter_x = width_at_lookahead / IMAGE_WIDTH
    
    # Center x coordinate
    x_center = IMAGE_WIDTH / 2.0
    x_offset_pixels = x_pixels - x_center
    
    # Convert x: pixels to meters (lateral)
    x_vehicle = x_offset_pixels * pixel_to_meter_x
    
    # Apply camera offset correction
    x_vehicle += CAMERA_OFFSET_X
    
    # Use lookahead_distance for y_vehicle
    y_vehicle = lookahead_distance
    
    # Clamp to reasonable bounds
    x_vehicle = np.clip(x_vehicle, -5.0, 5.0)
    y_vehicle = np.clip(y_vehicle, 0.0, 50.0)
    
    return x_vehicle, y_vehicle


def evaluate_lane_polynomial_at_y(coeffs: np.ndarray, y_image: float) -> float:
    """
    Evaluate lane polynomial at given y pixel coordinate.
    
    Args:
        coeffs: Polynomial coefficients [a, b, c] for x = ay^2 + by + c
        y_image: Y coordinate in image (0=top, height=bottom)
    
    Returns:
        X coordinate in pixels
    """
    return np.polyval(coeffs, y_image)


def analyze_frame(frame_idx: int, recording_file: Path, 
                  y_image_at_lookahead: float = 390.0) -> Dict:
    """
    Analyze a single frame's detection vs ground truth.
    
    Args:
        frame_idx: Frame index to analyze
        recording_file: Path to HDF5 recording file
        y_image_at_lookahead: Y pixel coordinate to evaluate lanes at (default 390px = ~8m)
    
    Returns:
        Dictionary with analysis results
    """
    with h5py.File(recording_file, 'r') as f:
        # Get ground truth
        gt_left_x = float(f['ground_truth/left_lane_x'][frame_idx])
        gt_right_x = float(f['ground_truth/right_lane_x'][frame_idx])
        gt_center_x = float(f['ground_truth/lane_center_x'][frame_idx])
        gt_width = gt_right_x - gt_left_x
        gt_desired_heading = float(f['ground_truth/desired_heading'][frame_idx])
        
        # Get detected values
        detected_left_x = float(f['perception/left_lane_x'][frame_idx])
        detected_right_x = float(f['perception/right_lane_x'][frame_idx])
        detected_width = detected_right_x - detected_left_x
        detected_center = (detected_left_x + detected_right_x) / 2.0
        
        # Get lane coefficients if available
        lane_coeffs = None
        if 'perception/lane_lines' in f:
            try:
                lane_lines_data = f['perception/lane_lines']
                # Lane lines might be stored as variable-length arrays
                # Try to get it for this frame
                if hasattr(lane_lines_data, 'shape') and len(lane_lines_data.shape) > 0:
                    # If it's a fixed-size array, try to read it
                    try:
                        coeffs_raw = lane_lines_data[frame_idx]
                        # Convert to numpy array if possible
                        if isinstance(coeffs_raw, np.ndarray):
                            lane_coeffs = coeffs_raw
                    except:
                        pass
            except:
                pass
        
        # Calculate errors
        left_error = detected_left_x - gt_left_x
        right_error = detected_right_x - gt_right_x
        center_error = detected_center - gt_center_x
        width_error = detected_width - gt_width
        
        # Try to reconstruct what pixel positions were detected
        # We can reverse-engineer from vehicle coords back to pixels
        # x_vehicle = (x_pixels - center) * pixel_to_meter + camera_offset
        # So: x_pixels = center + (x_vehicle - camera_offset) / pixel_to_meter
        
        fov_rad = np.radians(CAMERA_FOV)
        width_at_lookahead = 2.0 * LOOKAHEAD_DISTANCE * np.tan(fov_rad / 2.0)
        pixel_to_meter_x = width_at_lookahead / IMAGE_WIDTH
        x_center = IMAGE_WIDTH / 2.0
        
        detected_left_px = x_center + (detected_left_x - CAMERA_OFFSET_X) / pixel_to_meter_x
        detected_right_px = x_center + (detected_right_x - CAMERA_OFFSET_X) / pixel_to_meter_x
        
        # What pixel positions should ground truth correspond to?
        expected_left_px = x_center + (gt_left_x - CAMERA_OFFSET_X) / pixel_to_meter_x
        expected_right_px = x_center + (gt_right_x - CAMERA_OFFSET_X) / pixel_to_meter_x
        
        pixel_left_error = detected_left_px - expected_left_px
        pixel_right_error = detected_right_px - expected_right_px
        pixel_width_error = (detected_right_px - detected_left_px) - (expected_right_px - expected_left_px)
        
        # Get detection confidence and method
        confidence = float(f['perception/confidence'][frame_idx])
        detection_method = f['perception/detection_method'][frame_idx].decode('utf-8') if isinstance(
            f['perception/detection_method'][frame_idx], bytes) else str(f['perception/detection_method'][frame_idx])
        
        # Get timestamp
        timestamp = float(f['camera/timestamps'][frame_idx])
        
        result = {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'ground_truth': {
                'left_x': gt_left_x,
                'right_x': gt_right_x,
                'center_x': gt_center_x,
                'width': gt_width,
                'desired_heading_deg': gt_desired_heading,
            },
            'detected': {
                'left_x': detected_left_x,
                'right_x': detected_right_x,
                'center_x': detected_center,
                'width': detected_width,
            },
            'errors': {
                'left_x_m': left_error,
                'right_x_m': right_error,
                'center_x_m': center_error,
                'width_m': width_error,
            },
            'pixel_positions': {
                'detected_left_px': detected_left_px,
                'detected_right_px': detected_right_px,
                'expected_left_px': expected_left_px,
                'expected_right_px': expected_right_px,
                'pixel_left_error_px': pixel_left_error,
                'pixel_right_error_px': pixel_right_error,
                'pixel_width_error_px': pixel_width_error,
            },
            'detection_info': {
                'confidence': confidence,
                'method': detection_method,
            },
        }
        
        # If we have lane coefficients, add them
        if lane_coeffs is not None:
            result['lane_coeffs'] = lane_coeffs.tolist() if isinstance(lane_coeffs, np.ndarray) else lane_coeffs
        
        return result


def print_frame_analysis(frame_data: Dict, verbose: bool = False):
    """Print analysis for a single frame."""
    f = frame_data['frame_idx']
    gt = frame_data['ground_truth']
    det = frame_data['detected']
    err = frame_data['errors']
    px = frame_data['pixel_positions']
    info = frame_data['detection_info']
    
    print(f"\n{'='*100}")
    print(f"FRAME {f} (t={frame_data['timestamp']:.3f}s)")
    print(f"{'='*100}")
    print(f"Detection: {info['method']} (confidence: {info['confidence']:.2f})")
    print()
    
    print("GROUND TRUTH vs DETECTED (vehicle coordinates, meters):")
    print("-" * 100)
    print(f"  Left lane:   GT={gt['left_x']:7.3f}m  Detected={det['left_x']:7.3f}m  Error={err['left_x_m']:7.3f}m")
    print(f"  Right lane:  GT={gt['right_x']:7.3f}m  Detected={det['right_x']:7.3f}m  Error={err['right_x_m']:7.3f}m")
    print(f"  Center:      GT={gt['center_x']:7.3f}m  Detected={det['center_x']:7.3f}m  Error={err['center_x_m']:7.3f}m")
    print(f"  Width:       GT={gt['width']:7.3f}m  Detected={det['width']:7.3f}m  Error={err['width_m']:7.3f}m")
    print()
    
    if verbose:
        print("PIXEL POSITIONS (at y=390px, ~8m lookahead):")
        print("-" * 100)
        print(f"  Left lane:   Expected={px['expected_left_px']:7.1f}px  Detected={px['detected_left_px']:7.1f}px  Error={px['pixel_left_error_px']:7.1f}px")
        print(f"  Right lane:  Expected={px['expected_right_px']:7.1f}px  Detected={px['detected_right_px']:7.1f}px  Error={px['pixel_right_error_px']:7.1f}px")
        print(f"  Width:       Expected={px['expected_right_px']-px['expected_left_px']:7.1f}px  Detected={px['detected_right_px']-px['detected_left_px']:7.1f}px  Error={px['pixel_width_error_px']:7.1f}px")
        print()
    
    # Classification
    if abs(err['width_m']) > 1.0:
        print(f"  ⚠️  LARGE WIDTH ERROR: {err['width_m']:.3f}m")
    if abs(err['left_x_m']) > 1.0 or abs(err['right_x_m']) > 1.0:
        print(f"  ⚠️  LARGE POSITION ERROR: left={err['left_x_m']:.3f}m, right={err['right_x_m']:.3f}m")
    if abs(err['center_x_m']) > 0.5:
        print(f"  ⚠️  CENTER OFFSET: {err['center_x_m']:.3f}m")


def main():
    parser = argparse.ArgumentParser(description='Frame-by-frame perception analysis')
    parser.add_argument('recording_file', type=Path, help='Path to perception replay HDF5 file')
    parser.add_argument('--frames', type=str, default='0,10,20,30', 
                       help='Comma-separated frame indices to analyze (default: 0,10,20,30)')
    parser.add_argument('--all', action='store_true', 
                       help='Analyze all frames (summary statistics)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show pixel-level details')
    parser.add_argument('--output', type=Path,
                       help='Save detailed analysis to JSON file')
    
    args = parser.parse_args()
    
    if not args.recording_file.exists():
        print(f"Error: Recording file not found: {args.recording_file}")
        return 1
    
    print("=" * 100)
    print("FRAME-BY-FRAME PERCEPTION ANALYSIS")
    print("=" * 100)
    print(f"Recording: {args.recording_file.name}")
    print()
    
    with h5py.File(args.recording_file, 'r') as f:
        num_frames = len(f['camera/timestamps'])
        print(f"Total frames: {num_frames}")
    
    if args.all:
        # Analyze all frames and show statistics
        print("\nAnalyzing all frames...")
        all_results = []
        for frame_idx in range(num_frames):
            try:
                result = analyze_frame(frame_idx, args.recording_file)
                all_results.append(result)
            except Exception as e:
                print(f"Error analyzing frame {frame_idx}: {e}")
                continue
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved detailed analysis to: {args.output}")
        
        # Calculate statistics
        errors = {
            'left_x': [r['errors']['left_x_m'] for r in all_results],
            'right_x': [r['errors']['right_x_m'] for r in all_results],
            'center_x': [r['errors']['center_x_m'] for r in all_results],
            'width': [r['errors']['width_m'] for r in all_results],
        }
        
        pixel_errors = {
            'left_px': [r['pixel_positions']['pixel_left_error_px'] for r in all_results],
            'right_px': [r['pixel_positions']['pixel_right_error_px'] for r in all_results],
            'width_px': [r['pixel_positions']['pixel_width_error_px'] for r in all_results],
        }
        
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS (All Frames)")
        print("=" * 100)
        print("\nVehicle Coordinate Errors (meters):")
        print("-" * 100)
        for key, values in errors.items():
            values = np.array(values)
            print(f"  {key:12s}: mean={np.mean(np.abs(values)):7.3f}m, "
                  f"std={np.std(values):7.3f}m, max={np.max(np.abs(values)):7.3f}m, "
                  f"bias={np.mean(values):7.3f}m")
        
        print("\nPixel Position Errors (at y=390px, ~8m lookahead):")
        print("-" * 100)
        for key, values in pixel_errors.items():
            values = np.array(values)
            print(f"  {key:12s}: mean={np.mean(np.abs(values)):7.1f}px, "
                  f"std={np.std(values):7.1f}px, max={np.max(np.abs(values)):7.1f}px, "
                  f"bias={np.mean(values):7.1f}px")
        
        # Show worst frames
        print("\nWorst frames (by absolute width error):")
        print("-" * 100)
        sorted_frames = sorted(all_results, key=lambda x: abs(x['errors']['width_m']), reverse=True)
        for i, frame_data in enumerate(sorted_frames[:10]):
            print(f"  Frame {frame_data['frame_idx']:3d}: width_error={frame_data['errors']['width_m']:7.3f}m, "
                  f"left_err={frame_data['errors']['left_x_m']:7.3f}m, right_err={frame_data['errors']['right_x_m']:7.3f}m")
        
    else:
        # Analyze specific frames
        frame_indices = [int(f.strip()) for f in args.frames.split(',')]
        frame_indices = [f for f in frame_indices if 0 <= f < num_frames]
        
        all_results = []
        for frame_idx in frame_indices:
            try:
                result = analyze_frame(frame_idx, args.recording_file)
                all_results.append(result)
                print_frame_analysis(result, verbose=args.verbose)
            except Exception as e:
                print(f"Error analyzing frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved detailed analysis to: {args.output}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
