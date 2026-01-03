"""
Visualize detected lanes vs ground truth lanes on camera images.

This tool loads a recording and overlays detected lanes (red) and ground truth lanes (green)
on the actual camera images, making it easy to see where detection is failing.
"""

import argparse
import h5py
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List

# Camera parameters
IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 480.0
CAMERA_FOV = 85.0
CAMERA_OFFSET_X = 0.0411
LOOKAHEAD_DISTANCE = 8.0


def convert_vehicle_to_pixel_coords(x_vehicle: float, lookahead_distance: float = LOOKAHEAD_DISTANCE) -> float:
    """Convert vehicle x coordinate (meters) to pixel x coordinate."""
    fov_rad = np.radians(CAMERA_FOV)
    width_at_lookahead = 2.0 * lookahead_distance * np.tan(fov_rad / 2.0)
    pixel_to_meter_x = width_at_lookahead / IMAGE_WIDTH
    x_center = IMAGE_WIDTH / 2.0
    x_pixels = x_center + (x_vehicle - CAMERA_OFFSET_X) / pixel_to_meter_x
    return x_pixels


def draw_lane_line(image: np.ndarray, x_vehicle_left: float, x_vehicle_right: float,
                   lookahead_distance: float, color: Tuple[int, int, int], thickness: int = 3):
    """
    Draw lane lines on image using vehicle coordinates.
    
    Args:
        image: Input image
        x_vehicle_left: Left lane x position in vehicle coordinates (meters)
        x_vehicle_right: Right lane x position in vehicle coordinates (meters)
        lookahead_distance: Distance ahead (meters)
        color: Line color (BGR)
        thickness: Line thickness
    """
    h, w = image.shape[:2]
    
    # Convert vehicle coordinates to pixel coordinates at lookahead
    # For simplicity, draw lines at y positions corresponding to lookahead
    # We'll draw straight vertical lines at the x positions
    
    x_pixel_left = convert_vehicle_to_pixel_coords(x_vehicle_left, lookahead_distance)
    x_pixel_right = convert_vehicle_to_pixel_coords(x_vehicle_right, lookahead_distance)
    
    # Clamp to image bounds
    x_pixel_left = int(np.clip(x_pixel_left, 0, w - 1))
    x_pixel_right = int(np.clip(x_pixel_right, 0, w - 1))
    
    # Draw vertical lines (for now - could be enhanced to draw polynomial curves)
    # Draw from bottom 2/3 to bottom (where lanes are visible)
    y_start = int(h * 0.33)
    y_end = h
    
    cv2.line(image, (x_pixel_left, y_start), (x_pixel_left, y_end), color, thickness)
    cv2.line(image, (x_pixel_right, y_start), (x_pixel_right, y_end), color, thickness)
    
    # Draw center line
    x_center = (x_pixel_left + x_pixel_right) // 2
    cv2.line(image, (x_center, y_start), (x_center, y_end), color, thickness // 2)


def visualize_frame(recording_file: Path, frame_idx: int, output_file: Optional[Path] = None,
                    show: bool = True):
    """
    Visualize a single frame with detected and ground truth lanes overlaid.
    
    Args:
        recording_file: Path to HDF5 recording file
        frame_idx: Frame index to visualize
        output_file: Optional path to save output image
        show: Whether to display the image
    """
    with h5py.File(recording_file, 'r') as f:
        # Load camera image
        images = f['camera/images']
        if frame_idx >= len(images):
            print(f"Error: Frame {frame_idx} not found (max: {len(images) - 1})")
            return
        
        image = images[frame_idx]
        
        # Convert from RGB to BGR for OpenCV
        if image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Create a copy for drawing
        vis_image = image_bgr.copy()
        
        # Load ground truth
        gt_left_x = float(f['ground_truth/left_lane_x'][frame_idx])
        gt_right_x = float(f['ground_truth/right_lane_x'][frame_idx])
        gt_center_x = float(f['ground_truth/lane_center_x'][frame_idx])
        gt_width = gt_right_x - gt_left_x
        
        # Load detected values
        detected_left_x = float(f['perception/left_lane_x'][frame_idx])
        detected_right_x = float(f['perception/right_lane_x'][frame_idx])
        detected_center = (detected_left_x + detected_right_x) / 2.0
        detected_width = detected_right_x - detected_left_x
        
        # Calculate errors
        left_error = detected_left_x - gt_left_x
        right_error = detected_right_x - gt_right_x
        center_error = detected_center - gt_center_x
        width_error = detected_width - gt_width
        
        # Draw ground truth lanes (green)
        draw_lane_line(vis_image, gt_left_x, gt_right_x, LOOKAHEAD_DISTANCE, (0, 255, 0), 3)
        
        # Draw detected lanes (red)
        draw_lane_line(vis_image, detected_left_x, detected_right_x, LOOKAHEAD_DISTANCE, (0, 0, 255), 3)
        
        # Add text overlay with metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        y_offset = 20
        
        # Ground truth info (green text)
        cv2.putText(vis_image, f"GT: L={gt_left_x:.2f}m, R={gt_right_x:.2f}m, W={gt_width:.2f}m",
                   (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        # Detected info (red text)
        cv2.putText(vis_image, f"Det: L={detected_left_x:.2f}m, R={detected_right_x:.2f}m, W={detected_width:.2f}m",
                   (10, y_offset + 20), font, font_scale, (0, 0, 255), thickness)
        
        # Error info (yellow text)
        cv2.putText(vis_image, f"Err: L={left_error:.2f}m, R={right_error:.2f}m, W={width_error:.2f}m",
                   (10, y_offset + 40), font, font_scale, (0, 255, 255), thickness)
        
        # Frame info
        timestamp = float(f['camera/timestamps'][frame_idx])
        confidence = float(f['perception/confidence'][frame_idx])
        method = f['perception/detection_method'][frame_idx].decode('utf-8') if isinstance(
            f['perception/detection_method'][frame_idx], bytes) else str(f['perception/detection_method'][frame_idx])
        
        cv2.putText(vis_image, f"Frame {frame_idx} | t={timestamp:.2f}s | {method} (conf={confidence:.2f})",
                   (10, y_offset + 60), font, font_scale, (255, 255, 255), thickness)
        
        # Add legend
        cv2.putText(vis_image, "Green: Ground Truth | Red: Detected",
                   (10, vis_image.shape[0] - 10), font, font_scale, (255, 255, 255), thickness)
        
        # Save if requested
        if output_file:
            cv2.imwrite(str(output_file), vis_image)
            print(f"Saved visualization to: {output_file}")
        
        # Show if requested
        if show:
            cv2.imshow(f"Frame {frame_idx} - Detected (Red) vs Ground Truth (Green)", vis_image)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def visualize_error_frames(recording_file: Path, num_frames: int = 10, output_dir: Optional[Path] = None):
    """
    Visualize frames with the largest errors.
    
    Args:
        recording_file: Path to HDF5 recording file
        num_frames: Number of worst frames to visualize
        output_dir: Optional directory to save images
    """
    print(f"Finding {num_frames} frames with largest width errors...")
    
    # Find worst frames
    with h5py.File(recording_file, 'r') as f:
        n = len(f['camera/timestamps'])
        
        errors = []
        for i in range(n):
            gt_width = f['ground_truth/right_lane_x'][i] - f['ground_truth/left_lane_x'][i]
            detected_width = f['perception/right_lane_x'][i] - f['perception/left_lane_x'][i]
            width_error = abs(detected_width - gt_width)
            errors.append((i, width_error))
        
        # Sort by error (descending)
        errors.sort(key=lambda x: x[1], reverse=True)
        worst_frames = [frame_idx for frame_idx, _ in errors[:num_frames]]
    
    print(f"Worst frames: {worst_frames}")
    
    # Visualize each
    for frame_idx in worst_frames:
        output_file = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"frame_{frame_idx:06d}_comparison.png"
        
        print(f"\nVisualizing frame {frame_idx}...")
        visualize_frame(recording_file, frame_idx, output_file=output_file, show=False)
    
    if output_dir:
        print(f"\nSaved {num_frames} visualizations to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize detected vs ground truth lanes')
    parser.add_argument('recording_file', type=Path, help='Path to HDF5 recording file')
    parser.add_argument('--frame', type=int, help='Frame index to visualize')
    parser.add_argument('--worst', type=int, metavar='N', 
                       help='Visualize N worst frames (by width error)')
    parser.add_argument('--output', type=Path, help='Output image file (for single frame)')
    parser.add_argument('--output-dir', type=Path, help='Output directory (for multiple frames)')
    parser.add_argument('--no-show', action='store_true', help='Do not display images')
    
    args = parser.parse_args()
    
    if not args.recording_file.exists():
        print(f"Error: Recording file not found: {args.recording_file}")
        return 1
    
    if args.worst:
        visualize_error_frames(args.recording_file, num_frames=args.worst, output_dir=args.output_dir)
    elif args.frame is not None:
        visualize_frame(args.recording_file, args.frame, output_file=args.output, show=not args.no_show)
    else:
        print("Error: Must specify --frame or --worst")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

