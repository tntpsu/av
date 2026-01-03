"""
Test CV lane detection on Unity frames from recordings.
"""

import h5py
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.models.lane_detection import SimpleLaneDetector
from perception.inference import LaneDetectionInference


def extract_frames_from_recording(recording_path: str, max_frames: int = 10) -> list:
    """
    Extract frames from HDF5 recording.
    
    Args:
        recording_path: Path to HDF5 recording file
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of (image, frame_id) tuples
    """
    frames = []
    
    try:
        with h5py.File(recording_path, 'r') as f:
            if 'camera/images' not in f:
                print(f"Error: No camera images found in {recording_path}")
                return frames
            
            images = f['camera/images']
            num_frames = min(len(images), max_frames)
            
            # Sample frames evenly
            indices = np.linspace(0, len(images) - 1, num_frames, dtype=int)
            
            for idx in indices:
                image_data = images[idx]
                # HDF5 stores as uint8, convert to numpy array
                image = np.array(image_data)
                frames.append((image, idx))
            
            print(f"Extracted {len(frames)} frames from recording")
    
    except Exception as e:
        print(f"Error reading recording: {e}")
    
    return frames


def test_cv_detection(image: np.ndarray, detector: SimpleLaneDetector) -> tuple:
    """
    Test CV detection on a single image.
    
    Args:
        image: Input RGB image
        detector: CV lane detector
    
    Returns:
        Tuple of (lane_coeffs, num_lanes_detected)
    """
    lane_coeffs = detector.detect(image)
    num_lanes = sum(1 for c in lane_coeffs if c is not None)
    return lane_coeffs, num_lanes


def visualize_detection(image: np.ndarray, lane_coeffs: list, output_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize detected lanes on image.
    
    Args:
        image: Input RGB image
        lane_coeffs: Lane polynomial coefficients
        output_path: Optional path to save visualization
    
    Returns:
        Image with lanes drawn
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    colors = [(0, 255, 0), (255, 0, 0)]  # Green for left, Red for right
    
    for i, coeffs in enumerate(lane_coeffs):
        if coeffs is None:
            continue
        
        color = colors[i % len(colors)]
        
        # Generate points along lane
        y_points = np.linspace(h // 3, h, 50)
        x_points = np.polyval(coeffs, y_points)
        
        # Filter valid points
        valid_mask = (x_points >= 0) & (x_points < w)
        x_points = x_points[valid_mask]
        y_points = y_points[valid_mask]
        
        if len(x_points) < 2:
            continue
        
        # Draw lane line
        points = np.array([x_points, y_points], dtype=np.int32).T
        cv2.polylines(vis_image, [points], isClosed=False, color=color, thickness=3)
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image


def main():
    parser = argparse.ArgumentParser(description='Test CV lane detection on Unity frames')
    parser.add_argument('recording', type=str, help='Path to HDF5 recording file')
    parser.add_argument('--max-frames', type=int, default=10, help='Maximum frames to test')
    parser.add_argument('--output-dir', type=str, default='perception/test_outputs',
                       help='Directory to save visualization images')
    parser.add_argument('--visualize', action='store_true', help='Show visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames from recording
    print(f"Reading recording: {args.recording}")
    frames = extract_frames_from_recording(args.recording, args.max_frames)
    
    if not frames:
        print("No frames extracted. Exiting.")
        return
    
    # Initialize detector
    detector = SimpleLaneDetector()
    
    # Test on each frame
    results = []
    for image, frame_id in frames:
        lane_coeffs, num_lanes = test_cv_detection(image, detector)
        results.append((frame_id, num_lanes))
        
        print(f"Frame {frame_id}: Detected {num_lanes} lanes")
        
        # Visualize
        if args.visualize or num_lanes > 0:
            vis_image = visualize_detection(
                image, 
                lane_coeffs,
                str(output_dir / f"frame_{frame_id:06d}_lanes.png")
            )
            
            if args.visualize:
                # Show in window
                vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"Frame {frame_id}", vis_bgr)
                cv2.waitKey(1000)  # Wait 1 second
                cv2.destroyAllWindows()
    
    # Summary statistics
    total_frames = len(results)
    frames_with_lanes = sum(1 for _, num in results if num > 0)
    detection_rate = (frames_with_lanes / total_frames * 100) if total_frames > 0 else 0
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total frames tested: {total_frames}")
    print(f"Frames with lanes detected: {frames_with_lanes}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Average lanes per frame: {sum(num for _, num in results) / total_frames:.2f}")
    print(f"Visualizations saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()

