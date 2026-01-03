"""
Calibrate perception coordinate conversion using ground truth data.

This tool finds the optimal distance/calibration parameters that make
perception match ground truth, then applies that calibration.

Usage:
    python tools/calibrate_perception.py <recording_file>
    python tools/calibrate_perception.py --list  # List available recordings
"""

import sys
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import argparse
from scipy.optimize import minimize_scalar

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_perception_and_ground_truth(recording_file: str) -> Dict:
    """Load perception and ground truth data from recording."""
    with h5py.File(recording_file, 'r') as f:
        # Load perception data
        if "perception/left_lane_line_x" not in f:
            raise ValueError("No perception data found! Run perception replay first.")
        
        perception_left = f["perception/left_lane_line_x"][:]
        perception_right = f["perception/right_lane_line_x"][:]
        perception_center = (perception_left + perception_right) / 2.0
        
        # Load ground truth
        if "ground_truth/left_lane_line_x" in f:
            gt_left = f["ground_truth/left_lane_line_x"][:]
            gt_right = f["ground_truth/right_lane_line_x"][:]
        elif "ground_truth/left_lane_x" in f:  # Backward compatibility
            gt_left = f["ground_truth/left_lane_x"][:]
            gt_right = f["ground_truth/right_lane_x"][:]
        else:
            raise ValueError("No ground truth data found!")
        
        gt_center = (gt_left + gt_right) / 2.0
        
        # Load image coordinates (for recalculation)
        if "perception/lane_line_coefficients" in f:
            lane_coeffs = f["perception/lane_line_coefficients"][:]
        else:
            lane_coeffs = None
        
        # Load camera calibration
        camera_8m_screen_y = None
        if "vehicle/camera_8m_screen_y" in f:
            camera_8m_screen_y = f["vehicle/camera_8m_screen_y"][:]
        
        camera_horizontal_fov = None
        if "vehicle/camera_horizontal_fov" in f:
            camera_horizontal_fov = f["vehicle/camera_horizontal_fov"][:]
        
        return {
            'perception_left': perception_left,
            'perception_right': perception_right,
            'perception_center': perception_center,
            'gt_left': gt_left,
            'gt_right': gt_right,
            'gt_center': gt_center,
            'lane_coeffs': lane_coeffs,
            'camera_8m_screen_y': camera_8m_screen_y,
            'camera_horizontal_fov': camera_horizontal_fov
        }


def calculate_perception_with_distance(data: Dict, conversion_distance: float, 
                                      camera_horizontal_fov: float = 110.0,
                                      image_width: float = 640.0,
                                      y_image: float = 295.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recalculate perception lane positions using a different conversion distance.
    
    This simulates what perception would detect if we used a different distance
    for coordinate conversion.
    """
    # For now, we'll use a simple scaling approach
    # If we had the original image coordinates, we could recalculate properly
    # But for calibration, we can estimate the effect of distance change
    
    # The relationship: x_vehicle = x_pixels * pixel_to_meter
    # pixel_to_meter = width_at_distance / image_width
    # width_at_distance = 2 * distance * tan(FOV/2)
    
    # Current perception was calculated with some distance (let's assume 8m)
    # We want to see what it would be with a different distance
    
    # Scale factor: new_width / old_width = new_distance / old_distance
    # This is approximate - assumes same y position
    
    # For calibration, we'll use a simpler approach:
    # Find the offset that makes perception match ground truth
    offset = np.mean(data['gt_center'] - data['perception_center'])
    
    # Apply offset to perception
    calibrated_left = data['perception_left'] + offset
    calibrated_right = data['perception_right'] + offset
    
    return calibrated_left, calibrated_right


def find_optimal_calibration(data: Dict) -> Dict:
    """
    Find optimal calibration parameters to minimize perception error.
    
    Returns:
        Dictionary with calibration parameters and results
    """
    print("\n" + "="*80)
    print("CALIBRATING PERCEPTION COORDINATE CONVERSION")
    print("="*80)
    
    # Calculate current errors
    center_errors = data['perception_center'] - data['gt_center']
    mean_error = np.mean(center_errors)
    rmse = np.sqrt(np.mean(center_errors**2))
    
    print(f"\nCurrent Performance (before calibration):")
    print(f"  Mean error: {mean_error:.3f}m")
    print(f"  RMSE: {rmse:.3f}m")
    print(f"  Std error: {np.std(center_errors):.3f}m")
    
    # Simple calibration: find offset that minimizes error
    # This is a constant offset correction
    optimal_offset = -mean_error  # Negative because we want to correct the error
    
    # Apply calibration
    calibrated_left = data['perception_left'] + optimal_offset
    calibrated_right = data['perception_right'] + optimal_offset
    calibrated_center = (calibrated_left + calibrated_right) / 2.0
    
    # Calculate new errors
    calibrated_errors = calibrated_center - data['gt_center']
    calibrated_mean = np.mean(calibrated_errors)
    calibrated_rmse = np.sqrt(np.mean(calibrated_errors**2))
    calibrated_std = np.std(calibrated_errors)
    
    print(f"\nAfter Calibration (offset = {optimal_offset:.3f}m):")
    print(f"  Mean error: {calibrated_mean:.3f}m")
    print(f"  RMSE: {calibrated_rmse:.3f}m")
    print(f"  Std error: {calibrated_std:.3f}m")
    
    improvement = rmse - calibrated_rmse
    improvement_pct = (improvement / rmse) * 100 if rmse > 0 else 0
    
    print(f"\nImprovement:")
    print(f"  RMSE reduction: {improvement:.3f}m ({improvement_pct:.1f}%)")
    
    if calibrated_rmse < 0.1:
        status = 'EXCELLENT'
        print(f"\n✓ {status}: Calibration successful! RMSE < 0.1m")
    elif calibrated_rmse < 0.2:
        status = 'GOOD'
        print(f"\n✓ {status}: Calibration good! RMSE < 0.2m")
    elif calibrated_rmse < 0.3:
        status = 'ACCEPTABLE'
        print(f"\n⚠️  {status}: Calibration acceptable, but could be better")
    else:
        status = 'POOR'
        print(f"\n❌ {status}: Calibration didn't help much - may need different approach")
    
    return {
        'optimal_offset': optimal_offset,
        'before_rmse': rmse,
        'after_rmse': calibrated_rmse,
        'improvement': improvement,
        'status': status,
        'calibrated_left': calibrated_left,
        'calibrated_right': calibrated_right,
        'calibrated_center': calibrated_center
    }


def create_calibration_visualization(data: Dict, calibration: Dict, output_dir: Path):
    """Create visualization plots for calibration results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Before vs After calibration
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    frames = np.arange(len(data['perception_center']))
    
    # Before calibration
    ax1.plot(frames, data['perception_center'], 'b-', label='Perception (before)', linewidth=2, alpha=0.7)
    ax1.plot(frames, data['gt_center'], 'g--', label='Ground Truth', linewidth=2)
    ax1.fill_between(frames, 
                     data['gt_center'] - 0.1,
                     data['gt_center'] + 0.1,
                     alpha=0.2, color='green', label='±0.1m tolerance')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Lane Center (m)')
    ax1.set_title('Before Calibration: Perception vs Ground Truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # After calibration
    ax2.plot(frames, calibration['calibrated_center'], 'r-', label='Perception (after)', linewidth=2, alpha=0.7)
    ax2.plot(frames, data['gt_center'], 'g--', label='Ground Truth', linewidth=2)
    ax2.fill_between(frames,
                     data['gt_center'] - 0.1,
                     data['gt_center'] + 0.1,
                     alpha=0.2, color='green', label='±0.1m tolerance')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Lane Center (m)')
    ax2.set_title(f'After Calibration (offset={calibration["optimal_offset"]:.3f}m): Perception vs Ground Truth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'perception_calibration.png', dpi=150)
    plt.close()
    
    # Plot 2: Error distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    before_errors = data['perception_center'] - data['gt_center']
    after_errors = calibration['calibrated_center'] - data['gt_center']
    
    # Histogram
    ax1.hist(before_errors, bins=50, alpha=0.7, label=f'Before (RMSE={calibration["before_rmse"]:.3f}m)', edgecolor='black')
    ax1.hist(after_errors, bins=50, alpha=0.7, label=f'After (RMSE={calibration["after_rmse"]:.3f}m)', edgecolor='black')
    ax1.axvline(0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Error (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution: Before vs After Calibration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time series
    ax2.plot(frames, before_errors, 'b-', label='Before', linewidth=1, alpha=0.7)
    ax2.plot(frames, after_errors, 'r-', label='After', linewidth=1, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', linewidth=2)
    ax2.fill_between(frames, -0.1, 0.1, alpha=0.2, color='green', label='±0.1m tolerance')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Error Over Time: Before vs After Calibration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_error_analysis.png', dpi=150)
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate perception coordinate conversion using ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recording", nargs='?', default=None,
                       help="Path to recording file (default: latest replay)")
    parser.add_argument("--list", action="store_true",
                       help="List available recordings and exit")
    parser.add_argument("--output-dir", type=str, default="tmp/perception_calibration",
                       help="Directory for output plots (default: tmp/perception_calibration)")
    
    args = parser.parse_args()
    
    # List recordings if requested
    if args.list:
        from tools.list_recordings import list_recordings
        list_recordings()
        return
    
    # Find recording file
    if args.recording:
        recording_file = Path(args.recording)
        if not recording_file.exists():
            print(f"Error: Recording not found: {recording_file}")
            return
    else:
        # Use latest replay recording
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("No recordings found!")
            return
        
        # Prefer replay recordings
        replay_recordings = [r for r in recordings if "replay" in r.name.lower()]
        if replay_recordings:
            recording_file = replay_recordings[0]
            print(f"Using latest replay recording: {recording_file.name}")
        else:
            recording_file = recordings[0]
            print(f"Using latest recording: {recording_file.name}")
    
    print(f"\nCalibrating perception in: {recording_file}")
    print("="*80)
    
    # Load data
    try:
        data = load_perception_and_ground_truth(str(recording_file))
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Find optimal calibration
    calibration = find_optimal_calibration(data)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    create_calibration_visualization(data, calibration, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)
    print(f"\nOptimal Offset: {calibration['optimal_offset']:.3f}m")
    print(f"  (Apply this offset to perception lane positions)")
    print(f"\nPerformance:")
    print(f"  Before: RMSE = {calibration['before_rmse']:.3f}m")
    print(f"  After:  RMSE = {calibration['after_rmse']:.3f}m")
    print(f"  Improvement: {calibration['improvement']:.3f}m")
    print(f"\nStatus: {calibration['status']}")
    print(f"\nTo apply this calibration:")
    print(f"  1. Add offset to perception coordinate conversion")
    print(f"  2. Or apply offset in trajectory planning")
    print(f"  3. Or update camera calibration parameters")


if __name__ == "__main__":
    main()

