"""
Analyze trajectory planning accuracy against ground truth.

This script takes a recording (especially from perception replay) and analyzes:
1. Trajectory centering accuracy (does it follow lane center?)
2. Reference point accuracy (at lookahead distance)
3. Trajectory vs ground truth path comparison
4. Trajectory quality metrics (smoothness, curvature)

Usage:
    python tools/analyze_trajectory.py <recording_file>
    python tools/analyze_trajectory.py --list  # List available recordings
"""

import sys
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trajectory_data(recording_file: str) -> Dict:
    """Load trajectory and ground truth data from recording."""
    with h5py.File(recording_file, 'r') as f:
        # Check if trajectory data exists
        if "trajectory/timestamps" not in f:
            raise ValueError("No trajectory data found in recording!")
        
        # Load trajectory data
        traj_timestamps = f["trajectory/timestamps"][:]
        ref_x = f["trajectory/reference_point_x"][:] if "trajectory/reference_point_x" in f else None
        ref_y = f["trajectory/reference_point_y"][:] if "trajectory/reference_point_y" in f else None
        ref_heading = f["trajectory/reference_point_heading"][:] if "trajectory/reference_point_heading" in f else None
        
        # Load vehicle state (for ground truth comparison)
        vehicle_timestamps = f["vehicle/timestamps"][:]
        vehicle_positions = f["vehicle/position"][:]
        
        # Load ground truth data
        gt_left = None
        gt_right = None
        gt_center = None
        
        if "ground_truth/left_lane_line_x" in f:
            gt_left = f["ground_truth/left_lane_line_x"][:]
        elif "ground_truth/left_lane_x" in f:  # Backward compatibility
            gt_left = f["ground_truth/left_lane_x"][:]
            
        if "ground_truth/right_lane_line_x" in f:
            gt_right = f["ground_truth/right_lane_line_x"][:]
        elif "ground_truth/right_lane_x" in f:  # Backward compatibility
            gt_right = f["ground_truth/right_lane_x"][:]
            
        if "ground_truth/lane_center_x" in f:
            gt_center = f["ground_truth/lane_center_x"][:]
        
        # Load perception data (for understanding what trajectory was based on)
        perception_left = None
        perception_right = None
        
        if "perception/left_lane_line_x" in f:
            perception_left = f["perception/left_lane_line_x"][:]
        elif "perception/left_lane_x" in f:  # Backward compatibility
            perception_left = f["perception/left_lane_x"][:]
            
        if "perception/right_lane_line_x" in f:
            perception_right = f["perception/right_lane_line_x"][:]
        elif "perception/right_lane_x" in f:  # Backward compatibility
            perception_right = f["perception/right_lane_x"][:]
        
        # Load control commands (for understanding trajectory usage)
        control_timestamps = None
        lateral_errors = None
        heading_errors = None
        
        if "control/timestamps" in f:
            control_timestamps = f["control/timestamps"][:]
            if "control/lateral_error" in f:
                lateral_errors = f["control/lateral_error"][:]
            if "control/heading_error" in f:
                heading_errors = f["control/heading_error"][:]
        
        return {
            'traj_timestamps': traj_timestamps,
            'ref_x': ref_x,
            'ref_y': ref_y,
            'ref_heading': ref_heading,
            'vehicle_timestamps': vehicle_timestamps,
            'vehicle_positions': vehicle_positions,
            'gt_left': gt_left,
            'gt_right': gt_right,
            'gt_center': gt_center,
            'perception_left': perception_left,
            'perception_right': perception_right,
            'control_timestamps': control_timestamps,
            'lateral_errors': lateral_errors,
            'heading_errors': heading_errors
        }


def sync_timestamps(data: Dict, target_timestamps: np.ndarray) -> Dict:
    """Synchronize all data to target timestamps using nearest neighbor."""
    synced = {}
    
    for key, values in data.items():
        if values is None:
            synced[key] = None
            continue
            
        if key.endswith('_timestamps'):
            synced[key] = target_timestamps
            continue
        
        # For arrays, interpolate to target timestamps
        if isinstance(values, np.ndarray) and len(values) > 0:
            # Get source timestamps
            if key.startswith('ref_'):
                source_ts = data['traj_timestamps']
            elif key.startswith('gt_') or key.startswith('perception_'):
                source_ts = data['vehicle_timestamps']
            elif key.startswith('lateral_') or key.startswith('heading_'):
                source_ts = data['control_timestamps']
            else:
                source_ts = data['vehicle_timestamps']
            
            if source_ts is None or len(source_ts) == 0:
                synced[key] = None
                continue
            
            # Interpolate to target timestamps
            if len(values.shape) == 1:
                # 1D array - simple interpolation
                synced[key] = np.interp(target_timestamps, source_ts, values)
            else:
                # Multi-dimensional - interpolate each dimension
                synced[key] = np.array([
                    np.interp(target_timestamps, source_ts, values[:, i])
                    for i in range(values.shape[1])
                ]).T
        else:
            synced[key] = values
    
    return synced


def analyze_reference_point_accuracy(data: Dict) -> Dict:
    """Analyze reference point accuracy vs ground truth lane center."""
    print("\n" + "="*80)
    print("ANALYSIS 1: Reference Point Accuracy")
    print("="*80)
    
    if data['ref_x'] is None or data['gt_center'] is None:
        print("❌ Cannot analyze - missing reference point or ground truth data")
        return {'status': 'FAIL'}
    
    # Reference point is in vehicle coordinates (lateral offset from car)
    # Ground truth center is also in vehicle coordinates
    # They should match if trajectory is correctly centered
    
    ref_lateral = data['ref_x']  # Lateral position of reference point
    gt_center = data['gt_center']  # Ground truth lane center
    
    # Calculate errors
    ref_errors = ref_lateral - gt_center
    
    valid_mask = ~np.isnan(ref_errors) & ~np.isnan(gt_center)
    if np.sum(valid_mask) == 0:
        print("❌ No valid data for comparison")
        return {'status': 'FAIL'}
    
    ref_errors_valid = ref_errors[valid_mask]
    gt_center_valid = gt_center[valid_mask]
    
    mean_error = np.mean(ref_errors_valid)
    std_error = np.std(ref_errors_valid)
    max_error = np.max(np.abs(ref_errors_valid))
    rmse = np.sqrt(np.mean(ref_errors_valid**2))
    
    print(f"\nReference Point vs Ground Truth Lane Center:")
    print(f"  Mean error: {mean_error:.3f}m (positive = reference right of GT center)")
    print(f"  Std error: {std_error:.3f}m")
    print(f"  Max absolute error: {max_error:.3f}m")
    print(f"  RMSE: {rmse:.3f}m")
    
    # Percentiles
    abs_errors = np.abs(ref_errors_valid)
    p50 = np.percentile(abs_errors, 50)
    p90 = np.percentile(abs_errors, 90)
    p95 = np.percentile(abs_errors, 95)
    
    print(f"\nError Distribution:")
    print(f"  Median absolute error: {p50:.3f}m")
    print(f"  90th percentile: {p90:.3f}m")
    print(f"  95th percentile: {p95:.3f}m")
    
    # Status assessment
    if rmse < 0.1:
        status = 'PASS'
        print(f"\n✓ EXCELLENT: Reference point is very accurate (RMSE < 0.1m)")
    elif rmse < 0.2:
        status = 'PASS'
        print(f"\n✓ GOOD: Reference point is accurate (RMSE < 0.2m)")
    elif rmse < 0.5:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Reference point has moderate errors (RMSE < 0.5m)")
    else:
        status = 'FAIL'
        print(f"\n❌ CRITICAL: Reference point has large errors (RMSE >= 0.5m)")
    
    return {
        'status': status,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'rmse': rmse,
        'p50': p50,
        'p90': p90,
        'p95': p95
    }


def analyze_trajectory_centering(data: Dict) -> Dict:
    """Analyze if trajectory is centered between detected lanes."""
    print("\n" + "="*80)
    print("ANALYSIS 2: Trajectory Centering (vs Detected Lanes)")
    print("="*80)
    
    if data['ref_x'] is None or data['perception_left'] is None or data['perception_right'] is None:
        print("❌ Cannot analyze - missing reference point or perception data")
        return {'status': 'FAIL'}
    
    # Calculate detected lane center
    detected_center = (data['perception_left'] + data['perception_right']) / 2.0
    
    # Reference point should be at detected center (if trajectory is correctly centered)
    ref_lateral = data['ref_x']
    centering_errors = ref_lateral - detected_center
    
    valid_mask = ~np.isnan(centering_errors) & ~np.isnan(detected_center)
    if np.sum(valid_mask) == 0:
        print("❌ No valid data for comparison")
        return {'status': 'FAIL'}
    
    centering_errors_valid = centering_errors[valid_mask]
    
    mean_error = np.mean(centering_errors_valid)
    std_error = np.std(centering_errors_valid)
    max_error = np.max(np.abs(centering_errors_valid))
    rmse = np.sqrt(np.mean(centering_errors_valid**2))
    
    print(f"\nReference Point vs Detected Lane Center:")
    print(f"  Mean error: {mean_error:.3f}m (positive = reference right of detected center)")
    print(f"  Std error: {std_error:.3f}m")
    print(f"  Max absolute error: {max_error:.3f}m")
    print(f"  RMSE: {rmse:.3f}m")
    
    # Status assessment
    if rmse < 0.05:
        status = 'PASS'
        print(f"\n✓ EXCELLENT: Trajectory is perfectly centered (RMSE < 0.05m)")
    elif rmse < 0.1:
        status = 'PASS'
        print(f"\n✓ GOOD: Trajectory is well-centered (RMSE < 0.1m)")
    elif rmse < 0.2:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Trajectory has moderate centering errors (RMSE < 0.2m)")
    else:
        status = 'FAIL'
        print(f"\n❌ CRITICAL: Trajectory has large centering errors (RMSE >= 0.2m)")
        print(f"   This suggests trajectory planning is not correctly centering!")
    
    return {
        'status': status,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'rmse': rmse
    }


def analyze_trajectory_vs_ground_truth(data: Dict) -> Dict:
    """Compare trajectory reference point to ground truth path."""
    print("\n" + "="*80)
    print("ANALYSIS 3: Trajectory vs Ground Truth Path")
    print("="*80)
    
    if data['ref_x'] is None or data['gt_center'] is None:
        print("❌ Cannot analyze - missing reference point or ground truth data")
        return {'status': 'FAIL'}
    
    # Compare reference point to ground truth
    ref_lateral = data['ref_x']
    gt_center = data['gt_center']
    
    errors = ref_lateral - gt_center
    
    valid_mask = ~np.isnan(errors) & ~np.isnan(gt_center)
    if np.sum(valid_mask) == 0:
        print("❌ No valid data for comparison")
        return {'status': 'FAIL'}
    
    errors_valid = errors[valid_mask]
    
    mean_error = np.mean(errors_valid)
    std_error = np.std(errors_valid)
    max_error = np.max(np.abs(errors_valid))
    rmse = np.sqrt(np.mean(errors_valid**2))
    
    print(f"\nOverall Trajectory Accuracy (vs Ground Truth):")
    print(f"  Mean error: {mean_error:.3f}m")
    print(f"  Std error: {std_error:.3f}m")
    print(f"  Max absolute error: {max_error:.3f}m")
    print(f"  RMSE: {rmse:.3f}m")
    
    # Check if errors are systematic (bias) or random
    bias = np.mean(errors_valid)
    random_error = np.std(errors_valid)
    
    print(f"\nError Breakdown:")
    print(f"  Systematic bias: {bias:.3f}m")
    print(f"  Random error (std): {random_error:.3f}m")
    
    if abs(bias) > 0.1:
        print(f"  ⚠️  WARNING: Significant systematic bias detected!")
        print(f"     This suggests a calibration or offset issue")
    
    # Status assessment
    if rmse < 0.15:
        status = 'PASS'
        print(f"\n✓ EXCELLENT: Trajectory closely matches ground truth (RMSE < 0.15m)")
    elif rmse < 0.3:
        status = 'PASS'
        print(f"\n✓ GOOD: Trajectory matches ground truth reasonably well (RMSE < 0.3m)")
    elif rmse < 0.5:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Trajectory has moderate errors vs ground truth (RMSE < 0.5m)")
    else:
        status = 'FAIL'
        print(f"\n❌ CRITICAL: Trajectory has large errors vs ground truth (RMSE >= 0.5m)")
    
    return {
        'status': status,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'rmse': rmse,
        'bias': bias,
        'random_error': random_error
    }


def analyze_trajectory_quality(data: Dict) -> Dict:
    """Analyze trajectory quality metrics (smoothness, consistency)."""
    print("\n" + "="*80)
    print("ANALYSIS 4: Trajectory Quality Metrics")
    print("="*80)
    
    if data['ref_x'] is None:
        print("❌ Cannot analyze - missing reference point data")
        return {'status': 'FAIL'}
    
    ref_x = data['ref_x']
    ref_heading = data['ref_heading'] if data['ref_heading'] is not None else None
    
    valid_mask = ~np.isnan(ref_x)
    if np.sum(valid_mask) < 2:
        print("❌ Not enough valid data for quality analysis")
        return {'status': 'FAIL'}
    
    ref_x_valid = ref_x[valid_mask]
    
    # Smoothness: measure rate of change
    if len(ref_x_valid) > 1:
        dx = np.diff(ref_x_valid)
        smoothness = np.std(dx)  # Lower is smoother
        
        print(f"\nTrajectory Smoothness:")
        print(f"  Lateral change std: {smoothness:.3f}m/frame")
        
        if smoothness < 0.05:
            smooth_status = 'PASS'
            print(f"  ✓ EXCELLENT: Very smooth trajectory")
        elif smoothness < 0.1:
            smooth_status = 'PASS'
            print(f"  ✓ GOOD: Smooth trajectory")
        elif smoothness < 0.2:
            smooth_status = 'WARN'
            print(f"  ⚠️  WARNING: Moderate trajectory roughness")
        else:
            smooth_status = 'FAIL'
            print(f"  ❌ CRITICAL: Very rough/jumpy trajectory")
    else:
        smoothness = np.nan
        smooth_status = 'UNKNOWN'
    
    # Consistency: check for sudden jumps
    if len(ref_x_valid) > 1:
        large_jumps = np.sum(np.abs(dx) > 0.5)  # Jumps > 0.5m
        jump_rate = large_jumps / len(dx) if len(dx) > 0 else 0.0
        
        print(f"\nTrajectory Consistency:")
        print(f"  Large jumps (>0.5m): {large_jumps}/{len(dx)} ({jump_rate*100:.1f}%)")
        
        if jump_rate < 0.01:
            consistency_status = 'PASS'
            print(f"  ✓ EXCELLENT: Very consistent trajectory")
        elif jump_rate < 0.05:
            consistency_status = 'PASS'
            print(f"  ✓ GOOD: Consistent trajectory")
        elif jump_rate < 0.1:
            consistency_status = 'WARN'
            print(f"  ⚠️  WARNING: Some trajectory jumps detected")
        else:
            consistency_status = 'FAIL'
            print(f"  ❌ CRITICAL: Many trajectory jumps (unstable)")
    else:
        jump_rate = np.nan
        consistency_status = 'UNKNOWN'
    
    # Heading smoothness (if available)
    if ref_heading is not None:
        ref_heading_valid = ref_heading[valid_mask]
        if len(ref_heading_valid) > 1:
            dheading = np.diff(ref_heading_valid)
            # Handle angle wrapping
            dheading = np.where(dheading > 180, dheading - 360, dheading)
            dheading = np.where(dheading < -180, dheading + 360, dheading)
            heading_smoothness = np.std(dheading)
            
            print(f"\nHeading Smoothness:")
            print(f"  Heading change std: {heading_smoothness:.2f}°/frame")
            
            if heading_smoothness < 2.0:
                heading_status = 'PASS'
                print(f"  ✓ EXCELLENT: Very smooth heading")
            elif heading_smoothness < 5.0:
                heading_status = 'PASS'
                print(f"  ✓ GOOD: Smooth heading")
            else:
                heading_status = 'WARN'
                print(f"  ⚠️  WARNING: Rough heading changes")
        else:
            heading_smoothness = np.nan
            heading_status = 'UNKNOWN'
    else:
        heading_smoothness = np.nan
        heading_status = 'UNKNOWN'
    
    # Overall status
    statuses = [s for s in [smooth_status, consistency_status, heading_status] if s != 'UNKNOWN']
    if all(s == 'PASS' for s in statuses):
        overall_status = 'PASS'
    elif any(s == 'FAIL' for s in statuses):
        overall_status = 'FAIL'
    else:
        overall_status = 'WARN'
    
    return {
        'status': overall_status,
        'smoothness': smoothness,
        'jump_rate': jump_rate,
        'heading_smoothness': heading_smoothness
    }


def create_visualizations(data: Dict, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Reference point vs ground truth over time
    if data['ref_x'] is not None and data['gt_center'] is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valid_mask = ~np.isnan(data['ref_x']) & ~np.isnan(data['gt_center'])
        if np.sum(valid_mask) > 0:
            frames = np.arange(len(data['ref_x']))[valid_mask]
            ax.plot(frames, data['ref_x'][valid_mask], 'b-', label='Reference Point', linewidth=2)
            ax.plot(frames, data['gt_center'][valid_mask], 'g--', label='Ground Truth Center', linewidth=2)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Lateral Position (m)')
            ax.set_title('Reference Point vs Ground Truth Lane Center')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'trajectory_vs_ground_truth.png', dpi=150)
            plt.close()
    
    # Plot 2: Error distribution
    if data['ref_x'] is not None and data['gt_center'] is not None:
        errors = data['ref_x'] - data['gt_center']
        valid_errors = errors[~np.isnan(errors)]
        
        if len(valid_errors) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            ax1.hist(valid_errors, bins=50, edgecolor='black', alpha=0.7)
            ax1.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax1.axvline(np.mean(valid_errors), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_errors):.3f}m')
            ax1.set_xlabel('Error (m)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Reference Point Error Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Time series
            frames = np.arange(len(errors))[~np.isnan(errors)]
            ax2.plot(frames, valid_errors, 'b-', linewidth=1, alpha=0.7)
            ax2.axhline(0, color='r', linestyle='--', linewidth=2)
            ax2.axhline(np.mean(valid_errors), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_errors):.3f}m')
            ax2.fill_between(frames, 
                            np.mean(valid_errors) - np.std(valid_errors),
                            np.mean(valid_errors) + np.std(valid_errors),
                            alpha=0.2, color='orange', label=f'±1 std: {np.std(valid_errors):.3f}m')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Error (m)')
            ax2.set_title('Reference Point Error Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'trajectory_error_analysis.png', dpi=150)
            plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory planning accuracy against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recording", nargs='?', default=None,
                       help="Path to recording file (default: latest)")
    parser.add_argument("--list", action="store_true",
                       help="List available recordings and exit")
    parser.add_argument("--output-dir", type=str, default="tmp/trajectory_analysis",
                       help="Directory for output plots (default: tmp/trajectory_analysis)")
    
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
        # Use latest recording
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("No recordings found!")
            return
        
        # Prefer replay recordings (they have trajectory data)
        replay_recordings = [r for r in recordings if "replay" in r.name.lower()]
        if replay_recordings:
            recording_file = replay_recordings[0]
            print(f"Using latest replay recording: {recording_file.name}")
        else:
            recording_file = recordings[0]
            print(f"Using latest recording: {recording_file.name}")
    
    print(f"\nAnalyzing trajectory in: {recording_file}")
    print("="*80)
    
    # Load data
    try:
        data = load_trajectory_data(str(recording_file))
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Synchronize to trajectory timestamps
    if data['traj_timestamps'] is not None and len(data['traj_timestamps']) > 0:
        data = sync_timestamps(data, data['traj_timestamps'])
    else:
        print("❌ No trajectory timestamps found!")
        return
    
    # Run analyses
    results = {}
    results['ref_point_accuracy'] = analyze_reference_point_accuracy(data)
    results['trajectory_centering'] = analyze_trajectory_centering(data)
    results['trajectory_vs_gt'] = analyze_trajectory_vs_ground_truth(data)
    results['trajectory_quality'] = analyze_trajectory_quality(data)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    create_visualizations(data, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    statuses = [r['status'] for r in results.values() if 'status' in r]
    if all(s == 'PASS' for s in statuses):
        print("✓ ALL CHECKS PASSED: Trajectory planning is working correctly!")
    elif any(s == 'FAIL' for s in statuses):
        print("❌ SOME CHECKS FAILED: Trajectory planning needs attention")
    else:
        print("⚠️  SOME WARNINGS: Trajectory planning is mostly good but has issues")
    
    print(f"\nDetailed results:")
    for name, result in results.items():
        if 'status' in result:
            status_icon = {'PASS': '✓', 'WARN': '⚠️', 'FAIL': '❌'}.get(result['status'], '?')
            print(f"  {status_icon} {name.replace('_', ' ').title()}: {result['status']}")


if __name__ == "__main__":
    main()

