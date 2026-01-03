"""
Diagnose whether the issue is trajectory planning or steering control.

This script analyzes a recording to determine:
1. Is the reference point (trajectory) correct?
2. Is the steering following the trajectory?
3. Where is the problem: trajectory planning or control?

Usage:
    python tools/diagnose_trajectory_vs_steering.py [recording_file]
"""

import sys
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def diagnose(recording_file: str):
    """Diagnose trajectory vs steering issue."""
    print("=" * 70)
    print("TRAJECTORY vs STEERING DIAGNOSIS")
    print("=" * 70)
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Check what data is available
        has_trajectory = "trajectory/reference_point_x" in f
        has_control = "control/steering" in f
        has_ground_truth = "ground_truth/lane_center_x" in f
        has_perception = "perception/left_lane_line_x" in f
        
        print("Data availability:")
        print(f"  Trajectory data: {'✓' if has_trajectory else '✗'}")
        print(f"  Control data: {'✓' if has_control else '✗'}")
        print(f"  Ground truth: {'✓' if has_ground_truth else '✗'}")
        print(f"  Perception: {'✓' if has_perception else '✗'}")
        print()
        
        if not has_trajectory:
            print("❌ ERROR: No trajectory data found!")
            print("   This recording doesn't have trajectory planning data.")
            print("   Make sure you ran the normal AV stack (not ground truth follower).")
            return
        
        # Load trajectory data
        ref_x = f["trajectory/reference_point_x"][:]
        ref_y = f["trajectory/reference_point_y"][:]
        ref_heading = f["trajectory/reference_point_heading"][:]
        
        # Load control data
        steering = None
        lateral_error = None
        heading_error = None
        
        if has_control:
            steering = f["control/steering"][:]
            if "control/lateral_error" in f:
                lateral_error = f["control/lateral_error"][:]
            if "control/heading_error" in f:
                heading_error = f["control/heading_error"][:]
        
        # Load ground truth for comparison
        gt_center = None
        if has_ground_truth:
            gt_center = f["ground_truth/lane_center_x"][:]
        
        # Load perception for understanding trajectory source
        perception_center = None
        if has_perception:
            perception_left = f["perception/left_lane_line_x"][:]
            perception_right = f["perception/right_lane_line_x"][:]
            perception_center = (perception_left + perception_right) / 2.0
        
        # Align data by timestamps
        traj_timestamps = f["trajectory/timestamps"][:]
        if has_control:
            control_timestamps = f["control/timestamps"][:]
        else:
            control_timestamps = traj_timestamps
        
        # Find matching indices (simplified - assume aligned)
        min_len = min(len(ref_x), len(control_timestamps) if has_control else len(ref_x))
        ref_x = ref_x[:min_len]
        ref_y = ref_y[:min_len]
        ref_heading = ref_heading[:min_len]
        
        if steering is not None:
            steering = steering[:min_len]
        if lateral_error is not None:
            lateral_error = lateral_error[:min_len]
        if heading_error is not None:
            heading_error = heading_error[:min_len]
        if gt_center is not None:
            gt_center = gt_center[:min_len]
        if perception_center is not None:
            perception_center = perception_center[:min_len]
        
        print("=" * 70)
        print("ANALYSIS 1: TRAJECTORY REFERENCE POINT")
        print("=" * 70)
        print()
        
        # Check if reference point is reasonable
        ref_x_abs = np.abs(ref_x)
        max_ref_x = np.max(ref_x_abs)
        mean_ref_x = np.mean(ref_x_abs)
        std_ref_x = np.std(ref_x)
        
        print(f"Reference point X (lateral offset from car center):")
        print(f"  Mean absolute: {mean_ref_x:.3f}m")
        print(f"  Max absolute: {max_ref_x:.3f}m")
        print(f"  Std deviation: {std_ref_x:.3f}m")
        print()
        
        # Check reference point variation (should change on curves)
        ref_x_change = np.abs(np.diff(ref_x))
        mean_change = np.mean(ref_x_change)
        max_change = np.max(ref_x_change)
        
        print(f"Reference point variation (how much it changes):")
        print(f"  Mean change per frame: {mean_change:.3f}m")
        print(f"  Max change per frame: {max_change:.3f}m")
        print()
        
        if mean_ref_x < 0.1 and max_ref_x < 0.2:
            print("⚠️  WARNING: Reference point is very close to center!")
            print("   This suggests trajectory is not planning ahead on curves.")
            print("   → LIKELY TRAJECTORY ISSUE")
        elif mean_change < 0.01:
            print("⚠️  WARNING: Reference point barely changes!")
            print("   Trajectory is not adapting to curves.")
            print("   → LIKELY TRAJECTORY ISSUE")
        else:
            print("✓ Reference point shows variation (good for curves)")
        
        print()
        print("=" * 70)
        print("ANALYSIS 2: TRAJECTORY vs GROUND TRUTH")
        print("=" * 70)
        print()
        
        if gt_center is not None:
            # Compare reference point to ground truth lane center
            ref_vs_gt = ref_x - gt_center
            mean_error = np.mean(ref_vs_gt)
            rmse = np.sqrt(np.mean(ref_vs_gt**2))
            max_error = np.max(np.abs(ref_vs_gt))
            
            print(f"Reference point vs Ground truth lane center:")
            print(f"  Mean error: {mean_error:.3f}m")
            print(f"  RMSE: {rmse:.3f}m")
            print(f"  Max error: {max_error:.3f}m")
            print()
            
            if rmse > 0.5:
                print("❌ CRITICAL: Reference point has large errors vs ground truth!")
                print("   Trajectory is not following the correct path.")
                print("   → LIKELY TRAJECTORY ISSUE")
            elif rmse > 0.2:
                print("⚠️  WARNING: Reference point has moderate errors.")
                print("   Trajectory may need improvement.")
            else:
                print("✓ Reference point is close to ground truth (good)")
        else:
            print("No ground truth data available for comparison")
        
        print()
        print("=" * 70)
        print("ANALYSIS 3: STEERING RESPONSE")
        print("=" * 70)
        print()
        
        if steering is not None:
            steering_abs = np.abs(steering)
            mean_steering = np.mean(steering_abs)
            max_steering = np.max(steering_abs)
            std_steering = np.std(steering)
            
            print(f"Steering commands:")
            print(f"  Mean absolute: {mean_steering:.3f}")
            print(f"  Max absolute: {max_steering:.3f}")
            print(f"  Std deviation: {std_steering:.3f}")
            print()
            
            # Check if steering changes (should on curves)
            steering_change = np.abs(np.diff(steering))
            mean_steering_change = np.mean(steering_change)
            max_steering_change = np.max(steering_change)
            
            print(f"Steering variation:")
            print(f"  Mean change per frame: {mean_steering_change:.3f}")
            print(f"  Max change per frame: {max_steering_change:.3f}")
            print()
            
            if mean_steering < 0.05 and max_steering < 0.1:
                print("⚠️  WARNING: Steering is very small!")
                print("   Car is not turning enough.")
                print("   → COULD BE STEERING ISSUE (gains too low)")
                print("   → OR TRAJECTORY ISSUE (not planning enough curve)")
            elif mean_steering_change < 0.01:
                print("⚠️  WARNING: Steering barely changes!")
                print("   Car is not responding to curves.")
                print("   → COULD BE STEERING ISSUE (not responsive)")
                print("   → OR TRAJECTORY ISSUE (reference point not changing)")
            else:
                print("✓ Steering shows variation (good for curves)")
            
            # Check steering vs reference point correlation
            if len(ref_x) == len(steering):
                # Check if steering correlates with reference point
                # If ref_x is positive (right), steering should be positive (right)
                correlation = np.corrcoef(ref_x, steering)[0, 1]
                print()
                print(f"Steering vs Reference point correlation: {correlation:.3f}")
                if correlation < 0.3:
                    print("⚠️  WARNING: Low correlation between steering and reference point!")
                    print("   Steering is not following the reference point.")
                    print("   → LIKELY STEERING ISSUE (controller not working)")
                elif correlation > 0.7:
                    print("✓ Good correlation - steering follows reference point")
                else:
                    print("⚠️  Moderate correlation - may need improvement")
        else:
            print("No control data available")
        
        print()
        print("=" * 70)
        print("ANALYSIS 4: CONTROL ERRORS")
        print("=" * 70)
        print()
        
        if lateral_error is not None:
            lateral_error_abs = np.abs(lateral_error)
            mean_lateral_error = np.mean(lateral_error_abs)
            max_lateral_error = np.max(lateral_error_abs)
            
            print(f"Lateral error (how far car is from reference point):")
            print(f"  Mean absolute: {mean_lateral_error:.3f}m")
            print(f"  Max absolute: {max_lateral_error:.3f}m")
            print()
            
            if mean_lateral_error > 0.5:
                print("❌ CRITICAL: Large lateral errors!")
                print("   Car is not reaching the reference point.")
                print("   → LIKELY STEERING ISSUE (not aggressive enough)")
            elif mean_lateral_error > 0.2:
                print("⚠️  WARNING: Moderate lateral errors.")
                print("   Car may not be following trajectory well.")
            else:
                print("✓ Lateral errors are small (car is following trajectory)")
        
        if heading_error is not None:
            heading_error_abs = np.abs(heading_error)
            mean_heading_error = np.mean(heading_error_abs)
            max_heading_error = np.max(heading_error_abs)
            
            print(f"Heading error:")
            print(f"  Mean absolute: {np.degrees(mean_heading_error):.2f}°")
            print(f"  Max absolute: {np.degrees(max_heading_error):.2f}°")
            print()
        
        print()
        print("=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)
        print()
        
        # Determine likely issue
        trajectory_issues = []
        steering_issues = []
        
        if mean_ref_x < 0.1 or mean_change < 0.01:
            trajectory_issues.append("Reference point not varying enough")
        
        if gt_center is not None and rmse > 0.5:
            trajectory_issues.append("Reference point far from ground truth")
        
        if steering is not None:
            if mean_steering < 0.05:
                steering_issues.append("Steering too small")
            if correlation < 0.3:
                steering_issues.append("Steering not following reference point")
        
        if lateral_error is not None and mean_lateral_error > 0.5:
            steering_issues.append("Large lateral errors (car not reaching reference)")
        
        print("Likely issues:")
        if trajectory_issues:
            print("  TRAJECTORY ISSUES:")
            for issue in trajectory_issues:
                print(f"    - {issue}")
        else:
            print("  ✓ No obvious trajectory issues")
        
        if steering_issues:
            print("  STEERING ISSUES:")
            for issue in steering_issues:
                print(f"    - {issue}")
        else:
            print("  ✓ No obvious steering issues")
        
        print()
        print("Recommendations:")
        if trajectory_issues:
            print("  1. Check trajectory planner configuration")
            print("  2. Verify perception is giving correct lane positions")
            print("  3. Check reference_lookahead distance")
            print("  4. Run: python tools/analyze_trajectory.py <recording>")
        
        if steering_issues:
            print("  1. Increase steering gains (kp) in config")
            print("  2. Check controller is receiving correct reference point")
            print("  3. Verify steering limits are not too restrictive")
        
        if not trajectory_issues and not steering_issues:
            print("  System appears to be working correctly!")
            print("  If car still drives outside lane, check:")
            print("    - Perception accuracy")
            print("    - Coordinate conversion")
            print("    - Unity physics settings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose trajectory vs steering issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnose latest recording
  python tools/diagnose_trajectory_vs_steering.py
  
  # Diagnose specific recording
  python tools/diagnose_trajectory_vs_steering.py data/recordings/recording_20251231_120044.h5
        """
    )
    parser.add_argument("recording", nargs='?', default=None,
                        help="Path to recording file (default: latest)")
    
    args = parser.parse_args()
    
    # Find input recording
    if args.recording:
        input_file = args.recording
        if not Path(input_file).exists():
            print(f"Error: Recording not found: {input_file}")
            sys.exit(1)
    else:
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("No recordings found!")
            sys.exit(1)
        
        input_file = str(recordings[0])
        print(f"Using latest recording: {Path(input_file).name}\n")
    
    diagnose(input_file)

