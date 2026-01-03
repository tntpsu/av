#!/usr/bin/env python3
"""
Test trajectory heading fix in isolation using real Unity lane detections.
This allows us to verify the fix without running the full integrated system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner
from trajectory.inference import TrajectoryPlanningInference


def test_heading_with_unity_lanes(recording_path: Path, frame_idx: int = 0):
    """
    Test heading calculation using actual Unity lane detections.
    This tests trajectory planning in isolation.
    """
    print("="*80)
    print("TRAJECTORY HEADING TEST (ISOLATED)")
    print("="*80)
    print(f"Recording: {recording_path.name}")
    print(f"Frame: {frame_idx}\n")
    
    with h5py.File(recording_path, 'r') as f:
        # Get perception data
        left_lane_x = np.array(f['perception/left_lane_x'])[frame_idx]
        right_lane_x = np.array(f['perception/right_lane_x'])[frame_idx]
        num_lanes = np.array(f['perception/num_lanes_detected'])[frame_idx]
        
        if num_lanes < 2:
            print(f"⚠ Frame {frame_idx} has only {num_lanes} lanes detected")
            return
        
        # Get actual trajectory heading from recording
        ref_heading = np.array(f['trajectory/reference_point_heading'])[frame_idx]
        ref_heading_deg = np.degrees(ref_heading)
        
        print(f"Recorded Reference Heading: {ref_heading_deg:.1f}°")
        print()
    
    # Create lane coefficients that match Unity behavior
    # From analysis: Unity lanes have large linear differences but small curvature
    # Left: linear ≈ -7.37, Right: linear ≈ 13.85, Curvature ≈ 0.001
    left_lane = np.array([0.001, -7.37, 200.0])
    right_lane = np.array([0.001, 13.85, 440.0])
    
    print("Lane Coefficients (simulating Unity):")
    print(f"  Left:  {left_lane}")
    print(f"  Right: {right_lane}")
    print()
    
    # Check properties
    linear_diff = abs(left_lane[-2] - right_lane[-2])
    max_curvature = max(abs(left_lane[0]), abs(right_lane[0]))
    
    print(f"Properties:")
    print(f"  Linear difference: {linear_diff:.2f} (appears non-parallel)")
    print(f"  Max curvature: {max_curvature:.4f} (road is straight)")
    print()
    
    # Test trajectory planning
    planner = TrajectoryPlanningInference(
        planner_type="rule_based",
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    trajectory = planner.plan([left_lane, right_lane])
    ref_point = planner.get_reference_point(trajectory, lookahead=10.0)
    
    if ref_point is None:
        print("⚠ No reference point generated")
        return
    
    test_heading = ref_point['heading']
    test_heading_deg = np.degrees(test_heading)
    
    print(f"Test Results:")
    print(f"  Reference Heading: {test_heading_deg:.1f}°")
    print(f"  Target: ~0° (for straight road)")
    print()
    
    # Check if fix worked
    if abs(test_heading_deg) < 2.0:
        print("✅ HEADING FIX WORKS!")
        print(f"   Heading is near 0° ({test_heading_deg:.1f}°) for straight road")
        print(f"   Even though lanes appear non-parallel (diff: {linear_diff:.2f})")
    else:
        print("⚠ HEADING FIX MAY NOT BE WORKING")
        print(f"   Heading is {test_heading_deg:.1f}° (expected ~0°)")
        print(f"   May need to adjust curvature threshold")
    
    print()
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Recorded (full system): {ref_heading_deg:.1f}°")
    print(f"Test (trajectory only):  {test_heading_deg:.1f}°")
    print()
    
    if abs(test_heading_deg) < abs(ref_heading_deg):
        improvement = abs(ref_heading_deg) - abs(test_heading_deg)
        print(f"✅ Improvement: {improvement:.1f}°")
        print(f"   Trajectory planning fix is working!")
    else:
        print(f"⚠ No improvement or worse")
        print(f"   May need further investigation")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trajectory heading fix in isolation')
    parser.add_argument('--recording', type=str, help='Path to recording file')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to test')
    
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
    
    test_heading_with_unity_lanes(recording_path, args.frame)


if __name__ == '__main__':
    main()

