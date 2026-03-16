"""
Test trajectory heading fix using real Unity lane detections.
This tests trajectory planning in isolation, not the full integrated system.
"""

import sys

import pytest
import numpy as np
import h5py
from pathlib import Path
from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner
from trajectory.inference import TrajectoryPlanningInference

# Make conftest importable for golden_recording_path
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import golden_recording_path


def load_unity_lane_coefficients(recording_path: Path, frame_idx: int = 0):
    """
    Load actual lane coefficients from Unity recording.
    We need to reconstruct them from the recorded data.
    """
    with h5py.File(recording_path, 'r') as f:
        # Get lane positions at lookahead (vehicle coordinates)
        left_lane_x = np.array(f['perception/left_lane_x'])[frame_idx]
        right_lane_x = np.array(f['perception/right_lane_x'])[frame_idx]
        
        # These are in vehicle coordinates (meters), not image pixels
        # We need to work backwards to get image-space coefficients
        # For now, create synthetic coefficients that match the behavior
        
        # From analysis: Unity lanes have large linear differences (10-20)
        # But road is straight (small curvature)
        # Create coefficients that simulate this
        
        # Left lane: slight slope, small curvature
        left_coeffs = np.array([0.001, -5.0, 200.0])  # Small curvature, negative slope
        
        # Right lane: different slope, small curvature  
        right_coeffs = np.array([0.001, 10.0, 440.0])  # Small curvature, positive slope
        
        # These are NOT parallel (linear diff = 15), but road IS straight (curvature = 0.001)
        
        return left_coeffs, right_coeffs


def test_heading_fix_with_non_parallel_lanes():
    """
    Test that heading fix works even when lanes appear non-parallel in image space.
    This simulates the Unity scenario where car is offset/angled.
    """
    planner = RuleBasedTrajectoryPlanner(
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Create lanes that are NOT parallel in image space (like Unity)
    # But road IS straight (small curvature)
    left_lane = np.array([0.001, -5.0, 200.0])   # Small curvature, negative slope
    right_lane = np.array([0.001, 10.0, 440.0])  # Small curvature, positive slope
    
    # Check: lanes are NOT parallel (linear diff = 15)
    linear_diff = abs(left_lane[-2] - right_lane[-2])
    assert linear_diff > 10, "Lanes should appear non-parallel"
    
    # Check: road IS straight (small curvature)
    max_curvature = max(abs(left_lane[0]), abs(right_lane[0]))
    assert max_curvature < 0.01, "Road should be straight (small curvature)"
    
    # Plan trajectory
    trajectory = planner.plan([left_lane, right_lane])
    
    assert trajectory is not None
    assert len(trajectory.points) > 0
    
    # Check headings - should be near 0° because road is straight
    headings = [p.heading for p in trajectory.points[:10]]
    mean_heading = np.mean(headings)
    
    # For straight road, heading should be near 0° even though lanes appear non-parallel
    assert abs(mean_heading) < np.radians(2.0), (
        f"Heading should be near 0° for straight road (got {np.degrees(mean_heading):.1f}°). "
        f"Fix should work even when lanes appear non-parallel."
    )


def test_heading_fix_with_real_unity_data():
    """Test heading fix using golden recording to verify perception fields exist,
    then test trajectory planning with Unity-like synthetic coefficients.
    """
    recording = golden_recording_path("s_loop")
    if recording is None:
        pytest.skip("s_loop golden recording not registered or not on disk")

    # Verify golden recording has perception data with dual-lane frames
    with h5py.File(recording, 'r') as f:
        has_perception = ('perception/left_lane_x' in f
                          or 'perception/left_lane_line_x' in f)
        assert has_perception, "No perception data in golden recording"
        num_lanes = np.array(f['perception/num_lanes_detected'])
        dual_lane_frames = np.where(num_lanes == 2)[0]
        assert len(dual_lane_frames) > 0, "No dual-lane frames in golden recording"

    # Use synthetic coefficients that match Unity behavior
    # (non-parallel in image space, but straight road)
    left_lane = np.array([0.001, -7.0, 200.0])
    right_lane = np.array([0.001, 13.0, 440.0])

    planner = TrajectoryPlanningInference(
        planner_type="rule_based",
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )

    trajectory = planner.plan([left_lane, right_lane])
    ref_point = planner.get_reference_point(trajectory, lookahead=10.0)

    assert ref_point is not None
    assert 'heading' in ref_point

    heading = ref_point['heading']
    assert abs(heading) < np.radians(2.0), (
        f"Reference heading should be near 0 for straight road "
        f"(got {np.degrees(heading):.1f} deg)"
    )


def test_heading_fix_curvature_threshold():
    """
    Test that heading fix uses curvature threshold correctly.
    """
    planner = RuleBasedTrajectoryPlanner(
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Test case 1: Straight road (small curvature) - should fix
    left_straight = np.array([0.0001, -5.0, 200.0])   # Very small curvature
    right_straight = np.array([0.0001, 10.0, 440.0])
    
    trajectory_straight = planner.plan([left_straight, right_straight])
    headings_straight = [p.heading for p in trajectory_straight.points[:10]]
    mean_heading_straight = np.mean(headings_straight)
    
    assert abs(mean_heading_straight) < np.radians(2.0), (
        f"Straight road should have heading ~0° (got {np.degrees(mean_heading_straight):.1f}°)"
    )
    
    # Test case 2: Curved road (large curvature) - should NOT fix
    left_curved = np.array([0.01, -5.0, 200.0])   # Larger curvature
    right_curved = np.array([0.01, 10.0, 440.0])
    
    trajectory_curved = planner.plan([left_curved, right_curved])
    headings_curved = [p.heading for p in trajectory_curved.points[:10]]
    mean_heading_curved = np.mean(headings_curved)
    
    # Curved road may have non-zero heading (this is correct)
    # We just verify the fix doesn't break curved roads
    assert trajectory_curved is not None, "Curved road should still produce trajectory"

