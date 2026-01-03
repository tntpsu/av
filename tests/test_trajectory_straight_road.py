"""
Test trajectory planning for straight road scenarios.
Verifies that straight roads produce straight trajectories with zero heading.
"""

import pytest
import numpy as np
from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner


def test_parallel_lanes_produce_straight_center():
    """Test that parallel lanes produce a straight center lane."""
    planner = RuleBasedTrajectoryPlanner(
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Create parallel lanes (same linear coefficient)
    # Left lane at x=200, right lane at x=440 (parallel)
    left_lane = np.array([0.0, 0.0, 200.0])  # Straight line at x=200
    right_lane = np.array([0.0, 0.0, 440.0])  # Straight line at x=440
    
    trajectory = planner.plan([left_lane, right_lane])
    
    assert trajectory is not None
    assert len(trajectory.points) > 0
    
    # Check that center lane is straight (zero linear coefficient)
    # This is tested in _compute_center_lane, but we verify the result
    # by checking heading of trajectory points
    headings = [p.heading for p in trajectory.points]
    mean_heading = np.mean(headings)
    
    # For straight road, heading should be near 0°
    assert abs(mean_heading) < np.radians(2.0), f"Center lane should be straight (heading: {np.degrees(mean_heading):.1f}°)"


def test_center_lane_linear_coefficient_zeroed():
    """Test that center lane linear coefficient is zeroed for parallel lanes."""
    planner = RuleBasedTrajectoryPlanner(
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Create parallel lanes with slight noise
    # Left lane: x = 200 + 0.01*y (slight slope)
    # Right lane: x = 440 + 0.01*y (same slope, parallel)
    left_lane = np.array([0.0, 0.01, 200.0])
    right_lane = np.array([0.0, 0.01, 440.0])
    
    trajectory = planner.plan([left_lane, right_lane])
    
    assert trajectory is not None
    
    # Check headings - should be near 0° even though lanes have slope
    # (because they're parallel, center should be straight)
    headings = [p.heading for p in trajectory.points[:10]]  # Check first 10 points
    mean_heading = np.mean(headings)
    
    assert abs(mean_heading) < np.radians(2.0), f"Parallel lanes should produce straight center (heading: {np.degrees(mean_heading):.1f}°)"


def test_reference_heading_for_straight_road():
    """Test that reference heading is near 0° for straight roads."""
    from trajectory.inference import TrajectoryPlanningInference
    
    planner = TrajectoryPlanningInference(
        planner_type="rule_based",
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Create parallel straight lanes
    left_lane = np.array([0.0, 0.0, 200.0])
    right_lane = np.array([0.0, 0.0, 440.0])
    
    trajectory = planner.plan([left_lane, right_lane])
    ref_point = planner.get_reference_point(trajectory, lookahead=10.0)
    
    assert ref_point is not None
    assert 'heading' in ref_point
    
    # Reference heading should be near 0° for straight road
    heading = ref_point['heading']
    assert abs(heading) < np.radians(2.0), f"Reference heading should be near 0° (got {np.degrees(heading):.1f}°)"


def test_non_parallel_lanes_handled():
    """Test that non-parallel lanes are handled correctly."""
    planner = RuleBasedTrajectoryPlanner(
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Create non-parallel lanes (different slopes)
    left_lane = np.array([0.0, 0.01, 200.0])   # Slight slope
    right_lane = np.array([0.0, 0.05, 440.0])  # Different slope (not parallel)
    
    trajectory = planner.plan([left_lane, right_lane])
    
    # Should still produce a trajectory (even if not perfectly straight)
    assert trajectory is not None
    assert len(trajectory.points) > 0

