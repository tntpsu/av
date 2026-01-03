"""
Test reference point computation when car is offset (facing right/left).
Verifies that midpoint calculation works correctly even with different lane curvatures.
"""

import pytest
import numpy as np
from trajectory.inference import TrajectoryPlanningInference


def test_reference_point_when_car_offset_right():
    """Test reference point when car is offset to the right."""
    planner = TrajectoryPlanningInference(
        planner_type="rule_based",
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Scenario: Car is offset to the right
    # Left lane is further left, right lane is closer
    # Lane positions in vehicle coordinates (already converted)
    lane_positions = {
        'left_lane_x': -2.0,   # Left lane is 2m to the left
        'right_lane_x': 0.5    # Right lane is 0.5m to the right
    }
    
    # Create dummy trajectory (not used when lane_positions provided)
    from trajectory.models.trajectory_planner import Trajectory, TrajectoryPoint
    trajectory = Trajectory(
        points=[TrajectoryPoint(x=0.0, y=8.0, heading=0.0, velocity=8.0, curvature=0.0)],
        length=8.0
    )
    
    ref_point = planner.get_reference_point(
        trajectory,
        lookahead=8.0,
        lane_positions=lane_positions,
        use_direct=True
    )
    
    assert ref_point is not None
    assert 'x' in ref_point
    
    # Reference point should be at lane center
    expected_center = (-2.0 + 0.5) / 2.0  # -0.75m
    assert abs(ref_point['x'] - expected_center) < 0.1, (
        f"Reference point should be at lane center ({expected_center:.2f}m), "
        f"got {ref_point['x']:.2f}m"
    )
    
    # Heading should be 0° for straight road
    assert abs(ref_point['heading']) < np.radians(2.0), (
        f"Heading should be near 0° for straight road, got {np.degrees(ref_point['heading']):.1f}°"
    )


def test_reference_point_when_car_offset_left():
    """Test reference point when car is offset to the left."""
    planner = TrajectoryPlanningInference(
        planner_type="rule_based",
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Scenario: Car is offset to the left
    # Left lane is closer, right lane is further right
    lane_positions = {
        'left_lane_x': -0.5,   # Left lane is 0.5m to the left
        'right_lane_x': 2.0     # Right lane is 2m to the right
    }
    
    from trajectory.models.trajectory_planner import Trajectory, TrajectoryPoint
    trajectory = Trajectory(
        points=[TrajectoryPoint(x=0.0, y=8.0, heading=0.0, velocity=8.0, curvature=0.0)],
        length=8.0
    )
    
    ref_point = planner.get_reference_point(
        trajectory,
        lookahead=8.0,
        lane_positions=lane_positions,
        use_direct=True
    )
    
    assert ref_point is not None
    
    # Reference point should be at lane center
    expected_center = (-0.5 + 2.0) / 2.0  # 0.75m
    assert abs(ref_point['x'] - expected_center) < 0.1, (
        f"Reference point should be at lane center ({expected_center:.2f}m), "
        f"got {ref_point['x']:.2f}m"
    )


def test_reference_point_with_different_curvatures():
    """
    Test that midpoint calculation works even when lanes have different curvatures.
    
    When car is offset, one lane appears more curved than the other (perspective).
    The midpoint should still be correct because we're using lane positions
    already converted to vehicle coordinates.
    """
    planner = TrajectoryPlanningInference(
        planner_type="rule_based",
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Scenario: Car is far right, left lane appears more curved
    # But lane positions in vehicle coords are still accurate
    lane_positions = {
        'left_lane_x': -1.5,   # Left lane position (already in vehicle coords)
        'right_lane_x': 0.3    # Right lane position (already in vehicle coords)
    }
    
    from trajectory.models.trajectory_planner import Trajectory, TrajectoryPoint
    trajectory = Trajectory(
        points=[TrajectoryPoint(x=0.0, y=8.0, heading=0.0, velocity=8.0, curvature=0.0)],
        length=8.0
    )
    
    ref_point = planner.get_reference_point(
        trajectory,
        lookahead=8.0,
        lane_positions=lane_positions,
        use_direct=True
    )
    
    assert ref_point is not None
    
    # Midpoint should be correct regardless of curvature differences
    # because we're using vehicle coordinates, not image coordinates
    expected_center = (-1.5 + 0.3) / 2.0  # -0.6m
    assert abs(ref_point['x'] - expected_center) < 0.1, (
        f"Midpoint should be correct even with different curvatures. "
        f"Expected {expected_center:.2f}m, got {ref_point['x']:.2f}m"
    )


def test_reference_point_fallback_to_lane_coeffs():
    """Test that fallback to lane coefficients works when lane_positions not available."""
    planner = TrajectoryPlanningInference(
        planner_type="rule_based",
        image_width=640,
        image_height=480,
        camera_fov=60.0,
        camera_height=1.0
    )
    
    # Create lanes with different curvatures (simulating car offset)
    left_lane = np.array([0.05, -10.0, 200.0])   # High curvature
    right_lane = np.array([0.01, 5.0, 440.0])   # Lower curvature
    
    from trajectory.models.trajectory_planner import Trajectory, TrajectoryPoint
    trajectory = Trajectory(
        points=[TrajectoryPoint(x=0.0, y=8.0, heading=0.0, velocity=8.0, curvature=0.0)],
        length=8.0
    )
    
    # Use lane_coeffs (fallback method)
    ref_point = planner.get_reference_point(
        trajectory,
        lookahead=8.0,
        lane_coeffs=[left_lane, right_lane],
        lane_positions=None,  # Not provided, should use fallback
        use_direct=True
    )
    
    assert ref_point is not None
    assert 'x' in ref_point
    assert 'heading' in ref_point
    
    # Heading should be 0° for straight road (curvature check should apply)
    # Note: max curvature is 0.05, which is <= 0.05 threshold
    assert abs(ref_point['heading']) < np.radians(2.0), (
        f"Heading should be near 0° for straight road, got {np.degrees(ref_point['heading']):.1f}°"
    )

