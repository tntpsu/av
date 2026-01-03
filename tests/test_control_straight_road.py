"""
Test control system for straight road scenarios.
Verifies that control produces minimal steering for straight roads.
"""

import pytest
import numpy as np
from control.pid_controller import LateralController


def test_straight_road_produces_minimal_steering():
    """Test that straight road with centered car produces minimal steering."""
    controller = LateralController(
        kp=0.3,
        ki=0.0,
        kd=0.1,
        lookahead_distance=10.0,
        max_steering=0.5
    )
    
    # Reference point: straight ahead, centered
    reference_point = {
        'x': 0.0,  # Centered (no lateral offset)
        'y': 10.0,  # 10m ahead
        'heading': 0.0,  # Straight ahead
        'velocity': 10.0
    }
    
    # Car is centered, heading straight
    steering = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point
    )
    
    # Should produce minimal steering (near 0.0)
    assert abs(steering) < 0.05, f"Straight road should produce minimal steering (got {steering:.3f})"


def test_steering_direction_correct():
    """Test that steering direction is correct for lateral errors."""
    controller = LateralController(
        kp=0.3,
        ki=0.0,
        kd=0.1,
        lookahead_distance=10.0,
        max_steering=0.5,
        deadband=0.0
    )
    
    # Case 1: Car is LEFT of reference (ref_x > 0)
    # Should steer RIGHT (positive steering)
    reference_point_left = {
        'x': 0.5,  # Reference is 0.5m RIGHT of vehicle center
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering_left = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point_left
    )
    
    # Car is LEFT of ref → should steer RIGHT (positive)
    assert steering_left > 0.0, f"Car left of ref should steer right (got {steering_left:.3f})"
    
    # Case 2: Car is RIGHT of reference (ref_x < 0)
    # Should steer LEFT (negative steering)
    reference_point_right = {
        'x': -0.5,  # Reference is 0.5m LEFT of vehicle center
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering_right = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point_right
    )
    
    # Car is RIGHT of ref → should steer LEFT (negative)
    assert steering_right < 0.0, f"Car right of ref should steer left (got {steering_right:.3f})"


def test_heading_error_affects_steering():
    """Test that heading error affects steering."""
    controller = LateralController(
        kp=0.3,
        ki=0.0,
        kd=0.1,
        lookahead_distance=10.0,
        max_steering=0.5,
        heading_weight=0.5,
        lateral_weight=0.5
    )
    
    # Case 1: Car heading LEFT (negative heading error)
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,  # Desired: straight
        'velocity': 10.0
    }
    
    steering_left_heading = controller.compute_steering(
        current_heading=-0.1,  # Car heading left (10°)
        reference_point=reference_point
    )
    
    # Should steer RIGHT to correct (positive)
    assert steering_left_heading > 0.0, f"Left heading should steer right (got {steering_left_heading:.3f})"
    
    # Case 2: Car heading RIGHT (positive heading error)
    steering_right_heading = controller.compute_steering(
        current_heading=0.1,  # Car heading right (10°)
        reference_point=reference_point
    )
    
    # Should steer LEFT to correct (negative)
    assert steering_right_heading < 0.0, f"Right heading should steer left (got {steering_right_heading:.3f})"

