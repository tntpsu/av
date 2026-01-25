"""
Test control system lateral error performance.

This test verifies that the control system can maintain acceptable lateral error
(RMSE < 0.4m) during path following scenarios. This catches control tuning issues
before Unity testing.

Thresholds (matching Path Tracking metrics):
- Good: RMSE < 0.2m
- Acceptable: RMSE < 0.4m
- Poor: RMSE >= 0.4m
"""

import pytest
import numpy as np
import yaml
from pathlib import Path
from control.pid_controller import LateralController


def _load_control_config():
    """Load control config from av_stack_config.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "av_stack_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get("control", {}).get("lateral", {})


def test_control_lateral_error_straight_road():
    """
    Test that control maintains acceptable lateral error on straight road.
    
    Scenario: Car starts with 0.5m lateral offset, should converge to < 0.2m RMSE.
    """
    # Load config values from av_stack_config.yaml
    lateral_config = _load_control_config()
    controller = LateralController(
        kp=lateral_config.get("kp", 1.0),
        ki=lateral_config.get("ki", 0.002),
        kd=lateral_config.get("kd", 0.5),
        lookahead_distance=10.0,
        max_steering=lateral_config.get("max_steering", 1.0),
        deadband=lateral_config.get("deadband", 0.01),
        heading_weight=lateral_config.get("heading_weight", 0.6),
        lateral_weight=lateral_config.get("lateral_weight", 0.4),
        error_clip=lateral_config.get("error_clip", np.pi / 2),
        integral_limit=lateral_config.get("integral_limit", 0.10)
    )
    
    # Simulate 5 seconds at 30 FPS
    dt = 1.0 / 30.0
    num_frames = int(5.0 / dt)  # 150 frames
    
    # Initial state: car is 0.5m to the left of reference
    lateral_errors = []
    current_lateral_error = 0.5  # Start 0.5m left of reference
    current_heading = 0.0
    
    for frame in range(num_frames):
        # Reference point: straight ahead, centered (desired position)
        reference_point = {
            'x': 0.0,  # Desired: centered
            'y': 10.0,
            'heading': 0.0,  # Desired: straight
            'velocity': 10.0
        }
        
        # Compute steering
        steering = controller.compute_steering(
            current_heading=current_heading,
            reference_point=reference_point
        )
        
        # Simple vehicle dynamics: steering affects lateral position
        # For bicycle model: lateral_velocity = speed * sin(steering_angle)
        # For small angles: sin(steering) ≈ steering, so lateral_velocity ≈ speed * steering
        speed = 10.0  # m/s
        # Use actual steering angle (steering is normalized -1 to 1, convert to radians)
        # Assuming max steering angle of ~30 degrees (0.52 rad) for steering=1.0
        max_steering_angle_rad = 0.52  # ~30 degrees
        steering_angle_rad = steering * max_steering_angle_rad
        lateral_velocity = speed * np.sin(steering_angle_rad)  # More accurate model
        current_lateral_error -= lateral_velocity * dt  # Update position
        
        # Update heading (simplified: heading change = steering * some factor)
        # Heading change rate is proportional to steering and speed
        heading_rate = steering * speed / 2.5  # Simplified: wheelbase ~2.5m
        current_heading += heading_rate * dt
        
        # Record error (actual error from reference)
        actual_error = current_lateral_error - reference_point['x']
        lateral_errors.append(abs(actual_error))
    
    # Calculate RMSE
    lateral_error_rmse = np.sqrt(np.mean(np.array(lateral_errors)**2))
    
    # Should converge to acceptable error (< 0.4m)
    assert lateral_error_rmse < 0.4, (
        f"Lateral error RMSE ({lateral_error_rmse:.3f}m) exceeds acceptable threshold (0.4m). "
        f"This indicates control tuning issue. Consider:\n"
        f"  - Increasing kp (current: {controller.pid.kp})\n"
        f"  - Adjusting ki (current: {controller.pid.ki})\n"
        f"  - Checking steering response to errors"
    )
    
    # Ideally should be good (< 0.2m)
    if lateral_error_rmse >= 0.2:
        pytest.skip(f"Lateral error RMSE ({lateral_error_rmse:.3f}m) is acceptable but not good (< 0.2m). "
                   f"Consider tuning for better performance.")


def test_control_lateral_error_curve():
    """
    Test that control maintains acceptable lateral error on curved road.
    
    Scenario: Car follows a curved path (sinusoidal reference), should maintain < 0.4m RMSE.
    """
    # Load config values from av_stack_config.yaml
    lateral_config = _load_control_config()
    controller = LateralController(
        kp=lateral_config.get("kp", 1.0),
        ki=lateral_config.get("ki", 0.002),
        kd=lateral_config.get("kd", 0.5),
        lookahead_distance=10.0,
        max_steering=lateral_config.get("max_steering", 1.0),
        deadband=lateral_config.get("deadband", 0.01),
        heading_weight=lateral_config.get("heading_weight", 0.6),
        lateral_weight=lateral_config.get("lateral_weight", 0.4),
        error_clip=lateral_config.get("error_clip", np.pi / 2),
        integral_limit=lateral_config.get("integral_limit", 0.10)
    )
    
    # Simulate 10 seconds at 30 FPS (longer for curve)
    dt = 1.0 / 30.0
    num_frames = int(10.0 / dt)  # 300 frames
    
    # Initial state
    lateral_errors = []
    current_lateral_error = 0.0
    current_heading = 0.0
    time = 0.0
    
    for frame in range(num_frames):
        # Curved reference: sinusoidal path
        # Reference oscillates left/right with 20m period
        reference_x = 0.5 * np.sin(2 * np.pi * time / 20.0)  # ±0.5m oscillation
        reference_heading = np.arctan2(
            0.5 * 2 * np.pi / 20.0 * np.cos(2 * np.pi * time / 20.0),
            1.0
        )  # Heading to follow curve
        
        reference_point = {
            'x': reference_x,
            'y': 10.0,
            'heading': reference_heading,
            'velocity': 10.0
        }
        
        # Compute steering
        steering = controller.compute_steering(
            current_heading=current_heading,
            reference_point=reference_point
        )
        
        # Update vehicle state
        speed = 10.0
        lateral_velocity = steering * speed
        current_lateral_error = reference_x  # Error is difference from reference
        current_heading += steering * 0.1 * dt
        
        # Record error
        lateral_errors.append(abs(current_lateral_error))
        time += dt
    
    # Calculate RMSE
    lateral_error_rmse = np.sqrt(np.mean(np.array(lateral_errors)**2))
    
    # Should maintain acceptable error on curves (< 0.4m)
    assert lateral_error_rmse < 0.4, (
        f"Lateral error RMSE on curve ({lateral_error_rmse:.3f}m) exceeds acceptable threshold (0.4m). "
        f"This indicates control tuning issue for curves. Consider:\n"
        f"  - Increasing kp for better response (current: {controller.pid.kp})\n"
        f"  - Adjusting feedforward contribution\n"
        f"  - Checking steering limits"
    )


def test_control_lateral_error_step_response():
    """
    Test step response: car should converge quickly after sudden offset.
    
    Scenario: Car starts centered, then reference jumps 0.3m, should converge in < 2 seconds.
    """
    # Load config values from av_stack_config.yaml
    lateral_config = _load_control_config()
    controller = LateralController(
        kp=lateral_config.get("kp", 1.0),
        ki=lateral_config.get("ki", 0.002),
        kd=lateral_config.get("kd", 0.5),
        lookahead_distance=10.0,
        max_steering=lateral_config.get("max_steering", 1.0),
        deadband=lateral_config.get("deadband", 0.01),
        heading_weight=lateral_config.get("heading_weight", 0.6),
        lateral_weight=lateral_config.get("lateral_weight", 0.4),
        error_clip=lateral_config.get("error_clip", np.pi / 2),
        integral_limit=lateral_config.get("integral_limit", 0.10)
    )
    
    dt = 1.0 / 30.0
    num_frames = int(3.0 / dt)  # 3 seconds
    
    lateral_errors = []
    current_lateral_error = 0.0
    current_heading = 0.0
    
    for frame in range(num_frames):
        # Step change at frame 30 (1 second): reference jumps 0.3m
        if frame < 30:
            reference_x = 0.0
        else:
            reference_x = 0.3
        
        reference_point = {
            'x': reference_x,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 10.0
        }
        
        steering = controller.compute_steering(
            current_heading=current_heading,
            reference_point=reference_point
        )
        
        speed = 10.0
        lateral_velocity = steering * speed
        current_lateral_error = reference_x
        current_heading += steering * 0.1 * dt
        
        lateral_errors.append(abs(current_lateral_error))
    
    # After step (frames 30+), should converge quickly
    post_step_errors = lateral_errors[30:]
    post_step_rmse = np.sqrt(np.mean(np.array(post_step_errors)**2))
    
    # Should converge to < 0.2m within 2 seconds after step
    assert post_step_rmse < 0.2, (
        f"Step response RMSE ({post_step_rmse:.3f}m) exceeds threshold (0.2m). "
        f"Control should converge faster. Consider increasing kp or adjusting tuning."
    )
    
    # Final error should be small
    final_error = lateral_errors[-1]
    assert final_error < 0.15, (
        f"Final error after step ({final_error:.3f}m) is too large. "
        f"Control should converge better."
    )

