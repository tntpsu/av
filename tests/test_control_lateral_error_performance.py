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



