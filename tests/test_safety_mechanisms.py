"""
Tests for safety mechanisms (recovery mode, hard limits, emergency stop).
These tests verify that safety mechanisms trigger correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from control.pid_controller import LateralController, LongitudinalController, VehicleController


class TestSafetyMechanisms:
    """Test that safety mechanisms trigger correctly."""
    
    def test_lateral_error_in_control_command(self):
        """
        CRITICAL: Verify lateral_error is in control_command.
        
        Safety mechanisms in av_stack.py need lateral_error to trigger.
        This test ensures it's always present.
        """
        controller = VehicleController(
            lateral_kp=0.5, lateral_kd=0.1,
            longitudinal_kp=0.3, longitudinal_ki=0.05, longitudinal_kd=0.02
        )
        
        current_state = {
            'heading': 0.0,
            'speed': 8.0,
            'position': np.array([0.0, 0.0])
        }
        
        reference_point = {
            'x': 0.0,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        control_command = controller.compute_control(
            current_state, reference_point, return_metadata=True
        )
        
        # lateral_error must be in control_command for safety mechanisms
        assert 'lateral_error' in control_command, (
            "lateral_error must be in control_command for safety mechanisms to work!"
        )
        
        # Should be a number
        assert isinstance(control_command['lateral_error'], (int, float, np.number)), (
            f"lateral_error must be a number, got {type(control_command['lateral_error'])}"
        )
    
    def test_recovery_mode_threshold(self):
        """
        Test that recovery mode would trigger at 1.0m error.
        
        This test verifies the threshold logic (not the actual triggering in av_stack).
        """
        controller = VehicleController(
            lateral_kp=0.5, lateral_kd=0.1,
            longitudinal_kp=0.3, longitudinal_ki=0.05, longitudinal_kd=0.02
        )
        
        current_state = {
            'heading': 0.0,
            'speed': 8.0,
            'position': np.array([0.0, 0.0])  # Vehicle position (not used for lateral_error anymore)
        }
        
        reference_point = {
            'x': 1.5,  # Reference 1.5m right of vehicle center (vehicle frame)
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        control_command = controller.compute_control(
            current_state, reference_point, return_metadata=True
        )
        
        lateral_error_abs = abs(control_command['lateral_error'])
        
        # Error should be > 1.0m (recovery threshold)
        assert lateral_error_abs > 1.0, (
            f"Error {lateral_error_abs:.3f}m should be > 1.0m for recovery mode test"
        )
        
        # Verify lateral_error is present
        assert 'lateral_error' in control_command, (
            "lateral_error must be present for safety mechanisms"
        )
    
    def test_hard_limit_threshold(self):
        """
        Test that hard limit would trigger at 2.0m error.
        """
        controller = VehicleController(
            lateral_kp=0.5, lateral_kd=0.1,
            longitudinal_kp=0.3, longitudinal_ki=0.05, longitudinal_kd=0.02
        )
        
        current_state = {
            'heading': 0.0,
            'speed': 8.0,
            'position': np.array([0.0, 0.0])  # Vehicle position (not used for lateral_error anymore)
        }
        
        reference_point = {
            'x': 2.5,  # Reference 2.5m right of vehicle center (vehicle frame)
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        control_command = controller.compute_control(
            current_state, reference_point, return_metadata=True
        )
        
        lateral_error_abs = abs(control_command['lateral_error'])
        
        # Error should be > 2.0m (hard limit threshold)
        assert lateral_error_abs > 2.0, (
            f"Error {lateral_error_abs:.3f}m should be > 2.0m for hard limit test"
        )
    
    def test_emergency_stop_threshold(self):
        """
        Test that emergency stop would trigger at 3.0m error.
        """
        controller = VehicleController(
            lateral_kp=0.5, lateral_kd=0.1,
            longitudinal_kp=0.3, longitudinal_ki=0.05, longitudinal_kd=0.02
        )
        
        current_state = {
            'heading': 0.0,
            'speed': 8.0,
            'position': np.array([0.0, 0.0])  # Vehicle position (not used for lateral_error anymore)
        }
        
        reference_point = {
            'x': 3.5,  # Reference 3.5m right of vehicle center (vehicle frame)
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        control_command = controller.compute_control(
            current_state, reference_point, return_metadata=True
        )
        
        lateral_error_abs = abs(control_command['lateral_error'])
        
        # Error should be > 3.0m (emergency stop threshold)
        assert lateral_error_abs > 3.0, (
            f"Error {lateral_error_abs:.3f}m should be > 3.0m for emergency stop test"
        )

