"""
Integration tests for configuration system.
Tests that config parameters are correctly passed through all layers.
"""

import pytest
import sys
import tempfile
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from av_stack import AVStack, load_config
from control.pid_controller import VehicleController


class TestConfigParameterFlow:
    """Test that config parameters flow correctly from YAML to components."""
    
    def test_lateral_params_flow(self):
        """Test that lateral control parameters flow from config to controller."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'control': {
                    'lateral': {
                        'kp': 0.4,
                        'ki': 0.01,
                        'kd': 0.15,
                        'max_steering': 0.6,
                        'deadband': 0.025,
                        'heading_weight': 0.6,
                        'lateral_weight': 0.4,
                        'error_clip': 0.8,
                        'steering_smoothing_alpha': 0.65,
                        'curve_feedforward_gain': 1.2,
                        'curve_feedforward_threshold': 0.03,
                        'curve_feedforward_gain_min': 0.8,
                        'curve_feedforward_gain_max': 1.3,
                        'curve_feedforward_curvature_min': 0.004,
                        'curve_feedforward_curvature_max': 0.025,
                        'curve_feedforward_curvature_clamp': 0.03,
                        'straight_curvature_threshold': 0.012,
                        'steering_rate_curvature_min': 0.006,
                        'steering_rate_curvature_max': 0.02,
                        'steering_rate_scale_min': 0.55
                    }
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            av_stack = AVStack(
                bridge_url="http://localhost:8000",
                record_data=False,
                config_path=config_path
            )
            
            # Check that parameters reached the controller
            lateral_ctrl = av_stack.controller.lateral_controller
            assert lateral_ctrl.pid.kp == 0.4
            assert lateral_ctrl.pid.ki == 0.01
            assert lateral_ctrl.pid.kd == 0.15
            assert lateral_ctrl.max_steering == 0.6
            assert lateral_ctrl.deadband == 0.025
            assert lateral_ctrl.heading_weight == 0.6
            assert lateral_ctrl.lateral_weight == 0.4
            assert lateral_ctrl.error_clip == 0.8
            assert lateral_ctrl.steering_smoothing_alpha == 0.65
            assert lateral_ctrl.curve_feedforward_gain == 1.2
            assert lateral_ctrl.curve_feedforward_threshold == 0.03
            assert lateral_ctrl.curve_feedforward_gain_min == 0.8
            assert lateral_ctrl.curve_feedforward_gain_max == 1.3
            assert lateral_ctrl.curve_feedforward_curvature_min == 0.004
            assert lateral_ctrl.curve_feedforward_curvature_max == 0.025
            assert lateral_ctrl.curve_feedforward_curvature_clamp == 0.03
            assert lateral_ctrl.straight_curvature_threshold == 0.012
            assert lateral_ctrl.steering_rate_curvature_min == 0.006
            assert lateral_ctrl.steering_rate_curvature_max == 0.02
            assert lateral_ctrl.steering_rate_scale_min == 0.55
        finally:
            Path(config_path).unlink()
    
    def test_longitudinal_params_flow(self):
        """Test that longitudinal control parameters flow from config to controller."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'control': {
                    'longitudinal': {
                        'kp': 0.4,
                        'ki': 0.06,
                        'kd': 0.03,
                        'target_speed': 9.0,
                        'max_speed': 12.0,
                        'throttle_rate_limit': 0.06,
                        'brake_rate_limit': 0.12,
                        'throttle_smoothing_alpha': 0.55
                    }
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            av_stack = AVStack(
                bridge_url="http://localhost:8000",
                record_data=False,
                config_path=config_path
            )
            
            # Check that parameters reached the controller
            long_ctrl = av_stack.controller.longitudinal_controller
            assert long_ctrl.target_speed == 9.0
            assert long_ctrl.throttle_rate_limit == 0.06
            assert long_ctrl.brake_rate_limit == 0.12
            assert long_ctrl.throttle_smoothing_alpha == 0.55
            # Note: max_speed is set in VehicleController, not LongitudinalController
            # The safety config max_speed is used in _process_frame, not in controller
            assert long_ctrl.pid_throttle.kp == 0.4
            assert long_ctrl.pid_throttle.ki == 0.06
            assert long_ctrl.pid_throttle.kd == 0.03
        finally:
            Path(config_path).unlink()
    
    def test_trajectory_params_flow(self):
        """Test that trajectory planning parameters flow from config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'trajectory': {
                    'lookahead_distance': 30.0,
                    'target_speed': 9.0,
                    'reference_lookahead': 12.0,
                    'image_width': 800.0,
                    'image_height': 600.0,
                    'camera_fov': 70.0,
                    'camera_height': 0.6,
                    'bias_correction_threshold': 15.0
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            av_stack = AVStack(
                bridge_url="http://localhost:8000",
                record_data=False,
                config_path=config_path
            )
            
            # Check that parameters reached the planner
            planner = av_stack.trajectory_planner.planner
            assert planner.lookahead_distance == 30.0
            assert planner.target_speed == 9.0
            assert planner.image_width == 800.0
            assert planner.image_height == 600.0
            assert planner.camera_fov == 70.0
            assert planner.camera_height == 0.6
            assert planner.bias_correction_threshold == 15.0
        finally:
            Path(config_path).unlink()
    
    def test_safety_params_flow(self):
        """Test that safety parameters are stored correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'safety': {
                    'max_speed': 15.0,
                    'emergency_brake_threshold': 2.5,
                    'speed_prevention_threshold': 0.9,
                    'speed_prevention_brake_threshold': 0.95,
                    'speed_prevention_brake_amount': 0.3
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            av_stack = AVStack(
                bridge_url="http://localhost:8000",
                record_data=False,
                config_path=config_path
            )
            
            # Check that safety config is stored
            assert av_stack.safety_config['max_speed'] == 15.0
            assert av_stack.safety_config['emergency_brake_threshold'] == 2.5
            assert av_stack.safety_config['speed_prevention_threshold'] == 0.9
            assert av_stack.safety_config['speed_prevention_brake_threshold'] == 0.95
            assert av_stack.safety_config['speed_prevention_brake_amount'] == 0.3
        finally:
            Path(config_path).unlink()


class TestConfigDefaults:
    """Test that default values are used when config is missing."""
    
    def test_missing_config_uses_defaults(self):
        """Test that missing config file uses hardcoded defaults."""
        av_stack = AVStack(
            bridge_url="http://localhost:8000",
            record_data=False,
            config_path='/nonexistent/config.yaml'
        )
        
        # Should still initialize with defaults
        assert av_stack is not None
        assert av_stack.controller is not None
        
        # Check some default values
        lateral_ctrl = av_stack.controller.lateral_controller
        assert lateral_ctrl.pid.kp == 0.3  # Default
        assert lateral_ctrl.max_steering == 0.5  # Default
    
    def test_partial_config_uses_defaults_for_missing(self):
        """Test that partial config uses defaults for missing parameters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Only specify some parameters
            config_data = {
                'control': {
                    'lateral': {
                        'kp': 0.5  # Only specify kp
                    }
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            av_stack = AVStack(
                bridge_url="http://localhost:8000",
                record_data=False,
                config_path=config_path
            )
            
            # kp should be from config
            assert av_stack.controller.lateral_controller.pid.kp == 0.5
            
            # Other params should use defaults
            assert av_stack.controller.lateral_controller.pid.ki == 0.0  # Default
            assert av_stack.controller.lateral_controller.max_steering == 0.5  # Default
        finally:
            Path(config_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

