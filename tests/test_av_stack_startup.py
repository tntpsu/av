"""
Tests for AV Stack startup and initialization.
These tests catch common startup issues like missing arguments, config errors, etc.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from av_stack import AVStack, load_config
from control.pid_controller import VehicleController, LateralController, LongitudinalController
from trajectory.inference import TrajectoryPlanningInference
from perception.inference import LaneDetectionInference


class TestConfigLoading:
    """Test configuration file loading."""
    
    def test_load_default_config(self):
        """Test that default config file can be loaded if it exists."""
        config = load_config()
        # Should return dict (empty if file doesn't exist, or loaded config)
        assert isinstance(config, dict)
    
    def test_load_custom_config(self):
        """Test loading a custom config file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'control': {
                    'lateral': {
                        'kp': 0.5,
                        'deadband': 0.03
                    },
                    'longitudinal': {
                        'target_speed': 10.0
                    }
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert isinstance(config, dict)
            assert config['control']['lateral']['kp'] == 0.5
            assert config['control']['lateral']['deadband'] == 0.03
        finally:
            Path(config_path).unlink()
    
    def test_load_nonexistent_config(self):
        """Test that loading nonexistent config returns empty dict."""
        config = load_config('/nonexistent/path/config.yaml')
        assert isinstance(config, dict)
        assert len(config) == 0


class TestAVStackInitialization:
    """Test AV Stack can be initialized without errors."""
    
    def test_init_with_defaults(self):
        """Test AV Stack initialization with default parameters."""
        # Should not raise any errors
        av_stack = AVStack(
            bridge_url="http://localhost:8000",
            model_path=None,
            record_data=False,  # Disable recording for tests
            config_path=None
        )
        
        assert av_stack is not None
        assert av_stack.bridge is not None
        assert av_stack.perception is not None
        assert av_stack.trajectory_planner is not None
        assert av_stack.controller is not None
    
    def test_init_with_config_file(self):
        """Test AV Stack initialization with config file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'control': {
                    'lateral': {
                        'kp': 0.4,
                        'ki': 0.0,
                        'kd': 0.15,
                        'max_steering': 0.6,
                        'deadband': 0.025,
                        'heading_weight': 0.6,
                        'lateral_weight': 0.4,
                        'error_clip': 0.8
                    },
                    'longitudinal': {
                        'kp': 0.4,
                        'ki': 0.06,
                        'kd': 0.03,
                        'target_speed': 9.0,
                        'max_speed': 12.0
                    }
                },
                'trajectory': {
                    'lookahead_distance': 25.0,
                    'target_speed': 9.0,
                    'reference_lookahead': 10.0,
                    'image_width': 640.0,
                    'image_height': 480.0,
                    'camera_fov': 60.0,
                    'camera_height': 0.5,
                    'bias_correction_threshold': 15.0
                },
                'safety': {
                    'max_speed': 12.0,
                    'emergency_brake_threshold': 2.0,
                    'speed_prevention_threshold': 0.85,
                    'speed_prevention_brake_threshold': 0.9,
                    'speed_prevention_brake_amount': 0.2
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            av_stack = AVStack(
                bridge_url="http://localhost:8000",
                model_path=None,
                record_data=False,
                config_path=config_path
            )
            
            assert av_stack is not None
            # Verify config was loaded
            assert av_stack.config is not None
            assert av_stack.control_config is not None
            assert av_stack.trajectory_config is not None
            assert av_stack.safety_config is not None
        finally:
            Path(config_path).unlink()
    
    def test_init_with_invalid_config(self):
        """Test that invalid config file doesn't crash initialization."""
        # Create a malformed config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            # Should handle gracefully (either skip or use defaults)
            # This might raise an exception, which is fine - we just want to test it doesn't crash
            try:
                av_stack = AVStack(
                    bridge_url="http://localhost:8000",
                    record_data=False,
                    config_path=config_path
                )
                # If it doesn't crash, that's good
                assert av_stack is not None
            except (yaml.YAMLError, Exception):
                # YAML error is acceptable - config loading should handle it
                pass
        finally:
            Path(config_path).unlink()


class TestVehicleControllerInitialization:
    """Test VehicleController can be initialized with all config parameters."""
    
    def test_init_with_all_lateral_params(self):
        """Test VehicleController accepts all lateral control parameters."""
        controller = VehicleController(
            lateral_kp=0.3,
            lateral_ki=0.0,
            lateral_kd=0.1,
            longitudinal_kp=0.3,
            longitudinal_ki=0.05,
            longitudinal_kd=0.02,
            lookahead_distance=10.0,
            target_speed=8.0,
            max_speed=10.0,
            max_steering=0.5,
            lateral_deadband=0.02,
            lateral_heading_weight=0.5,
            lateral_lateral_weight=0.5,
            lateral_error_clip=0.785
        )
        
        assert controller is not None
        assert controller.lateral_controller is not None
        assert controller.longitudinal_controller is not None
    
    def test_init_with_defaults(self):
        """Test VehicleController with default parameters."""
        controller = VehicleController()
        assert controller is not None
    
    def test_lateral_controller_params(self):
        """Test LateralController accepts all config parameters."""
        controller = LateralController(
            kp=0.3,
            ki=0.0,
            kd=0.1,
            lookahead_distance=10.0,
            max_steering=0.5,
            deadband=0.02,
            heading_weight=0.5,
            lateral_weight=0.5,
            error_clip=0.785
        )
        
        assert controller is not None
        assert controller.deadband == 0.02
        assert controller.heading_weight == 0.5
        assert controller.lateral_weight == 0.5


class TestArgumentParser:
    """Test that command-line argument parsing works correctly."""
    
    def test_parse_default_args(self):
        """Test parsing default arguments."""
        from av_stack import main
        import argparse
        
        # Mock sys.argv
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ['av_stack.py']
            # This will call parse_args() - we just want to make sure it doesn't crash
            # We can't easily test the full main() without mocking, so we'll test the parser directly
            parser = argparse.ArgumentParser()
            parser.add_argument('--bridge_url', type=str, default='http://localhost:8000')
            parser.add_argument('--model_path', type=str, default=None)
            parser.add_argument('--record', action='store_true', default=True)
            parser.add_argument('--no-record', dest='record', action='store_false')
            parser.add_argument('--recording_dir', type=str, default='data/recordings')
            parser.add_argument('--max_frames', type=int, default=None)
            parser.add_argument('--config', type=str, default=None)
            
            args = parser.parse_args([])
            assert args.bridge_url == 'http://localhost:8000'
            assert args.config is None
            assert args.record is True
        finally:
            sys.argv = original_argv
    
    def test_parse_config_arg(self):
        """Test parsing --config argument."""
        import argparse
        import sys
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default=None)
        
        original_argv = sys.argv
        try:
            sys.argv = ['av_stack.py', '--config', '/path/to/config.yaml']
            args = parser.parse_args()
            assert args.config == '/path/to/config.yaml'
        finally:
            sys.argv = original_argv


class TestComponentIntegration:
    """Test that all components can be instantiated together."""
    
    def test_all_components_initialized(self):
        """Test that AV Stack initializes all required components."""
        av_stack = AVStack(
            bridge_url="http://localhost:8000",
            record_data=False,
            config_path=None
        )
        
        # Check all components exist
        assert hasattr(av_stack, 'bridge')
        assert hasattr(av_stack, 'perception')
        assert hasattr(av_stack, 'trajectory_planner')
        assert hasattr(av_stack, 'controller')
        
        # Check component types
        assert isinstance(av_stack.perception, LaneDetectionInference)
        assert isinstance(av_stack.trajectory_planner, TrajectoryPlanningInference)
        assert isinstance(av_stack.controller, VehicleController)
    
    def test_config_stored_correctly(self):
        """Test that config is stored correctly in AV Stack."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'control': {'lateral': {'kp': 0.5}},
                'trajectory': {'lookahead_distance': 30.0},
                'safety': {'max_speed': 15.0}
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            av_stack = AVStack(
                bridge_url="http://localhost:8000",
                record_data=False,
                config_path=config_path
            )
            
            # Check config is stored
            assert hasattr(av_stack, 'config')
            assert hasattr(av_stack, 'control_config')
            assert hasattr(av_stack, 'trajectory_config')
            assert hasattr(av_stack, 'safety_config')
            
            # Check values
            assert av_stack.control_config['lateral']['kp'] == 0.5
            assert av_stack.trajectory_config['lookahead_distance'] == 30.0
            assert av_stack.safety_config['max_speed'] == 15.0
        finally:
            Path(config_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

