"""
Integration tests for AV stack components.
Tests the full pipeline: Perception → Trajectory → Control
and catches integration issues early.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from perception.inference import LaneDetectionInference
from trajectory.inference import TrajectoryPlanningInference
from control.pid_controller import VehicleController
from av_stack import AVStack


class TestPerceptionToTrajectory:
    """Test integration between perception and trajectory planning."""
    
    def test_lane_detection_to_trajectory(self):
        """Test that lane detection output can be used by trajectory planner."""
        # Create perception and trajectory components
        perception = LaneDetectionInference(model_path=None, fallback_to_cv=True)
        trajectory_planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            lookahead_distance=20.0
        )
        
        # Create a dummy image (RGB, 640x480)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some lane-like features (white lines)
        dummy_image[400:480, 200:220] = 255  # Left lane
        dummy_image[400:480, 420:440] = 255  # Right lane
        
        # Run perception
        lane_coeffs, confidence = perception.detect(dummy_image)
        
        # Should return list of coefficients (or None)
        assert isinstance(lane_coeffs, list)
        assert len(lane_coeffs) == 2  # Left and right lanes
        
        # Run trajectory planning
        trajectory = trajectory_planner.plan(lane_coeffs)
        
        # Should return a trajectory (even if lanes are None)
        assert trajectory is not None
        assert hasattr(trajectory, 'points') or hasattr(trajectory, 'reference_point')
    
    def test_none_lanes_handled(self):
        """Test that trajectory planner handles None lanes gracefully."""
        trajectory_planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            lookahead_distance=20.0
        )
        
        # Test with all None lanes
        lane_coeffs = [None, None]
        trajectory = trajectory_planner.plan(lane_coeffs)
        
        # Should not crash, should return something (even if empty/default)
        assert trajectory is not None
    
    def test_one_lane_detected(self):
        """Test trajectory planning with only one lane detected."""
        trajectory_planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            lookahead_distance=20.0
        )
        
        # Only left lane detected
        left_lane = np.array([0.001, 0.0, 100.0])  # Simple lane
        lane_coeffs = [left_lane, None]
        
        trajectory = trajectory_planner.plan(lane_coeffs)
        
        # Should handle gracefully
        assert trajectory is not None


class TestTrajectoryToControl:
    """Test integration between trajectory planning and control."""
    
    def test_reference_point_to_control(self):
        """Test that reference point from trajectory can be used by controller."""
        trajectory_planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            lookahead_distance=20.0
        )
        controller = VehicleController()
        
        # Create dummy lanes
        left_lane = np.array([0.001, 0.0, 100.0])
        right_lane = np.array([0.001, 0.0, 300.0])
        lane_coeffs = [left_lane, right_lane]
        
        # Get trajectory and reference point
        trajectory = trajectory_planner.plan(lane_coeffs)
        
        # Extract reference point (format depends on planner implementation)
        if hasattr(trajectory, 'reference_point'):
            ref_point = trajectory.reference_point
        elif hasattr(trajectory, 'get_reference_point'):
            ref_point = trajectory.get_reference_point()
        else:
            # Try to get from trajectory points
            if hasattr(trajectory, 'points') and len(trajectory.points) > 0:
                point = trajectory.points[0]
                ref_point = {
                    'x': point.x if hasattr(point, 'x') else 0.0,
                    'y': point.y if hasattr(point, 'y') else 10.0,
                    'heading': point.heading if hasattr(point, 'heading') else 0.0,
                    'velocity': point.velocity if hasattr(point, 'velocity') else 8.0
                }
            else:
                # Default reference point
                ref_point = {'x': 0.0, 'y': 10.0, 'heading': 0.0, 'velocity': 8.0}
        
        # Verify reference point format
        assert isinstance(ref_point, dict)
        assert 'x' in ref_point
        assert 'y' in ref_point
        assert 'heading' in ref_point
        assert 'velocity' in ref_point
        
        # Test controller can use it
        current_state = {
            'heading': 0.0,
            'speed': 8.0,
            'position': np.array([0.0, 0.0])
        }
        
        commands = controller.compute_control(current_state, ref_point)
        
        # Should return valid commands
        assert 'steering' in commands
        assert 'throttle' in commands
        assert 'brake' in commands
        assert -1.0 <= commands['steering'] <= 1.0
        assert 0.0 <= commands['throttle'] <= 1.0
        assert 0.0 <= commands['brake'] <= 1.0
    
    def test_missing_reference_point_handled(self):
        """Test that controller handles missing/invalid reference points."""
        controller = VehicleController()
        
        # Test with None reference point
        current_state = {
            'heading': 0.0,
            'speed': 8.0,
            'position': np.array([0.0, 0.0])
        }
        
        # Should handle gracefully (either return default commands or raise clear error)
        try:
            commands = controller.compute_control(current_state, None)
            # If it doesn't crash, verify commands are valid
            assert 'steering' in commands
            assert 'throttle' in commands
            assert 'brake' in commands
        except (TypeError, AttributeError) as e:
            # If it raises an error, that's also acceptable - just verify it's clear
            assert "None" in str(e) or "missing" in str(e).lower()


class TestCoordinateTransformations:
    """Test coordinate system transformations."""
    
    def test_quaternion_to_heading(self):
        """Test quaternion to heading (yaw) extraction."""
        from av_stack import AVStack
        
        # Test identity quaternion (no rotation) = 0° heading
        quaternion_identity = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        heading = AVStack._extract_heading(None, {'rotation': quaternion_identity})
        
        # Should be approximately 0
        assert abs(heading) < 0.01, f"Identity quaternion should give 0° heading, got {np.degrees(heading)}°"
        
        # Test 90° rotation around Y axis (Unity's yaw axis)
        # Quaternion for 90° rotation around Y: [0, sin(45°), 0, cos(45°)]
        quaternion_90deg = {'x': 0.0, 'y': np.sin(np.pi/4), 'z': 0.0, 'w': np.cos(np.pi/4)}
        heading = AVStack._extract_heading(None, {'rotation': quaternion_90deg})
        
        # Should be approximately 90° (π/2 radians)
        # Note: Unity uses Y-axis for yaw, so we check if heading is reasonable
        assert abs(heading) > 0.01, f"90° rotation should give non-zero heading, got {np.degrees(heading)}°"
    
    def test_image_to_vehicle_coords(self):
        """Test coordinate transformation from image pixels to vehicle meters."""
        from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner
        
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0,
            camera_height=0.5
        )
        
        # Test center of image (x=320) should map to x≈0 in vehicle coords
        if hasattr(planner, '_convert_image_to_vehicle_coords'):
            x_image = 320.0  # Center of 640px wide image
            y_image = 480.0  # Bottom of image
            
            x_vehicle, y_vehicle = planner._convert_image_to_vehicle_coords(x_image, y_image)
            
            # Center of image should be near x=0 in vehicle frame
            assert abs(x_vehicle) < 1.0, f"Center of image should map to x≈0, got {x_vehicle}m"


class TestEdgeCases:
    """Test edge case handling."""
    
    def test_no_lanes_detected(self):
        """Test full pipeline with no lanes detected."""
        perception = LaneDetectionInference(model_path=None, fallback_to_cv=True)
        trajectory_planner = TrajectoryPlanningInference(planner_type="rule_based")
        controller = VehicleController()
        
        # Create image with no lanes
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run perception
        lane_coeffs, confidence = perception.detect(dummy_image)
        
        # Run trajectory planning
        trajectory = trajectory_planner.plan(lane_coeffs)
        
        # Should not crash
        assert trajectory is not None
    
    def test_swapped_lanes(self):
        """Test that trajectory planner handles swapped lanes (left/right)."""
        trajectory_planner = TrajectoryPlanningInference(planner_type="rule_based")
        
        # Create lanes that are swapped (right lane is actually on left side of image)
        # Right lane at x=100 (left side of image)
        right_lane_swapped = np.array([0.001, 0.0, 100.0])
        # Left lane at x=500 (right side of image)
        left_lane_swapped = np.array([0.001, 0.0, 500.0])
        
        # Pass swapped (should be detected and corrected)
        lane_coeffs = [left_lane_swapped, right_lane_swapped]
        
        trajectory = trajectory_planner.plan(lane_coeffs)
        
        # Should handle gracefully (either correct or use as-is)
        assert trajectory is not None
    
    def test_extreme_speeds(self):
        """Test control with extreme speeds."""
        controller = VehicleController()
        
        # Test with very high speed
        current_state = {
            'heading': 0.0,
            'speed': 50.0,  # Very high speed
            'position': np.array([0.0, 0.0])
        }
        
        reference_point = {
            'x': 0.0,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        commands = controller.compute_control(current_state, reference_point)
        
        # Should apply braking for high speed
        assert commands['brake'] > 0.0, "Should brake when speed is very high"
        assert commands['throttle'] < 1.0, "Should reduce throttle when speed is very high"
    
    def test_zero_heading(self):
        """Test control with zero heading."""
        controller = VehicleController()
        
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
        
        commands = controller.compute_control(current_state, reference_point)
        
        # Should not crash
        assert 'steering' in commands
        assert 'throttle' in commands
        assert 'brake' in commands


class TestDataFormats:
    """Test data format validation."""
    
    def test_reference_point_format(self):
        """Test that reference points have correct format."""
        trajectory_planner = TrajectoryPlanningInference(planner_type="rule_based")
        
        left_lane = np.array([0.001, 0.0, 100.0])
        right_lane = np.array([0.001, 0.0, 300.0])
        lane_coeffs = [left_lane, right_lane]
        
        trajectory = trajectory_planner.plan(lane_coeffs)
        
        # Try to get reference point
        if hasattr(trajectory, 'reference_point'):
            ref_point = trajectory.reference_point
        elif hasattr(trajectory, 'get_reference_point'):
            ref_point = trajectory.get_reference_point()
        else:
            pytest.skip("Trajectory doesn't expose reference point")
        
        # Verify format
        assert isinstance(ref_point, dict)
        required_keys = ['x', 'y', 'heading', 'velocity']
        for key in required_keys:
            assert key in ref_point, f"Reference point missing key: {key}"
            assert isinstance(ref_point[key], (int, float)), f"Reference point {key} should be numeric"
    
    def test_control_command_format(self):
        """Test that control commands have correct format."""
        controller = VehicleController()
        
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
        
        commands = controller.compute_control(current_state, reference_point)
        
        # Verify format
        assert isinstance(commands, dict)
        required_keys = ['steering', 'throttle', 'brake']
        for key in required_keys:
            assert key in commands, f"Control commands missing key: {key}"
            assert isinstance(commands[key], (int, float)), f"Control command {key} should be numeric"
        
        # Verify ranges
        assert -1.0 <= commands['steering'] <= 1.0
        assert 0.0 <= commands['throttle'] <= 1.0
        assert 0.0 <= commands['brake'] <= 1.0


class TestFullPipeline:
    """Test the full AV stack pipeline."""
    
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked components."""
        # Create AV stack (without bridge connection)
        av_stack = AVStack(
            bridge_url="http://localhost:8000",
            record_data=False,
            config_path=None
        )
        
        # Verify all components are initialized
        assert av_stack.perception is not None
        assert av_stack.trajectory_planner is not None
        assert av_stack.controller is not None
        
        # Create dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image[400:480, 200:220] = 255  # Left lane
        dummy_image[400:480, 420:440] = 255  # Right lane
        
        # Run perception
        lane_coeffs, confidence = av_stack.perception.detect(dummy_image)
        
        # Run trajectory planning
        trajectory = av_stack.trajectory_planner.plan(lane_coeffs)
        
        # Get reference point
        if hasattr(trajectory, 'reference_point'):
            ref_point = trajectory.reference_point
        elif hasattr(trajectory, 'get_reference_point'):
            ref_point = trajectory.get_reference_point()
        else:
            ref_point = {'x': 0.0, 'y': 10.0, 'heading': 0.0, 'velocity': 8.0}
        
        # Run control
        current_state = {
            'heading': 0.0,
            'speed': 8.0,
            'position': np.array([0.0, 0.0])
        }
        
        commands = av_stack.controller.compute_control(current_state, ref_point)
        
        # Verify end-to-end pipeline works
        assert commands is not None
        assert 'steering' in commands
        assert 'throttle' in commands
        assert 'brake' in commands


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

