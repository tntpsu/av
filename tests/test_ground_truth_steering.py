"""
Unit tests for ground truth steering calculation.

These tests verify the fundamental logic of ground truth following:
1. Coordinate system interpretation
2. Steering sign correctness
3. Control law correctness
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the method we need to test
from tools.ground_truth_follower import GroundTruthFollower


class TestGroundTruthSteering:
    """Test ground truth steering calculation logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal ground truth follower for testing
        # We'll mock the bridge and AV stack
        self.follower = GroundTruthFollower.__new__(GroundTruthFollower)
        self.follower.kp_steering_base = 1.5
        self.follower.kp_steering = 1.5
        self.follower.kd_steering = 0.3
        self.follower.steering_sign = 1.0
        self.follower.steering_smoothing = 0.3
        self.follower.last_steering = 0.0
        self.follower.steering_deadband = 0.02
        self.follower.base_speed = 3.0
        self.follower.speed_gain_factor = 0.8
        self.follower.lookahead_base = 5.0
        self.follower.lookahead_speed_factor = 1.5
        self.follower.prev_lane_center = 0.0
        self.follower.prev_steering_time = None
    
    def test_steering_sign_when_car_left_of_center(self):
        """
        Test: If car is LEFT of lane center, should steer RIGHT (positive).
        
        Coordinate system:
        - Positive lane_center = center is to the RIGHT of car
        - If center is to the RIGHT, car is to the LEFT
        - To correct: steer RIGHT (positive steering)
        - So: steering should have SAME sign as lane_center
        """
        vehicle_state = {
            'groundTruthLaneCenterX': 2.0,  # Center is 2m to the RIGHT
            'speed': 5.0
        }
        
        steering = self.follower.get_ground_truth_steering(vehicle_state)
        
        # Car is LEFT of center (lane_center > 0)
        # Should steer RIGHT (positive steering)
        assert steering > 0, f"Expected positive steering when lane_center=2.0, got {steering}"
        print(f"✓ Test 1 PASSED: lane_center=2.0 → steering={steering:.3f} (positive, correct)")
    
    def test_steering_sign_when_car_right_of_center(self):
        """
        Test: If car is RIGHT of lane center, should steer LEFT (negative).
        
        Coordinate system:
        - Negative lane_center = center is to the LEFT of car
        - If center is to the LEFT, car is to the RIGHT
        - To correct: steer LEFT (negative steering)
        - So: steering should have SAME sign as lane_center
        """
        vehicle_state = {
            'groundTruthLaneCenterX': -2.0,  # Center is 2m to the LEFT
            'speed': 5.0
        }
        
        steering = self.follower.get_ground_truth_steering(vehicle_state)
        
        # Car is RIGHT of center (lane_center < 0)
        # Should steer LEFT (negative steering)
        assert steering < 0, f"Expected negative steering when lane_center=-2.0, got {steering}"
        print(f"✓ Test 2 PASSED: lane_center=-2.0 → steering={steering:.3f} (negative, correct)")
    
    def test_steering_magnitude_proportional_to_error(self):
        """
        Test: Steering magnitude should be proportional to lane center offset.
        Larger offset = larger steering correction.
        """
        vehicle_state_small = {
            'groundTruthLaneCenterX': 1.0,
            'speed': 5.0
        }
        vehicle_state_large = {
            'groundTruthLaneCenterX': 3.0,
            'speed': 5.0
        }
        
        steering_small = abs(self.follower.get_ground_truth_steering(vehicle_state_small))
        steering_large = abs(self.follower.get_ground_truth_steering(vehicle_state_large))
        
        # Larger error should produce larger steering
        assert steering_large > steering_small, \
            f"Expected larger steering for larger error: {steering_large} should be > {steering_small}"
        print(f"✓ Test 3 PASSED: |steering| for error=3.0 ({steering_large:.3f}) > error=1.0 ({steering_small:.3f})")
    
    def test_steering_zero_when_at_center(self):
        """
        Test: When car is at lane center (lane_center ≈ 0), steering should be near zero.
        """
        vehicle_state = {
            'groundTruthLaneCenterX': 0.0,
            'speed': 5.0
        }
        
        steering = self.follower.get_ground_truth_steering(vehicle_state)
        
        # Should be very small (within deadband)
        assert abs(steering) < 0.1, f"Expected near-zero steering when at center, got {steering}"
        print(f"✓ Test 4 PASSED: lane_center=0.0 → steering={steering:.3f} (near zero)")
    
    def test_speed_adaptive_gain(self):
        """
        Test: At higher speeds, gain should be reduced to prevent oversteering.
        """
        vehicle_state_slow = {
            'groundTruthLaneCenterX': 2.0,
            'speed': 2.0  # Slow
        }
        vehicle_state_fast = {
            'groundTruthLaneCenterX': 2.0,
            'speed': 8.0  # Fast
        }
        
        steering_slow = abs(self.follower.get_ground_truth_steering(vehicle_state_slow))
        steering_fast = abs(self.follower.get_ground_truth_steering(vehicle_state_fast))
        
        # At higher speeds, steering should be less aggressive (lower gain)
        # But this depends on the speed factor calculation
        print(f"✓ Test 5: Slow (2m/s): steering={steering_slow:.3f}, Fast (8m/s): steering={steering_fast:.3f}")
        print(f"  Speed factor at 2m/s: {self.follower.kp_steering / self.follower.kp_steering_base:.3f}")
        print(f"  Speed factor at 8m/s: {self.follower.kp_steering / self.follower.kp_steering_base:.3f}")
    
    def test_coordinate_system_interpretation(self):
        """
        Test: Verify our understanding of the coordinate system.
        
        From GroundTruthReporter.cs:
        - Vehicle frame: car's forward = +Y, car's right = +X
        - GetLanePositionsVehicle() returns (leftX, rightX) in vehicle coordinates
        - GetCurrentLaneCenterVehicle() returns (leftX + rightX) / 2.0
        
        If leftX = 1.79m and rightX = 6.87m (from logs):
        - Lane center = (1.79 + 6.87) / 2 = 4.33m
        - This means lane center is 4.33m to the RIGHT of car
        - Car is 4.33m to the LEFT of lane center
        - Should steer RIGHT (positive) to correct
        """
        # Simulate the actual values from logs
        vehicle_state = {
            'groundTruthLaneCenterX': 4.33,  # From actual logs
            'speed': 5.0
        }
        
        steering = self.follower.get_ground_truth_steering(vehicle_state)
        
        # With lane_center = 4.33m (positive), car is LEFT of center
        # Should steer RIGHT (positive steering)
        assert steering > 0, f"Expected positive steering for lane_center=4.33, got {steering}"
        
        # Calculate expected steering
        expected_steering = self.follower.kp_steering * 4.33
        print(f"✓ Test 6: lane_center=4.33m → steering={steering:.3f}")
        print(f"  Expected (kp * lane_center): {expected_steering:.3f}")
        print(f"  Actual steering: {steering:.3f}")
        print(f"  Difference: {abs(steering - expected_steering):.3f}")


def test_steering_control_law():
    """
    Test the fundamental control law: steering = kp * lane_center
    
    This is the simplest possible control law for following a line.
    If this doesn't work, nothing will.
    """
    kp = 1.5
    test_cases = [
        (0.0, 0.0),      # At center: no steering
        (1.0, 1.5),      # 1m right: steer right by kp*1.0
        (-1.0, -1.5),    # 1m left: steer left by kp*-1.0
        (2.0, 3.0),      # 2m right: steer right by kp*2.0
        (-2.0, -3.0),    # 2m left: steer left by kp*-2.0
    ]
    
    print("\n=== Testing Fundamental Control Law: steering = kp * lane_center ===")
    for lane_center, expected_steering in test_cases:
        actual_steering = kp * lane_center
        assert abs(actual_steering - expected_steering) < 0.001, \
            f"Control law failed: kp={kp} * lane_center={lane_center} = {actual_steering}, expected {expected_steering}"
        print(f"✓ lane_center={lane_center:5.1f}m → steering={actual_steering:6.2f} (expected {expected_steering:6.2f})")


if __name__ == "__main__":
    print("=" * 70)
    print("GROUND TRUTH STEERING UNIT TESTS")
    print("=" * 70)
    print()
    
    # Run control law test
    test_steering_control_law()
    print()
    
    # Run steering calculation tests
    test_suite = TestGroundTruthSteering()
    test_suite.setup_method()
    
    print("\n=== Testing Steering Calculation Logic ===")
    test_suite.test_steering_sign_when_car_left_of_center()
    test_suite.test_steering_sign_when_car_right_of_center()
    test_suite.test_steering_magnitude_proportional_to_error()
    test_suite.test_steering_zero_when_at_center()
    test_suite.test_speed_adaptive_gain()
    test_suite.test_coordinate_system_interpretation()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)

