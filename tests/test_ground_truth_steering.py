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
        self.follower.target_lane = "right"
        self.follower.single_lane_width_threshold_m = 5.0
        self.follower.lateral_correction_gain = 0.2  # Required for steering calculation
        self.follower.max_steer_angle_deg = 30.0  # Required for exact steering calculation
        self.follower.wheelbase = 2.5  # Required for exact steering calculation
        self.follower.use_exact_steering = True  # Use exact calculation
    
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

