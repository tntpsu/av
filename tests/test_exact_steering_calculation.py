"""
Comprehensive tests for exact steering calculation.

These tests can run BEFORE Unity to verify:
1. Steering calculation is mathematically correct
2. Coordinate system interpretation is correct
3. Path curvature to steering conversion is correct
4. Lateral correction works correctly

These tests can be tweaked based on recorded data from Unity runs.
"""

import numpy as np
import math
from typing import Tuple


class ExactSteeringCalculator:
    """
    Exact steering calculation (matches ground_truth_follower.py).
    
    This is a pure Python implementation that can be tested without Unity.
    """
    
    def __init__(self, 
                 wheelbase: float = 2.5,
                 max_steer_angle_deg: float = 30.0,
                 lateral_correction_gain: float = 0.2,
                 steering_deadband: float = 0.02):
        """
        Initialize calculator.
        
        Args:
            wheelbase: Car wheelbase (meters)
            max_steer_angle_deg: Maximum steering angle (degrees)
            lateral_correction_gain: Gain for lateral error correction
            steering_deadband: Deadband for steering (meters)
        """
        self.wheelbase = wheelbase
        self.max_steer_angle_deg = max_steer_angle_deg
        self.max_steer_angle_rad = np.deg2rad(max_steer_angle_deg)
        self.lateral_correction_gain = lateral_correction_gain
        self.steering_deadband = steering_deadband
    
    def calculate_steering(self, 
                          path_curvature: float,
                          lane_center: float) -> Tuple[float, dict]:
        """
        Calculate exact steering from path curvature and lateral error.
        
        Args:
            path_curvature: Path curvature (1/meters)
            lane_center: Lateral error (meters, positive = center to right)
            
        Returns:
            (steering, metadata) where:
            - steering: Steering command (-1.0 to 1.0)
            - metadata: Dict with calculation breakdown
        """
        metadata = {
            'path_curvature': path_curvature,
            'lane_center': lane_center,
            'raw_steering': 0.0,
            'lateral_correction': 0.0,
            'calculated_steering_angle_deg': 0.0
        }
        
        # EXACT STEERING FROM PATH CURVATURE (bicycle model)
        if abs(path_curvature) > 1e-6:
            # Calculate exact steering angle from curvature
            # Bicycle model: steering_angle = atan(wheelbase * curvature)
            steering_angle_rad = np.arctan(self.wheelbase * path_curvature)
            
            # Convert to Unity steering input [-1, 1]
            # Unity: steerInput * maxSteerAngle = actual steering angle
            # So: steerInput = steering_angle_rad / maxSteerAngle_rad
            exact_steering = steering_angle_rad / self.max_steer_angle_rad
            
            # Add small lateral correction to stay centered in lane
            lateral_correction = self.lateral_correction_gain * lane_center
            exact_steering = exact_steering + lateral_correction
            
            # Clip to valid range
            exact_steering = np.clip(exact_steering, -1.0, 1.0)
            
            metadata['raw_steering'] = exact_steering - lateral_correction
            metadata['lateral_correction'] = lateral_correction
            metadata['calculated_steering_angle_deg'] = exact_steering * self.max_steer_angle_deg
            
            return exact_steering, metadata
        else:
            # Straight path (curvature = 0) - only need lateral correction
            if abs(lane_center) > self.steering_deadband:
                exact_steering = self.lateral_correction_gain * lane_center
                exact_steering = np.clip(exact_steering, -1.0, 1.0)
                
                metadata['raw_steering'] = 0.0
                metadata['lateral_correction'] = exact_steering
                metadata['calculated_steering_angle_deg'] = exact_steering * self.max_steer_angle_deg
                
                return exact_steering, metadata
            else:
                metadata['raw_steering'] = 0.0
                metadata['lateral_correction'] = 0.0
                metadata['calculated_steering_angle_deg'] = 0.0
                return 0.0, metadata


def test_straight_path_at_center():
    """Test: Straight path, car at center -> no steering."""
    calc = ExactSteeringCalculator()
    steering, metadata = calc.calculate_steering(0.0, 0.0)
    assert steering == 0.0, f"Expected 0.0, got {steering}"
    print("✓ PASS: Straight path at center -> no steering")


def test_straight_path_left_of_center():
    """Test: Straight path, car left of center -> steer right (positive)."""
    calc = ExactSteeringCalculator()
    steering, metadata = calc.calculate_steering(0.0, 2.0)  # Positive = car left
    assert steering > 0.0, f"Expected positive steering, got {steering}"
    expected = 0.2 * 2.0  # lateral_correction_gain * lane_center
    assert abs(steering - expected) < 0.01, f"Expected ~{expected}, got {steering}"
    print(f"✓ PASS: Car left of center -> steer right (steering={steering:.3f})")


def test_straight_path_right_of_center():
    """Test: Straight path, car right of center -> steer left (negative)."""
    calc = ExactSteeringCalculator()
    steering, metadata = calc.calculate_steering(0.0, -2.0)  # Negative = car right
    assert steering < 0.0, f"Expected negative steering, got {steering}"
    expected = 0.2 * -2.0
    assert abs(steering - expected) < 0.01, f"Expected ~{expected}, got {steering}"
    print(f"✓ PASS: Car right of center -> steer left (steering={steering:.3f})")


def test_curved_path_right_turn():
    """Test: Right turn (positive curvature) -> positive steering."""
    calc = ExactSteeringCalculator()
    curvature = 0.02  # 1/meters (50m radius)
    steering, metadata = calc.calculate_steering(curvature, 0.0)
    
    # Verify calculation: steering_angle = atan(wheelbase * curvature)
    expected_angle_rad = np.arctan(calc.wheelbase * curvature)
    expected_steering = expected_angle_rad / calc.max_steer_angle_rad
    
    assert steering > 0.0, f"Expected positive steering for right turn, got {steering}"
    assert abs(steering - expected_steering) < 0.01, \
        f"Expected {expected_steering:.3f}, got {steering:.3f}"
    print(f"✓ PASS: Right turn -> positive steering (steering={steering:.3f}, expected={expected_steering:.3f})")


def test_curved_path_left_turn():
    """Test: Left turn (negative curvature) -> negative steering."""
    calc = ExactSteeringCalculator()
    curvature = -0.02  # Left turn
    steering, metadata = calc.calculate_steering(curvature, 0.0)
    
    expected_angle_rad = np.arctan(calc.wheelbase * curvature)
    expected_steering = expected_angle_rad / calc.max_steer_angle_rad
    
    assert steering < 0.0, f"Expected negative steering for left turn, got {steering}"
    assert abs(steering - expected_steering) < 0.01, \
        f"Expected {expected_steering:.3f}, got {steering:.3f}"
    print(f"✓ PASS: Left turn -> negative steering (steering={steering:.3f}, expected={expected_steering:.3f})")


def test_curved_path_with_lateral_error():
    """Test: Curved path + lateral error -> combines both."""
    calc = ExactSteeringCalculator()
    curvature = 0.02  # Right turn
    lane_center = 1.0  # Car left of center
    
    steering, metadata = calc.calculate_steering(curvature, lane_center)
    
    # Should have both curvature steering and lateral correction
    expected_curvature_steering = np.arctan(calc.wheelbase * curvature) / calc.max_steer_angle_rad
    expected_lateral_correction = calc.lateral_correction_gain * lane_center
    expected_steering = expected_curvature_steering + expected_lateral_correction
    
    assert abs(steering - expected_steering) < 0.01, \
        f"Expected {expected_steering:.3f}, got {steering:.3f}"
    print(f"✓ PASS: Curved path + lateral error combines correctly (steering={steering:.3f})")


def test_steering_clipping():
    """Test: Steering is clipped to [-1.0, 1.0] range."""
    calc = ExactSteeringCalculator()
    curvature = 1.0  # Very sharp turn
    steering, metadata = calc.calculate_steering(curvature, 0.0)
    
    assert -1.0 <= steering <= 1.0, f"Steering {steering} not in [-1, 1] range"
    print(f"✓ PASS: Steering clipped correctly (steering={steering:.3f})")


def test_coordinate_system():
    """Test: Verify coordinate system interpretation."""
    calc = ExactSteeringCalculator()
    
    # Car LEFT of center (lane_center > 0) -> should steer RIGHT (positive)
    steering_left, _ = calc.calculate_steering(0.0, 2.0)
    assert steering_left > 0.0, "Car left of center should steer right (positive)"
    
    # Car RIGHT of center (lane_center < 0) -> should steer LEFT (negative)
    steering_right, _ = calc.calculate_steering(0.0, -2.0)
    assert steering_right < 0.0, "Car right of center should steer left (negative)"
    
    print("✓ PASS: Coordinate system interpretation correct")


def test_mathematical_correctness():
    """Test: Verify mathematical correctness of calculation."""
    calc = ExactSteeringCalculator()
    
    # Test case 1: Straight, centered
    steering, _ = calc.calculate_steering(0.0, 0.0)
    assert abs(steering) < 0.01, f"Expected 0.0, got {steering}"
    
    # Test case 2: Straight, 2m left -> 0.2 * 2.0 = 0.4
    steering, _ = calc.calculate_steering(0.0, 2.0)
    assert abs(steering - 0.4) < 0.01, f"Expected 0.4, got {steering}"
    
    # Test case 3: Right turn, centered
    steering, _ = calc.calculate_steering(0.02, 0.0)
    assert steering > 0.0, f"Right turn should give positive steering, got {steering}"
    
    # Test case 4: Left turn, centered
    steering, _ = calc.calculate_steering(-0.02, 0.0)
    assert steering < 0.0, f"Left turn should give negative steering, got {steering}"
    
    print("✓ PASS: Mathematical correctness verified")


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING EXACT STEERING CALCULATION")
    print("=" * 70)
    print()
    
    try:
        test_straight_path_at_center()
        test_straight_path_left_of_center()
        test_straight_path_right_of_center()
        test_curved_path_right_turn()
        test_curved_path_left_turn()
        test_curved_path_with_lateral_error()
        test_steering_clipping()
        test_coordinate_system()
        test_mathematical_correctness()
        
        print()
        print("=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        raise
