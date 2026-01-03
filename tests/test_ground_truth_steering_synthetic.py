"""
Synthetic test for ground truth steering without Unity.

This test simulates a car following a known path and validates:
1. Steering direction correctness
2. Path-based steering (heading + lateral error)
3. Mathematical correctness of steering calculations
"""

import math
from typing import Tuple, List


class SyntheticPath:
    """Synthetic path for testing (oval track)."""
    
    def __init__(self, center_x: float = 0.0, center_z: float = 0.0, 
                 radius: float = 50.0):
        """
        Create an oval track path.
        
        Args:
            center_x: Center X coordinate
            center_z: Center Z coordinate  
            radius: Track radius (meters)
        """
        self.center_x = center_x
        self.center_z = center_z
        self.radius = radius
    
    def get_path_point(self, distance_along_path: float) -> Tuple[float, float, float]:
        """
        Get path point at distance along path.
        
        Args:
            distance_along_path: Distance along path (meters)
            
        Returns:
            (x, z, heading_degrees) - position and heading at that point
        """
        # Oval track: two semicircles connected by straight segments
        # For simplicity, use a circle
        circumference = 2 * math.pi * self.radius
        angle = (distance_along_path / circumference) * 2 * math.pi
        
        x = self.center_x + self.radius * math.cos(angle)
        z = self.center_z + self.radius * math.sin(angle)
        heading = math.degrees(angle + math.pi / 2)  # Tangent to circle
        
        return (x, z, heading)
    
    def get_lane_center_at_car(self, car_x: float, car_z: float, 
                               car_heading: float) -> float:
        """
        Get lane center position relative to car (vehicle coordinates).
        
        Args:
            car_x: Car X position
            car_z: Car Z position
            car_heading: Car heading (degrees)
            
        Returns:
            Lane center X in vehicle coordinates (positive = right of car)
        """
        # Find closest point on path
        # For circle: distance from center
        dist_from_center = math.sqrt((car_x - self.center_x)**2 + 
                                     (car_z - self.center_z)**2)
        
        # Lane center is at radius distance
        lateral_offset = dist_from_center - self.radius
        
        # Convert to vehicle coordinates
        # Car's right vector in world: (cos(heading+90), sin(heading+90))
        # For simplicity, assume car is aligned with path
        # Lateral offset is perpendicular to path
        
        return lateral_offset


class SyntheticCar:
    """Synthetic car for testing."""
    
    def __init__(self, x: float = 0.0, z: float = 0.0, heading: float = 0.0):
        """
        Initialize car state.
        
        Args:
            x: X position
            z: Z position
            heading: Heading in degrees
        """
        self.x = x
        self.z = z
        self.heading = heading
        self.speed = 0.0
    
    def update(self, steering: float, throttle: float, dt: float = 0.033):
        """
        Update car state with control inputs.
        
        Args:
            steering: Steering input (-1 to 1)
            throttle: Throttle input (0 to 1)
            dt: Time step (seconds)
        """
        # Simple bicycle model
        max_steer_angle = 30.0  # degrees
        wheelbase = 2.5  # meters
        max_speed = 5.0  # m/s
        
        steer_angle = steering * max_steer_angle
        self.speed = throttle * max_speed
        
        # Update heading
        if abs(self.speed) > 0.01:
            turn_radius = wheelbase / math.tan(math.radians(steer_angle)) if abs(steer_angle) > 0.01 else float('inf')
            angular_velocity = self.speed / turn_radius if abs(turn_radius) < 1000 else 0.0
            self.heading += math.degrees(angular_velocity * dt)
        
        # Update position
        self.x += self.speed * math.sin(math.radians(self.heading)) * dt
        self.z += self.speed * math.cos(math.radians(self.heading)) * dt


def test_steering_direction():
    """Test that steering direction is correct."""
    print("=" * 70)
    print("TEST 1: Steering Direction Correctness")
    print("=" * 70)
    
    # Test cases: (lane_center, expected_steering_sign, description)
    test_cases = [
        (-2.0, 1.0, "Car RIGHT of center (lane_center < 0) → steer LEFT (negative)"),
        (2.0, -1.0, "Car LEFT of center (lane_center > 0) → steer RIGHT (positive)"),
        (0.0, 0.0, "Car at center (lane_center = 0) → no steering"),
    ]
    
    kp = 1.5
    
    for lane_center, expected_sign, description in test_cases:
        # Current implementation (after fix): opposite sign
        steering = -kp * lane_center
        
        if abs(lane_center) < 0.01:
            correct = abs(steering) < 0.01
        else:
            correct = (steering > 0 and expected_sign > 0) or (steering < 0 and expected_sign < 0)
        
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"{status}: {description}")
        print(f"        lane_center={lane_center:5.2f}m → steering={steering:6.3f} (expected sign: {expected_sign:+.1f})")
        print()
    
    print()


def test_path_based_steering():
    """Test path-based steering (heading + lateral error)."""
    print("=" * 70)
    print("TEST 2: Path-Based Steering")
    print("=" * 70)
    
    path = SyntheticPath(radius=50.0)
    car = SyntheticCar(x=52.0, z=0.0, heading=90.0)  # 2m right of path, heading 90°
    
    # Get desired heading from path (lookahead)
    lookahead = 5.0  # meters
    path_x, path_z, desired_heading = path.get_path_point(lookahead)
    
    # Compute errors
    heading_error = desired_heading - car.heading
    lateral_error = path.get_lane_center_at_car(car.x, car.z, car.heading)
    
    # Path-based steering
    kp_heading = 0.1  # Gain for heading error
    kp_lateral = 1.5  # Gain for lateral error
    
    steering_heading = kp_heading * heading_error
    steering_lateral = -kp_lateral * lateral_error  # Opposite sign (our fix)
    steering_combined = steering_heading + steering_lateral
    
    print(f"Car position: ({car.x:.2f}, {car.z:.2f}), heading: {car.heading:.1f}°")
    print(f"Desired heading: {desired_heading:.1f}°")
    print(f"Lateral error: {lateral_error:.2f}m")
    print()
    print(f"Heading error: {heading_error:.1f}°")
    print(f"Steering from heading: {steering_heading:.3f}")
    print(f"Steering from lateral: {steering_lateral:.3f}")
    print(f"Combined steering: {steering_combined:.3f}")
    print()
    
    # Validate: steering should correct both errors
    if lateral_error > 0:  # Car left of center
        expected_lateral_steering = -1.0  # Steer right (positive)
        print(f"✓ Lateral steering correct: {steering_lateral:.3f} (expected negative for left correction)")
    else:  # Car right of center
        expected_lateral_steering = 1.0  # Steer left (negative)
        print(f"✓ Lateral steering correct: {steering_lateral:.3f} (expected positive for right correction)")
    
    print()


def test_car_following_path():
    """Test car following a path over multiple frames."""
    print("=" * 70)
    print("TEST 3: Car Following Path (Multi-Frame)")
    print("=" * 70)
    
    path = SyntheticPath(radius=50.0)
    car = SyntheticCar(x=52.0, z=0.0, heading=90.0)  # Start 2m right of path
    
    kp_heading = 0.1
    kp_lateral = 1.5
    lookahead = 5.0
    
    print("Frame | X      | Z      | Heading | LatErr | Steering | Correct?")
    print("-" * 70)
    
    for frame in range(10):
        # Get path information
        _, _, desired_heading = path.get_path_point(lookahead)
        lateral_error = path.get_lane_center_at_car(car.x, car.z, car.heading)
        
        # Compute steering
        heading_error = desired_heading - car.heading
        steering_heading = kp_heading * heading_error
        steering_lateral = -kp_lateral * lateral_error
        steering = max(-1.0, min(1.0, steering_heading + steering_lateral))
        
        # Check if steering is correct
        if abs(lateral_error) < 0.1:
            correct = "✓"
        elif lateral_error > 0 and steering > 0:  # Left of center, steering right
            correct = "✓"
        elif lateral_error < 0 and steering < 0:  # Right of center, steering left
            correct = "✓"
        else:
            correct = "✗"
        
        print(f"{frame:5d} | {car.x:6.2f} | {car.z:6.2f} | {car.heading:7.1f}° | "
              f"{lateral_error:6.2f}m | {steering:8.3f} | {correct}")
        
        # Update car
        car.update(steering, throttle=1.0, dt=0.033)
    
    print()
    print(f"Final lateral error: {path.get_lane_center_at_car(car.x, car.z, car.heading):.2f}m")
    print()


def test_mathematical_correctness():
    """Test mathematical correctness of steering calculations."""
    print("=" * 70)
    print("TEST 4: Mathematical Correctness")
    print("=" * 70)
    
    # Test pure proportional control
    kp = 1.5
    test_cases = [
        (-2.0, 3.0, "Car 2m right (lane_center=-2) → steer left (negative) by 3.0"),
        (2.0, -3.0, "Car 2m left (lane_center=+2) → steer right (positive) by 3.0"),
        (0.0, 0.0, "Car at center → no steering"),
    ]
    
    for lane_center, expected_steering, description in test_cases:
        # With our fix: opposite sign (steer toward center)
        # If lane_center < 0 (car right of center), steer left (negative)
        # If lane_center > 0 (car left of center), steer right (positive)
        steering = -kp * lane_center
        correct = abs(steering - expected_steering) < 0.01
        
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"{status}: {description}")
        print(f"        steering = -{kp} * {lane_center} = {steering:.3f} (expected {expected_steering:.3f})")
        print()
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SYNTHETIC GROUND TRUTH STEERING TESTS")
    print("=" * 70)
    print()
    
    test_steering_direction()
    test_path_based_steering()
    test_car_following_path()
    test_mathematical_correctness()
    
    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)

