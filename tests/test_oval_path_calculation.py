"""
Unit test for oval path calculation logic.

This test simulates the C# Unity path calculation to verify:
1. GetOvalCenterPoint(t) is continuous
2. GetOvalDirection(t) never returns zero
3. GetDistanceAlongPath() and FindTFromDistance() are inverse functions
4. Path can be traversed smoothly from start to end
"""

import numpy as np
import pytest


class RoadGeneratorSimulator:
    """Python simulation of Unity's RoadGenerator.GetOvalCenterPoint and GetOvalDirection"""
    
    def __init__(self, straight_length=200.0, turn_radius=50.0):
        self.straight_length = straight_length
        self.turn_radius = turn_radius
    
    def get_oval_center_point(self, t):
        """Simulate GetOvalCenterPoint(t) - returns (x, z) position"""
        # Normalize t to 0-1 range (simulate Mathf.Repeat)
        t = t % 1.0
        if t < 0:
            t = t + 1.0
        
        # Segment 0: Top straight (t: 0 to 0.25)
        # Segment 1: Right curve (t: 0.25 to 0.5)
        # Segment 2: Bottom straight (t: 0.5 to 0.75)
        # Segment 3: Left curve (t: 0.75 to 1.0)
        
        segment_length = 0.25
        
        if t < 0.25:
            # Top straight section
            local_t = t / segment_length
            z = self.turn_radius
            x = np.interp(local_t, [0, 1], [-self.straight_length * 0.5, self.straight_length * 0.5])
            return np.array([x, z])
        elif t < 0.5:
            # Right curve
            local_t = (t - 0.25) / segment_length
            angle = np.interp(local_t, [0, 1], [90, -90]) * np.pi / 180.0
            cx = self.straight_length * 0.5
            x = cx + self.turn_radius * np.cos(angle)
            z = 0 + self.turn_radius * np.sin(angle)
            return np.array([x, z])
        elif t < 0.75:
            # Bottom straight section
            local_t = (t - 0.5) / segment_length
            z = -self.turn_radius
            x = np.interp(local_t, [0, 1], [self.straight_length * 0.5, -self.straight_length * 0.5])
            return np.array([x, z])
        else:
            # Left curve
            local_t = (t - 0.75) / segment_length
            angle = np.interp(local_t, [0, 1], [-90, 90]) * np.pi / 180.0
            cx = -self.straight_length * 0.5
            x = cx - self.turn_radius * np.cos(angle)
            z = 0 + self.turn_radius * np.sin(angle)
            return np.array([x, z])
    
    def get_oval_direction(self, t, delta=0.0025):
        """Simulate GetOvalDirection(t) - returns normalized direction vector"""
        # Normalize t
        t = t % 1.0
        if t < 0:
            t = t + 1.0
        
        # Handle wrap-around at boundaries (same logic as C# fix)
        if t < delta:
            # Near start: don't wrap backward
            t_prev = 0.0
            t_next = t + delta
        elif t > (1.0 - delta):
            # Near end: don't wrap forward
            t_prev = t - delta
            t_next = 0.999  # Just before wrap point (same as C# fix)
        else:
            # Middle range: normal calculation
            t_prev = t - delta
            t_next = t + delta
        
        prev = self.get_oval_center_point(t_prev)
        next = self.get_oval_center_point(t_next)
        
        dir_vec = next - prev
        dir_sqr_magnitude = np.sum(dir_vec**2)
        
        if dir_sqr_magnitude < 1e-8:
            return np.array([1.0, 0.0])  # Return right direction as fallback
        
        return dir_vec / np.sqrt(dir_sqr_magnitude)


def get_distance_along_path(road_gen, t0, t1, samples=1000):
    """Simulate GetDistanceAlongPath(t0, t1)"""
    distance = 0.0
    prev_point = road_gen.get_oval_center_point(t0)
    
    for i in range(1, samples + 1):
        t = np.interp(i, [0, samples], [t0, t1])
        current_point = road_gen.get_oval_center_point(t)
        distance += np.linalg.norm(current_point - prev_point)
        prev_point = current_point
    
    return distance


def find_t_from_distance(road_gen, target_distance, path_length, tolerance=0.001, max_iterations=20):
    """Simulate FindTFromDistance() using binary search"""
    if target_distance < 0:
        target_distance = 0.0
    
    if target_distance <= 0.001:
        return 0.0
    
    # Check if close to end
    epsilon = 0.01
    if target_distance >= (path_length - epsilon):
        return 0.999  # Just before wrap point
    
    # Binary search
    low = 0.0
    high = 1.0
    
    for _ in range(max_iterations):
        mid = (low + high) * 0.5
        distance_at_mid = get_distance_along_path(road_gen, 0.0, mid, samples=1000)
        
        if abs(distance_at_mid - target_distance) < tolerance:
            return mid
        
        if distance_at_mid < target_distance:
            low = mid
        else:
            high = mid
    
    return (low + high) * 0.5


def test_path_closed():
    """Test that path is closed (t=1.0 wraps to t=0.0)"""
    road_gen = RoadGeneratorSimulator()
    
    # Test that t=1.0 wraps to t=0.0 (path is closed)
    point_at_0 = road_gen.get_oval_center_point(0.0)
    point_at_1 = road_gen.get_oval_center_point(1.0)  # Should wrap to 0.0
    dist = np.linalg.norm(point_at_1 - point_at_0)
    assert dist < 0.1, f"Path not closed: distance between t=0 and t=1 is {dist}"


def test_direction_never_zero():
    """Test that GetOvalDirection never returns zero vector"""
    road_gen = RoadGeneratorSimulator()
    
    # Test at many points along the path
    test_points = np.linspace(0, 1, 1000, endpoint=False)  # Exclude 1.0 to avoid wrap
    
    for t in test_points:
        direction = road_gen.get_oval_direction(t)
        dir_magnitude = np.linalg.norm(direction)
        
        assert dir_magnitude > 0.9, f"Direction magnitude too small at t={t}: {dir_magnitude}"
        assert not np.isnan(direction).any(), f"Direction has NaN at t={t}"
        assert not np.isinf(direction).any(), f"Direction has Inf at t={t}"


def test_distance_monotonic():
    """Test that GetDistanceAlongPath increases monotonically"""
    road_gen = RoadGeneratorSimulator()
    
    # Calculate path length
    path_length = get_distance_along_path(road_gen, 0.0, 1.0, samples=1000)
    
    # Test that distance increases as we go along path
    test_points = np.linspace(0, 1, 100)
    distances = [get_distance_along_path(road_gen, 0.0, t, samples=1000) for t in test_points]
    
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i-1] - 0.1, \
            f"Distance not monotonic: dist[{i-1}]={distances[i-1]:.3f}, dist[{i}]={distances[i]:.3f}"
    
    print(f"Path length: {path_length:.2f}m")


def test_find_t_from_distance_inverse():
    """Test that FindTFromDistance correctly inverts GetDistanceAlongPath"""
    road_gen = RoadGeneratorSimulator()
    path_length = get_distance_along_path(road_gen, 0.0, 1.0, samples=1000)
    
    # Test at various t values
    test_t_values = np.linspace(0.01, 0.99, 50)
    
    for original_t in test_t_values:
        # Get distance for this t
        distance = get_distance_along_path(road_gen, 0.0, original_t, samples=1000)
        
        # Find t back from distance
        found_t = find_t_from_distance(road_gen, distance, path_length)
        
        # Should be close
        error = abs(found_t - original_t)
        assert error < 0.02, \
            f"FindTFromDistance failed: original_t={original_t:.3f}, found_t={found_t:.3f}, error={error:.3f}, distance={distance:.2f}"


def test_full_path_traversal():
    """Test that we can traverse the full path smoothly"""
    road_gen = RoadGeneratorSimulator()
    path_length = get_distance_along_path(road_gen, 0.0, 1.0, samples=1000)
    
    # Simulate traveling along the path at constant speed
    reference_speed = 5.0  # m/s
    time_step = 0.1  # 10 Hz
    total_time = path_length / reference_speed  # Time to complete one lap
    
    t_values = []
    positions = []
    directions = []
    
    for elapsed_time in np.arange(0, total_time * 1.1, time_step):  # Go slightly past one lap
        distance_traveled = reference_speed * elapsed_time
        normalized_distance = distance_traveled % path_length
        
        t = find_t_from_distance(road_gen, normalized_distance, path_length)
        pos = road_gen.get_oval_center_point(t)
        direction = road_gen.get_oval_direction(t)
        
        t_values.append(t)
        positions.append(pos)
        directions.append(direction)
    
    # Check that t values increase (with wrap-around)
    for i in range(1, len(t_values)):
        # t should generally increase, or wrap from ~1.0 to ~0.0
        if t_values[i] < t_values[i-1]:
            # Wrap occurred - previous should be close to 1.0, current close to 0.0
            assert t_values[i-1] > 0.9, f"Unexpected wrap: t[{i-1}]={t_values[i-1]:.3f}, t[{i}]={t_values[i]:.3f}"
            assert t_values[i] < 0.1, f"Unexpected wrap: t[{i-1}]={t_values[i-1]:.3f}, t[{i}]={t_values[i]:.3f}"
    
    # Check that direction never becomes zero
    for i, dir_vec in enumerate(directions):
        dir_magnitude = np.linalg.norm(dir_vec)
        assert dir_magnitude > 0.9, f"Zero direction at step {i}, t={t_values[i]:.3f}"
    
    print(f"Successfully traversed {len(t_values)} steps along path")
    print(f"Path length: {path_length:.2f}m, Travel time: {total_time:.2f}s")


if __name__ == "__main__":
    print("=" * 80)
    print("Running Oval Path Calculation Tests")
    print("=" * 80)
    print()
    
    try:
        test_path_closed()
        print("✓ Path closed test passed")
        
        test_direction_never_zero()
        print("✓ Direction never zero test passed")
        
        test_distance_monotonic()
        print("✓ Distance monotonic test passed")
        
        test_find_t_from_distance_inverse()
        print("✓ FindTFromDistance inverse test passed")
        
        test_full_path_traversal()
        print("✓ Full path traversal test passed")
        
        print()
        print("=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise

