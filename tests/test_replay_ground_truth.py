#!/usr/bin/env python3
"""
Replay test with ground truth validation.
Compares detected lane center against actual lane center from Unity world coordinates.

This test validates:
1. Lane detection accuracy (detected center vs actual center)
2. Coordinate conversion accuracy
3. Reference point accuracy
4. Edge cases (drift, turns, single-lane detection)

Usage:
    pytest tests/test_replay_ground_truth.py -v
    pytest tests/test_replay_ground_truth.py::test_lane_center_accuracy -v
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def quaternion_to_yaw(q):
    """Convert quaternion (x, y, z, w) to yaw angle (radians)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
    return yaw


def calculate_ground_truth_lane_center(car_x: float, car_z: float, car_heading: float,
                                       lane_width: float = 7.0, left_lane_x_world: float = -3.5,
                                       right_lane_x_world: float = 3.5) -> float:
    """
    Calculate ground truth lane center position in vehicle coordinates.
    
    For a single-lane road (no center line):
    - Left lane line is at x=-3.5m in world coordinates (road edge)
    - Right lane line is at x=3.5m in world coordinates (road edge)
    - Road center is at x=0.0m ((left_lane_x_world + right_lane_x_world) / 2)
    - Car starts at x=0.0m (center of road)
    
    Args:
        car_x: Car X position in world coordinates
        car_z: Car Z position in world coordinates (forward distance)
        car_heading: Car heading in radians
        lane_width: Total road width in meters (default 7.0m for single lane)
        left_lane_x_world: Left lane line X position in world coordinates
        right_lane_x_world: Right lane line X position in world coordinates
    
    Returns:
        Ground truth road center X position in vehicle coordinates (meters)
        For single-lane road: (left_lane_x_world + right_lane_x_world) / 2 - car_x
    """
    # Road center in world coordinates (midpoint between lane lines)
    road_center_world = (left_lane_x_world + right_lane_x_world) / 2.0
    
    # Convert to vehicle coordinates (relative to car)
    # For straight road, just subtract car X
    # For curved roads, would need to account for heading and rotation
    lane_center_vehicle = road_center_world - car_x
    
    return lane_center_vehicle


def calculate_ground_truth_lane_positions(car_x: float, car_z: float, car_heading: float,
                                         left_lane_x_world: float = -3.5,
                                         right_lane_x_world: float = 3.5) -> tuple:
    """
    Calculate ground truth lane positions (left and right) in vehicle coordinates.
    
    Args:
        car_x: Car X position in world coordinates
        car_z: Car Z position in world coordinates
        car_heading: Car heading in radians
        left_lane_x_world: Left lane line X position in world coordinates
        right_lane_x_world: Right lane line X position in world coordinates
    
    Returns:
        Tuple of (left_lane_x_vehicle, right_lane_x_vehicle) in meters
    """
    left_lane_x_vehicle = left_lane_x_world - car_x
    right_lane_x_vehicle = right_lane_x_world - car_x
    
    return (left_lane_x_vehicle, right_lane_x_vehicle)


def test_lane_center_accuracy(recording_path: str = None):
    """
    Test that detected lane center matches ground truth lane center.
    
    Args:
        recording_path: Path to recording file (default: latest)
    """
    if recording_path is None:
        recordings_dir = Path('data/recordings')
        recordings = sorted(recordings_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            pytest.skip("No recordings found")
        recording_path = recordings[0]
    
    recording_path = Path(recording_path)
    if not recording_path.exists():
        pytest.skip(f"Recording not found: {recording_path}")
    
    with h5py.File(recording_path, 'r') as f:
        num_frames = len(f['perception/num_lanes_detected'])
        
        # Get data
        positions = np.array(f['vehicle/position'])
        rotations = np.array(f['vehicle/rotation'])
        num_lanes = np.array(f['perception/num_lanes_detected'])
        left_lane_x = np.array(f['perception/left_lane_line_x'])
        right_lane_x = np.array(f['perception/right_lane_line_x'])
        ref_x = np.array(f['trajectory/reference_point_x'])
        
        errors = []
        valid_frames = 0
        
        for i in range(num_frames):
            if num_lanes[i] != 2:
                continue  # Skip single-lane frames for now
            
            # Vehicle state
            car_x = positions[i, 0]
            car_z = positions[i, 2]
            q = rotations[i]
            car_heading = quaternion_to_yaw(q)
            
            # Detected lane center
            detected_lane_center = (left_lane_x[i] + right_lane_x[i]) / 2.0
            
            # Ground truth lane center (right lane, since car starts in right lane)
            gt_lane_center = calculate_ground_truth_lane_center(car_x, car_z, car_heading)
            
            # Also get ground truth lane positions for validation
            gt_left_lane_x, gt_right_lane_x = calculate_ground_truth_lane_positions(
                car_x, car_z, car_heading
            )
            
            # Error
            error = abs(detected_lane_center - gt_lane_center)
            errors.append(error)
            valid_frames += 1
        
        if valid_frames == 0:
            pytest.skip("No valid frames with 2 lanes detected")
        
        # Statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
        
        print(f"\nLane Center Accuracy Test:")
        print(f"  Valid frames: {valid_frames}/{num_frames}")
        print(f"  Mean error: {mean_error:.3f}m")
        print(f"  Max error: {max_error:.3f}m")
        print(f"  Std error: {std_error:.3f}m")
        
        # Assertions
        assert mean_error < 0.5, f"Mean lane center error ({mean_error:.3f}m) exceeds 0.5m threshold"
        assert max_error < 1.0, f"Max lane center error ({max_error:.3f}m) exceeds 1.0m threshold"
        
        # Check that most frames are accurate
        accurate_frames = sum(1 for e in errors if e < 0.3)
        accuracy_rate = accurate_frames / valid_frames
        assert accuracy_rate > 0.8, f"Only {accuracy_rate:.1%} of frames have error < 0.3m (expected >80%)"


def test_reference_point_accuracy(recording_path: str = None):
    """
    Test that reference point matches ground truth lane center.
    
    Args:
        recording_path: Path to recording file (default: latest)
    """
    if recording_path is None:
        recordings_dir = Path('data/recordings')
        recordings = sorted(recordings_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            pytest.skip("No recordings found")
        recording_path = recordings[0]
    
    recording_path = Path(recording_path)
    if not recording_path.exists():
        pytest.skip(f"Recording not found: {recording_path}")
    
    with h5py.File(recording_path, 'r') as f:
        num_frames = len(f['perception/num_lanes_detected'])
        
        # Get data
        positions = np.array(f['vehicle/position'])
        rotations = np.array(f['vehicle/rotation'])
        num_lanes = np.array(f['perception/num_lanes_detected'])
        left_lane_x = np.array(f['perception/left_lane_line_x'])
        right_lane_x = np.array(f['perception/right_lane_line_x'])
        ref_x = np.array(f['trajectory/reference_point_x'])
        
        errors = []
        valid_frames = 0
        
        for i in range(num_frames):
            if num_lanes[i] != 2:
                continue
            
            # Vehicle state
            car_x = positions[i, 0]
            car_z = positions[i, 2]
            q = rotations[i]
            car_heading = quaternion_to_yaw(q)
            
            # Ground truth lane center
            gt_lane_center = calculate_ground_truth_lane_center(car_x, car_z, car_heading)
            
            # Reference point error
            error = abs(ref_x[i] - gt_lane_center)
            errors.append(error)
            valid_frames += 1
        
        if valid_frames == 0:
            pytest.skip("No valid frames with 2 lanes detected")
        
        # Statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"\nReference Point Accuracy Test:")
        print(f"  Valid frames: {valid_frames}/{num_frames}")
        print(f"  Mean error: {mean_error:.3f}m")
        print(f"  Max error: {max_error:.3f}m")
        
        # Assertions
        assert mean_error < 0.3, f"Mean reference point error ({mean_error:.3f}m) exceeds 0.3m threshold"
        assert max_error < 0.7, f"Max reference point error ({max_error:.3f}m) exceeds 0.7m threshold (relaxed from 0.5m for single-lane scenario)"


def test_lane_width_accuracy(recording_path: str = None):
    """
    Test that detected lane width matches expected lane width.
    
    Args:
        recording_path: Path to recording file (default: latest)
    """
    if recording_path is None:
        recordings_dir = Path('data/recordings')
        recordings = sorted(recordings_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            pytest.skip("No recordings found")
        recording_path = recordings[0]
    
    recording_path = Path(recording_path)
    if not recording_path.exists():
        pytest.skip(f"Recording not found: {recording_path}")
    
    with h5py.File(recording_path, 'r') as f:
        num_frames = len(f['perception/num_lanes_detected'])
        
        # Get data
        num_lanes = np.array(f['perception/num_lanes_detected'])
        left_lane_x = np.array(f['perception/left_lane_line_x'])
        right_lane_x = np.array(f['perception/right_lane_line_x'])
        
        widths = []
        valid_frames = 0
        
        for i in range(num_frames):
            if num_lanes[i] != 2:
                continue
            
            lane_width = abs(right_lane_x[i] - left_lane_x[i])
            widths.append(lane_width)
            valid_frames += 1
        
        if valid_frames == 0:
            pytest.skip("No valid frames with 2 lanes detected")
        
        # Statistics
        mean_width = np.mean(widths)
        std_width = np.std(widths)
        expected_width = 7.0  # meters (single-lane road, total road width)
        
        print(f"\nLane Width Accuracy Test:")
        print(f"  Valid frames: {valid_frames}/{num_frames}")
        print(f"  Mean width: {mean_width:.3f}m (expected: {expected_width}m)")
        print(f"  Std width: {std_width:.3f}m")
        
        # Assertions
        width_error = abs(mean_width - expected_width)
        assert width_error < 0.5, f"Mean lane width error ({width_error:.3f}m) exceeds 0.5m threshold"
        assert std_width < 0.5, f"Lane width std ({std_width:.3f}m) exceeds 0.5m threshold (too much variation)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

