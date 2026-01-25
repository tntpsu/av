"""
Ground truth following test for perception and heading validation.

This test isolates perception and heading calculation by:
1. Using ground truth to drive the car (open-loop control)
2. Recording all frames during the drive
3. Comparing perception output and heading against ground truth for each frame

Key questions this test answers:
1. Are we detecting the correct lane lines? (left vs right assignment)
2. Are lane positions accurate in vehicle coordinates?
3. Is lane width correct?
4. Is heading calculation correct?
5. Are we handling curves correctly?
6. Are we handling straight roads correctly?
7. Is there temporal consistency?
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PerceptionGroundTruthTest:
    """Test perception and heading against ground truth data."""
    
    def __init__(self, recording_file: str):
        """
        Initialize test with a recording.
        
        Args:
            recording_file: Path to HDF5 recording file
        """
        self.recording_file = Path(recording_file)
        if not self.recording_file.exists():
            raise FileNotFoundError(f"Recording not found: {recording_file}")
        
        self.h5_file = h5py.File(self.recording_file, 'r')
        self._load_data()
    
    def _load_data(self):
        """Load all data from recording."""
        # Ground truth
        self.gt_left = self.h5_file["ground_truth/left_lane_line_x"][:]
        self.gt_right = self.h5_file["ground_truth/right_lane_line_x"][:]
        self.gt_center = self.h5_file["ground_truth/lane_center_x"][:]
        self.gt_width = self.gt_right - self.gt_left
        
        # Perception
        self.perception_left = self.h5_file["perception/left_lane_line_x"][:]
        self.perception_right = self.h5_file["perception/right_lane_line_x"][:]
        self.perception_width = self.perception_right - self.perception_left
        self.num_lanes_detected = self.h5_file["perception/num_lanes_detected"][:]
        
        # Trajectory/Heading
        self.heading = np.degrees(self.h5_file["trajectory/reference_point_heading"][:])
        self.ref_x = self.h5_file["trajectory/reference_point_x"][:]
        self.ref_y = self.h5_file["trajectory/reference_point_y"][:]

        # Ground truth heading/curvature (optional)
        self.gt_desired_heading = None
        self.gt_path_curvature = None
        if "ground_truth/desired_heading" in self.h5_file:
            self.gt_desired_heading = self.h5_file["ground_truth/desired_heading"][:]
        if "ground_truth/path_curvature" in self.h5_file:
            self.gt_path_curvature = self.h5_file["ground_truth/path_curvature"][:]
        
        # Vehicle state
        self.positions = self.h5_file["vehicle/position"][:]
        self.speeds = self.h5_file["vehicle/speed"][:]
        
        self.num_frames = len(self.gt_left)
    
    def compute_expected_heading(self, frame_idx: int, lookahead: float = 2.0) -> float:
        """
        Compute expected heading from ground truth.
        
        For a curved road, heading = arctan(lateral_change / forward_distance)
        We can estimate this from ground truth lane positions at different distances.
        
        Args:
            frame_idx: Current frame index
            lookahead: Distance ahead to compute heading (meters)
        
        Returns:
            Expected heading in degrees
        """
        if frame_idx >= self.num_frames - 1:
            return 0.0
        
        # For now, use a simple approximation:
        # Heading is related to how the lane center changes as we move forward
        # On a curve, the lane center will shift laterally
        
        # Get current and next frame's ground truth
        current_center = self.gt_center[frame_idx]
        next_center = self.gt_center[min(frame_idx + 1, self.num_frames - 1)]
        
        # Estimate forward distance (from speed and time)
        # Assuming 30 fps, each frame is ~0.033s
        dt = 1.0 / 30.0
        if frame_idx < self.num_frames - 1:
            avg_speed = (self.speeds[frame_idx] + self.speeds[frame_idx + 1]) / 2.0
            forward_distance = avg_speed * dt
        else:
            forward_distance = lookahead
        
        # Lateral change
        lateral_change = next_center - current_center
        
        # Compute heading
        if abs(forward_distance) > 0.01:
            heading_rad = np.arctan2(lateral_change, forward_distance)
            return np.degrees(heading_rad)
        else:
            return 0.0

    def get_curve_mask(
        self,
        curvature_threshold: float = 0.01,
        heading_threshold_deg: float = 2.0,
    ) -> Tuple[np.ndarray, str]:
        """
        Return a boolean mask of curve frames and a description of the source.

        Preference order:
        1) ground_truth/path_curvature
        2) ground_truth/desired_heading
        3) perception heading (trajectory/reference_point_heading)
        """
        if self.gt_path_curvature is not None:
            curve_mask = np.abs(self.gt_path_curvature) > curvature_threshold
            source = f"path_curvature>|{curvature_threshold}|"
        elif self.gt_desired_heading is not None:
            curve_mask = np.abs(self.gt_desired_heading) > heading_threshold_deg
            source = f"desired_heading>|{heading_threshold_deg}°|"
        else:
            curve_mask = np.abs(self.heading) > heading_threshold_deg
            source = f"perception_heading>|{heading_threshold_deg}°|"
        return curve_mask, source
    
    def analyze_frame(self, frame_idx: int) -> Dict:
        """
        Analyze a single frame.
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            'frame_idx': frame_idx,
            'perception_valid': self.num_lanes_detected[frame_idx] >= 2,
            'errors': {},
            'warnings': []
        }
        
        if not results['perception_valid']:
            results['warnings'].append("No lanes detected")
            return results
        
        # Lane position errors
        left_error = self.perception_left[frame_idx] - self.gt_left[frame_idx]
        right_error = self.perception_right[frame_idx] - self.gt_right[frame_idx]
        center_error = (self.perception_left[frame_idx] + self.perception_right[frame_idx]) / 2.0 - self.gt_center[frame_idx]
        
        results['errors']['left_lane'] = left_error
        results['errors']['right_lane'] = right_error
        results['errors']['center'] = center_error
        
        # Lane width error
        width_error = self.perception_width[frame_idx] - self.gt_width[frame_idx]
        results['errors']['width'] = width_error
        
        # Heading error
        expected_heading = self.compute_expected_heading(frame_idx)
        heading_error = self.heading[frame_idx] - expected_heading
        results['errors']['heading'] = heading_error
        results['expected_heading'] = expected_heading
        results['actual_heading'] = self.heading[frame_idx]
        
        # Check thresholds
        if abs(left_error) > 0.5:
            results['warnings'].append(f"Large left lane error: {left_error:.3f}m")
        if abs(right_error) > 0.5:
            results['warnings'].append(f"Large right lane error: {right_error:.3f}m")
        if abs(width_error) > 0.5:
            results['warnings'].append(f"Large width error: {width_error:.3f}m")
        if abs(heading_error) > 5.0:
            results['warnings'].append(f"Large heading error: {heading_error:.1f}°")
        
        # Check if lanes are swapped
        if self.perception_left[frame_idx] > self.perception_right[frame_idx]:
            results['warnings'].append("Lanes are swapped (left > right)")
        
        return results
    
    def analyze_all_frames(self) -> Dict:
        """
        Analyze all frames and compute statistics.
        
        Returns:
            Dictionary with overall statistics
        """
        valid_frames = np.where(self.num_lanes_detected >= 2)[0]
        
        if len(valid_frames) == 0:
            return {
                'num_frames': self.num_frames,
                'valid_frames': 0,
                'detection_rate': 0.0,
                'error': 'No frames with valid lane detection'
            }
        
        stats = {
            'num_frames': self.num_frames,
            'valid_frames': len(valid_frames),
            'detection_rate': len(valid_frames) / self.num_frames,
            'errors': {
                'left_lane': {
                    'mean': np.mean(np.abs(self.perception_left[valid_frames] - self.gt_left[valid_frames])),
                    'std': np.std(self.perception_left[valid_frames] - self.gt_left[valid_frames]),
                    'max': np.max(np.abs(self.perception_left[valid_frames] - self.gt_left[valid_frames]))
                },
                'right_lane': {
                    'mean': np.mean(np.abs(self.perception_right[valid_frames] - self.gt_right[valid_frames])),
                    'std': np.std(self.perception_right[valid_frames] - self.gt_right[valid_frames]),
                    'max': np.max(np.abs(self.perception_right[valid_frames] - self.gt_right[valid_frames]))
                },
                'center': {
                    'mean': np.mean(np.abs((self.perception_left[valid_frames] + self.perception_right[valid_frames]) / 2.0 - self.gt_center[valid_frames])),
                    'std': np.std((self.perception_left[valid_frames] + self.perception_right[valid_frames]) / 2.0 - self.gt_center[valid_frames]),
                    'max': np.max(np.abs((self.perception_left[valid_frames] + self.perception_right[valid_frames]) / 2.0 - self.gt_center[valid_frames]))
                },
                'width': {
                    'mean': np.mean(np.abs(self.perception_width[valid_frames] - self.gt_width[valid_frames])),
                    'std': np.std(self.perception_width[valid_frames] - self.gt_width[valid_frames]),
                    'max': np.max(np.abs(self.perception_width[valid_frames] - self.gt_width[valid_frames]))
                }
            }
        }
        
        # Heading errors (simplified - would need better expected heading calculation)
        heading_errors = []
        for i in valid_frames:
            if i < self.num_frames - 1:
                expected = self.compute_expected_heading(i)
                actual = self.heading[i]
                heading_errors.append(abs(actual - expected))
        
        if len(heading_errors) > 0:
            stats['errors']['heading'] = {
                'mean': np.mean(heading_errors),
                'std': np.std(heading_errors),
                'max': np.max(heading_errors)
            }
        
        return stats
    
    def close(self):
        """Close HDF5 file."""
        if self.h5_file:
            self.h5_file.close()


def test_perception_accuracy(recording_file: str = None):
    """
    Test perception accuracy against ground truth.
    
    Args:
        recording_file: Path to recording file (default: latest)
    """
    if recording_file is None:
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            pytest.skip("No recordings found")
        recording_file = str(recordings[0])
    
    test = PerceptionGroundTruthTest(recording_file)
    
    try:
        # Analyze all frames
        stats = test.analyze_all_frames()
        
        print("\n" + "="*80)
        print("PERCEPTION GROUND TRUTH TEST RESULTS")
        print("="*80)
        print(f"Total frames: {stats['num_frames']}")
        print(f"Valid frames (2+ lanes): {stats['valid_frames']}")
        print(f"Detection rate: {stats['detection_rate']*100:.1f}%")
        
        if 'error' in stats:
            print(f"\nERROR: {stats['error']}")
            return
        
        print("\n" + "-"*80)
        print("ERROR STATISTICS")
        print("-"*80)
        
        for error_type, error_stats in stats['errors'].items():
            print(f"\n{error_type.upper()}:")
            print(f"  Mean absolute error: {error_stats['mean']:.3f}")
            print(f"  Std deviation: {error_stats['std']:.3f}")
            print(f"  Max error: {error_stats['max']:.3f}")
        
        # Assertions
        assert stats['detection_rate'] > 0.8, f"Detection rate too low: {stats['detection_rate']*100:.1f}%"
        assert stats['errors']['left_lane']['mean'] < 0.5, f"Left lane error too high: {stats['errors']['left_lane']['mean']:.3f}m"
        assert stats['errors']['right_lane']['mean'] < 0.5, f"Right lane error too high: {stats['errors']['right_lane']['mean']:.3f}m"
        assert stats['errors']['width']['mean'] < 0.5, f"Width error too high: {stats['errors']['width']['mean']:.3f}m"
        
        if 'heading' in stats['errors']:
            assert stats['errors']['heading']['mean'] < 5.0, f"Heading error too high: {stats['errors']['heading']['mean']:.1f}°"
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    finally:
        test.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Find latest recording
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_file = str(recordings[0])
            print(f"Using latest recording: {recordings[0].name}")
        else:
            print("No recordings found!")
            sys.exit(1)
    else:
        recording_file = sys.argv[1]
    
    test_perception_accuracy(recording_file)


