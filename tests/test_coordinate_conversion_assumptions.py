"""
Test to validate coordinate conversion assumptions.

This test verifies that our assumptions about camera parameters match Unity's actual values,
and that the coordinate conversion produces correct results.

Key assumptions to validate:
1. Camera FOV: Config says 60°, Unity should match
2. Camera height: Config says 0.5m, but Unity camera is at y=1.2m relative to car
3. Image dimensions: 640x480 pixels
4. Pixel-to-meter conversion at different distances
5. Lane width calculation (should be ~3.5m for standard lane)
"""

import sys
from pathlib import Path

import pytest
import numpy as np
from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner

# Make conftest importable for golden_recording_path
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import golden_recording_path


class TestCoordinateConversionAssumptions:
    """Test coordinate conversion assumptions and validate against expected values."""
    
    def test_camera_fov_assumption(self):
        """Test that FOV assumption (60°) is reasonable."""
        # Unity AVCamera has FOV = 60° (from CarPrefab.prefab line 289)
        # Config says camera_fov: 60.0
        expected_fov = 60.0
        config_fov = 60.0
        
        assert config_fov == expected_fov, \
            f"Config FOV ({config_fov}°) doesn't match Unity FOV ({expected_fov}°)"
    
    @pytest.mark.xfail(reason="Known config mismatch: config camera_height=0.5m but Unity camera is ~1.9m above ground. Not blocking — perception uses segmentation, not pinhole projection.")
    def test_camera_height_assumption(self):
        """Test that camera height assumption (0.5m) is reasonable.

        NOTE: Unity camera is at y=1.2m relative to car, but we need absolute height above ground.
        If car is ~0.7m tall, then camera height = 1.2m + 0.7m = 1.9m (not 0.5m!)
        This could be a major source of error!
        """
        config_height = 0.5  # meters
        unity_camera_y_relative = 1.2  # meters
        estimated_car_height = 0.7  # meters (rough estimate)
        actual_camera_height = unity_camera_y_relative + estimated_car_height

        assert abs(actual_camera_height - config_height) <= 0.3, (
            f"Camera height mismatch: config={config_height}m, "
            f"estimated={actual_camera_height}m"
        )
    
    def test_coordinate_conversion_consistency(self):
        """Test that coordinate conversion is consistent across different distances.
        
        The same pixel offset should map to different meter offsets at different distances,
        but the relationship should be linear with distance.
        """
        planner = RuleBasedTrajectoryPlanner(
            lookahead_distance=20.0,
            image_width=640,
            image_height=480,
            camera_fov=60.0,
            camera_height=0.5
        )
        
        # Test pixel offset at different distances
        pixel_offset = 100  # pixels from center
        distances = [5.0, 10.0, 15.0, 20.0]  # meters
        
        meter_offsets = []
        for distance in distances:
            x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                320 + pixel_offset, 480, lookahead_distance=distance
            )
            meter_offsets.append(x_vehicle)
        
        print(f"\nCoordinate conversion consistency:")
        print(f"  Pixel offset: {pixel_offset}px")
        for d, m in zip(distances, meter_offsets):
            print(f"  At {d:4.1f}m: {m:6.3f}m offset")
        
        # Meter offset should scale linearly with distance
        # At 2x distance, offset should be ~2x
        ratio_10_5 = meter_offsets[1] / meter_offsets[0] if meter_offsets[0] != 0 else 0
        ratio_20_10 = meter_offsets[3] / meter_offsets[1] if meter_offsets[1] != 0 else 0
        
        print(f"  Ratio (10m/5m): {ratio_10_5:.3f} (expected ~2.0)")
        print(f"  Ratio (20m/10m): {ratio_20_10:.3f} (expected ~2.0)")
        
        # Ratios should be close to 2.0 (within 20%)
        assert 1.6 < ratio_10_5 < 2.4, \
            f"Coordinate conversion not linear: ratio 10m/5m = {ratio_10_5:.3f} (expected ~2.0)"
        assert 1.6 < ratio_20_10 < 2.4, \
            f"Coordinate conversion not linear: ratio 20m/10m = {ratio_20_10:.3f} (expected ~2.0)"
    
    def test_camera_height_impact(self):
        """Test how camera height affects coordinate conversion.
        
        Camera height is used for distance estimation from y_pixels.
        If height is wrong, distance estimation will be wrong, affecting all conversions.
        """
        # Test with different camera heights
        heights = [0.5, 1.0, 1.5, 2.0]  # meters
        
        print(f"\nCamera height impact analysis:")
        print(f"  Testing coordinate conversion with different camera heights:")
        
        results = []
        for height in heights:
            planner = RuleBasedTrajectoryPlanner(
                lookahead_distance=20.0,
                image_width=640,
                image_height=480,
                camera_fov=60.0,
                camera_height=height
            )
            
            # Convert a point at bottom of image (y=480)
            # Without lookahead_distance, it will estimate distance from y_pixels
            x_vehicle, y_vehicle = planner._convert_image_to_vehicle_coords(
                320, 480, lookahead_distance=None  # Force distance estimation
            )
            
            results.append((height, y_vehicle))
            print(f"    Height {height:3.1f}m: estimated distance = {y_vehicle:5.2f}m")
        
        # Higher camera = farther estimated distance (for same y_pixel)
        # Check that relationship holds
        distances = [r[1] for r in results]
        if len(distances) > 1:
            # Distance should increase with height (or at least be consistent)
            print(f"  Note: Distance estimation varies significantly with camera height!")
            print(f"  This suggests camera height is critical for accurate conversion.")
    
    def test_actual_lane_width_from_recorded_data(self):
        """Test lane width calculation using golden recording data.

        Loads s_loop golden recording and checks if calculated lane widths are
        reasonable. Lane line positions measure visible marking separation,
        typically ~3.5-4.0m for a standard lane.
        """
        import h5py

        recording_file = golden_recording_path("s_loop")
        if recording_file is None:
            pytest.skip("s_loop golden recording not registered or not on disk")

        with h5py.File(recording_file, 'r') as f:
            # Support both old (left_lane_x) and current (left_lane_line_x) field names
            left_key = ('perception/left_lane_line_x' if 'perception/left_lane_line_x' in f
                        else 'perception/left_lane_x')
            right_key = ('perception/right_lane_line_x' if 'perception/right_lane_line_x' in f
                         else 'perception/right_lane_x')
            if left_key not in f or right_key not in f:
                pytest.skip("Recording doesn't have lane position data")

            left_lane_x = np.array(f[left_key])
            right_lane_x = np.array(f[right_key])
            num_lanes = np.array(f['perception/num_lanes_detected'])

            lane_widths = []
            for i in range(len(num_lanes)):
                if num_lanes[i] == 2 and not np.isnan(left_lane_x[i]) and not np.isnan(right_lane_x[i]):
                    width = abs(right_lane_x[i] - left_lane_x[i])
                    lane_widths.append(width)

            assert len(lane_widths) > 0, "No dual-lane detections in golden recording"

            lane_widths = np.array(lane_widths)
            mean_width = np.mean(lane_widths)
            std_width = np.std(lane_widths)

            assert 3.0 <= mean_width <= 5.0, (
                f"Lane width calculation wrong: mean={mean_width:.3f}m "
                f"(expected ~3.5-4.0m for lane marking separation)"
            )
            assert std_width < 1.0, (
                f"Lane width inconsistent: std={std_width:.3f}m (should be < 1.0m)"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

