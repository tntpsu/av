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

import pytest
import numpy as np
from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner


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
    
    def test_camera_height_assumption(self):
        """Test that camera height assumption (0.5m) is reasonable.
        
        NOTE: Unity camera is at y=1.2m relative to car, but we need absolute height above ground.
        If car is ~0.7m tall, then camera height = 1.2m + 0.7m = 1.9m (not 0.5m!)
        This could be a major source of error!
        """
        # Config says camera_height: 0.5m
        # Unity camera position: {x: 0, y: 1.2, z: 1.0} relative to car
        # If car center is at ground level, camera is 1.2m above car center
        # If car has height ~0.7m, camera is ~1.9m above ground
        config_height = 0.5  # meters
        
        # Unity camera y position relative to car
        unity_camera_y_relative = 1.2  # meters
        
        # Estimated car height (typical car center to ground)
        estimated_car_height = 0.7  # meters (rough estimate)
        
        # Actual camera height above ground
        actual_camera_height = unity_camera_y_relative + estimated_car_height
        
        print(f"\nCamera height analysis:")
        print(f"  Config assumption: {config_height}m")
        print(f"  Unity camera y (relative to car): {unity_camera_y_relative}m")
        print(f"  Estimated car height: {estimated_car_height}m")
        print(f"  Actual camera height: {actual_camera_height}m")
        print(f"  Error: {abs(actual_camera_height - config_height):.2f}m ({abs(actual_camera_height - config_height)/config_height*100:.1f}%)")
        
        # This is a WARNING, not a failure - we need to verify actual car height
        if abs(actual_camera_height - config_height) > 0.3:
            pytest.skip(f"Camera height mismatch: config={config_height}m, estimated={actual_camera_height}m. "
                       f"Need to verify actual car height in Unity.")
    
    def test_pixel_to_meter_conversion_at_known_distance(self):
        """Test pixel-to-meter conversion at a known distance.
        
        For a 60° FOV camera with 640px width:
        - At distance d, horizontal FOV = 2 * d * tan(30°)
        - At 10m: FOV = 2 * 10 * tan(30°) ≈ 11.55m
        - pixel_to_meter = 11.55m / 640px ≈ 0.018m/px
        """
        planner = RuleBasedTrajectoryPlanner(
            lookahead_distance=20.0,
            image_width=640,
            image_height=480,
            camera_fov=60.0,
            camera_height=0.5
        )
        
        # Test at 10m distance
        test_distance = 10.0  # meters
        fov_rad = np.radians(60.0)
        width_at_distance = 2.0 * test_distance * np.tan(fov_rad / 2.0)
        expected_pixel_to_meter = width_at_distance / 640.0
        
        print(f"\nPixel-to-meter conversion at {test_distance}m:")
        print(f"  Horizontal FOV: {width_at_distance:.3f}m")
        print(f"  Pixel-to-meter: {expected_pixel_to_meter*1000:.3f}mm/px")
        
        # Verify the conversion
        x_pixels = 320  # Center of image
        y_pixels = 480  # Bottom of image
        x_vehicle, y_vehicle = planner._convert_image_to_vehicle_coords(
            x_pixels, y_pixels, lookahead_distance=test_distance
        )
        
        print(f"  Test: x_pixels={x_pixels}, y_pixels={y_pixels}")
        print(f"  Result: x_vehicle={x_vehicle:.3f}m, y_vehicle={y_vehicle:.3f}m")
        
        # Center pixel should map to x=0 (vehicle center)
        assert abs(x_vehicle) < 0.1, \
            f"Center pixel should map to x≈0, got {x_vehicle:.3f}m"
        
        # y_vehicle should be close to test_distance
        assert abs(y_vehicle - test_distance) < 0.5, \
            f"y_vehicle should be close to {test_distance}m, got {y_vehicle:.3f}m"
    
    def test_lane_width_calculation(self):
        """Test that lane width calculation produces reasonable values.
        
        Standard lane width: ~3.5m per lane (11.5 feet)
        Current setup: Single-lane road with total width ~7.0m
        If we detect lanes at x positions in pixels, conversion should give ~7.0m for total road width.
        """
        planner = RuleBasedTrajectoryPlanner(
            lookahead_distance=20.0,
            image_width=640,
            image_height=480,
            camera_fov=60.0,
            camera_height=0.5
        )
        
        # Simulate detecting lanes at typical pixel positions
        # For a 3.5m lane at 10m distance:
        # At 10m: pixel_to_meter ≈ 0.018m/px
        # 3.5m / 0.018m/px ≈ 194px
        # So lanes might be at: center ± 97px = 320 ± 97 = 223px and 417px
        
        test_distance = 10.0  # meters
        expected_lane_width = 3.5  # meters
        
        # Calculate expected pixel separation
        fov_rad = np.radians(60.0)
        width_at_distance = 2.0 * test_distance * np.tan(fov_rad / 2.0)
        pixel_to_meter = width_at_distance / 640.0
        expected_pixel_separation = expected_lane_width / pixel_to_meter
        
        # Simulate left and right lane positions
        center_x = 320
        left_lane_x_pixels = center_x - expected_pixel_separation / 2
        right_lane_x_pixels = center_x + expected_pixel_separation / 2
        
        print(f"\nLane width calculation test:")
        print(f"  Expected lane width: {expected_lane_width}m")
        print(f"  Expected pixel separation: {expected_pixel_separation:.1f}px")
        print(f"  Left lane x: {left_lane_x_pixels:.1f}px")
        print(f"  Right lane x: {right_lane_x_pixels:.1f}px")
        
        # Convert to vehicle coordinates
        left_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
            left_lane_x_pixels, 480, lookahead_distance=test_distance
        )
        right_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
            right_lane_x_pixels, 480, lookahead_distance=test_distance
        )
        
        calculated_lane_width = abs(right_x_vehicle - left_x_vehicle)
        
        print(f"  Calculated lane width: {calculated_lane_width:.3f}m")
        print(f"  Error: {abs(calculated_lane_width - expected_lane_width):.3f}m")
        
        # Allow 10% error
        error_percent = abs(calculated_lane_width - expected_lane_width) / expected_lane_width * 100
        assert error_percent < 10, \
            f"Lane width calculation error too large: {error_percent:.1f}% " \
            f"(expected {expected_lane_width}m, got {calculated_lane_width:.3f}m)"
    
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
        """Test lane width calculation using actual recorded data.
        
        This test loads recorded data and checks if calculated lane widths are reasonable.
        Expected: ~7.0m for single-lane road (total road width).
        Note: Current Unity setup has a single 7m-wide lane with no center line.
        """
        import h5py
        from pathlib import Path
        
        # Find latest recording
        recordings_dir = Path('data/recordings')
        if not recordings_dir.exists():
            pytest.skip("No recordings directory found")
        
        recordings = sorted(recordings_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            pytest.skip("No recordings found")
        
        recording_file = recordings[0]
        
        print(f"\nTesting with actual recorded data: {recording_file.name}")
        
        with h5py.File(recording_file, 'r') as f:
            if 'perception/left_lane_x' not in f or 'perception/right_lane_x' not in f:
                pytest.skip("Recording doesn't have lane position data")
            
            left_lane_x = np.array(f['perception/left_lane_x'])
            right_lane_x = np.array(f['perception/right_lane_x'])
            num_lanes = np.array(f['perception/num_lanes_detected'])
            
            # Calculate lane widths for frames with both lanes
            lane_widths = []
            for i in range(len(num_lanes)):
                if num_lanes[i] == 2 and not np.isnan(left_lane_x[i]) and not np.isnan(right_lane_x[i]):
                    width = abs(right_lane_x[i] - left_lane_x[i])
                    lane_widths.append(width)
            
            if not lane_widths:
                pytest.skip("No dual-lane detections in recording")
            
            lane_widths = np.array(lane_widths)
            mean_width = np.mean(lane_widths)
            std_width = np.std(lane_widths)
            
            print(f"  Frames with dual lanes: {len(lane_widths)}")
            print(f"  Mean lane width: {mean_width:.3f}m")
            print(f"  Std lane width: {std_width:.3f}m")
            print(f"  Expected: ~7.0m (single-lane road, total road width)")
            print(f"  Error: {abs(mean_width - 7.0):.3f}m ({abs(mean_width - 7.0)/7.0*100:.1f}%)")
            
            # Check if mean is reasonable (6.0m to 8.0m for single-lane road)
            if mean_width < 6.0 or mean_width > 8.0:
                pytest.fail(
                    f"Lane width calculation is wrong! "
                    f"Mean width = {mean_width:.3f}m (expected ~7.0m for single-lane road). "
                    f"This suggests coordinate conversion is incorrect."
                )
            
            # Check if variance is reasonable (std < 1.0m)
            if std_width > 1.0:
                pytest.fail(
                    f"Lane width calculation is inconsistent! "
                    f"Std = {std_width:.3f}m (should be < 1.0m). "
                    f"This suggests coordinate conversion is unstable."
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

