"""
Tests using real recorded data to catch integration issues.
These tests use actual Unity recordings to validate behavior.
"""

import pytest
import numpy as np
import h5py
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from control.pid_controller import LateralController
from trajectory.inference import TrajectoryPlanningInference


class TestReferencePointValidation:
    """Validate reference points from real data."""
    
    def test_reference_points_in_meters(self):
        """Verify reference points are in meters, not pixels."""
        # Find latest recording
        recordings = sorted(
            Path('data/recordings').glob('*.h5'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not recordings:
            pytest.skip("No recordings available")
        
        latest = recordings[0]
        
        with h5py.File(latest, 'r') as f:
            ref_x = f['trajectory/reference_point_x'][:]
            
            # Reference points should be in meters
            # For a straight road, ref_x should be ~0m (center)
            # If ref_x is 300-500, it's in pixels!
            max_abs_ref_x = np.max(np.abs(ref_x))
            
            assert max_abs_ref_x < 10.0, (
                f"ref_x values look like pixels ({max_abs_ref_x:.1f}), not meters! "
                f"Expected < 10m, got {max_abs_ref_x:.1f}. "
                f"Coordinate conversion may not be applied!"
            )
            
            # Mean should be close to 0 for centered reference
            mean_ref_x = np.mean(ref_x)
            assert abs(mean_ref_x) < 2.0, (
                f"Mean ref_x should be ~0m (centered), got {mean_ref_x:.3f}m. "
                f"Reference point may be biased!"
            )
    
    def test_reference_heading_reasonable(self):
        """Verify reference heading is reasonable."""
        recordings = sorted(
            Path('data/recordings').glob('*.h5'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not recordings:
            pytest.skip("No recordings available")
        
        latest = recordings[0]
        
        with h5py.File(latest, 'r') as f:
            ref_heading = f['trajectory/reference_point_heading'][:]
            
            # For a straight road, heading should be ~0°
            # If heading is 90°, coordinate system is wrong!
            mean_heading_deg = np.degrees(np.mean(ref_heading))
            
            assert abs(mean_heading_deg) < 45.0, (
                f"Mean ref_heading should be ~0° (straight), got {mean_heading_deg:.1f}°. "
                f"Coordinate system or heading calculation may be wrong!"
            )


class TestSteeringDirectionWithRealData:
    """Test steering direction using real vehicle positions."""
    
    def test_steering_corrects_error(self):
        """Verify steering actually corrects lateral error."""
        recordings = sorted(
            Path('data/recordings').glob('*.h5'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not recordings:
            pytest.skip("No recordings available")
        
        latest = recordings[0]
        
        with h5py.File(latest, 'r') as f:
            positions = f['vehicle/position'][:]
            rotations = f['vehicle/rotation'][:]
            steerings = f['control/steering'][:]
            ref_x = f['trajectory/reference_point_x'][:]
            ref_y = f['trajectory/reference_point_y'][:]
            
            # Extract headings
            def extract_heading(rot):
                x, y, z, w = rot[0], rot[1], rot[2], rot[3]
                siny_cosp = 2.0 * (w * y + z * x)
                cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                return np.arctan2(siny_cosp, cosy_cosp)
            
            headings = np.array([extract_heading(rot) for rot in rotations])
            x_positions = positions[:, 0]
            
            # Calculate lateral errors
            # NOTE: ref_x is already in vehicle coordinates (lateral offset from vehicle center)
            # We should use ref_x directly, not ref_x - vehicle_x
            lateral_errors = []
            for i in range(min(len(positions), len(ref_x))):
                # ref_x is already the lateral error in vehicle frame
                # No need to subtract vehicle position or apply coordinate transformation
                lat_err = ref_x[i]
                lateral_errors.append(lat_err)
            
            lateral_errors = np.array(lateral_errors)
            
            # Check correlation
            if len(steerings) == len(lateral_errors):
                correlation = np.corrcoef(steerings, lateral_errors)[0, 1]
                
                # Correlation should be POSITIVE
                # ref_x > 0 (target RIGHT) → steering > 0 (steer RIGHT) → positive correlation
                # ref_x < 0 (target LEFT) → steering < 0 (steer LEFT) → positive correlation
                # NOTE: ref_x is already the lateral error in vehicle frame
                assert correlation > 0, (
                    f"Steering and lateral error correlation should be POSITIVE, "
                    f"got {correlation:.3f}. Steering direction may be inverted!"
                )
                
                # Check magnitude - correlation should be reasonably strong
                assert abs(correlation) > 0.3, (
                    f"Steering correlation is weak ({correlation:.3f}). "
                    f"Steering may not be responding to errors!"
                )
    


class TestCoordinateSystemValidation:
    """Validate coordinate system transformations."""
    
    def test_vehicle_position_reasonable(self):
        """Verify coordinate system is correct by checking lane-relative reference point.

        vehicle/position stores Unity world coordinates (hundreds of meters) and cannot
        distinguish pixels from world positions. Instead, check trajectory/reference_point_x
        which is lane-relative (should always be < 5m from lane center).
        If values are 300-500, coordinate conversion is broken.
        """
        recordings = sorted(
            Path('data/recordings').glob('*.h5'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not recordings:
            pytest.skip("No recordings available")

        latest = recordings[0]

        with h5py.File(latest, 'r') as f:
            if 'trajectory/reference_point_x' not in f:
                pytest.skip("No reference_point_x in recording")
            ref_x = f['trajectory/reference_point_x'][:]
            max_abs_ref_x = np.max(np.abs(ref_x))

            assert max_abs_ref_x < 5.0, (
                f"reference_point_x looks wrong ({max_abs_ref_x:.2f}m). "
                f"Expected lane-relative value < 5m. "
                f"If value is 300-500, coordinate conversion is not being applied!"
            )


class TestControlResponse:
    """Test that control actually works with real data."""
    
    
    def test_steering_saturation_not_excessive(self):
        """Verify steering is not saturated too often."""
        recordings = sorted(
            Path('data/recordings').glob('*.h5'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not recordings:
            pytest.skip("No recordings available")
        
        latest = recordings[0]
        
        with h5py.File(latest, 'r') as f:
            steerings = f['control/steering'][:]
            
            # Steering should not be saturated most of the time
            saturation_rate = np.sum(np.abs(steerings) > 0.45) / len(steerings)
            
            assert saturation_rate < 0.5, (
                f"Steering is saturated {saturation_rate*100:.1f}% of the time! "
                f"Expected < 50%, got {saturation_rate*100:.1f}%. "
                f"Gains may be too high or errors too large!"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

