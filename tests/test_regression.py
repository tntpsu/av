"""
Regression tests to verify fixes don't make things worse.

These tests compare before/after metrics to catch performance degradation.
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDriftRateRegression:
    """Test that fixes don't make drift rate worse."""
    
    def test_camera_offset_doesnt_increase_drift(self):
        """
        CRITICAL REGRESSION TEST: Verify camera offset fix doesn't make drift worse.
        
        This test would catch if camera offset direction fix makes drift worse.
        Requires baseline recording (before fix) and current recording (after fix).
        """
        recordings = sorted(
            Path('data/recordings').glob('*.h5'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if len(recordings) < 2:
            pytest.skip("Need at least 2 recordings to compare (before/after fix)")
        
        # Get latest (after fix) and previous (before fix)
        latest = recordings[0]
        previous = recordings[1]
        
        with h5py.File(latest, 'r') as f_latest:
            world_x_latest = f_latest['vehicle/position'][:, 0]
            timestamps_latest = f_latest['vehicle/timestamps'][:]
            duration_latest = timestamps_latest[-1] - timestamps_latest[0]
            drift_rate_latest = (world_x_latest[-1] - world_x_latest[0]) / duration_latest if duration_latest > 0 else 0.0
        
        with h5py.File(previous, 'r') as f_prev:
            world_x_prev = f_prev['vehicle/position'][:, 0]
            timestamps_prev = f_prev['vehicle/timestamps'][:]
            duration_prev = timestamps_prev[-1] - timestamps_prev[0]
            drift_rate_prev = (world_x_prev[-1] - world_x_prev[0]) / duration_prev if duration_prev > 0 else 0.0
        
        # Drift rate should not increase (get worse) after fix
        # Allow 10% tolerance for measurement variance
        drift_increase = abs(drift_rate_latest) - abs(drift_rate_prev)
        tolerance = abs(drift_rate_prev) * 0.1
        
        assert drift_increase <= tolerance, (
            f"Drift rate got WORSE after fix! "
            f"Previous: {drift_rate_prev:.4f} m/s, "
            f"Latest: {drift_rate_latest:.4f} m/s, "
            f"Increase: {drift_increase:.4f} m/s. "
            f"This indicates the fix made things worse."
        )
    
    def test_reference_point_variance_doesnt_increase(self):
        """
        REGRESSION TEST: Verify fixes don't increase reference point variance.
        
        High variance can prevent adaptive corrections from triggering.
        """
        recordings = sorted(
            Path('data/recordings').glob('*.h5'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if len(recordings) < 2:
            pytest.skip("Need at least 2 recordings to compare")
        
        latest = recordings[0]
        previous = recordings[1]
        
        with h5py.File(latest, 'r') as f_latest:
            if 'trajectory/reference_point_x' not in f_latest:
                pytest.skip("No reference point data in latest recording")
            ref_x_latest = f_latest['trajectory/reference_point_x'][:]
            variance_latest = np.std(ref_x_latest)
        
        with h5py.File(previous, 'r') as f_prev:
            if 'trajectory/reference_point_x' not in f_prev:
                pytest.skip("No reference point data in previous recording")
            ref_x_prev = f_prev['trajectory/reference_point_x'][:]
            variance_prev = np.std(ref_x_prev)
        
        # Variance should not increase significantly (allow 20% increase)
        variance_increase = variance_latest - variance_prev
        tolerance = variance_prev * 0.2
        
        assert variance_increase <= tolerance, (
            f"Reference point variance increased significantly! "
            f"Previous: {variance_prev:.4f}m, "
            f"Latest: {variance_latest:.4f}m, "
            f"Increase: {variance_increase:.4f}m. "
            f"High variance can prevent adaptive corrections from triggering."
        )


class TestFixEffectiveness:
    """Test that fixes actually improve system behavior."""
    
    def test_camera_offset_reduces_bias(self):
        """
        EFFECTIVENESS TEST: Verify camera offset reduces reference point bias.
        
        This test verifies that the fix actually works, not just that code exists.
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
                pytest.skip("No reference point data")
            
            ref_x = f['trajectory/reference_point_x'][:]
            bias_mean = np.mean(ref_x)
            bias_abs = abs(bias_mean)
            
            # With camera offset, bias should be small (< 0.05m)
            # This verifies the fix is effective
            assert bias_abs < 0.05, (
                f"Camera offset should reduce bias. "
                f"Current bias: {bias_mean:.4f}m, expected < 0.05m. "
                f"This indicates camera offset is not effective."
            )

