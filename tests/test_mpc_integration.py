"""
Integration tests for Phase 2.3: VehicleController + MPC + Recorder wiring.

Tests verify:
- MPC-disabled (PP-only) mode is backward-compatible
- MPC-enabled mode switches regimes correctly
- All 9 new fields propagate through compute_control result dict
- HDF5 recorder creates and fills MPC datasets
- Config loading for mpc_sloop.yaml and mpc_highway.yaml

Run:  pytest tests/test_mpc_integration.py -v
Gate: 9/9 pass
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent


def _base_cfg(regime_enabled: bool = False, **mpc_overrides) -> dict:
    """Build minimal config dict for VehicleController integration tests."""
    mpc = {
        "horizon_steps": 10,
        "dt": 0.033,
        "wheelbase_m": 2.5,
        "q_lat": 10.0,
        "q_heading": 5.0,
        "q_speed": 2.0,
        "r_steer": 0.5,
        "r_accel": 0.2,
        "r_steer_rate": 1.0,
        "r_accel_rate": 0.1,
        "max_lat_error_m": 2.0,
        "max_heading_error_rad": 0.5,
        "max_accel_mps2": 3.0,
        "max_decel_mps2": 4.0,
        "max_steer_norm": 0.7,
        "max_steer_rate_per_step": 0.15,
        "max_consecutive_failures": 5,
        "min_horizon_steps": 5,
        "horizon_speed_scale": 1.0,
    }
    mpc.update(mpc_overrides)

    return {
        "control": {
            "lateral": {
                "control_mode": "pure_pursuit",
                "max_steering": 0.7,
                "pp_max_steering_jerk": 18.0,
                "pp_min_lookahead": 0.5,
                "pp_max_steering_rate": 0.4,
            },
            "longitudinal": {
                "target_speed": 12.0,
                "max_speed": 12.0,
                "kp": 0.25,
                "ki": 0.02,
                "kd": 0.01,
                "max_accel": 1.2,
                "max_decel": 2.4,
                "max_jerk": 0.7,
            },
            "regime": {
                "enabled": regime_enabled,
                "pp_max_speed_mps": 10.0,
                "upshift_hysteresis_mps": 1.0,
                "downshift_hysteresis_mps": 1.0,
                "blend_frames": 15,
                "min_hold_frames": 2,  # short for tests
            },
        },
        "trajectory": {"mpc": mpc},
        "safety": {"max_speed": 12.0},
    }


def _make_vehicle_controller(regime_enabled: bool = False, **mpc_overrides):
    """Instantiate VehicleController with optional MPC enabled."""
    from control.pid_controller import VehicleController
    cfg = _base_cfg(regime_enabled=regime_enabled, **mpc_overrides)
    return VehicleController(full_config=cfg)


def _state(speed: float = 8.0) -> dict:
    return {"heading": 0.0, "speed": speed, "position": (0.0, 0.0)}


def _ref(speed: float = 12.0) -> dict:
    return {
        "x": 0.0,
        "y": 5.0,
        "heading": 0.0,
        "velocity": speed,
        "curvature": 0.0,
    }


# ---------------------------------------------------------------------------
# Tests: PP-only mode (MPC disabled)
# ---------------------------------------------------------------------------


class TestMPCDisabled:
    """MPC disabled — must remain backward-compatible."""

    def test_returns_steering_throttle_brake(self):
        vc = _make_vehicle_controller(regime_enabled=False)
        result = vc.compute_control(_state(), _ref())
        assert "steering" in result
        assert "throttle" in result
        assert "brake" in result

    def test_mpc_fields_zero_filled_in_metadata(self):
        vc = _make_vehicle_controller(regime_enabled=False)
        result = vc.compute_control(_state(), _ref(), return_metadata=True)
        assert result["mpc_feasible"] is False
        assert result["mpc_solve_time_ms"] == 0.0
        assert result["mpc_e_lat"] == 0.0
        assert result["mpc_e_heading"] == 0.0
        assert result["mpc_kappa_ref"] == 0.0
        assert result["mpc_fallback_active"] is False
        assert result["mpc_consecutive_failures"] == 0

    def test_regime_stays_pure_pursuit(self):
        vc = _make_vehicle_controller(regime_enabled=False)
        # Even at high speed, regime selector is disabled
        result = vc.compute_control(_state(speed=14.0), _ref(), return_metadata=True)
        assert result["regime"] == 0  # PURE_PURSUIT

    def test_blend_weight_is_one_in_pp_mode(self):
        vc = _make_vehicle_controller(regime_enabled=False)
        result = vc.compute_control(_state(), _ref(), return_metadata=True)
        assert result["regime_blend_weight"] == 1.0


# ---------------------------------------------------------------------------
# Tests: MPC enabled mode
# ---------------------------------------------------------------------------


class TestMPCEnabled:
    """MPC enabled — regime switching and field propagation."""

    def test_mpc_controller_initialized(self):
        vc = _make_vehicle_controller(regime_enabled=True)
        assert vc._mpc_controller is not None

    def test_low_speed_stays_pp(self):
        """Below pp_max_speed_mps=10.0 → PURE_PURSUIT."""
        vc = _make_vehicle_controller(regime_enabled=True)
        result = vc.compute_control(_state(speed=6.0), _ref(), return_metadata=True)
        assert result["regime"] == 0  # PURE_PURSUIT

    def test_high_speed_shifts_to_mpc(self):
        """Above pp_max_speed_mps + upshift_hysteresis → MPC after hold."""
        vc = _make_vehicle_controller(regime_enabled=True)
        # Drive at 12 m/s for min_hold_frames+1 iterations to trigger upshift
        for _ in range(4):
            result = vc.compute_control(_state(speed=12.0), _ref(), return_metadata=True)
        assert result["regime"] == 1  # LINEAR_MPC

    def test_mpc_fields_present_in_metadata(self):
        vc = _make_vehicle_controller(regime_enabled=True)
        # Force MPC regime quickly
        for _ in range(4):
            result = vc.compute_control(_state(speed=12.0), _ref(), return_metadata=True)
        # All 9 fields must exist
        for field in [
            "mpc_feasible", "mpc_solve_time_ms", "mpc_e_lat",
            "mpc_e_heading", "mpc_kappa_ref", "mpc_fallback_active",
            "mpc_consecutive_failures", "regime", "regime_blend_weight",
        ]:
            assert field in result, f"Missing field: {field}"

    def test_steering_finite_in_mpc_mode(self):
        vc = _make_vehicle_controller(regime_enabled=True)
        for _ in range(4):
            result = vc.compute_control(_state(speed=12.0), _ref(), return_metadata=True)
        assert np.isfinite(result["steering"])

    def test_reset_clears_regime_state(self):
        vc = _make_vehicle_controller(regime_enabled=True)
        # Push to MPC
        for _ in range(4):
            vc.compute_control(_state(speed=12.0), _ref(), return_metadata=True)
        vc.reset()
        # After reset, low speed must be PP again
        result = vc.compute_control(_state(speed=5.0), _ref(), return_metadata=True)
        assert result["regime"] == 0  # PURE_PURSUIT back after reset


# ---------------------------------------------------------------------------
# Tests: HDF5 recorder MPC datasets
# ---------------------------------------------------------------------------


class TestRecorderMPCDatasets:
    """Verify recorder creates 9 new MPC datasets and writes data."""

    @pytest.fixture
    def rec_dir(self, tmp_path):
        """DataRecorder takes a directory — it creates recording_*.h5 inside."""
        return str(tmp_path)

    def _find_h5(self, directory: str) -> str:
        """Glob for the HDF5 file created by DataRecorder inside directory."""
        import glob
        files = glob.glob(f"{directory}/*.h5")
        assert files, f"No .h5 file found in {directory}"
        return files[0]

    def test_mpc_datasets_created(self, rec_dir):
        """All 9 MPC datasets exist in a freshly opened HDF5 file."""
        import h5py
        from data.recorder import DataRecorder

        rec = DataRecorder(rec_dir)
        rec.close()

        h5_path = self._find_h5(rec_dir)
        with h5py.File(h5_path, "r") as f:
            for key in [
                "control/mpc_feasible",
                "control/mpc_solve_time_ms",
                "control/mpc_e_lat",
                "control/mpc_e_heading",
                "control/mpc_kappa_ref",
                "control/mpc_fallback_active",
                "control/mpc_consecutive_failures",
                "control/regime",
                "control/regime_blend_weight",
            ]:
                assert key in f, f"Dataset missing: {key}"

    def test_mpc_datasets_filled_after_write(self, rec_dir):
        """Writing frames populates MPC datasets (not stuck at shape (0,))."""
        import h5py
        from data.recorder import DataRecorder
        from data.formats.data_format import ControlCommand, RecordingFrame

        rec = DataRecorder(rec_dir)

        # Build 3 synthetic frames with non-default MPC values
        frames = []
        for i in range(3):
            cc = ControlCommand(
                timestamp=float(i) * 0.033,
                steering=0.1 * i,
                throttle=0.5,
                brake=0.0,
                mpc_feasible=True,
                mpc_solve_time_ms=1.5 + i,
                mpc_e_lat=0.1 * i,
                mpc_e_heading=0.05 * i,
                mpc_kappa_ref=0.002,
                mpc_fallback_active=False,
                mpc_consecutive_failures=0,
                regime=1,
                regime_blend_weight=1.0,
            )
            frame = RecordingFrame(
                timestamp=float(i) * 0.033,
                frame_id=i,
                control_command=cc,
            )
            frames.append(frame)

        rec._write_control_commands(frames)
        rec.close()

        h5_path = self._find_h5(rec_dir)
        with h5py.File(h5_path, "r") as f:
            assert f["control/mpc_feasible"].shape[0] == 3
            assert f["control/regime"][:].tolist() == [1, 1, 1]
            np.testing.assert_allclose(
                f["control/mpc_solve_time_ms"][:], [1.5, 2.5, 3.5], rtol=1e-4
            )


# ---------------------------------------------------------------------------
# Tests: Config file validity
# ---------------------------------------------------------------------------


class TestConfigFiles:
    """Verify mpc_sloop.yaml and mpc_highway.yaml are valid and MPC-enabled."""

    def _load(self, name: str) -> dict:
        path = _PROJECT_ROOT / "config" / name
        assert path.exists(), f"Config not found: {path}"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_mpc_sloop_regime_enabled(self):
        cfg = self._load("mpc_sloop.yaml")
        assert cfg["control"]["regime"]["enabled"] is True

    def test_mpc_sloop_q_lat_tuned(self):
        cfg = self._load("mpc_sloop.yaml")
        assert cfg["trajectory"]["mpc"]["mpc_q_lat"] == pytest.approx(12.0)

    def test_mpc_highway_regime_enabled(self):
        cfg = self._load("mpc_highway.yaml")
        assert cfg["control"]["regime"]["enabled"] is True

    def test_mpc_highway_r_steer_rate_tuned(self):
        cfg = self._load("mpc_highway.yaml")
        assert cfg["trajectory"]["mpc"]["mpc_r_steer_rate"] == pytest.approx(2.0)
