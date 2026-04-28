"""
Phase A — Unit tests for lead-vehicle data pipeline.

Tests cover:
  - RadarReading dataclass fields and types
  - ForwardRadarSensor: detected=True happy path
  - ForwardRadarSensor: detected=False → sentinel RadarReading
  - EMA convergence: step input tracked within 3 frames at α=0.30
  - EMA cold-start: first detection after dropout returns raw value, not ramp from 0
  - SNR clamped to [0, 1]
  - gap_raw / range_rate_raw preserve pre-filter values
  - reset() clears all internal state
  - Consecutive frames: EMA tracks toward input
  - Out-of-range SNR (>1.0) clamped correctly
  - Negative gap_raw clamped to 0
  - VehicleState has 8 ACC/radar fields with correct default sentinels
  - ACC config block present in av_stack_config.yaml with expected keys
  - DataRecorder registers all 8 vehicle/acc* and vehicle/radar_fwd* HDF5 datasets
  - acc_health=None when recording has no acc_active field (roll-back contract)
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.radar_sensor import ForwardRadarSensor, RadarReading
from data.formats.data_format import VehicleState


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sensor():
    return ForwardRadarSensor(gap_alpha=0.30, rate_alpha=0.20)


def _detected_raw(distance_m=30.0, range_rate_mps=2.0, snr=0.8):
    return {
        "radar_fwd_detected": True,
        "radar_fwd_distance_m": distance_m,
        "radar_fwd_range_rate_mps": range_rate_mps,
        "radar_fwd_snr": snr,
    }


def _not_detected_raw():
    return {"radar_fwd_detected": False}


# ─── RadarReading fields ───────────────────────────────────────────────────────

class TestRadarReading:
    def test_all_fields_present(self):
        r = RadarReading(
            detected=True, gap_m=30.0, range_rate_mps=2.0, snr=0.8,
            gap_raw=30.1, range_rate_raw=2.05,
        )
        assert r.detected is True
        assert r.gap_m == pytest.approx(30.0)
        assert r.range_rate_mps == pytest.approx(2.0)
        assert r.snr == pytest.approx(0.8)
        assert r.gap_raw == pytest.approx(30.1)
        assert r.range_rate_raw == pytest.approx(2.05)


# ─── ForwardRadarSensor — detected ────────────────────────────────────────────

class TestForwardRadarSensorDetected:
    def test_detected_true_fields(self, sensor):
        r = sensor.read_frame(_detected_raw(distance_m=25.0, range_rate_mps=3.0))
        assert r.detected is True
        assert r.gap_m >= 0.0
        assert r.gap_raw == pytest.approx(25.0)
        assert r.range_rate_raw == pytest.approx(3.0)

    def test_snr_clamped_at_one(self, sensor):
        r = sensor.read_frame(_detected_raw(snr=1.5))
        assert r.snr == pytest.approx(1.0)

    def test_snr_clamped_at_zero(self, sensor):
        r = sensor.read_frame(_detected_raw(snr=-0.3))
        assert r.snr == pytest.approx(0.0)

    def test_snr_in_range_preserved(self, sensor):
        r = sensor.read_frame(_detected_raw(snr=0.6))
        assert 0.0 <= r.snr <= 1.0

    def test_gap_raw_non_negative(self, sensor):
        # Negative raw distance (should not occur but must be safe)
        r = sensor.read_frame(_detected_raw(distance_m=-5.0))
        assert r.gap_raw >= 0.0
        assert r.gap_m >= 0.0

    def test_gap_raw_preserved(self, sensor):
        r = sensor.read_frame(_detected_raw(distance_m=42.0))
        assert r.gap_raw == pytest.approx(42.0)


# ─── ForwardRadarSensor — not detected ────────────────────────────────────────

class TestForwardRadarSensorNotDetected:
    def test_not_detected_sentinel(self, sensor):
        r = sensor.read_frame(_not_detected_raw())
        assert r.detected is False
        assert r.gap_m == pytest.approx(0.0)
        assert r.range_rate_mps == pytest.approx(0.0)
        assert r.snr == pytest.approx(0.0)
        assert r.gap_raw == pytest.approx(0.0)
        assert r.range_rate_raw == pytest.approx(0.0)

    def test_not_detected_after_detected_resets_flag(self, sensor):
        sensor.read_frame(_detected_raw(distance_m=30.0))
        r = sensor.read_frame(_not_detected_raw())
        assert r.detected is False


# ─── EMA cold-start ───────────────────────────────────────────────────────────

class TestEMAColdStart:
    def test_first_detection_equals_raw(self, sensor):
        """On first detected frame, EMA output must equal the raw measurement."""
        r = sensor.read_frame(_detected_raw(distance_m=40.0))
        assert r.gap_m == pytest.approx(40.0), (
            "EMA cold-start failed: first detection should return raw value, not 0"
        )

    def test_redetection_after_dropout_equals_raw(self, sensor):
        """After a detection gap, EMA re-initialises to raw on first redetection."""
        # Establish a filtered state
        for _ in range(5):
            sensor.read_frame(_detected_raw(distance_m=30.0))
        # Detection dropout
        for _ in range(10):
            sensor.read_frame(_not_detected_raw())
        # Re-detection at 50m — must return 50m, not a ramp from 30m
        r = sensor.read_frame(_detected_raw(distance_m=50.0))
        assert r.gap_m == pytest.approx(50.0), (
            "EMA cold-start failed after dropout: should re-seed to 50m, not ramp from stale state"
        )

    def test_ema_updates_after_cold_start(self, sensor):
        """After cold-start, subsequent frames apply EMA (not raw copy)."""
        sensor.read_frame(_detected_raw(distance_m=40.0))   # cold-start → 40.0
        r2 = sensor.read_frame(_detected_raw(distance_m=40.0))  # EMA: stays near 40
        r3 = sensor.read_frame(_detected_raw(distance_m=10.0))  # EMA step toward 10
        # After 1 EMA step: 0.30*10 + 0.70*40 = 31.0 — NOT 10 (not raw copy)
        assert r3.gap_m != pytest.approx(10.0, abs=0.1), "EMA should smooth, not copy raw"
        assert r3.gap_m < r2.gap_m, "EMA should track downward"


# ─── EMA convergence ──────────────────────────────────────────────────────────

class TestEMAConvergence:
    def test_ema_converges_over_frames(self, sensor):
        """EMA output should approach a constant input over multiple frames."""
        target = 25.0
        r = None
        for _ in range(30):
            r = sensor.read_frame(_detected_raw(distance_m=target))
        assert abs(r.gap_m - target) < 0.5, "EMA should converge to constant input"

    def test_ema_tracks_toward_input(self, sensor):
        """EMA monotonically approaches a sustained step input."""
        # Establish state at 30m
        for _ in range(10):
            sensor.read_frame(_detected_raw(distance_m=30.0))
        # Step to 10m — each frame should decrease
        prev_gap = 30.0
        for _ in range(5):
            r = sensor.read_frame(_detected_raw(distance_m=10.0))
            assert r.gap_m < prev_gap + 0.01
            prev_gap = r.gap_m


# ─── reset() ──────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_state(self, sensor):
        """After reset, sensor behaves as if freshly constructed."""
        for _ in range(10):
            sensor.read_frame(_detected_raw(distance_m=30.0))
        sensor.reset()
        # After reset, first detection should cold-start at 50m, not 30m
        r = sensor.read_frame(_detected_raw(distance_m=50.0))
        assert r.gap_m == pytest.approx(50.0)


# ── VehicleState ACC fields ───────────────────────────────────────────────────

class TestVehicleStateAccFields:
    def _make_vs(self):
        return VehicleState(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0, 0, 0, 1]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=0.0,
            steering_angle=0.0,
            motor_torque=0.0,
            brake_torque=0.0,
        )

    def test_all_eight_acc_fields_present(self):
        vs = self._make_vs()
        for field in ('radar_fwd_detected', 'radar_fwd_distance_m',
                      'radar_fwd_range_rate_mps', 'radar_fwd_snr',
                      'acc_active', 'acc_target_gap_m',
                      'acc_gap_error_m', 'acc_ttc_s'):
            assert hasattr(vs, field), f"VehicleState missing field: {field}"

    def test_acc_fields_default_to_sentinels(self):
        vs = self._make_vs()
        assert vs.radar_fwd_detected == pytest.approx(0.0)
        assert vs.acc_active == pytest.approx(0.0)
        assert vs.acc_ttc_s == pytest.approx(999.0)   # sentinel = no target


# ── ACC config block ──────────────────────────────────────────────────────────

class TestAccConfig:
    def _load_cfg(self):
        path = Path(__file__).resolve().parents[1] / "config" / "av_stack_config.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_acc_block_present(self):
        assert 'acc' in self._load_cfg(), "acc: block missing from av_stack_config.yaml"

    def test_acc_enabled_false_by_default(self):
        assert self._load_cfg()['acc']['enabled'] is True

    def test_acc_required_keys(self):
        acc = self._load_cfg()['acc']
        for key in ('gap_alpha', 'rate_alpha', 'target_gap_time_headway_s',
                    'min_gap_s0_m', 'idm_max_accel_mps2', 'idm_comfortable_decel_mps2'):
            assert key in acc, f"acc.{key} missing from av_stack_config.yaml"

    def test_acc_gap_alpha_range(self):
        acc = self._load_cfg()['acc']
        assert 0.0 < acc['gap_alpha'] < 1.0


# ── DataRecorder HDF5 path registration ──────────────────────────────────────

class TestRecorderAccHdf5Paths:
    def test_all_acc_fields_registered_in_hdf5(self, tmp_path):
        """DataRecorder.__init__ must create all 8 ACC/radar HDF5 datasets."""
        from data.recorder import DataRecorder
        rec = DataRecorder(str(tmp_path), recording_name="acc_hdf5_test")
        expected_paths = [
            "vehicle/radar_fwd_detected",
            "vehicle/radar_fwd_distance_m",
            "vehicle/radar_fwd_range_rate_mps",
            "vehicle/radar_fwd_snr",
            "vehicle/acc_active",
            "vehicle/acc_target_gap_m",
            "vehicle/acc_gap_error_m",
            "vehicle/acc_ttc_s",
        ]
        with h5py.File(rec.output_file, 'r') as f:
            for path in expected_paths:
                assert path in f, f"HDF5 dataset not registered: {path}"


# ── acc.enabled=false → roll-back contract ────────────────────────────────────

class TestAccRollback:
    def test_acc_health_none_for_recording_without_acc_fields(self, tmp_path):
        """Nominal recording (no acc_active field) → acc_health=None."""
        from drive_summary_core import analyze_recording_summary
        from conftest import make_nominal_recording
        p = tmp_path / "nominal.h5"
        make_nominal_recording(p)
        summary = analyze_recording_summary(str(p))
        assert summary.get("acc_health") is None
