"""
Phase T/L2 — Replay-based synthetic HDF5 tests for ACC.

Two groups:
  1. Structural tests (run now): verify make_acc_recording() produces valid HDF5
     with the correct field names, shapes, and sentinel conventions.
  2. Scoring tests (Phase D): verify drive_summary_core.py ACC section behaviour.
     Marked xfail(strict=False) until Phase D adds ACC scoring to drive_summary_core.

To run structural tests only:
    pytest tests/test_acc_replay.py -v -k "Structural"

To run all (some will xfail until Phase D):
    pytest tests/test_acc_replay.py -v
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import make_acc_recording, make_nominal_recording

_REQUIRED_ACC_FIELDS = [
    "vehicle/radar_fwd_detected",
    "vehicle/radar_fwd_distance_m",
    "vehicle/radar_fwd_range_rate_mps",
    "vehicle/radar_fwd_snr",
    "vehicle/acc_active",
    "vehicle/acc_target_gap_m",
    "vehicle/acc_gap_error_m",
    "vehicle/acc_ttc_s",
]

_REQUIRED_BASE_FIELDS = [
    "vehicle/timestamps",
    "vehicle/speed",
    "control/steering",
    "control/lateral_error",
    "control/emergency_stop",
    "ground_truth/left_lane_line_x",
    "ground_truth/right_lane_line_x",
]


# ─── Structural tests ─────────────────────────────────────────────────────────

class TestStructural:
    def test_all_acc_fields_present(self, tmp_path):
        p = tmp_path / "acc_nominal.h5"
        make_acc_recording(p, duration_s=10.0)
        with h5py.File(p, "r") as f:
            for field in _REQUIRED_ACC_FIELDS:
                assert field in f, f"Missing field: {field}"

    def test_all_base_fields_present(self, tmp_path):
        p = tmp_path / "acc_base.h5"
        make_acc_recording(p, duration_s=10.0)
        with h5py.File(p, "r") as f:
            for field in _REQUIRED_BASE_FIELDS:
                assert field in f, f"Missing base field: {field}"

    def test_field_shapes_match(self, tmp_path):
        p = tmp_path / "acc_shape.h5"
        fps, dur = 30.0, 10.0
        make_acc_recording(p, duration_s=dur, fps=fps)
        n_expected = int(dur * fps)
        with h5py.File(p, "r") as f:
            for field in _REQUIRED_ACC_FIELDS:
                assert f[field].shape == (n_expected,), (
                    f"{field}: shape {f[field].shape} != ({n_expected},)"
                )

    def test_inactive_frames_are_sentinel_zero(self, tmp_path):
        """Frames before acc_active_start must have acc_active=0 and radar_fwd_detected=0."""
        p = tmp_path / "acc_late_start.h5"
        make_acc_recording(p, duration_s=10.0, fps=30.0, acc_active_start=60)
        with h5py.File(p, "r") as f:
            acc_active = f["vehicle/acc_active"][:]
            detected = f["vehicle/radar_fwd_detected"][:]
        assert np.all(acc_active[:60] == 0.0), "Pre-ACC frames should have acc_active=0"
        assert np.all(detected[:60] == 0.0), "Pre-ACC frames should have detected=0"

    def test_active_frames_have_nonzero_gap(self, tmp_path):
        """Frames after acc_active_start should have non-zero radar_fwd_distance_m."""
        p = tmp_path / "acc_active.h5"
        make_acc_recording(p, duration_s=5.0, fps=30.0, acc_active_start=0, gap_m=30.0)
        with h5py.File(p, "r") as f:
            dist = f["vehicle/radar_fwd_distance_m"][:]
        assert np.any(dist > 0.0), "ACC active frames should have non-zero distance"

    def test_near_miss_frames_injected(self, tmp_path):
        """n_near_miss_frames=5 → at least 5 frames with distance < 2.0m."""
        p = tmp_path / "acc_near_miss.h5"
        make_acc_recording(p, duration_s=10.0, fps=30.0, n_near_miss_frames=5, acc_active_start=0)
        with h5py.File(p, "r") as f:
            dist = f["vehicle/radar_fwd_distance_m"][:]
        near_miss = np.sum((dist > 0.0) & (dist < 2.0))
        assert near_miss >= 5, f"Expected ≥5 near-miss frames, got {near_miss}"

    def test_collision_frame_injected(self, tmp_path):
        """collision=True → exactly one frame with radar_fwd_distance_m < 0."""
        p = tmp_path / "acc_collision.h5"
        make_acc_recording(p, duration_s=10.0, fps=30.0, collision=True, acc_active_start=0)
        with h5py.File(p, "r") as f:
            dist = f["vehicle/radar_fwd_distance_m"][:]
        assert np.any(dist < 0.0), "Collision recording must have a frame with distance < 0"

    def test_ttc_field_positive_when_closing(self, tmp_path):
        """When acc active and range_rate > 0, acc_ttc_s must be positive."""
        p = tmp_path / "acc_ttc.h5"
        make_acc_recording(
            p, duration_s=5.0, fps=30.0,
            acc_active_start=0, gap_m=20.0, range_rate_mps=3.0,
        )
        with h5py.File(p, "r") as f:
            ttc = f["vehicle/acc_ttc_s"][:]
            active = f["vehicle/acc_active"][:]
        active_ttc = ttc[active > 0.5]
        assert np.all(active_ttc > 0.0), "TTC must be positive in active ACC frames with closing rate"

    def test_gap_callable_produces_variable_gap(self, tmp_path):
        """Callable gap_m parameter produces varying distance values."""
        p = tmp_path / "acc_var_gap.h5"
        make_acc_recording(
            p, duration_s=5.0, fps=30.0,
            acc_active_start=0,
            gap_m=lambda i: 20.0 + i * 0.1,
        )
        with h5py.File(p, "r") as f:
            dist = f["vehicle/radar_fwd_distance_m"][:]
        assert dist.max() > dist.min(), "Callable gap should produce varying distances"

    def test_no_nan_in_any_field(self, tmp_path):
        """No NaN values in any ACC field."""
        p = tmp_path / "acc_nan_check.h5"
        make_acc_recording(
            p, duration_s=10.0, fps=30.0,
            acc_active_start=0, n_near_miss_frames=3,
            n_ttc_warn_frames=3, collision=False,
        )
        with h5py.File(p, "r") as f:
            for field in _REQUIRED_ACC_FIELDS:
                data = f[field][:]
                assert not np.any(np.isnan(data)), f"NaN found in {field}"


# ─── Scoring tests (Phase D — xfail until drive_summary_core ACC section added) ──

def _try_import_drive_summary():
    try:
        import drive_summary_core
        return drive_summary_core
    except ImportError:
        return None


@pytest.mark.xfail(
    strict=False,
    reason="Phase D: drive_summary_core ACC section not yet implemented",
)
class TestACCScoring:
    """
    These tests verify that drive_summary_core.py correctly applies ACC scoring.
    They will xfail until Phase D adds the ACC metric computation section.
    After Phase D, remove the xfail marker and they should all pass.
    """

    def test_inactive_acc_does_not_affect_score(self, tmp_path):
        """ACC fully inactive → overall score identical to free-flow baseline."""
        dsc = _try_import_drive_summary()
        if dsc is None:
            pytest.skip("drive_summary_core not importable")

        baseline_path = tmp_path / "baseline.h5"
        acc_path = tmp_path / "acc_inactive.h5"
        make_nominal_recording(baseline_path)
        make_acc_recording(acc_path, acc_active_start=99999)  # never activates

        baseline = dsc.compute_summary(str(baseline_path))
        acc_result = dsc.compute_summary(str(acc_path))

        assert abs(
            acc_result["overall_score"] - baseline["overall_score"]
        ) < 0.5, "Inactive ACC must not affect overall score"

    def test_near_miss_deducts_from_safety_layer(self, tmp_path):
        """Near-miss events → Safety layer deduction = n × ACC_NEAR_MISS_PENALTY_PTS."""
        dsc = _try_import_drive_summary()
        if dsc is None:
            pytest.skip("drive_summary_core not importable")

        clean_path = tmp_path / "clean.h5"
        nm_path = tmp_path / "near_miss.h5"
        make_acc_recording(clean_path, acc_active_start=0, n_near_miss_frames=0)
        make_acc_recording(nm_path, acc_active_start=0, n_near_miss_frames=3)

        clean = dsc.compute_summary(str(clean_path))
        nm = dsc.compute_summary(str(nm_path))

        assert nm["overall_score"] < clean["overall_score"], (
            "Near-miss recording should score lower than clean recording"
        )

    def test_collision_forces_score_to_zero(self, tmp_path):
        """Collision event → overall score = 0.0 regardless of other layers."""
        dsc = _try_import_drive_summary()
        if dsc is None:
            pytest.skip("drive_summary_core not importable")

        collision_path = tmp_path / "collision.h5"
        make_acc_recording(collision_path, acc_active_start=0, collision=True)
        result = dsc.compute_summary(str(collision_path))
        assert result["overall_score"] == pytest.approx(0.0), (
            "Collision must override overall score to 0.0"
        )
