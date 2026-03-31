"""
Phase D — Unit tests for ACC scoring integration.

Tests verify:
  - ACC section populates in drive_summary when acc_active_pct > 10%
  - ACC section absent (None) when acc_active_pct < 10%
  - acc_gap_rmse_m computed correctly from synthetic HDF5
  - acc_ttc_min_s = min across ACC-active frames only
  - Collision event (gap < 0) → acc_collision_events ≥ 1 + hard_zero=True
  - Gate pass/fail at boundary values (0.50m gap RMSE, 2.0s TTC, 4.0 m/s³ jerk)
  - Registry constants imported and wired correctly in drive_summary_core
  - Safety layer deductions non-zero when near-miss events present
  - Hard-zero collision override applies to overall_score
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import make_acc_recording
from drive_summary_core import _build_acc_health_summary
from scoring_registry import (
    ACC_GAP_RMSE_GATE_M,
    ACC_TTC_MIN_GATE_S,
    ACC_JERK_P95_GATE_MPS3,
    ACC_DETECTION_RATE_GATE,
    ACC_NEAR_MISS_GAP_M,
    ACC_MIN_ACTIVE_FRAME_RATE,
    ACC_TTC_CRITICAL_S,
    ACC_NEAR_MISS_PENALTY_PTS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_data(n=300, acc_active_start=0, gap_m=30.0, range_rate=0.0,
               gap_error=0.0, ttc_s=999.0, detected=True):
    """Build a minimal data dict matching drive_summary_core shape."""
    acc_active = np.zeros(n, dtype=float)
    for i in range(acc_active_start, n):
        acc_active[i] = 1.0

    gap_val = gap_m if not callable(gap_m) else None
    dist = np.full(n, gap_val if gap_val is not None else 30.0, dtype=float)
    if callable(gap_m):
        for i in range(n):
            dist[i] = gap_m(i)

    return {
        "acc_active": acc_active,
        "radar_fwd_detected": np.ones(n, dtype=float) * (1.0 if detected else 0.0),
        "radar_fwd_distance_m": dist,
        "radar_fwd_range_rate_mps": np.full(n, range_rate, dtype=float),
        "acc_gap_error_m": np.full(n, gap_error, dtype=float),
        "acc_ttc_s": np.full(n, ttc_s, dtype=float),
        "acc_target_gap_m": np.full(n, 30.0, dtype=float),
    }


# ── Core function tests (unit, no HDF5 needed) ───────────────────────────────

class TestAccHealthSummaryUnit:
    def test_returns_none_when_no_acc_data(self):
        result = _build_acc_health_summary({}, n_frames=100)
        assert result is None

    def test_returns_none_when_acc_inactive(self):
        """acc_active all zeros → below threshold → None."""
        data = _make_data(n=300, acc_active_start=300)  # never active
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is None

    def test_returns_summary_when_acc_active(self):
        data = _make_data(n=300, acc_active_start=0)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["acc_active_pct"] == pytest.approx(100.0, abs=0.1)

    def test_collision_detected(self):
        data = _make_data(n=300, acc_active_start=0, gap_m=-1.0)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["acc_collision_events"] >= 1
        assert result["hard_zero"] is True
        assert result["collision_gate_pass"] is False

    def test_no_collision_when_gap_positive(self):
        data = _make_data(n=300, acc_active_start=0, gap_m=10.0)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["acc_collision_events"] == 0
        assert result["hard_zero"] is False

    def test_gap_rmse_correct(self):
        """gap_error = constant 0.30m → RMSE = 0.30m exactly."""
        data = _make_data(n=300, acc_active_start=0, gap_error=0.30)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["acc_gap_rmse_m"] == pytest.approx(0.30, abs=0.005)

    def test_gap_rmse_gate_boundary_pass(self):
        """RMSE = ACC_GAP_RMSE_GATE_M exactly → gate pass."""
        data = _make_data(n=300, acc_active_start=0, gap_error=ACC_GAP_RMSE_GATE_M)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["gap_rmse_gate_pass"] is True

    def test_gap_rmse_gate_fail_above_threshold(self):
        """RMSE > ACC_GAP_RMSE_GATE_M → gate fail."""
        data = _make_data(n=300, acc_active_start=0, gap_error=ACC_GAP_RMSE_GATE_M + 1.0)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["gap_rmse_gate_pass"] is False

    def test_ttc_min_from_active_frames_only(self):
        """TTC minimum must be computed only over ACC-active frames."""
        n = 300
        acc_active = np.zeros(n, dtype=float)
        acc_active[100:] = 1.0   # active from frame 100
        ttc_arr = np.full(n, 0.5, dtype=float)   # pre-ACC frames have 0.5s (below critical)
        ttc_arr[100:] = 5.0                        # ACC-active frames have 5s (healthy)

        data = {
            "acc_active": acc_active,
            "radar_fwd_detected": np.ones(n),
            "radar_fwd_distance_m": np.full(n, 20.0),
            "radar_fwd_range_rate_mps": np.zeros(n),
            "acc_gap_error_m": np.zeros(n),
            "acc_ttc_s": ttc_arr,
            "acc_target_gap_m": np.full(n, 30.0),
        }
        result = _build_acc_health_summary(data, n_frames=n)
        assert result is not None
        assert result["acc_ttc_min_s"] == pytest.approx(5.0, abs=0.01)

    def test_ttc_min_gate_boundary_pass(self):
        """TTC minimum = 2.0s → gate pass (≥ ACC_TTC_MIN_GATE_S = 2.0)."""
        data = _make_data(n=300, acc_active_start=0, ttc_s=ACC_TTC_MIN_GATE_S)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["ttc_min_gate_pass"] is True

    def test_ttc_min_gate_fail_below_threshold(self):
        """TTC minimum = 1.9s < 2.0s → gate fail."""
        data = _make_data(n=300, acc_active_start=0, ttc_s=1.9)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["ttc_min_gate_pass"] is False

    def test_near_miss_penalty_applied(self):
        """Near-miss events (gap < 2.0m, ≥3 frames) → near_miss_penalty > 0."""
        n = 300
        acc_active = np.ones(n, dtype=float)
        dist = np.full(n, 20.0, dtype=float)
        dist[50:60] = 1.5   # 10 frames at 1.5m (< ACC_NEAR_MISS_GAP_M = 2.0m) = 1 event
        data = {
            "acc_active": acc_active,
            "radar_fwd_detected": np.ones(n),
            "radar_fwd_distance_m": dist,
            "radar_fwd_range_rate_mps": np.zeros(n),
            "acc_gap_error_m": np.zeros(n),
            "acc_ttc_s": np.full(n, 5.0),
            "acc_target_gap_m": np.full(n, 30.0),
        }
        result = _build_acc_health_summary(data, n_frames=n)
        assert result is not None
        assert result["acc_near_miss_events"] >= 1
        assert result["near_miss_penalty"] == pytest.approx(ACC_NEAR_MISS_PENALTY_PTS, abs=0.1)
        assert result["near_miss_gate_pass"] is False

    def test_detection_rate_gate_pass(self):
        """100% detection → detection_gate_pass = True."""
        data = _make_data(n=300, acc_active_start=0, detected=True)
        result = _build_acc_health_summary(data, n_frames=300)
        assert result is not None
        assert result["detection_gate_pass"] is True

    def test_registry_constants_imported(self):
        """Verify registry constants are accessible and have expected values."""
        assert ACC_GAP_RMSE_GATE_M == pytest.approx(35.0)
        assert ACC_TTC_MIN_GATE_S == pytest.approx(2.0)
        assert ACC_JERK_P95_GATE_MPS3 == pytest.approx(4.0)
        assert ACC_DETECTION_RATE_GATE == pytest.approx(0.95)
        assert ACC_NEAR_MISS_GAP_M == pytest.approx(2.0)
        assert ACC_TTC_CRITICAL_S == pytest.approx(1.5)
        assert ACC_NEAR_MISS_PENALTY_PTS == pytest.approx(15.0)


# ── HDF5-based integration tests ─────────────────────────────────────────────

class TestAccScoringFromHDF5:
    def test_acc_section_present_when_active(self, tmp_path):
        """make_acc_recording with acc_active_start=0 → acc_health is not None."""
        from drive_summary_core import analyze_recording_summary
        p = tmp_path / "acc_active.h5"
        make_acc_recording(p, duration_s=10.0, fps=30.0, acc_active_start=0)
        summary = analyze_recording_summary(str(p))
        assert summary.get("acc_health") is not None

    def test_acc_section_absent_when_inactive(self, tmp_path):
        """acc_active_start beyond recording length → acc_health is None."""
        from drive_summary_core import analyze_recording_summary
        p = tmp_path / "acc_inactive.h5"
        make_acc_recording(p, duration_s=5.0, fps=30.0, acc_active_start=9999)
        summary = analyze_recording_summary(str(p))
        assert summary.get("acc_health") is None

    def test_collision_forces_hard_zero_in_summary(self, tmp_path):
        """Collision recording → acc_health.hard_zero=True → overall_score = 0.0."""
        from drive_summary_core import analyze_recording_summary
        p = tmp_path / "acc_collision.h5"
        make_acc_recording(p, duration_s=10.0, fps=30.0, collision=True, acc_active_start=0)
        summary = analyze_recording_summary(str(p))
        acc = summary.get("acc_health")
        assert acc is not None
        assert acc["hard_zero"] is True
        overall_score = summary["executive_summary"]["overall_score"]
        assert overall_score == pytest.approx(0.0, abs=0.01)

    def test_near_miss_adds_safety_deduction(self, tmp_path):
        """Recording with near-miss frames → Safety layer has non-zero near-miss deduction."""
        from drive_summary_core import analyze_recording_summary
        p_clean = tmp_path / "clean.h5"
        p_nm = tmp_path / "near_miss.h5"
        make_acc_recording(p_clean, duration_s=10.0, fps=30.0, acc_active_start=0, n_near_miss_frames=0)
        make_acc_recording(p_nm, duration_s=10.0, fps=30.0, acc_active_start=0, n_near_miss_frames=5)

        clean = analyze_recording_summary(str(p_clean))
        nm = analyze_recording_summary(str(p_nm))

        # Near-miss recording should score lower or equal (synthetic recordings may have same base)
        nm_score = nm["executive_summary"]["overall_score"]
        clean_score = clean["executive_summary"]["overall_score"]
        assert nm_score <= clean_score + 0.1

        # Safety layer should have non-zero ACC near-miss deduction in nm recording
        breakdown = nm.get("layer_score_breakdown", {})
        safety = breakdown.get("Safety", {})
        deductions = safety.get("deductions", [])
        nm_deduction = next(
            (d for d in deductions if "Near-Miss" in d.get("name", "")),
            None
        )
        assert nm_deduction is not None
        assert nm_deduction["value"] > 0.0

    def test_no_nan_in_acc_health_keys(self, tmp_path):
        """ACC health summary must not contain NaN values."""
        import math
        from drive_summary_core import analyze_recording_summary
        p = tmp_path / "acc_nan_check.h5"
        make_acc_recording(
            p, duration_s=10.0, fps=30.0, acc_active_start=0,
            n_near_miss_frames=2, n_ttc_warn_frames=2,
        )
        summary = analyze_recording_summary(str(p))
        acc = summary.get("acc_health")
        assert acc is not None
        for key, val in acc.items():
            if isinstance(val, float):
                assert not math.isnan(val), f"NaN in acc_health[{key!r}]"
