"""Tests for tools/speed_utilization.py (synthetic HDF5, no Unity).

The metric answers "did the car use the road it was given?" — the one thing no
existing layer scores. These tests pin the eligibility rules (startup, ACC
following, e-stop, braking toward a lower limit), the two ratios, and the
binding-cap attribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.speed_utilization import (  # noqa: E402
    STARTUP_S,
    UNDER_UTILISED_RATIO,
    compute_speed_utilization,
    format_summary,
)

FPS = 13.0
DT = 1.0 / FPS


def _write(path: Path, *, n: int = 800, speed, limit=10.0, target=12.0, preview=None,
           acc_active=None, estop=None, caps=None, limiter=None, drop=()) -> Path:
    ts = np.arange(n, dtype=np.float64) * DT
    speed = np.broadcast_to(np.asarray(speed, dtype=np.float32), (n,)).copy()
    limit = np.broadcast_to(np.asarray(limit, dtype=np.float32), (n,)).copy()
    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=ts)
        f.create_dataset("vehicle/speed", data=speed)
        if "speed_limit" not in drop:
            f.create_dataset("vehicle/speed_limit", data=limit)
        f.create_dataset("vehicle/speed_limit_preview", data=np.broadcast_to(
            np.asarray(preview if preview is not None else limit, dtype=np.float32), (n,)).copy())
        if acc_active is not None:
            f.create_dataset("vehicle/acc_active", data=np.asarray(acc_active, dtype=np.float32))
        if estop is not None:
            f.create_dataset("control/emergency_stop", data=np.asarray(estop, dtype=np.int8))
        if target is not None:
            f.create_dataset("control/target_speed_raw", data=np.full(n, target, dtype=np.float32))
        for key, val in (caps or {}).items():
            f.create_dataset(key, data=np.broadcast_to(np.asarray(val, dtype=np.float32), (n,)).copy())
        if limiter is not None:
            f.create_dataset("control/speed_governor_active_limiter",
                             data=np.array(limiter, dtype="S32"))
    return path


class TestRatios:

    def test_car_at_limit_scores_one(self, tmp_path):
        s = compute_speed_utilization(_write(tmp_path / "a.h5", speed=10.0, limit=10.0, target=12.0))
        assert s["status"] == "ok"
        assert s["vs_limit_median"] == pytest.approx(1.0, abs=1e-6)
        assert s["vs_allowed_median"] == pytest.approx(1.0, abs=1e-6)
        assert s["under_utilised_pct"] == 0.0
        assert s["proposed_gate_pass"] is True

    def test_research_target_separates_the_two_ratios(self, tmp_path):
        """highway_65 shape: limit 29 m/s, target 12, car at 11 — 0.38 vs limit is the
        product truth, 0.92 vs allowed is the system truth."""
        s = compute_speed_utilization(_write(tmp_path / "b.h5", speed=11.0, limit=29.0, target=12.0))
        assert s["vs_limit_median"] == pytest.approx(11 / 29, abs=1e-3)
        assert s["vs_allowed_median"] == pytest.approx(11 / 12, abs=1e-3)
        assert s["proposed_gate_pass"] is True

    def test_governor_starvation_fails_vs_allowed(self, tmp_path):
        """s_loop shape: limit 8, target 12, car at 3.9 → 0.49 both ways."""
        s = compute_speed_utilization(_write(tmp_path / "c.h5", speed=3.9, limit=8.0, target=12.0))
        assert s["vs_allowed_median"] == pytest.approx(3.9 / 8.0, abs=1e-3)
        assert s["under_utilised_pct"] == pytest.approx(100.0)
        assert s["proposed_gate_pass"] is False

    def test_under_utilised_pct_counts_time_below_threshold(self, tmp_path):
        n = 800
        speed = np.full(n, 10.0); speed[400:] = 5.0          # second half at 0.5
        s = compute_speed_utilization(_write(tmp_path / "d.h5", n=n, speed=speed, limit=10.0, target=12.0))
        # startup (10 s = 130 frames) is excluded from the first half only
        elig_first = 400 - int(STARTUP_S * FPS) - 1
        expected = 100.0 * 400 / (400 + elig_first)
        assert s["under_utilised_pct"] == pytest.approx(expected, abs=1.0)
        assert UNDER_UTILISED_RATIO == 0.70


class TestEligibility:

    def test_startup_frames_excluded(self, tmp_path):
        n = 800
        speed = np.full(n, 10.0); speed[: int(STARTUP_S * FPS)] = 0.0     # stationary during startup
        s = compute_speed_utilization(_write(tmp_path / "e.h5", n=n, speed=speed, limit=10.0))
        assert s["vs_limit_median"] == pytest.approx(1.0, abs=1e-6)
        assert s["vs_limit_p10"] == pytest.approx(1.0, abs=1e-6)

    def test_acc_following_frames_excluded(self, tmp_path):
        """A slow lead sets the speed; following it is not under-utilisation."""
        n = 800
        speed = np.full(n, 10.0); speed[400:] = 4.0
        acc = np.zeros(n); acc[400:] = 1.0
        s = compute_speed_utilization(_write(tmp_path / "f.h5", n=n, speed=speed, limit=10.0, acc_active=acc))
        assert s["vs_limit_median"] == pytest.approx(1.0, abs=1e-6)
        assert s["under_utilised_pct"] == 0.0

    def test_emergency_stop_frames_excluded(self, tmp_path):
        n = 800
        speed = np.full(n, 10.0); speed[600:] = 0.0
        es = np.zeros(n); es[600:] = 1
        s = compute_speed_utilization(_write(tmp_path / "g.h5", n=n, speed=speed, limit=10.0, estop=es))
        assert s["under_utilised_pct"] == 0.0

    def test_braking_toward_lower_limit_excluded(self, tmp_path):
        """Slowing from a 15 m/s limit toward a 7 m/s curve ahead is correct driving,
        not under-utilisation — a hairpin's own limit is what it is judged against."""
        n = 800
        speed = np.full(n, 15.0); speed[400:] = 8.0
        preview = np.full(n, 15.0); preview[400:] = 7.0          # lower limit ahead
        s = compute_speed_utilization(_write(tmp_path / "h.h5", n=n, speed=speed, limit=15.0, preview=preview))
        assert s["under_utilised_pct"] == 0.0

    def test_insufficient_frames_reports_status(self, tmp_path):
        s = compute_speed_utilization(_write(tmp_path / "i.h5", n=100, speed=10.0, limit=10.0))
        assert s["status"] == "insufficient_eligible_frames"
        assert "n/a" in format_summary(s)

    def test_missing_speed_limit_returns_none(self, tmp_path):
        assert compute_speed_utilization(_write(tmp_path / "j.h5", speed=10.0, drop=("speed_limit",))) is None
        assert "n/a" in format_summary(None)


class TestAttribution:

    def test_binding_cap_is_the_smallest_cap(self, tmp_path):
        """hill_highway shape: target 12, comfort 8.9, velocity profile 7.0, curve cap 6.6
        → the curve cap is what holds the car at 6.4."""
        caps = {
            "control/velocity_profile_speed_mps": 7.0,
            "control/speed_governor_curve_cap_speed": 6.6,
            "control/speed_governor_comfort_speed": 8.86,
        }
        s = compute_speed_utilization(_write(tmp_path / "k.h5", speed=6.4, limit=11.18, target=12.0, caps=caps))
        attr = s["binding_cap_at_under_utilised_pct"]
        assert attr.get("curve_cap", 0) == pytest.approx(100.0)
        assert "target" not in attr

    def test_target_is_binding_on_highway(self, tmp_path):
        caps = {"control/speed_governor_comfort_speed": 40.0, "control/speed_governor_curve_cap_speed": np.nan}
        s = compute_speed_utilization(_write(tmp_path / "l.h5", speed=11.0, limit=29.0, target=12.0, caps=caps))
        # vs allowed is 0.92 → not under-utilised → no attribution needed
        assert s["under_utilised_pct"] == 0.0
        assert s["binding_cap_at_under_utilised_pct"] == {}

    def test_active_limiter_field_is_reported_when_present(self, tmp_path):
        n = 800
        limiter = [b"planner"] * n
        s = compute_speed_utilization(_write(tmp_path / "m.h5", n=n, speed=4.0, limit=10.0, target=12.0, limiter=limiter))
        assert s["binding_cap_at_under_utilised_pct"]["_active_limiter_field"]["planner"] == pytest.approx(100.0)

    def test_format_summary_mentions_both_ratios_and_gate(self, tmp_path):
        s = compute_speed_utilization(_write(tmp_path / "n.h5", speed=3.9, limit=8.0, target=12.0))
        txt = format_summary(s)
        assert "vs posted limit" in txt and "vs allowed" in txt and "would FAIL" in txt
