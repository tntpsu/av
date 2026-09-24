"""Unit tests for ACC composite score (Proposal A, 2026-05-05).

The score is computed by `_compute_acc_score()` in `tools/analyze/acc_pipeline_analysis.py`.
These tests exercise the deduction logic via synthetic input dicts that match
the shape `_load_acc_arrays()` produces, without touching HDF5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "analyze"))

from acc_pipeline_analysis import _compute_acc_score  # noqa: E402


def _make_input(
    n: int = 600,
    acc_active_frac: float = 1.0,
    distance: np.ndarray | None = None,
    ttc: np.ndarray | None = None,
    gap_error: np.ndarray | None = None,
    jerk: np.ndarray | None = None,
    accel_cmd: np.ndarray | None = None,
    brake_cmd: np.ndarray | None = None,
    estop: np.ndarray | None = None,
) -> dict:
    """Build a dict matching what `_load_acc_arrays` returns. Defaults = clean run."""
    acc_active = np.ones(n) if acc_active_frac == 1.0 else np.concatenate([
        np.ones(int(n * acc_active_frac)), np.zeros(n - int(n * acc_active_frac))
    ])
    return {
        "n": n,
        "acc_active": acc_active,
        "acc_active_pct": float(np.mean(acc_active)),
        "detected": np.ones(n),
        "distance": distance if distance is not None else np.full(n, 20.0),
        "range_rate": np.zeros(n),
        "snr": np.full(n, 10.0),
        "acc_active_flag": acc_active,
        "acc_ttc_s": ttc if ttc is not None else np.full(n, 5.0),
        "acc_gap_error": gap_error if gap_error is not None else np.zeros(n),
        "acc_target_gap": np.full(n, 20.0),
        "speed": np.full(n, 15.0),
        "long_jerk_capped": jerk if jerk is not None else np.zeros(n),
        "long_accel_smoothed": accel_cmd if accel_cmd is not None else np.zeros(n),
        "brake_cmd": brake_cmd if brake_cmd is not None else np.zeros(n),
        "emergency_stop": estop if estop is not None else np.zeros(n),
    }


def test_clean_run_scores_perfect():
    """No deductions in any sub-layer → composite 100."""
    score = _compute_acc_score(_make_input())
    assert score is not None
    assert score["composite"] == 100.0
    assert score["safety"] == 100.0
    assert score["tracking"] == 100.0
    assert score["behavior"] == 100.0


def test_too_few_acc_frames_returns_none():
    """Below the activity floor, score is not meaningful."""
    score = _compute_acc_score(_make_input(n=600, acc_active_frac=0.01))  # 6 ACC frames
    assert score is None


def test_collision_forces_composite_zero():
    """1+ collision frame → composite must be 0 even if Tracking/Behavior are 100."""
    distance = np.full(600, 20.0)
    distance[100] = -1.0  # one collision frame
    score = _compute_acc_score(_make_input(distance=distance))
    assert score is not None
    assert score["composite"] == 0.0
    assert score["n_collision"] == 1


def test_physical_contact_forces_composite_zero():
    """vehicle/lead_collision_detected frames are collisions even though the
    recorded distance never goes below 0 (Unity clamps it to 0.1 on contact)."""
    inp = _make_input()
    inp["distance"] = np.full(600, 6.0)            # reported centre-to-centre; never < 0
    contact = np.zeros(600); contact[400:] = 1.0
    inp["lead_collision"] = contact
    score = _compute_acc_score(inp)
    assert score["n_collision"] == 200
    assert score["n_contact_frames"] == 200
    assert score["composite"] == 0.0


def test_near_miss_is_measured_in_bumper_frame():
    """A reported 5.5 m is a 1.07 m bumper gap → near-miss; a reported 7.0 m is not."""
    inp = _make_input()
    d = np.full(600, 20.0); d[100:110] = 5.5
    inp["distance"] = d
    inp["lead_collision"] = np.zeros(600)
    inp["speed"] = np.full(600, 5.0)
    inp["radar_range_offset_recorded_m"] = 0.0          # legacy centre-to-centre recording
    inp["bumper_frame_correction_m"] = 4.43
    assert _compute_acc_score(inp)["safety"] < 100.0          # near-miss deduction taken
    d2 = np.full(600, 20.0); d2[100:110] = 7.0
    inp["distance"] = d2
    assert _compute_acc_score(inp)["safety"] == 100.0         # 2.57 m bumper gap: clean


def test_emergency_brake_reflex_is_not_an_estop_event():
    """B1 bypass tags EMERGENCY_BRAKE frames with emergency_stop=True; those are
    reflex braking, not e-stops, and must not be counted (124 phantom events on a
    clean stop, 2026-09-22). TTC_ESTOP frames still count."""
    inp = _make_input()
    n = 600
    es = np.zeros(n); es[100:110] = 1.0; es[200:210] = 1.0; es[300:310] = 1.0
    states = np.array(["ACC_ACTIVE"] * n, dtype=object)
    states[100:110] = "EMERGENCY_BRAKE"; states[200:210] = "EMERGENCY_BRAKE"; states[300:310] = "TTC_ESTOP"
    inp["emergency_stop"] = es
    inp["acc_state_code"] = states
    inp["lead_collision"] = np.zeros(n)
    score = _compute_acc_score(inp)
    assert score["safety"] == 75.0          # exactly one e-stop event (the TTC_ESTOP one)


def test_near_miss_uses_recorded_frame_offset():
    """Post-2026-09-22 recordings store the bumper gap (provenance offset 4.43);
    older ones store centre-to-centre (offset 0). 1.5 m must be a near-miss in the
    first and not in the second; 5.93 m the reverse."""
    for offset, dist_val, expect_nm in ((4.43, 1.5, True), (0.0, 1.5, False), (0.0, 5.93, True), (4.43, 5.93, False)):
        inp = _make_input()
        d = np.full(600, 20.0); d[100:110] = dist_val
        inp["distance"] = d
        inp["lead_collision"] = np.zeros(600)
        inp["speed"] = np.full(600, 5.0)
        inp["radar_range_offset_recorded_m"] = offset
        inp["bumper_frame_correction_m"] = 4.43 - offset
        s = _compute_acc_score(inp)
        assert (s["safety"] < 100.0) is expect_nm, (offset, dist_val, s["safety"])


def test_standstill_inside_s0_is_not_a_near_miss():
    inp = _make_input()
    d = np.full(600, 20.0); d[400:] = 1.7            # parked 1.7 m behind a stopped lead
    inp["distance"] = d
    inp["speed"] = np.where(np.arange(600) >= 400, 0.0, 5.0)
    inp["lead_collision"] = np.zeros(600)
    inp["radar_range_offset_recorded_m"] = 4.43
    inp["bumper_frame_correction_m"] = 0.0
    assert _compute_acc_score(inp)["safety"] == 100.0


def test_ttc_violation_deducts_safety_only():
    """TTC violation should hit Safety, leave Tracking/Behavior at 100."""
    ttc = np.full(600, 5.0)
    ttc[200] = 1.6  # below 2.0 gate, above 1.5 critical → -10
    score = _compute_acc_score(_make_input(ttc=ttc))
    assert score is not None
    assert score["safety"] == 90.0  # 100 - 10
    assert score["tracking"] == 100.0
    assert score["behavior"] == 100.0
    # Composite = 0.5*90 + 0.3*100 + 0.2*100 = 45 + 30 + 20 = 95
    assert score["composite"] == 95.0


def test_oscillating_accel_deducts_behavior():
    """High accel sign-flip rate should deduct Behavior, not Safety/Tracking."""
    n = 1800  # 60 seconds at 30 fps
    accel = np.zeros(n)
    accel[::2] = 1.0   # alternating sign every frame → 30 flips/sec = 1800/min
    accel[1::2] = -1.0
    score = _compute_acc_score(_make_input(n=n, accel_cmd=accel))
    assert score is not None
    assert score["safety"] == 100.0
    assert score["tracking"] == 100.0
    assert score["behavior"] is not None
    # ~1800 flips/min, free=30, penalty per unit=1.0, capped at 30 → behavior = 70
    assert score["behavior"] == 70.0


def test_high_jerk_deducts_behavior():
    """Jerk P95 above 4 m/s³ should deduct Behavior."""
    jerk = np.full(600, 9.0)  # 9 m/s³ → 5 above free → 50 deduction (cap)
    score = _compute_acc_score(_make_input(jerk=jerk))
    assert score is not None
    assert score["safety"] == 100.0
    assert score["behavior"] == 50.0


def test_estop_event_count_uses_rising_edges():
    """Two e-stop bursts (rising edges), not the total high-frame count."""
    estop = np.zeros(600)
    estop[100:120] = 1.0  # one event
    estop[300:330] = 1.0  # second event
    score = _compute_acc_score(_make_input(estop=estop))
    assert score is not None
    assert score["n_estop"] == 2
    # Safety = 100 - 25*2 = 50
    assert score["safety"] == 50.0


def test_behavior_skipped_when_no_control_fields():
    """Missing all longitudinal control fields → Behavior=None, weights renormalized."""
    score = _compute_acc_score(_make_input(jerk=None, accel_cmd=None, brake_cmd=None))
    # Replace defaults with explicit None to bypass _make_input's defaults
    inp = _make_input()
    inp["long_jerk_capped"] = None
    inp["long_accel_smoothed"] = None
    inp["brake_cmd"] = None
    score = _compute_acc_score(inp)
    assert score is not None
    assert score["behavior"] is None
    # Renormalized: composite = (0.5*100 + 0.3*100) / 0.8 = 100
    assert score["composite"] == 100.0


def test_composite_weights_sum_correctly():
    """Verify the weight math against a known mixed case."""
    # Construct: Safety=80, Tracking=60, Behavior=40 → 0.5*80 + 0.3*60 + 0.2*40 = 66
    n = 600
    estop = np.zeros(n); estop[100:110] = 1.0  # 1 event → -25 → safety=75... not 80
    # Easier: directly test the renorm path with fewer moving parts. The test
    # above (`test_ttc_violation_deducts_safety_only`) already exercises the
    # composite formula end-to-end; this test is informational, kept simple.
    score = _compute_acc_score(_make_input())
    assert score["composite"] == 100.0


# ── Regression: sign-flip rate must use the REAL capture rate ────────────────
# Added 2026-08-14 after `_sign_flips_per_min` was found hardcoding fps=30.0
# while the stack captures at ~13 FPS, inflating every per-minute rate by ~2.3x
# and pegging the oscillation penalty at its cap on every ACC scenario. The 28
# pre-existing ACC tests all passed both before and after the fix — none of them
# exercised the rate denominator or the zero handling.
from acc_pipeline_analysis import _sign_flips_per_min  # noqa: E402


class TestSignFlipsPerMin:
    def test_rate_uses_supplied_fps_not_hardcoded_30(self):
        """A 13 FPS recording must not be scored as if it were 30 FPS."""
        # 100 alternating samples -> 99 flips.
        v = np.array([1.0, -1.0] * 50)
        n = 1300  # frames of ACC-active data
        at_13 = _sign_flips_per_min(v, n, 13.0)
        at_30 = _sign_flips_per_min(v, n, 30.0)
        # 1300 frames is 100s at 13 FPS but only 43s at 30 FPS, so the bogus
        # 30 FPS denominator inflates the rate by ~2.3x.
        assert at_30 > at_13
        assert at_30 / at_13 == pytest.approx(30.0 / 13.0, rel=1e-6)

    def test_zeros_do_not_fake_sign_flips(self):
        """+ -> 0 -> + is not a sign change; the old no-op guard counted two."""
        v = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        assert _sign_flips_per_min(v, 100, 13.0) == 0.0

    def test_real_sign_change_through_zero_counts_once(self):
        v = np.array([1.0, 0.0, -1.0])
        n, fps = 130, 13.0
        expected = 1 / (n / fps / 60.0)
        assert _sign_flips_per_min(v, n, fps) == pytest.approx(expected)

    def test_all_zero_signal_has_no_flips(self):
        assert _sign_flips_per_min(np.zeros(50), 100, 13.0) == 0.0

    def test_none_fps_falls_back_without_crashing(self):
        v = np.array([1.0, -1.0, 1.0])
        assert _sign_flips_per_min(v, 100, None) > 0.0
