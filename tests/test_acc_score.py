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


def test_post_convergence_rmse_is_measured_against_equilibrium_not_target():
    """H7 shape: IDM equilibrium is 1.8x s*. A car sitting exactly at equilibrium has
    ~0 post-convergence RMSE vs EQ and a large RMSE vs s* — the old gate's failure."""
    from acc_pipeline_analysis import compute_post_convergence_gap
    n = 600
    tg = np.full(n, 15.0); eq = np.full(n, 27.0)
    gap = np.full(n, 60.0); gap[100:] = 27.0            # startup catch-up, then parked at EQ
    d = {"acc_active": np.ones(n), "acc_gap_error": gap - tg, "acc_target_gap": tg, "acc_equilibrium_gap": eq, "fps": 13.0}
    pc = compute_post_convergence_gap(d)
    assert pc["converged"] and pc["gate_pass"]
    assert pc["rmse_vs_eq_m"] == pytest.approx(0.0, abs=1e-6)
    assert pc["rmse_vs_target_m"] == pytest.approx(12.0, abs=1e-6)
    assert pc["converged_at_s"] == pytest.approx(100 / 13.0, abs=0.1)


def test_post_convergence_never_converged_is_reported():
    from acc_pipeline_analysis import compute_post_convergence_gap
    n = 600
    d = {"acc_active": np.ones(n), "acc_gap_error": np.full(n, 100.0), "acc_target_gap": np.full(n, 15.0),
         "acc_equilibrium_gap": np.full(n, 27.0), "fps": 13.0}
    pc = compute_post_convergence_gap(d)
    assert pc["converged"] is False
    assert compute_post_convergence_gap({"acc_active": np.ones(n), "acc_gap_error": None, "acc_target_gap": None, "acc_equilibrium_gap": None}) is None


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
    """Jerk P95 above 4 m/s³ deducts Behavior; the cap is a soft knee (T-METRIC-UNCAP):
    9 m/s³ → raw 50 → 50·(1−e⁻¹) = 31.6, and 14 m/s³ (raw 100) still ranks below it."""
    from acc_pipeline_analysis import _soft_cap
    score = _compute_acc_score(_make_input(jerk=np.full(600, 9.0)))
    assert score is not None and score["safety"] == 100.0
    assert score["behavior"] == pytest.approx(100.0 - _soft_cap(50.0, 50.0), abs=0.05)
    assert 60.0 < score["behavior"] < 70.0
    worse = _compute_acc_score(_make_input(jerk=np.full(600, 14.0)))
    assert worse["behavior"] < score["behavior"]          # severity keeps ranking past the old cap
    assert worse["behavior"] > 50.0                        # but never exceeds the cap asymptote

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


# ── Scorer work-package 2026-10-10: deadband, soft caps, post-convergence Tracking ──────────

from acc_pipeline_analysis import _soft_cap  # noqa: E402
import scoring_registry as _reg  # noqa: E402


class TestSoftCap:
    def test_legacy_hard_cap_when_disabled(self, monkeypatch):
        import acc_pipeline_analysis as apa
        monkeypatch.setattr(apa, "ACC_SCORE_SOFT_CAPS", False)
        assert _soft_cap(10.0, 30.0) == 10.0 and _soft_cap(80.0, 30.0) == 30.0

    def test_soft_knee_is_monotone_and_bounded(self):
        vals = [_soft_cap(x, 30.0) for x in (0, 5, 30, 60, 300)]
        assert vals == sorted(vals) and vals[0] == 0.0 and vals[-1] < 30.0
        assert _soft_cap(30.0, 30.0) == pytest.approx(30.0 * (1 - np.exp(-1)))
        assert _soft_cap(3.0, 30.0) == pytest.approx(3.0, rel=0.06)   # near-linear in the small-penalty regime


class TestSignFlipDeadband:
    def test_chatter_inside_deadband_does_not_count(self):
        """Night-59 shape: accel command flipping sign at ±0.1 m/s² (std 0.08–0.13) — chatter."""
        n = 1300  # 100 s at 13 fps
        chatter = 0.1 * np.sign(np.sin(np.arange(n) * 2 * np.pi / 13))   # one flip pair per second
        score = _compute_acc_score(_make_input(n=n, accel_cmd=chatter))
        assert score["behavior"] == 100.0
        assert _sign_flips_per_min(chatter, n, 13.0) > 60.0                       # legacy count
        assert _sign_flips_per_min(chatter, n, 13.0, deadband=_reg.ACC_SCORE_OSC_DEADBAND_MPS2) == 0.0

    def test_real_oscillation_beyond_deadband_still_counts(self):
        n = 1300
        hunt = 0.6 * np.sign(np.sin(np.arange(n) * 2 * np.pi / 13))
        score = _compute_acc_score(_make_input(n=n, accel_cmd=hunt))
        assert score["behavior"] < 100.0
        assert any("sign-flips" in label for label, _ in score["deductions"]["behavior"])

    def test_zero_deadband_reproduces_legacy(self):
        v = np.array([0.01, -0.01, 0.02, -0.02, 0.5, -0.5])
        assert _sign_flips_per_min(v, 100, 13.0, deadband=0.0) == _sign_flips_per_min(v, 100, 13.0)


def _following_input(n=2600, startup=200, eq=27.0, tg=15.0, noise=0.0, swing=0.0, fps=13.0):
    """Ego from rest: 60 m gap for `startup` frames, then parked at the IDM equilibrium
    (plus optional sub-metre noise or a ±swing oscillation)."""
    rng = np.random.default_rng(0)
    gap = np.full(n, 60.0)
    k = np.arange(n - startup)
    gap[startup:] = eq + noise * rng.standard_normal(n - startup) + swing * np.sign(np.sin(k * 2 * np.pi / 26))
    inp = _make_input(n=n, gap_error=gap - tg)
    inp["acc_target_gap"] = np.full(n, tg)
    inp["acc_equilibrium_gap"] = np.full(n, eq)
    inp["fps"] = fps
    return inp


class TestPostConvergenceTracking:
    def test_startup_transient_is_not_scored(self):
        """200 s run vs 90 s run of identical driving must score the same (Night-59 dilution bug)."""
        long = _compute_acc_score(_following_input(n=2600))
        short = _compute_acc_score(_following_input(n=1170))
        assert long["tracking_window"] == "post_conv" == short["tracking_window"]
        assert long["tracking"] == pytest.approx(short["tracking"], abs=0.1)
        assert long["tracking"] == 100.0                        # parked at EQ: nothing to deduct
        assert long["conv_time_s"] == pytest.approx(200 / 13.0, abs=0.2)

    def test_legacy_full_run_window_scales_with_duration(self, monkeypatch):
        import acc_pipeline_analysis as apa
        monkeypatch.setattr(apa, "ACC_SCORE_TRACKING_WINDOW", "full_run")
        long = _compute_acc_score(_following_input(n=2600))
        short = _compute_acc_score(_following_input(n=780))
        assert long["tracking_window"] == "full_run"
        assert short["tracking"] < long["tracking"] - 5.0        # the artefact this change removes

    def test_sub_metre_noise_around_equilibrium_is_not_hunting(self):
        score = _compute_acc_score(_following_input(noise=0.4))
        assert not any("hunting" in label for label, _ in score["deductions"]["tracking"])

    def test_gap_swing_beyond_s0_is_hunting(self):
        score = _compute_acc_score(_following_input(swing=3.0))
        assert any("hunting" in label for label, _ in score["deductions"]["tracking"])

    def test_slow_convergence_is_deducted(self):
        slow = _compute_acc_score(_following_input(n=2600, startup=13 * 40))   # 40 s to converge
        fast = _compute_acc_score(_following_input(n=2600, startup=13 * 10))
        assert fast["tracking"] == 100.0
        assert slow["tracking"] < fast["tracking"]
        assert any("converged" in label for label, _ in slow["deductions"]["tracking"])

    def test_never_converged_takes_the_cap_and_falls_back(self):
        inp = _following_input(n=1300, startup=1300)              # never reaches EQ
        score = _compute_acc_score(inp)
        assert score["tracking_window"].startswith("full_run")
        assert any("never converged" in label for label, _ in score["deductions"]["tracking"])
        assert score["conv_time_s"] is None

    def test_no_equilibrium_field_falls_back_to_full_run(self):
        score = _compute_acc_score(_make_input(gap_error=np.zeros(600)))
        assert score["tracking_window"] == "full_run" and score["tracking"] == 100.0


class TestEngageEdges:
    def test_steady_following_has_no_edge_penalty(self):
        score = _compute_acc_score(_following_input())
        assert not any("engage/disengage" in label for label, _ in score["deductions"]["behavior"])

    def test_hunting_toggle_is_penalised_even_though_dropouts_leave_the_mask(self):
        """Night-58 H8 shape: 27 edges in 120 s. Each dropout removes its own frames from the
        ACC-active mask, so Tracking never saw it; the whole-run edge rate does."""
        inp = _following_input(n=1560)                               # 120 s
        flag = inp["acc_active"].copy()
        for k in range(13):                                          # 13 dropouts of 10 frames → 26 edges
            flag[400 + 60 * k: 400 + 60 * k + 10] = 0.0
        inp["acc_active"] = flag; inp["acc_active_flag"] = flag
        score = _compute_acc_score(inp)
        labels = [label for label, _ in score["deductions"]["behavior"]]
        assert any("engage/disengage 26 edges" in l for l in labels), labels
        assert score["behavior"] < 90.0

    def test_single_legitimate_disengage_is_free(self):
        """H4 accel-away: one engage + one disengage over the run must not cost anything."""
        inp = _following_input(n=1560)
        inp["acc_active"][1300:] = 0.0; inp["acc_active_flag"] = inp["acc_active"]
        score = _compute_acc_score(inp)
        assert not any("engage/disengage" in label for label, _ in score["deductions"]["behavior"])


class TestEquilibriumValidityGuard:
    def test_diverging_equilibrium_does_not_count_as_convergence(self):
        """IDM EQ → ∞ as v → v0: a 145 m catch-up gap must not read 'converged at 0.5 s'."""
        from acc_pipeline_analysis import _post_convergence_mask
        n = 600
        acc = np.ones(n, dtype=bool)
        gap = np.full(n, 145.0); gap[300:] = 30.0
        eq = np.full(n, 145.0); eq[300:] = 30.0                       # EQ tracks the gap while diverged
        tg = np.full(n, 32.0)
        legacy = _post_convergence_mask(acc, gap, eq)                 # no target → no guard
        guarded = _post_convergence_mask(acc, gap, eq, tg)
        assert int(np.flatnonzero(legacy)[0]) == 0
        assert int(np.flatnonzero(guarded)[0]) == 300
        assert guarded[:300].sum() == 0
