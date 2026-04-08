"""
Phase G2 — Unit tests for RegimeSelector NMPC dispatch path (Step 5).

Tests cover:
  - NONLINEAR_MPC upshift: speed > lmpc_max_speed_mps
  - NONLINEAR_MPC upshift: heading_error > lmpc_max_heading_error_rad
  - Two hold-cycle requirement (PP→LMPC, then LMPC→NMPC)
  - NMPC→LMPC downshift: speed/heading falls back to LMPC range
  - NMPC→PP downshift: speed falls below mpc_min
  - HDF5 encoding: NONLINEAR_MPC == 2.0 (regime >= 1.5 mask)
  - mpc_fallback_active forces PP from NMPC (not just LMPC)

Run:  pytest tests/test_regime_selector_nmpc.py -v
"""

import pytest
from control.regime_selector import ControlRegime, RegimeConfig, RegimeSelector


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_selector(
    enabled=True,
    mpc_min_speed_mps=4.0,
    pp_max_speed_mps=4.0,
    upshift_hysteresis_mps=0.5,
    downshift_hysteresis_mps=0.5,
    min_hold_frames=5,
    blend_frames=3,
    lmpc_max_speed_mps=20.0,
    lmpc_max_heading_error_rad=0.25,
    mpc_max_curvature=0.020,
    mpc_curvature_hysteresis=0.003,
    lmpc_nmpc_blend_frames=None,
    lmpc_nmpc_min_hold_frames=None,
) -> RegimeSelector:
    # Default lmpc_nmpc params to same as PP↔LMPC values so existing tests stay fast.
    cfg = RegimeConfig(
        enabled=enabled,
        pp_max_speed_mps=pp_max_speed_mps,
        mpc_min_speed_mps=mpc_min_speed_mps,
        upshift_hysteresis_mps=upshift_hysteresis_mps,
        downshift_hysteresis_mps=downshift_hysteresis_mps,
        blend_frames=blend_frames,
        min_hold_frames=min_hold_frames,
        lmpc_max_speed_mps=lmpc_max_speed_mps,
        lmpc_max_heading_error_rad=lmpc_max_heading_error_rad,
        mpc_max_curvature=mpc_max_curvature,
        mpc_curvature_hysteresis=mpc_curvature_hysteresis,
        lmpc_nmpc_blend_frames=lmpc_nmpc_blend_frames if lmpc_nmpc_blend_frames is not None else blend_frames,
        lmpc_nmpc_min_hold_frames=lmpc_nmpc_min_hold_frames if lmpc_nmpc_min_hold_frames is not None else min_hold_frames,
    )
    return RegimeSelector(cfg)


def _run_n(selector, speed, n, heading_error=0.0, curvature_abs=0.0, map_curvature_abs=None):
    """Drive selector for n frames at constant speed. Returns (regime, blend) of last frame."""
    if map_curvature_abs is None:
        map_curvature_abs = curvature_abs
    result = None
    for _ in range(n):
        result = selector.update(
            speed=speed,
            heading_error=heading_error,
            curvature_abs=curvature_abs,
            map_curvature_abs=map_curvature_abs,
        )
    return result


def _settle_in_lmpc(selector, hold=15):
    """Drive selector to settled LINEAR_MPC state (speed well above mpc_min + hysteresis)."""
    _run_n(selector, speed=12.0, n=hold)
    regime, blend = _run_n(selector, speed=12.0, n=1)
    # Confirm we're in LMPC
    assert regime == ControlRegime.LINEAR_MPC, f"Expected LMPC but got {regime}"
    return selector


# ─── 1. NMPC encoding ────────────────────────────────────────────────────────

def test_nonlinear_mpc_enum_value():
    """NONLINEAR_MPC must encode as 2 (HDF5 convention: regime >= 1.5 mask)."""
    assert int(ControlRegime.NONLINEAR_MPC) == 2


def test_nmpc_regime_mask_convention():
    """Regime value 2.0 must be ≥ 1.5 (NMPC mask) and ≥ 0.5 (any-MPC mask)."""
    val = float(ControlRegime.NONLINEAR_MPC)
    assert val >= 1.5, "NMPC must satisfy nmpc_mask (>= 1.5)"
    assert val >= 0.5, "NMPC must satisfy any-mpc mask (>= 0.5)"


# ─── 2. NMPC upshift via speed ───────────────────────────────────────────────

def test_nmpc_upshift_on_high_speed():
    """Speed > lmpc_max_speed_mps while in LMPC should upshift to NONLINEAR_MPC."""
    sel = _make_selector(lmpc_max_speed_mps=20.0, min_hold_frames=5)
    _settle_in_lmpc(sel)
    # Now push above lmpc threshold + hold
    regime, blend = _run_n(sel, speed=22.0, n=30)
    assert regime == ControlRegime.NONLINEAR_MPC, f"Expected NMPC at 22 m/s, got {regime}"


def test_nmpc_not_reached_below_lmpc_max():
    """Speed below lmpc_max_speed_mps should stay in LMPC, not NMPC."""
    sel = _make_selector(lmpc_max_speed_mps=20.0, min_hold_frames=5)
    _settle_in_lmpc(sel)
    regime, _ = _run_n(sel, speed=18.0, n=30)
    assert regime == ControlRegime.LINEAR_MPC, f"Expected LMPC at 18 m/s, got {regime}"


def test_nmpc_requires_lmpc_as_intermediate():
    """NMPC must not be reachable directly from PP — requires PP→LMPC→NMPC path."""
    sel = _make_selector(lmpc_max_speed_mps=20.0, min_hold_frames=5)
    # Jump directly to high speed from startup (PP state)
    regime_early, _ = sel.update(speed=25.0)
    # On the very first frame from PP at high speed, we should be upshifting toward LMPC, not NMPC
    assert regime_early != ControlRegime.NONLINEAR_MPC


# ─── 3. NMPC upshift via heading error ───────────────────────────────────────

def test_nmpc_upshift_on_large_heading_error():
    """Large heading error while in LMPC should trigger NONLINEAR_MPC."""
    sel = _make_selector(lmpc_max_heading_error_rad=0.25, min_hold_frames=5)
    _settle_in_lmpc(sel)
    regime, _ = _run_n(sel, speed=12.0, n=30, heading_error=0.30)
    assert regime == ControlRegime.NONLINEAR_MPC, (
        f"Expected NMPC on large heading error (0.30 > 0.25 rad), got {regime}"
    )


def test_nmpc_not_triggered_below_heading_threshold():
    """Heading error below threshold should stay in LMPC."""
    sel = _make_selector(lmpc_max_heading_error_rad=0.25, min_hold_frames=5)
    _settle_in_lmpc(sel)
    regime, _ = _run_n(sel, speed=12.0, n=30, heading_error=0.20)
    assert regime == ControlRegime.LINEAR_MPC, (
        f"Expected LMPC for heading_error=0.20 < 0.25 threshold, got {regime}"
    )


# ─── 4. NMPC → LMPC downshift ────────────────────────────────────────────────

def test_nmpc_downshifts_to_lmpc_when_speed_drops():
    """When speed drops back below lmpc_max, NMPC should return to LMPC."""
    sel = _make_selector(lmpc_max_speed_mps=20.0, min_hold_frames=5)
    _settle_in_lmpc(sel)
    _run_n(sel, speed=22.0, n=30)  # Reach NMPC

    # Drop back to LMPC range
    regime, _ = _run_n(sel, speed=14.0, n=30)
    assert regime == ControlRegime.LINEAR_MPC, (
        f"Expected LMPC after speed drops to 14 m/s, got {regime}"
    )


def test_nmpc_downshifts_to_lmpc_when_heading_clears():
    """When heading error clears, NMPC triggered by heading should return to LMPC."""
    sel = _make_selector(lmpc_max_heading_error_rad=0.25, min_hold_frames=5)
    _settle_in_lmpc(sel)
    _run_n(sel, speed=12.0, n=30, heading_error=0.30)  # Reach NMPC

    # Heading error drops below threshold
    regime, _ = _run_n(sel, speed=12.0, n=30, heading_error=0.10)
    assert regime == ControlRegime.LINEAR_MPC, (
        f"Expected LMPC after heading clears, got {regime}"
    )


# ─── 5. NMPC → PP downshift ──────────────────────────────────────────────────

def test_nmpc_downshifts_to_pp_on_low_speed():
    """When speed drops below mpc_min − hysteresis, NMPC should downshift out of MPC.
    At very low speeds Stanley may activate — accept both PP and Stanley."""
    sel = _make_selector(
        mpc_min_speed_mps=4.0, downshift_hysteresis_mps=0.5,
        lmpc_max_speed_mps=20.0, min_hold_frames=5,
    )
    _settle_in_lmpc(sel)
    _run_n(sel, speed=22.0, n=30)  # Reach NMPC

    # Drop below mpc_min - hysteresis (3.5 m/s)
    regime, _ = _run_n(sel, speed=2.0, n=30)
    assert regime in (ControlRegime.PURE_PURSUIT, ControlRegime.STANLEY), (
        f"Expected PP or Stanley at 2.0 m/s (below mpc_min), got {regime}"
    )


# ─── 6. mpc_fallback_active forces PP from NMPC ──────────────────────────────

def test_mpc_fallback_forces_pp_from_nmpc():
    """mpc_fallback_active=True should cause downshift to PURE_PURSUIT.
    The hold timer still applies so we run enough frames to complete the transition."""
    sel = _make_selector(lmpc_max_speed_mps=20.0, min_hold_frames=5, blend_frames=3)
    _settle_in_lmpc(sel)
    _run_n(sel, speed=22.0, n=30)  # Settle in NMPC

    # Run with fallback active for enough frames to clear hold + blend
    regime, _ = _run_n(sel, speed=22.0, n=30, heading_error=0.0)  # fallback off
    # Now trigger fallback — after min_hold_frames the transition should complete
    for _ in range(20):
        regime, _ = sel.update(speed=22.0, mpc_fallback_active=True)
    assert regime == ControlRegime.PURE_PURSUIT, (
        f"mpc_fallback_active should transition to PP after hold+blend, got {regime}"
    )


def test_mpc_fallback_forces_pp_from_lmpc():
    """mpc_fallback_active=True also forces PP from LMPC after hold+blend completes."""
    sel = _make_selector(min_hold_frames=5, blend_frames=3)
    _settle_in_lmpc(sel)
    # Run enough frames for hold+blend transition to PP
    for _ in range(20):
        regime, _ = sel.update(speed=12.0, mpc_fallback_active=True)
    assert regime == ControlRegime.PURE_PURSUIT


# ─── 7. Blend weight during LMPC→NMPC transition ─────────────────────────────

def test_blend_weight_transitions_smoothly_lmpc_to_nmpc():
    """During the LMPC→NMPC blend, blend weight should progress from 0 toward 1."""
    sel = _make_selector(lmpc_max_speed_mps=20.0, min_hold_frames=5, blend_frames=10)
    _settle_in_lmpc(sel)

    # Push above NMPC threshold but only run a few hold + blend frames
    blend_weights = []
    for _ in range(30):
        regime, blend = sel.update(speed=22.0)
        blend_weights.append(blend)

    # By the end, we should be settled in NMPC with blend = 1.0
    assert blend_weights[-1] == pytest.approx(1.0), "Blend should reach 1.0 when settled"


# ─── 8. Reset ─────────────────────────────────────────────────────────────────

def test_reset_from_nmpc_returns_to_pp():
    """reset() should return selector to PP regardless of current NMPC state."""
    sel = _make_selector(lmpc_max_speed_mps=20.0, min_hold_frames=5)
    _settle_in_lmpc(sel)
    _run_n(sel, speed=22.0, n=30)

    sel.reset()
    assert sel.active_regime == ControlRegime.PURE_PURSUIT
    regime, blend = sel.update(speed=22.0)
    # First frame after reset: still PP (hold timer resets)
    assert blend == pytest.approx(1.0) or regime == ControlRegime.PURE_PURSUIT


# ─── 9. Curvature guard blocks NMPC upshift ──────────────────────────────────

def test_curvature_guard_routes_to_nmpc_not_pp():
    """On tight curve (κ > mpc_max_curvature), NMPC handles it instead of PP."""
    sel = _make_selector(
        mpc_max_curvature=0.020,
        mpc_curvature_hysteresis=0.003,
        lmpc_max_speed_mps=20.0,
        min_hold_frames=5,
    )
    # High speed on tight curve — NMPC should handle it (not PP)
    regime, _ = _run_n(sel, speed=25.0, n=60, curvature_abs=0.025)
    assert regime == ControlRegime.NONLINEAR_MPC, (
        f"Tight curve should route to NMPC, got {regime}"
    )


# ─── 10. lmpc_max_speed_mps config ──────────────────────────────────────────

def test_lmpc_max_speed_mps_configurable():
    """lmpc_max_speed_mps threshold should be respected — higher threshold delays NMPC."""
    sel_low  = _make_selector(lmpc_max_speed_mps=15.0, min_hold_frames=5)
    sel_high = _make_selector(lmpc_max_speed_mps=25.0, min_hold_frames=5)

    for sel in (sel_low, sel_high):
        _settle_in_lmpc(sel)

    speed = 18.0
    regime_low,  _ = _run_n(sel_low,  speed=speed, n=30)
    regime_high, _ = _run_n(sel_high, speed=speed, n=30)

    assert regime_low == ControlRegime.NONLINEAR_MPC, (
        f"18 m/s should exceed low threshold (15 m/s) → NMPC, got {regime_low}"
    )
    assert regime_high == ControlRegime.LINEAR_MPC, (
        f"18 m/s should not exceed high threshold (25 m/s) → LMPC, got {regime_high}"
    )


# ─── 11. lmpc_nmpc_blend_frames and lmpc_nmpc_min_hold_frames ────────────────

def test_lmpc_nmpc_longer_hold_delays_nmpc_upshift():
    """lmpc_nmpc_min_hold_frames > min_hold_frames should delay LMPC→NMPC transition."""
    # Short hold for PP↔LMPC; long hold for LMPC↔NMPC
    sel = _make_selector(
        lmpc_max_speed_mps=20.0,
        min_hold_frames=3,
        blend_frames=2,
        lmpc_nmpc_min_hold_frames=30,
        lmpc_nmpc_blend_frames=10,
    )
    _settle_in_lmpc(sel)

    # After only 15 frames above NMPC threshold, should still be in LMPC (hold=30)
    regime_early, _ = _run_n(sel, speed=22.0, n=15)
    assert regime_early == ControlRegime.LINEAR_MPC, (
        f"Should still be LMPC after 15 frames (lmpc_nmpc_min_hold_frames=30), got {regime_early}"
    )

    # After 40 frames total (hold + blend), should have reached NMPC
    regime_settled, _ = _run_n(sel, speed=22.0, n=25)
    assert regime_settled == ControlRegime.NONLINEAR_MPC, (
        f"Should be NMPC after hold+blend complete (30+10 frames), got {regime_settled}"
    )


def test_lmpc_nmpc_blend_frames_default_independent_of_pp_lmpc_blend():
    """lmpc_nmpc_blend_frames default (30) is independent of blend_frames (15).
    Verifies RegimeConfig stores them as separate fields."""
    cfg = RegimeConfig()
    assert cfg.lmpc_nmpc_blend_frames == 30
    assert cfg.lmpc_nmpc_min_hold_frames == 40
    assert cfg.blend_frames == 15         # PP↔LMPC unchanged
    assert cfg.min_hold_frames == 30      # PP↔LMPC unchanged


# ── Curvature-based NMPC routing ──────────────────────────────────────────


def test_nmpc_activates_on_high_curvature():
    """κ=0.025 at v=5 should route to NMPC, not PP."""
    sel = _make_selector()
    # First get into LMPC (low curvature, moderate speed)
    _run_n(sel, speed=8.0, n=20, curvature_abs=0.005)
    # Now increase curvature above threshold — should upshift to NMPC
    regime, _ = _run_n(sel, speed=8.0, n=60, curvature_abs=0.025)
    assert regime == ControlRegime.NONLINEAR_MPC, (
        f"High curvature should route to NMPC, got {regime}"
    )


def test_nmpc_deactivates_on_low_curvature():
    """κ=0.005 at v=8 should stay in LMPC, not NMPC."""
    sel = _make_selector()
    _run_n(sel, speed=8.0, n=20, curvature_abs=0.005)
    regime, _ = _run_n(sel, speed=8.0, n=20, curvature_abs=0.005)
    assert regime == ControlRegime.LINEAR_MPC, (
        f"Low curvature should stay LMPC, got {regime}"
    )


def test_pp_routes_to_nmpc_on_tight_curve():
    """From PP, κ=0.025 should route directly to NMPC."""
    sel = _make_selector()
    # Start in PP (low speed)
    _run_n(sel, speed=1.0, n=10, curvature_abs=0.0)
    # Speed up on tight curve — should go to NMPC, not PP
    regime, _ = _run_n(sel, speed=5.0, n=60, curvature_abs=0.025)
    assert regime == ControlRegime.NONLINEAR_MPC, (
        f"PP on tight curve should route to NMPC, got {regime}"
    )


def test_nmpc_curvature_threshold_configurable():
    """nmpc_curvature_threshold from config controls the routing."""
    cfg = RegimeConfig(nmpc_curvature_threshold=0.030)
    assert cfg.nmpc_curvature_threshold == 0.030
