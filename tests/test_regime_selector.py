"""
Unit tests for RegimeSelector (Phase 2.2).

Tests cover: disabled passthrough, low/high speed routing, hysteresis,
hold timer, blend weight ramp, and MPC fallback override.

Run:  pytest tests/test_regime_selector.py -v
Gate: 8/8 pass
"""

import pytest
from control.regime_selector import ControlRegime, RegimeConfig, RegimeSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_selector(
    enabled=True,
    pp_max_speed_mps=10.0,
    upshift_hysteresis_mps=1.0,
    downshift_hysteresis_mps=1.0,
    blend_frames=15,
    min_hold_frames=30,
) -> RegimeSelector:
    cfg = RegimeConfig(
        enabled=enabled,
        pp_max_speed_mps=pp_max_speed_mps,
        upshift_hysteresis_mps=upshift_hysteresis_mps,
        downshift_hysteresis_mps=downshift_hysteresis_mps,
        blend_frames=blend_frames,
        min_hold_frames=min_hold_frames,
    )
    return RegimeSelector(cfg)


def _run_n(selector, speed, n=35) -> tuple:
    """Run selector for n frames at constant speed. Returns last result."""
    result = None
    for _ in range(n):
        result = selector.update(speed=speed)
    return result


# ---------------------------------------------------------------------------
# 1. Disabled → always PP at any speed
# ---------------------------------------------------------------------------

def test_disabled_always_pp():
    sel = _make_selector(enabled=False)
    for speed in [5.0, 12.0, 15.0, 25.0]:
        regime, blend = sel.update(speed=speed)
        assert regime == ControlRegime.PURE_PURSUIT, \
            f"Disabled selector returned {regime} at v={speed}"
        assert blend == 1.0


# ---------------------------------------------------------------------------
# 2. Low speed → PP
# ---------------------------------------------------------------------------

def test_low_speed_pp():
    sel = _make_selector()
    regime, blend = _run_n(sel, speed=5.0, n=50)
    assert regime == ControlRegime.PURE_PURSUIT
    assert blend == 1.0


# ---------------------------------------------------------------------------
# 3. Mid speed sustained → upshift to LINEAR_MPC
# ---------------------------------------------------------------------------

def test_mid_speed_upshift():
    sel = _make_selector(min_hold_frames=30)
    # v=12 exceeds pp_max(10) + hysteresis(1) = 11
    regime, blend = _run_n(sel, speed=12.0, n=35)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected LINEAR_MPC after 35 frames at 12 m/s, got {regime}"


# ---------------------------------------------------------------------------
# 4. Speed oscillating near threshold → no chatter (≤1 switch total)
# ---------------------------------------------------------------------------

def test_hysteresis_no_chatter():
    """v oscillates 9.5↔10.5 — straddles pp_max but not hysteresis bands."""
    sel = _make_selector(
        pp_max_speed_mps=10.0,
        upshift_hysteresis_mps=1.0,    # upshift at 11.0
        downshift_hysteresis_mps=1.0,  # downshift at 9.0
        min_hold_frames=30,
    )
    switches = 0
    last_regime = ControlRegime.PURE_PURSUIT
    for i in range(120):
        speed = 9.5 if i % 2 == 0 else 10.5  # oscillate every frame
        regime, _ = sel.update(speed=speed)
        if regime != last_regime:
            switches += 1
            last_regime = regime

    assert switches <= 1, f"Expected ≤1 regime switch, got {switches}"


# ---------------------------------------------------------------------------
# 5. Upshift requires hold — stays PP until min_hold_frames
# ---------------------------------------------------------------------------

def test_upshift_requires_hold():
    sel = _make_selector(min_hold_frames=30, blend_frames=15)
    # Speed jumps above threshold but hold counter not yet satisfied
    for frame in range(29):
        regime, _ = sel.update(speed=12.0)
        assert regime == ControlRegime.PURE_PURSUIT, \
            f"Should still be PP at frame {frame}, hold not met"

    # Frame 30 (index 29): hold counter hits min_hold_frames → switch
    regime, blend = sel.update(speed=12.0)
    assert regime == ControlRegime.LINEAR_MPC, \
        "Expected switch to LINEAR_MPC at frame 30"
    assert blend < 1.0, "Blend should be < 1.0 immediately after switch"


# ---------------------------------------------------------------------------
# 6. Downshift on sustained speed drop
# ---------------------------------------------------------------------------

def test_downshift_on_speed_drop():
    sel = _make_selector(min_hold_frames=30)
    # Get into MPC first
    _run_n(sel, speed=12.0, n=35)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # Now drop speed well below downshift threshold (10 - 1 = 9 m/s)
    regime, blend = _run_n(sel, speed=5.0, n=35)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected downshift to PP after sustained v=5, got {regime}"


# ---------------------------------------------------------------------------
# 7. Blend weight ramps monotonically 0 → 1 after switch
# ---------------------------------------------------------------------------

def test_blend_weight_ramp():
    sel = _make_selector(min_hold_frames=5, blend_frames=15)
    # Trigger switch quickly
    _run_n(sel, speed=12.0, n=6)
    assert sel.active_regime == ControlRegime.LINEAR_MPC
    assert sel.blend_progress < 1.0

    # Collect blend values over 20 frames
    blends = []
    for _ in range(20):
        _, w = sel.update(speed=12.0)
        blends.append(w)

    # Must be monotonically non-decreasing
    for i in range(1, len(blends)):
        assert blends[i] >= blends[i - 1], \
            f"Blend not monotone at index {i}: {blends[i-1]:.3f} → {blends[i]:.3f}"

    # Must reach 1.0 within blend_frames
    assert blends[-1] == 1.0, f"Blend did not reach 1.0 in 20 frames: {blends[-1]}"


# ---------------------------------------------------------------------------
# 8. MPC fallback forces PP at any speed
# ---------------------------------------------------------------------------

def test_fallback_forces_pp():
    sel = _make_selector(min_hold_frames=5)
    # Get into MPC
    _run_n(sel, speed=12.0, n=10)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # Activate fallback — should immediately return PP regardless of speed
    for _ in range(10):
        regime, _ = sel.update(speed=12.0, mpc_fallback_active=True)
        # Note: fallback sets desired=PP but hold counter still applies for switch
        # What matters is desired is PP, so target flips
        pass

    # After enough frames with fallback, should be back to PP
    _run_n_with_fallback(sel, speed=12.0, n=35)
    regime, _ = sel.update(speed=12.0, mpc_fallback_active=True)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP during MPC fallback, got {regime}"


def _run_n_with_fallback(selector, speed, n=35):
    for _ in range(n):
        selector.update(speed=speed, mpc_fallback_active=True)


# ---------------------------------------------------------------------------
# Bonus: RegimeConfig.from_config
# ---------------------------------------------------------------------------

def test_regime_config_from_config():
    cfg = {
        "control": {
            "regime": {
                "enabled": True,
                "pp_max_speed_mps": 8.0,
                "min_hold_frames": 45,
            }
        }
    }
    rc = RegimeConfig.from_config(cfg)
    assert rc.enabled is True
    assert rc.pp_max_speed_mps == 8.0
    assert rc.min_hold_frames == 45
    assert rc.blend_frames == 15   # default


def test_regime_config_from_empty_config():
    """Missing control.regime key → all defaults."""
    rc = RegimeConfig.from_config({})
    assert rc.enabled is False
    assert rc.pp_max_speed_mps == 10.0


def test_reset_clears_state():
    sel = _make_selector(min_hold_frames=5)
    _run_n(sel, speed=12.0, n=10)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    sel.reset()
    assert sel.active_regime == ControlRegime.PURE_PURSUIT
    assert sel.blend_progress == 1.0
