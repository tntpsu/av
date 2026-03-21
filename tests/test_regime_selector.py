"""
Unit tests for RegimeSelector (Phase 2.2 + Phase 2.9 Stanley regime).

Tests cover: disabled passthrough, low/high speed routing, hysteresis,
hold timer, blend weight ramp, MPC fallback override, and Stanley low-speed regime.

Run:  pytest tests/test_regime_selector.py -v
Gate: 32/32 pass
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
    mpc_min_speed_mps=None,
    mpc_max_curvature=0.020,
    mpc_curvature_hysteresis=0.003,
) -> RegimeSelector:
    cfg = RegimeConfig(
        enabled=enabled,
        pp_max_speed_mps=pp_max_speed_mps,
        mpc_min_speed_mps=mpc_min_speed_mps if mpc_min_speed_mps is not None else pp_max_speed_mps,
        mpc_max_curvature=mpc_max_curvature,
        mpc_curvature_hysteresis=mpc_curvature_hysteresis,
        upshift_hysteresis_mps=upshift_hysteresis_mps,
        downshift_hysteresis_mps=downshift_hysteresis_mps,
        blend_frames=blend_frames,
        min_hold_frames=min_hold_frames,
    )
    return RegimeSelector(cfg)


def _run_n(selector, speed, n=35, curvature_abs=0.0) -> tuple:
    """Run selector for n frames at constant speed. Returns last result."""
    result = None
    for _ in range(n):
        result = selector.update(speed=speed, curvature_abs=curvature_abs)
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


# ---------------------------------------------------------------------------
# Stanley low-speed regime (Phase 2.9)
# ---------------------------------------------------------------------------

def _make_stanley_selector(
    stanley_max_speed_mps=5.0,
    stanley_up_hyst=0.5,
    stanley_down_hyst=0.5,
    stanley_blend_frames=10,
    stanley_min_hold_frames=15,
    pp_max_speed_mps=10.0,
) -> RegimeSelector:
    cfg = RegimeConfig(
        enabled=True,
        stanley_enabled=True,
        stanley_max_speed_mps=stanley_max_speed_mps,
        stanley_upshift_hysteresis_mps=stanley_up_hyst,
        stanley_downshift_hysteresis_mps=stanley_down_hyst,
        stanley_blend_frames=stanley_blend_frames,
        stanley_min_hold_frames=stanley_min_hold_frames,
        pp_max_speed_mps=pp_max_speed_mps,
        mpc_min_speed_mps=pp_max_speed_mps,  # preserve old speed logic for Stanley tests
        min_hold_frames=30,
    )
    return RegimeSelector(cfg)


def test_stanley_enum_value():
    """ControlRegime.STANLEY must be -1 for HDF5 backward compatibility."""
    assert int(ControlRegime.STANLEY) == -1


def test_stanley_downshift_on_low_speed():
    """Sustained low speed → Stanley after hold frames."""
    sel = _make_stanley_selector(stanley_max_speed_mps=5.0, stanley_down_hyst=0.5)
    # 4.4 m/s < 5.0 - 0.5 = 4.5 → should trigger downshift
    regime, blend = _run_n(sel, speed=4.4, n=50)
    assert regime == ControlRegime.STANLEY, \
        f"Expected STANLEY at v=4.4, got {regime}"
    assert blend == 1.0, "Blend should be fully settled after 50 frames"


def test_stanley_stays_pp_above_downshift_threshold():
    """Speed above lower threshold → stays PP (no Stanley)."""
    sel = _make_stanley_selector(stanley_max_speed_mps=5.0, stanley_down_hyst=0.5)
    # 4.6 m/s > 4.5 lower threshold → should stay PP
    regime, blend = _run_n(sel, speed=4.6, n=50)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP at v=4.6 (above lower threshold), got {regime}"


def test_stanley_upshift_to_pp():
    """After Stanley: speed rises above upper threshold → upshift to PP."""
    sel = _make_stanley_selector(stanley_max_speed_mps=5.0, stanley_up_hyst=0.5)
    # Settle into Stanley
    _run_n(sel, speed=3.0, n=50)
    assert sel.active_regime == ControlRegime.STANLEY

    # Speed rises above 5.0 + 0.5 = 5.5 m/s
    regime, blend = _run_n(sel, speed=6.0, n=50)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected upshift to PP at v=6.0, got {regime}"
    assert blend == 1.0


def test_stanley_hysteresis_no_chatter():
    """Speed oscillates around stanley_max → no more than 2 switches total."""
    sel = _make_stanley_selector(
        stanley_max_speed_mps=5.0,
        stanley_down_hyst=0.5,
        stanley_up_hyst=0.5,
        stanley_min_hold_frames=5,
    )
    switches = 0
    last_regime = ControlRegime.PURE_PURSUIT
    # Oscillate within the hysteresis band [4.5, 5.5] — no clean threshold crossing
    for i in range(120):
        speed = 4.6 if i % 2 == 0 else 5.4
        regime, _ = sel.update(speed=speed)
        if regime != last_regime:
            switches += 1
            last_regime = regime
    assert switches <= 2, f"Expected ≤2 switches in hysteresis band, got {switches}"


def test_stanley_full_speed_sweep():
    """Speed sweep 2→7→14 m/s exercises Stanley, PP, and LMPC regimes."""
    sel = _make_stanley_selector(
        stanley_max_speed_mps=5.0, pp_max_speed_mps=10.0,
        stanley_min_hold_frames=5,
    )
    # Override PP->LMPC hold to keep the test fast
    sel.config.min_hold_frames = 5
    regimes_seen = set()
    for speed in [2.0] * 20 + [7.0] * 20 + [14.0] * 20:
        regime, _ = sel.update(speed=speed)
        regimes_seen.add(regime)
    assert ControlRegime.STANLEY in regimes_seen, "Expected STANLEY in sweep"
    assert ControlRegime.PURE_PURSUIT in regimes_seen, "Expected PP in sweep"
    assert ControlRegime.LINEAR_MPC in regimes_seen, "Expected LMPC in sweep"


def test_stanley_disabled_when_stanley_enabled_false():
    """stanley_enabled=False → Stanley never activates even at low speed."""
    cfg = RegimeConfig(
        enabled=True,
        stanley_enabled=False,
        stanley_max_speed_mps=5.0,
    )
    sel = RegimeSelector(cfg)
    regime, _ = _run_n(sel, speed=2.0, n=50)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP when stanley_enabled=False, got {regime}"


def test_stanley_disabled_when_regime_disabled():
    """enabled=False → always PP, never Stanley."""
    cfg = RegimeConfig(enabled=False, stanley_enabled=True, stanley_max_speed_mps=5.0)
    sel = RegimeSelector(cfg)
    regime, blend = sel.update(speed=1.0)
    assert regime == ControlRegime.PURE_PURSUIT
    assert blend == 1.0


def test_stanley_blend_frames_shorter_than_pp():
    """Stanley blend is 10 frames; PP↔LMPC blend is 15 — verify Stanley uses shorter."""
    sel = _make_stanley_selector(
        stanley_blend_frames=10,
        stanley_min_hold_frames=5,
    )
    sel.config.blend_frames = 15  # PP↔LMPC uses 15
    # Trigger Stanley
    _run_n(sel, speed=3.0, n=6)  # hold=5 → switches on frame 6
    assert sel.active_regime == ControlRegime.STANLEY
    # Collect blend after first switch
    blends = [sel.blend_progress]
    for _ in range(10):
        _, w = sel.update(speed=3.0)
        blends.append(w)
    assert blends[-1] == 1.0, "Stanley blend should reach 1.0 in 10 frames"


def test_stanley_inhibit_curvature_blocks_downshift_from_mpc():
    """On highway curve (κ>threshold), brake below Stanley speed → stay LINEAR_MPC."""
    cfg = RegimeConfig(
        enabled=True,
        stanley_enabled=True,
        stanley_max_speed_mps=5.0,
        stanley_upshift_hysteresis_mps=0.5,
        stanley_downshift_hysteresis_mps=0.5,
        stanley_blend_frames=5,
        stanley_min_hold_frames=5,
        stanley_inhibit_curvature_abs=0.005,
        mpc_min_speed_mps=3.0,
        mpc_max_curvature=0.020,
        mpc_curvature_hysteresis=0.003,
        upshift_hysteresis_mps=1.0,
        downshift_hysteresis_mps=1.0,
        pp_max_speed_mps=10.0,
        min_hold_frames=5,
        blend_frames=10,
    )
    sel = RegimeSelector(cfg)
    _run_n(sel, speed=12.0, n=25, curvature_abs=0.01)
    assert sel.active_regime == ControlRegime.LINEAR_MPC
    regime, _ = _run_n(sel, speed=4.4, n=25, curvature_abs=0.01)
    assert regime == ControlRegime.LINEAR_MPC, (
        f"Expected LINEAR_MPC (Stanley inhibited on curve), got {regime}"
    )


def test_stanley_exits_when_curvature_spikes_while_slow():
    """If already in Stanley, entering a sharp curve forces exit from Stanley."""
    cfg = RegimeConfig(
        enabled=True,
        stanley_enabled=True,
        stanley_max_speed_mps=5.0,
        stanley_upshift_hysteresis_mps=0.5,
        stanley_downshift_hysteresis_mps=0.5,
        stanley_blend_frames=5,
        stanley_min_hold_frames=5,
        stanley_inhibit_curvature_abs=0.005,
        mpc_min_speed_mps=3.0,
        mpc_max_curvature=0.020,
        mpc_curvature_hysteresis=0.003,
        upshift_hysteresis_mps=1.0,
        downshift_hysteresis_mps=1.0,
        pp_max_speed_mps=10.0,
        min_hold_frames=5,
        blend_frames=10,
    )
    sel = RegimeSelector(cfg)
    _run_n(sel, speed=3.0, n=30, curvature_abs=0.0001)
    assert sel.active_regime == ControlRegime.STANLEY
    regime, _ = _run_n(sel, speed=3.0, n=30, curvature_abs=0.015)
    assert regime != ControlRegime.STANLEY


def test_stanley_inhibit_disabled_zero_threshold_legacy():
    """stanley_inhibit_curvature_abs=0 restores downshift on curve (legacy)."""
    cfg = RegimeConfig(
        enabled=True,
        stanley_enabled=True,
        stanley_max_speed_mps=5.0,
        stanley_upshift_hysteresis_mps=0.5,
        stanley_downshift_hysteresis_mps=0.5,
        stanley_blend_frames=5,
        stanley_min_hold_frames=5,
        stanley_inhibit_curvature_abs=0.0,
        mpc_min_speed_mps=3.0,
        mpc_max_curvature=0.020,
        mpc_curvature_hysteresis=0.003,
        upshift_hysteresis_mps=1.0,
        downshift_hysteresis_mps=1.0,
        pp_max_speed_mps=10.0,
        min_hold_frames=5,
        blend_frames=10,
    )
    sel = RegimeSelector(cfg)
    _run_n(sel, speed=12.0, n=25, curvature_abs=0.01)
    regime, _ = _run_n(sel, speed=4.4, n=25, curvature_abs=0.01)
    assert regime == ControlRegime.STANLEY


def test_stanley_from_config_loaded():
    """RegimeConfig.from_config loads stanley_* fields correctly."""
    cfg_dict = {
        "control": {
            "regime": {
                "enabled": True,
                "stanley_enabled": True,
                "stanley_max_speed_mps": 6.0,
                "stanley_upshift_hysteresis_mps": 0.3,
                "stanley_blend_frames": 8,
                "stanley_inhibit_curvature_abs": 0.006,
            }
        }
    }
    rc = RegimeConfig.from_config(cfg_dict)
    assert rc.stanley_enabled is True
    assert rc.stanley_max_speed_mps == 6.0
    assert rc.stanley_upshift_hysteresis_mps == 0.3
    assert rc.stanley_blend_frames == 8
    assert rc.stanley_min_hold_frames == 15  # default
    assert rc.stanley_inhibit_curvature_abs == 0.006


# ---------------------------------------------------------------------------
# Step 4: Curvature guard tests
# ---------------------------------------------------------------------------

def test_curvature_guard_forces_pp_on_tight_curve():
    """MPC active at 8 m/s, κ=0.025 (R40) → curvature guard forces PP."""
    sel = _make_selector(mpc_min_speed_mps=3.0, min_hold_frames=5)
    # Settle into MPC at 8 m/s with gentle curvature
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # Curvature rises above 0.020 → should force PP
    regime, _ = _run_n(sel, speed=8.0, n=10, curvature_abs=0.025)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP at κ=0.025 (tight curve), got {regime}"


def test_curvature_guard_allows_mpc_on_gentle_curve():
    """κ=0.010 (R100) at 8 m/s → MPC stays active."""
    sel = _make_selector(mpc_min_speed_mps=3.0, min_hold_frames=5)
    # Settle into MPC
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # κ=0.010 < 0.020 → MPC should stay
    regime, _ = _run_n(sel, speed=8.0, n=10, curvature_abs=0.010)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC at κ=0.010 (gentle curve), got {regime}"


def test_curvature_guard_hysteresis():
    """Oscillate κ between 0.018 and 0.021 → no chatter (≤1 switch)."""
    sel = _make_selector(mpc_min_speed_mps=3.0, min_hold_frames=5)
    # Start in MPC
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    switches = 0
    last_regime = sel.active_regime
    for i in range(120):
        kappa = 0.018 if i % 2 == 0 else 0.021
        regime, _ = sel.update(speed=8.0, curvature_abs=kappa)
        if regime != last_regime:
            switches += 1
            last_regime = regime

    # At most 1 transition: MPC→PP when κ=0.021 exceeds threshold.
    # Hysteresis (0.017) prevents re-engaging MPC at κ=0.018.
    assert switches <= 1, f"Expected ≤1 curvature switch, got {switches}"


def test_mpc_activates_at_low_speed():
    """v=5.5 m/s with low κ → MPC (was PP at old 10.0 threshold)."""
    sel = _make_selector(mpc_min_speed_mps=3.0, min_hold_frames=5)
    # 5.5 > 3.0 + 1.0 = 4.0 → should upshift to MPC
    regime, _ = _run_n(sel, speed=5.5, n=10, curvature_abs=0.005)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC at v=5.5 with mpc_min=3.0, got {regime}"


def test_mpc_blocked_below_min_speed():
    """v=1.5 m/s → should stay PP (or Stanley), never MPC."""
    sel = _make_selector(mpc_min_speed_mps=3.0, min_hold_frames=5, pp_max_speed_mps=10.0)
    sel.config.stanley_enabled = False  # isolate speed logic
    regime, _ = _run_n(sel, speed=1.5, n=20, curvature_abs=0.005)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP at v=1.5 (below mpc_min), got {regime}"


def test_curvature_zero_does_not_affect_speed_logic():
    """κ=0.0 (no data) → curvature guard inactive, pure speed logic."""
    sel = _make_selector(mpc_min_speed_mps=3.0, min_hold_frames=5)
    # With κ=0.0, speed logic should work normally
    regime, _ = _run_n(sel, speed=5.5, n=10, curvature_abs=0.0)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC at v=5.5 with κ=0.0, got {regime}"


def test_full_sweep_speed_and_curvature():
    """Sweep v and κ → verify correct regimes across the envelope."""
    sel = _make_selector(
        mpc_min_speed_mps=3.0, min_hold_frames=3,
        mpc_max_curvature=0.020, mpc_curvature_hysteresis=0.003,
    )
    sel.config.stanley_enabled = False  # isolate PP/MPC logic

    # Low speed → PP
    _run_n(sel, speed=1.5, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.PURE_PURSUIT

    # Medium speed, low κ → MPC
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # Medium speed, high κ → PP (curvature guard)
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.025)
    assert sel.active_regime == ControlRegime.PURE_PURSUIT

    # Back to low κ — must clear hysteresis (κ < 0.017) to re-engage MPC
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.010)
    assert sel.active_regime == ControlRegime.LINEAR_MPC


def test_from_config_loads_curvature_fields():
    """RegimeConfig.from_config loads mpc_min_speed_mps and curvature fields."""
    cfg_dict = {
        "control": {
            "regime": {
                "enabled": True,
                "mpc_min_speed_mps": 5.0,
                "mpc_max_curvature": 0.015,
                "mpc_curvature_hysteresis": 0.002,
            }
        }
    }
    rc = RegimeConfig.from_config(cfg_dict)
    assert rc.mpc_min_speed_mps == 5.0
    assert rc.mpc_max_curvature == 0.015
    assert rc.mpc_curvature_hysteresis == 0.002
