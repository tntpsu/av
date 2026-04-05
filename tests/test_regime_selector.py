"""
Unit tests for RegimeSelector (lateral accel budget + Stanley low-speed regime).

Tests cover: disabled passthrough, lateral accel budget PP↔MPC routing, hysteresis
dead band, hold timer, blend weight ramp, MPC fallback override, Stanley low-speed
regime, cross-track budget sweep, and deprecated param warnings.

Decision logic:
  Stanley:  v < stanley_max_speed_mps − hysteresis (speed-based, not a proxy)
  PP→MPC:   κ×v² < threshold − hyst  AND  v > min_speed_absolute
  MPC→PP:   κ×v² > threshold + hyst  OR   v < min_speed_absolute

Default thresholds: threshold=1.5 m/s², hyst=0.3, min_speed=2.0 m/s
  → PP→MPC upshift:   a_lat < 1.2 m/s²
  → MPC→PP downshift:  a_lat > 1.8 m/s²

Run:  pytest tests/test_regime_selector.py -v
"""

import logging
import pytest
from control.regime_selector import ControlRegime, RegimeConfig, RegimeSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_selector(
    enabled=True,
    mpc_max_lateral_accel_mps2=1.5,
    mpc_lateral_accel_hysteresis_mps2=0.3,
    mpc_min_speed_absolute_mps=2.0,
    blend_frames=15,
    min_hold_frames=30,
    stanley_enabled=False,
) -> RegimeSelector:
    """Create a selector with lateral accel budget params (new architecture)."""
    cfg = RegimeConfig(
        enabled=enabled,
        mpc_max_lateral_accel_mps2=mpc_max_lateral_accel_mps2,
        mpc_lateral_accel_hysteresis_mps2=mpc_lateral_accel_hysteresis_mps2,
        mpc_min_speed_absolute_mps=mpc_min_speed_absolute_mps,
        blend_frames=blend_frames,
        min_hold_frames=min_hold_frames,
        stanley_enabled=stanley_enabled,
    )
    return RegimeSelector(cfg)


def _run_n(selector, speed, n=35, curvature_abs=0.0) -> tuple:
    """Run selector for n frames at constant speed. Returns last result."""
    result = None
    for _ in range(n):
        result = selector.update(speed=speed, curvature_abs=curvature_abs)
    return result


def _run_n_with_fallback(selector, speed, n=35):
    for _ in range(n):
        selector.update(speed=speed, mpc_fallback_active=True)


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
# 2. Speed below absolute floor → PP (not a proxy — MPC needs nonzero speed)
# ---------------------------------------------------------------------------

def test_speed_floor_forces_pp():
    """v < mpc_min_speed_absolute (2.0 m/s) → PP regardless of a_lat."""
    sel = _make_selector(min_hold_frames=5)
    sel.config.stanley_enabled = False
    regime, _ = _run_n(sel, speed=1.5, n=20, curvature_abs=0.001)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP at v=1.5 (below speed floor), got {regime}"


def test_speed_floor_forces_pp_at_zero_speed():
    """v=0 → PP. Zero speed is below mpc_min_speed_absolute."""
    sel = _make_selector(min_hold_frames=5)
    sel.config.stanley_enabled = False
    regime, _ = _run_n(sel, speed=0.0, n=20)
    assert regime == ControlRegime.PURE_PURSUIT


# ---------------------------------------------------------------------------
# 3. Low a_lat → upshift to LINEAR_MPC
# ---------------------------------------------------------------------------

def test_low_alat_upshift():
    """v=12 m/s, κ=0.005 → a_lat=0.72 < 1.2 → MPC after hold timer."""
    sel = _make_selector(min_hold_frames=30)
    # a_lat = 0.005 × 144 = 0.72; well below upshift threshold 1.2
    regime, blend = _run_n(sel, speed=12.0, n=35, curvature_abs=0.005)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected LINEAR_MPC at a_lat=0.72, got {regime}"


def test_zero_curvature_upshift():
    """κ=0.0 → a_lat=0 → MPC at any speed above floor."""
    sel = _make_selector(min_hold_frames=5)
    regime, _ = _run_n(sel, speed=5.5, n=10, curvature_abs=0.0)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC at v=5.5, κ=0 (a_lat=0), got {regime}"


# ---------------------------------------------------------------------------
# 4. High a_lat → downshift to PP
# ---------------------------------------------------------------------------

def test_high_alat_downshift():
    """MPC active, then a_lat rises above threshold + hyst → PP."""
    sel = _make_selector(min_hold_frames=5)
    # Settle into MPC: v=12, κ=0.005 → a_lat=0.72
    _run_n(sel, speed=12.0, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # Increase curvature: v=8.7, κ=0.025 → a_lat=1.89 > 1.8
    regime, _ = _run_n(sel, speed=8.7, n=10, curvature_abs=0.025)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP at a_lat=1.89, got {regime}"


def test_mpc_stays_when_alat_below_downshift():
    """MPC active, a_lat in dead band [1.2, 1.8] → stays MPC."""
    sel = _make_selector(min_hold_frames=5)
    # Settle into MPC
    _run_n(sel, speed=12.0, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # a_lat = 0.010 × 144 = 1.44 — in dead band (< 1.8 downshift)
    regime, _ = _run_n(sel, speed=12.0, n=10, curvature_abs=0.010)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC to stay at a_lat=1.44 (in dead band), got {regime}"


# ---------------------------------------------------------------------------
# 5. Lateral accel budget hysteresis → no chatter in dead band
# ---------------------------------------------------------------------------

def test_budget_hysteresis_no_chatter():
    """a_lat oscillates between 1.3 and 1.7 (in dead band) → ≤1 switch."""
    sel = _make_selector(min_hold_frames=5)
    # Settle into MPC at low a_lat
    _run_n(sel, speed=12.0, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    switches = 0
    last_regime = sel.active_regime
    for i in range(120):
        # Oscillate a_lat: v=10, κ alternates
        # κ=0.013 → a_lat=1.30, κ=0.017 → a_lat=1.70
        kappa = 0.013 if i % 2 == 0 else 0.017
        regime, _ = sel.update(speed=10.0, curvature_abs=kappa)
        if regime != last_regime:
            switches += 1
            last_regime = regime

    assert switches <= 1, f"Expected ≤1 switch in dead band, got {switches}"


def test_budget_hysteresis_band():
    """Verify exact dead band boundaries: upshift at <1.2, no upshift at 1.25."""
    # a_lat = 1.25 → NOT < threshold-hyst (1.2) → stays PP
    sel = _make_selector(min_hold_frames=5)
    # v=10, κ=0.0125 → a_lat=1.25
    regime, _ = _run_n(sel, speed=10.0, n=20, curvature_abs=0.0125)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP at a_lat=1.25 (inside dead band), got {regime}"

    # a_lat = 1.10 → < 1.2 → should upshift to MPC
    sel2 = _make_selector(min_hold_frames=5)
    # v=10, κ=0.011 → a_lat=1.10
    regime, _ = _run_n(sel2, speed=10.0, n=20, curvature_abs=0.011)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC at a_lat=1.10 (below upshift threshold), got {regime}"


# ---------------------------------------------------------------------------
# 6. Upshift requires hold timer
# ---------------------------------------------------------------------------

def test_upshift_requires_hold():
    sel = _make_selector(min_hold_frames=30, blend_frames=15)
    # a_lat=0 (straight), speed=12 → desired=MPC but hold not met
    for frame in range(29):
        regime, _ = sel.update(speed=12.0, curvature_abs=0.0)
        assert regime == ControlRegime.PURE_PURSUIT, \
            f"Should still be PP at frame {frame}, hold not met"

    # Frame 30: hold counter hits min_hold_frames → switch
    regime, blend = sel.update(speed=12.0, curvature_abs=0.0)
    assert regime == ControlRegime.LINEAR_MPC, \
        "Expected switch to LINEAR_MPC at frame 30"
    assert blend < 1.0, "Blend should be < 1.0 immediately after switch"


# ---------------------------------------------------------------------------
# 7. Blend weight ramps monotonically 0 → 1 after switch
# ---------------------------------------------------------------------------

def test_blend_weight_ramp():
    sel = _make_selector(min_hold_frames=5, blend_frames=15)
    # Trigger switch quickly
    _run_n(sel, speed=12.0, n=6, curvature_abs=0.0)
    assert sel.active_regime == ControlRegime.LINEAR_MPC
    assert sel.blend_progress < 1.0

    # Collect blend values over 20 frames
    blends = []
    for _ in range(20):
        _, w = sel.update(speed=12.0, curvature_abs=0.0)
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
    _run_n(sel, speed=12.0, n=10, curvature_abs=0.0)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # Activate fallback — after hold frames, should be back to PP
    _run_n_with_fallback(sel, speed=12.0, n=35)
    regime, _ = sel.update(speed=12.0, mpc_fallback_active=True)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP during MPC fallback, got {regime}"


# ---------------------------------------------------------------------------
# 9. Reset clears state
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    sel = _make_selector(min_hold_frames=5)
    _run_n(sel, speed=12.0, n=10, curvature_abs=0.0)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    sel.reset()
    assert sel.active_regime == ControlRegime.PURE_PURSUIT
    assert sel.blend_progress == 1.0
    assert sel.last_lateral_accel == 0.0


# ---------------------------------------------------------------------------
# 10. last_lateral_accel property tracks κ×v²
# ---------------------------------------------------------------------------

def test_last_lateral_accel_property():
    """last_lateral_accel returns κ×v² from the most recent update."""
    sel = _make_selector(min_hold_frames=5)
    sel.update(speed=10.0, curvature_abs=0.015)
    assert abs(sel.last_lateral_accel - 1.5) < 0.01, \
        f"Expected a_lat≈1.5, got {sel.last_lateral_accel}"


# ---------------------------------------------------------------------------
# 11. peek() returns current state without advancing
# ---------------------------------------------------------------------------

def test_peek_does_not_advance():
    sel = _make_selector(min_hold_frames=5)
    _run_n(sel, speed=12.0, n=6, curvature_abs=0.0)
    regime1, blend1 = sel.peek()
    regime2, blend2 = sel.peek()
    assert regime1 == regime2
    assert blend1 == blend2


# ---------------------------------------------------------------------------
# Cross-track budget tests (the specific scenarios from the plan)
# ---------------------------------------------------------------------------

def test_budget_r40_at_8p7_forces_pp():
    """mixed_radius R40 corner: v=8.7, κ=0.025 → a_lat=1.89 > 1.8 → PP.

    This is THE root cause scenario: MPC was stuck on R40 under the old
    speed-threshold logic because 8.7 m/s fell in the hysteresis dead band.
    The lateral accel budget correctly identifies this as beyond MPC's
    linear tire region.
    """
    sel = _make_selector(min_hold_frames=5)
    # Start in MPC on a gentle section
    _run_n(sel, speed=11.4, n=10, curvature_abs=0.002)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # Enter R40 corner: a_lat = 0.025 × 8.7² = 1.89
    regime, _ = _run_n(sel, speed=8.7, n=10, curvature_abs=0.025)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP on R40 at 8.7 m/s (a_lat=1.89), got {regime}"


def test_budget_r200_at_11p4_keeps_mpc():
    """mixed_radius R200 section: v=11.4, κ=0.005 → a_lat=0.65 → MPC."""
    sel = _make_selector(min_hold_frames=5)
    regime, _ = _run_n(sel, speed=11.4, n=10, curvature_abs=0.005)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC on R200 at 11.4 m/s (a_lat=0.65), got {regime}"


def test_budget_r100_at_11p4_keeps_mpc():
    """hill_highway R100: v=11.4, κ=0.010 → a_lat=1.30 → MPC stays (in dead band).

    a_lat=1.30 is between upshift (1.2) and downshift (1.8) thresholds.
    Can't upshift from PP, but if already in MPC it stays — correct behavior.
    """
    sel = _make_selector(min_hold_frames=5)
    # Enter MPC on gentle section first
    _run_n(sel, speed=11.4, n=10, curvature_abs=0.002)  # a_lat=0.26
    assert sel.active_regime == ControlRegime.LINEAR_MPC
    # Corner: a_lat=1.30 < 1.8 → MPC stays
    regime, _ = _run_n(sel, speed=11.4, n=10, curvature_abs=0.010)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC on R100 at 11.4 m/s (a_lat=1.30), got {regime}"


def test_budget_r50_at_9p9_forces_pp():
    """Circle/oval R50: v=9.9, κ=0.020 → a_lat=1.96 → PP."""
    sel = _make_selector(min_hold_frames=5)
    _run_n(sel, speed=11.4, n=10, curvature_abs=0.002)  # start MPC
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    regime, _ = _run_n(sel, speed=9.9, n=10, curvature_abs=0.020)
    assert regime == ControlRegime.PURE_PURSUIT, \
        f"Expected PP on R50 at 9.9 m/s (a_lat=1.96), got {regime}"


def test_budget_r60_at_8p9_keeps_mpc():
    """mixed_radius R60: v=8.9, κ=0.017 → a_lat=1.35 → MPC stays (in dead band)."""
    sel = _make_selector(min_hold_frames=5)
    # Enter MPC on gentle section first
    _run_n(sel, speed=11.4, n=10, curvature_abs=0.002)  # a_lat=0.26
    assert sel.active_regime == ControlRegime.LINEAR_MPC
    # R60 corner: a_lat=1.35 < 1.8 → MPC stays
    regime, _ = _run_n(sel, speed=8.9, n=10, curvature_abs=0.017)
    assert regime == ControlRegime.LINEAR_MPC, \
        f"Expected MPC on R60 at 8.9 m/s (a_lat=1.35), got {regime}"


@pytest.mark.parametrize("track,speed,kappa,expected_regime", [
    # Track corners from the cross-track impact table in the plan
    ("highway R500", 11.4, 0.002, ControlRegime.LINEAR_MPC),     # a_lat=0.26
    ("sweeping R300", 11.4, 0.003, ControlRegime.LINEAR_MPC),    # a_lat=0.43
    ("autobahn R600", 11.4, 0.002, ControlRegime.LINEAR_MPC),    # a_lat=0.22
    ("hill R100", 11.4, 0.010, ControlRegime.LINEAR_MPC),        # a_lat=1.30
    ("mixed R200", 11.4, 0.005, ControlRegime.LINEAR_MPC),       # a_lat=0.65
    ("mixed R150", 11.4, 0.007, ControlRegime.LINEAR_MPC),       # a_lat=0.87
    ("mixed R60", 8.9, 0.017, ControlRegime.LINEAR_MPC),         # a_lat=1.33
    ("mixed R40", 8.7, 0.025, ControlRegime.PURE_PURSUIT),       # a_lat=1.89
    ("circle R50", 9.9, 0.020, ControlRegime.PURE_PURSUIT),      # a_lat=1.96
    ("oval R50", 9.9, 0.020, ControlRegime.PURE_PURSUIT),        # a_lat=1.96
], ids=lambda x: x if isinstance(x, str) else "")
def test_budget_sweep_all_tracks(track, speed, kappa, expected_regime):
    """Parameterized cross-track budget sweep.

    Each row represents the tightest corner on a track at its typical
    operating speed. The lateral accel budget must route every track
    correctly without per-track overrides.
    """
    sel = _make_selector(min_hold_frames=3)
    sel.config.stanley_enabled = False

    a_lat = kappa * speed * speed
    threshold = 1.5
    hyst = 0.3

    if expected_regime == ControlRegime.PURE_PURSUIT:
        # Start in MPC to test downshift
        _run_n(sel, speed=11.4, n=10, curvature_abs=0.002)
        assert sel.active_regime == ControlRegime.LINEAR_MPC
        regime, _ = _run_n(sel, speed=speed, n=10, curvature_abs=kappa)
    elif a_lat > threshold - hyst:
        # a_lat in dead band — can only stay in MPC, can't upshift from PP
        _run_n(sel, speed=11.4, n=10, curvature_abs=0.002)
        assert sel.active_regime == ControlRegime.LINEAR_MPC
        regime, _ = _run_n(sel, speed=speed, n=10, curvature_abs=kappa)
    else:
        # Test upshift from PP (a_lat below upshift threshold)
        regime, _ = _run_n(sel, speed=speed, n=10, curvature_abs=kappa)

    assert regime == expected_regime, \
        f"{track}: v={speed}, κ={kappa}, a_lat={kappa * speed**2:.2f} → " \
        f"expected {expected_regime.name}, got {regime.name}"


# ---------------------------------------------------------------------------
# Full envelope sweep
# ---------------------------------------------------------------------------

def test_full_sweep_speed_and_budget():
    """Sweep v and κ → verify correct regimes across the envelope."""
    sel = _make_selector(min_hold_frames=3)
    sel.config.stanley_enabled = False

    # Low speed → PP (speed floor)
    _run_n(sel, speed=1.5, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.PURE_PURSUIT

    # Medium speed, low κ → MPC (a_lat = 0.32)
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.005)
    assert sel.active_regime == ControlRegime.LINEAR_MPC

    # Medium speed, high κ → PP (a_lat = 64×0.025 = 1.6; from MPC, need >1.8)
    # Actually 1.6 < 1.8, so stays MPC. Use higher κ to exceed downshift.
    # v=8, κ=0.030 → a_lat=1.92 > 1.8
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.030)
    assert sel.active_regime == ControlRegime.PURE_PURSUIT

    # Back to low κ → must clear upshift threshold (a_lat < 1.2)
    # v=8, κ=0.010 → a_lat=0.64 < 1.2 → MPC
    _run_n(sel, speed=8.0, n=10, curvature_abs=0.010)
    assert sel.active_regime == ControlRegime.LINEAR_MPC


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_regime_config_from_config():
    cfg = {
        "control": {
            "regime": {
                "enabled": True,
                "mpc_max_lateral_accel_mps2": 2.0,
                "mpc_lateral_accel_hysteresis_mps2": 0.4,
                "mpc_min_speed_absolute_mps": 3.0,
                "min_hold_frames": 45,
            }
        }
    }
    rc = RegimeConfig.from_config(cfg)
    assert rc.enabled is True
    assert rc.mpc_max_lateral_accel_mps2 == 2.0
    assert rc.mpc_lateral_accel_hysteresis_mps2 == 0.4
    assert rc.mpc_min_speed_absolute_mps == 3.0
    assert rc.min_hold_frames == 45
    assert rc.blend_frames == 15   # default


def test_regime_config_from_empty_config():
    """Missing control.regime key → all defaults."""
    rc = RegimeConfig.from_config({})
    assert rc.enabled is False
    assert rc.mpc_max_lateral_accel_mps2 == 1.5
    assert rc.mpc_lateral_accel_hysteresis_mps2 == 0.3
    assert rc.mpc_min_speed_absolute_mps == 2.0


def test_from_config_loads_lateral_accel_fields():
    """RegimeConfig.from_config loads new lateral accel budget fields."""
    cfg_dict = {
        "control": {
            "regime": {
                "enabled": True,
                "mpc_max_lateral_accel_mps2": 1.8,
                "mpc_lateral_accel_hysteresis_mps2": 0.25,
                "mpc_min_speed_absolute_mps": 2.5,
            }
        }
    }
    rc = RegimeConfig.from_config(cfg_dict)
    assert rc.mpc_max_lateral_accel_mps2 == 1.8
    assert rc.mpc_lateral_accel_hysteresis_mps2 == 0.25
    assert rc.mpc_min_speed_absolute_mps == 2.5


def test_deprecated_params_log_warning(caplog):
    """Old proxy params trigger deprecation warning when loaded from config."""
    cfg_dict = {
        "control": {
            "regime": {
                "enabled": True,
                "mpc_min_speed_mps": 5.0,
                "mpc_max_curvature": 0.015,
                "mpc_curvature_hysteresis": 0.002,
                "pp_max_speed_mps": 12.0,
                "upshift_hysteresis_mps": 1.5,
                "downshift_hysteresis_mps": 1.5,
            }
        }
    }
    with caplog.at_level(logging.WARNING):
        rc = RegimeConfig.from_config(cfg_dict)

    # All 6 deprecated params should trigger a warning
    deprecated_keys = [
        "mpc_min_speed_mps", "mpc_max_curvature", "mpc_curvature_hysteresis",
        "pp_max_speed_mps", "upshift_hysteresis_mps", "downshift_hysteresis_mps",
    ]
    for key in deprecated_keys:
        assert any(key in record.message for record in caplog.records), \
            f"Expected deprecation warning for '{key}'"

    # Values should still be loaded (backward compat)
    assert rc.mpc_min_speed_mps == 5.0
    assert rc.mpc_max_curvature == 0.015


def test_from_config_loads_deprecated_curvature_fields():
    """Deprecated fields still load correctly for backward compat."""
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


# ---------------------------------------------------------------------------
# Stanley low-speed regime (Phase 2.9 — unchanged by lateral accel budget)
# ---------------------------------------------------------------------------

def _make_stanley_selector(
    stanley_max_speed_mps=5.0,
    stanley_up_hyst=0.5,
    stanley_down_hyst=0.5,
    stanley_blend_frames=10,
    stanley_min_hold_frames=15,
    stanley_inhibit_curvature_abs=0.004,
    mpc_max_lateral_accel_mps2=1.5,
    mpc_lateral_accel_hysteresis_mps2=0.3,
    mpc_min_speed_absolute_mps=2.0,
) -> RegimeSelector:
    cfg = RegimeConfig(
        enabled=True,
        stanley_enabled=True,
        stanley_max_speed_mps=stanley_max_speed_mps,
        stanley_upshift_hysteresis_mps=stanley_up_hyst,
        stanley_downshift_hysteresis_mps=stanley_down_hyst,
        stanley_blend_frames=stanley_blend_frames,
        stanley_min_hold_frames=stanley_min_hold_frames,
        stanley_inhibit_curvature_abs=stanley_inhibit_curvature_abs,
        mpc_max_lateral_accel_mps2=mpc_max_lateral_accel_mps2,
        mpc_lateral_accel_hysteresis_mps2=mpc_lateral_accel_hysteresis_mps2,
        mpc_min_speed_absolute_mps=mpc_min_speed_absolute_mps,
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


def test_stanley_no_downshift_above_threshold():
    """Speed above lower threshold → no Stanley (goes to MPC or PP, not Stanley)."""
    sel = _make_stanley_selector(stanley_max_speed_mps=5.0, stanley_down_hyst=0.5)
    # 4.6 m/s > 4.5 lower threshold → should NOT enter Stanley
    # With κ=0, a_lat=0 < 1.2 → upshifts to MPC (correct with budget logic)
    regime, blend = _run_n(sel, speed=4.6, n=50)
    assert regime != ControlRegime.STANLEY, \
        f"Expected no Stanley at v=4.6 (above lower threshold), got {regime}"


def test_stanley_exits_on_speed_rise():
    """After Stanley: speed rises above upper threshold → exits Stanley.

    With κ=0, a_lat=0 → budget allows MPC upshift directly from Stanley.
    The key assertion is that Stanley is exited, not specifically to PP.
    """
    sel = _make_stanley_selector(stanley_max_speed_mps=5.0, stanley_up_hyst=0.5)
    # Settle into Stanley
    _run_n(sel, speed=3.0, n=50)
    assert sel.active_regime == ControlRegime.STANLEY

    # Speed rises above 5.0 + 0.5 = 5.5 m/s
    regime, blend = _run_n(sel, speed=6.0, n=50)
    assert regime != ControlRegime.STANLEY, \
        f"Expected to exit Stanley at v=6.0, got {regime}"
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
    """Speed sweep exercises Stanley, PP, and LMPC regimes.

    With the lateral accel budget, PP only appears when a_lat is in the
    dead band or above the upshift threshold. We use curvature to create
    a PP-worthy a_lat at mid speed.
    """
    sel = _make_stanley_selector(
        stanley_max_speed_mps=5.0,
        stanley_min_hold_frames=5,
    )
    sel.config.min_hold_frames = 5

    regimes_seen = set()
    # Phase 1: low speed → Stanley (v=2.0, κ=0)
    for _ in range(20):
        regime, _ = sel.update(speed=2.0, curvature_abs=0.0)
        regimes_seen.add(regime)

    # Phase 2: mid speed, high curvature → PP (v=7.0, κ=0.030 → a_lat=1.47)
    # From Stanley, exits to PP because a_lat is in dead band (1.2-1.8)
    # and not below upshift threshold
    for _ in range(20):
        regime, _ = sel.update(speed=7.0, curvature_abs=0.030)
        regimes_seen.add(regime)

    # Phase 3: high speed, straight → MPC (v=14.0, κ=0 → a_lat=0)
    for _ in range(20):
        regime, _ = sel.update(speed=14.0, curvature_abs=0.0)
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
    # With stanley_enabled=False and v=2.0 > min_speed_absolute=2.0,
    # a_lat=0 → should upshift to MPC
    assert regime in (ControlRegime.PURE_PURSUIT, ControlRegime.LINEAR_MPC), \
        f"Expected PP or MPC when stanley_enabled=False, got {regime}"
    assert regime != ControlRegime.STANLEY


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
    """On highway curve (κ>threshold), brake below Stanley speed → stay LINEAR_MPC.

    With lateral accel budget: v=12, κ=0.01 → a_lat=1.44 < 1.8 → MPC stays.
    Then v=4.4, κ=0.01 → a_lat=0.194 < 1.8 → MPC still stays (no downshift).
    Stanley inhibited by curvature → doesn't override.
    """
    cfg = RegimeConfig(
        enabled=True,
        stanley_enabled=True,
        stanley_max_speed_mps=5.0,
        stanley_upshift_hysteresis_mps=0.5,
        stanley_downshift_hysteresis_mps=0.5,
        stanley_blend_frames=5,
        stanley_min_hold_frames=5,
        stanley_inhibit_curvature_abs=0.005,
        mpc_max_lateral_accel_mps2=1.5,
        mpc_lateral_accel_hysteresis_mps2=0.3,
        mpc_min_speed_absolute_mps=2.0,
        min_hold_frames=5,
        blend_frames=10,
    )
    sel = RegimeSelector(cfg)
    # Enter MPC with low κ first (a_lat = 0.002×144 = 0.288 < 1.2)
    _run_n(sel, speed=12.0, n=25, curvature_abs=0.002)
    assert sel.active_regime == ControlRegime.LINEAR_MPC
    # Then enter curve: v=4.4, κ=0.01 → a_lat=0.194 < 1.8 → MPC stays
    # Stanley inhibited by curvature → doesn't override
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
        mpc_max_lateral_accel_mps2=1.5,
        mpc_lateral_accel_hysteresis_mps2=0.3,
        mpc_min_speed_absolute_mps=2.0,
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
        mpc_max_lateral_accel_mps2=1.5,
        mpc_lateral_accel_hysteresis_mps2=0.3,
        mpc_min_speed_absolute_mps=2.0,
        min_hold_frames=5,
        blend_frames=10,
    )
    sel = RegimeSelector(cfg)
    # Enter MPC with low κ first (a_lat = 0.002×144 = 0.288 < 1.2)
    _run_n(sel, speed=12.0, n=25, curvature_abs=0.002)
    assert sel.active_regime == ControlRegime.LINEAR_MPC
    # Drop speed: v=4.4, κ=0.01 → a_lat=0.194 (MPC stays, no downshift needed)
    # But Stanley wants to activate: 4.4 < 4.5 (5.0-0.5), inhibit disabled → STANLEY
    # _compute_desired: Stanley check fires first → returns STANLEY
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
