"""Tests for jerk event classifier and e_lat dropout rate limiter.

WHAT THESE TESTS COVER
-----------------------
1. ``TestJerkEventClassifier``
   - Classifier returns empty list when no events exceed threshold.
   - Dropout recovery event is correctly labelled ``dropout_recovery``.
   - Dropout start event is correctly labelled ``dropout_start``.
   - Stale-to-fresh transition is labelled ``stale_recovery``.
   - Curve oscillation (large e_lat, gate active) is labelled ``curve_oscillation``.
   - Straight noise (small e_lat, no kappa) is labelled ``straight_noise``.
   - Gap-artifact frames (anomalous dt) are filtered out.

2. ``TestElatRampRateLimiter``
   - With rate limiter OFF (0.0): a large step passes through unchanged.
   - With rate limiter ON: a large step is capped to the configured rate.
   - Multiple consecutive frames ramp up to target rather than jumping.
   - Normal small curve drift (< rate limit) passes through unmodified.
   - Ramp state resets to zero on ``reset()``.
   - ``elat_ramp_active`` flag is True only when clamping occurred.
"""

import numpy as np
import pytest

from control.mpc_controller import MPCController, MPCParams
from tools.drive_summary_core import classify_jerk_events


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_WHEELBASE_M   = 2.5
_MAX_STEER_RAD = 0.5236
_DT            = 0.033


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mpc(ramp_rate: float = 0.0, **extra) -> MPCController:
    """Build a minimal MPCController with optional dropout rate limiter."""
    cfg = {"trajectory": {"mpc": {
        "mpc_horizon": 20, "mpc_dt": 0.1,
        "mpc_wheelbase_m": _WHEELBASE_M, "mpc_max_steer_rad": _MAX_STEER_RAD,
        "mpc_q_lat": 0.5, "mpc_q_heading": 5.0, "mpc_q_speed": 2.0,
        "mpc_r_steer": 0.05, "mpc_r_accel": 0.2, "mpc_r_steer_rate": 3.5,
        "mpc_delta_rate_max": 0.15, "mpc_max_accel": 3.0, "mpc_max_decel": 4.0,
        "mpc_v_min": 1.0, "mpc_v_max": 20.0, "mpc_speed_adaptive_horizon": False,
        "mpc_bias_enabled": False, "mpc_max_solve_time_ms": 500.0,
        "mpc_max_consecutive_failures": 5,
        "mpc_first_step_rate_enabled": True,
        "mpc_elat_ema_alpha": 1.0,
        "mpc_elat_deadband_m": 0.0,
        "mpc_elat_ramp_rate_m_per_frame": ramp_rate,
        **extra,
    }}}
    return MPCController(cfg)


def _solve(ctrl: MPCController, e_lat: float, e_heading: float = 0.0, speed: float = 8.0) -> dict:
    """Single-step solve wrapper."""
    return ctrl.compute_steering(
        e_lat=e_lat,
        e_heading=e_heading,
        current_speed=speed,
        last_delta_norm=0.0,
        kappa_ref=0.0,
        v_target=speed,
        v_max=speed + 2.0,
        dt=_DT,
    )


def _build_jerk_data(
    n: int = 30,
    dt: float = 0.033,
    steering_profile: list = None,
    e_lat: list = None,
    stale: list = None,
    silent: list = None,
    kappa: list = None,
    gate: list = None,
) -> dict:
    """Build minimal signal dict for ``classify_jerk_events``."""
    time = np.arange(n) * dt
    steer = np.array(steering_profile if steering_profile else [0.0] * n, dtype=float)
    return {
        "time":                          time,
        "steering":                      steer,
        "mpc_e_lat":                     np.array(e_lat if e_lat else [0.0] * n, dtype=float),
        "pp_stale_hold_active":          np.array(stale if stale else [0.0] * n, dtype=float),
        "diag_silent_elat_dropout_active": np.array(silent if silent else [0.0] * n, dtype=float),
        "curvature_map_abs":             np.array(kappa if kappa else [0.0] * n, dtype=float),
        "diag_heading_zero_gate_active": np.array(gate if gate else [0.0] * n, dtype=float),
    }


# ---------------------------------------------------------------------------
# Jerk event classifier tests
# ---------------------------------------------------------------------------

class TestJerkEventClassifier:

    def test_no_events_below_threshold(self):
        """Classifier returns empty list when no frame exceeds jerk threshold."""
        data = _build_jerk_data(n=20, steering_profile=[0.0] * 20)
        events = classify_jerk_events(data, steering_jerk_threshold_deg_per_s2=18.0)
        assert events == [], "Expected no events for flat steering"

    def test_dropout_recovery_classified(self):
        """A snap from e_lat=0 → large value with silent flag drop → dropout_recovery."""
        n = 20
        dt = 0.033
        # Frames 0-9: silent dropout (e_lat≈0, silent=1)
        # Frame 10: recovery — e_lat jumps, silent=0, large steer change
        steer = [0.0] * n
        steer[10] = 0.40   # sharp recovery steer
        steer[11] = 0.05

        e_lat  = [0.0] * n
        e_lat[10] = 0.25   # snapped back

        silent = [0.0] * n
        for i in range(0, 10):
            silent[i] = 1.0   # was in dropout
        # frame 10: silent=0 (just recovered)

        data = _build_jerk_data(
            n=n, dt=dt, steering_profile=steer, e_lat=e_lat, silent=silent,
        )
        events = classify_jerk_events(data, steering_jerk_threshold_deg_per_s2=5.0)
        causes = [e["cause"] for e in events]
        assert "dropout_recovery" in causes, (
            f"Expected 'dropout_recovery' in events but got: {causes}"
        )

    def test_dropout_start_classified(self):
        """When e_lat snaps to ≈0 and silent flag turns ON → dropout_start."""
        n = 20
        dt = 0.033
        steer = [0.1] * n
        steer[8]  = 0.1
        steer[9]  = 0.0   # large step down (steeer back to 0)
        steer[10] = 0.0

        e_lat  = [0.2] * n
        e_lat[9]  = 0.0   # dropout starts
        e_lat[10] = 0.0

        silent = [0.0] * n
        silent[9]  = 1.0  # dropout flag flips on
        silent[10] = 1.0

        data = _build_jerk_data(
            n=n, dt=dt, steering_profile=steer, e_lat=e_lat, silent=silent,
        )
        events = classify_jerk_events(data, steering_jerk_threshold_deg_per_s2=2.0)
        causes = [e["cause"] for e in events]
        assert "dropout_start" in causes, (
            f"Expected 'dropout_start' in events but got: {causes}"
        )

    def test_stale_recovery_classified(self):
        """When stale flag clears, jerk should be labelled stale_recovery."""
        n = 20
        dt = 0.033
        steer = [0.0] * n
        steer[10] = 0.35   # big step when stale clears
        steer[11] = 0.05

        stale = [0.0] * n
        for i in range(5, 10):
            stale[i] = 1.0   # stale hold was active
        # frame 10: stale=0 (just cleared)

        data = _build_jerk_data(
            n=n, dt=dt, steering_profile=steer, stale=stale,
        )
        events = classify_jerk_events(data, steering_jerk_threshold_deg_per_s2=5.0)
        causes = [e["cause"] for e in events]
        assert "stale_recovery" in causes, (
            f"Expected 'stale_recovery' in events but got: {causes}"
        )

    def test_curve_oscillation_classified(self):
        """Large e_lat on curve with heading gate active → curve_oscillation."""
        n = 30
        dt = 0.033
        # Oscillating steering on curve
        steer = [0.1 * (-1) ** i for i in range(n)]

        e_lat = [0.3] * n
        kappa = [0.01] * n
        gate  = [1.0] * n   # heading gate latched

        data = _build_jerk_data(
            n=n, dt=dt, steering_profile=steer, e_lat=e_lat, kappa=kappa, gate=gate,
        )
        events = classify_jerk_events(data, steering_jerk_threshold_deg_per_s2=2.0)
        causes = [e["cause"] for e in events]
        assert "curve_oscillation" in causes, (
            f"Expected 'curve_oscillation' in events but got: {causes}"
        )

    def test_straight_noise_classified(self):
        """Small e_lat on straight (kappa≈0) → straight_noise."""
        n = 30
        dt = 0.033
        # Alternating small steering (sign-flip oscillation)
        steer = [0.05 * (-1) ** i for i in range(n)]
        e_lat = [0.05] * n   # small, noise-level
        kappa = [0.0] * n    # straight

        data = _build_jerk_data(
            n=n, dt=dt, steering_profile=steer, e_lat=e_lat, kappa=kappa,
        )
        events = classify_jerk_events(data, steering_jerk_threshold_deg_per_s2=2.0)
        causes = [e["cause"] for e in events]
        assert "straight_noise" in causes, (
            f"Expected 'straight_noise' in events but got: {causes}"
        )

    def test_gap_artifact_frames_excluded(self):
        """Events at frames with anomalous dt (>1.5× median) are not reported."""
        n = 20
        dt = 0.033
        time = np.arange(n, dtype=float) * dt
        time[10] += 0.2   # large gap at frame 10 (anomalous)

        steer = np.zeros(n, dtype=float)
        steer[10] = 0.5   # huge steer change — should be filtered as gap artifact

        data = {
            "time":    time,
            "steering": steer,
        }
        events = classify_jerk_events(data, steering_jerk_threshold_deg_per_s2=5.0)
        frames = [e["frame"] for e in events]
        assert 10 not in frames, (
            "Frame 10 (adjacent to anomalous gap) should be excluded as artifact"
        )

    def test_event_dict_has_required_keys(self):
        """Each event dict includes all required keys."""
        n = 20
        steer = [0.0] * n
        steer[10] = 0.6
        steer[11] = 0.0
        data = _build_jerk_data(n=n, steering_profile=steer)
        events = classify_jerk_events(data, steering_jerk_threshold_deg_per_s2=2.0)
        for ev in events:
            for key in ("frame", "time", "jerk_deg_s2", "cause", "description",
                        "e_lat", "kappa", "stale", "silent", "gate"):
                assert key in ev, f"Missing key '{key}' in event dict: {ev}"


# ---------------------------------------------------------------------------
# E_lat dropout recovery rate limiter tests
# ---------------------------------------------------------------------------

class TestElatRampRateLimiter:

    def test_rate_limiter_off_large_step_passes_through(self):
        """With rate limiter disabled (0.0), a 0 → 0.25m step is not clamped."""
        ctrl = _make_mpc(ramp_rate=0.0)
        # Warm up at zero
        for _ in range(5):
            _solve(ctrl, e_lat=0.0)
        # Large step input
        result = _solve(ctrl, e_lat=0.25)
        assert not result["elat_ramp_active"], (
            "Rate limiter should be inactive when elat_ramp_rate_m_per_frame=0.0"
        )
        # The raw e_lat passed in (0.25) should equal the QP input (no clamping)
        assert result["e_lat_raw"] == pytest.approx(0.25)
        assert result["e_lat_input"] == pytest.approx(0.25)

    def test_rate_limiter_on_clamps_large_step(self):
        """With rate limiter enabled, a 0 → 0.25m step is capped to ramp_rate per frame."""
        ramp_rate = 0.05
        ctrl = _make_mpc(ramp_rate=ramp_rate)
        # Warm up at zero so _prev_elat_ramp = 0
        for _ in range(5):
            _solve(ctrl, e_lat=0.0)
        # Large step
        result = _solve(ctrl, e_lat=0.25)
        assert result["elat_ramp_active"], "Rate limiter should have fired"
        assert result["e_lat_input"] == pytest.approx(ramp_rate, abs=1e-9), (
            f"Expected e_lat_input={ramp_rate}, got {result['e_lat_input']:.4f}"
        )

    def test_rate_limiter_ramps_to_target_over_multiple_frames(self):
        """Rate limiter accumulates over multiple frames until target is reached."""
        ramp_rate = 0.05
        target    = 0.25
        ctrl      = _make_mpc(ramp_rate=ramp_rate)
        # Warm up
        for _ in range(5):
            _solve(ctrl, e_lat=0.0)
        # Apply constant target for enough frames to converge
        n_needed = int(np.ceil(target / ramp_rate))
        inputs_seen = []
        for _ in range(n_needed + 2):
            r = _solve(ctrl, e_lat=target)
            inputs_seen.append(r["e_lat_input"])
        # Final value must reach the target
        assert inputs_seen[-1] == pytest.approx(target, abs=ramp_rate + 1e-9), (
            f"Expected e_lat_input to converge to {target}, got {inputs_seen[-1]:.4f}"
        )
        # Values should be monotonically increasing (or plateau)
        diffs = np.diff(inputs_seen)
        assert np.all(diffs >= -1e-9), (
            f"Rate-limited e_lat should be non-decreasing toward target: {inputs_seen}"
        )

    def test_rate_limiter_does_not_affect_small_step(self):
        """A step smaller than ramp_rate passes through WITHOUT clamping."""
        ramp_rate = 0.10
        ctrl = _make_mpc(ramp_rate=ramp_rate)
        for _ in range(5):
            _solve(ctrl, e_lat=0.0)
        small_step = 0.03   # < ramp_rate
        result = _solve(ctrl, e_lat=small_step)
        assert not result["elat_ramp_active"], "Should not clamp a step smaller than ramp_rate"
        assert result["e_lat_input"] == pytest.approx(small_step, abs=1e-9)

    def test_rate_limiter_state_resets_on_reset(self):
        """After reset(), the limiter's internal state returns to zero."""
        ramp_rate = 0.05
        ctrl = _make_mpc(ramp_rate=ramp_rate)
        # Pump some non-zero e_lat in
        for _ in range(10):
            _solve(ctrl, e_lat=0.30)
        # Reset controller
        ctrl.reset()
        # Now a step from 0 → 0.25 should be limited from 0, not from 0.30
        result = _solve(ctrl, e_lat=0.25)
        assert result["elat_ramp_active"], "After reset, limiter should start from 0 again"
        assert result["e_lat_input"] == pytest.approx(ramp_rate, abs=1e-9)

    def test_ramp_active_flag_false_for_normal_drift(self):
        """On a gentle curve (e_lat grows <0.05 m/frame) the flag stays False."""
        ramp_rate = 0.05
        ctrl = _make_mpc(ramp_rate=ramp_rate)
        max_active = 0
        e_lat_val = 0.0
        # Simulate smooth 0.01 m/frame drift (well below ramp_rate)
        for _ in range(20):
            e_lat_val += 0.01
            result = _solve(ctrl, e_lat=e_lat_val)
            max_active += int(result["elat_ramp_active"])
        assert max_active == 0, (
            f"Rate limiter should not fire for gradual drift: {max_active} events"
        )

    def test_rate_limiter_production_default_is_disabled(self):
        """Production default (ramp_rate=0.0) means limiter never fires."""
        ctrl = _make_mpc(ramp_rate=0.0)
        for _ in range(5):
            _solve(ctrl, e_lat=0.0)
        result = _solve(ctrl, e_lat=0.50)   # large step
        assert not result["elat_ramp_active"], (
            "Production default must have rate limiter disabled"
        )
        assert result["e_lat_input"] == pytest.approx(0.50)
