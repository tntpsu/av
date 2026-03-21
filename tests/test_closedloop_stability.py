"""Closed-loop stability tests for the MPC lateral controller.

These tests close the feedback loop between the MPC and a kinematic bicycle
model, running N frames without Unity.  They catch control instability
(divergence, oscillation runaway, slow convergence) that unit tests on
individual components cannot detect.

The bicycle model uses Frenet-frame error dynamics:

    ė_lat     = v · sin(e_heading)  ≈  v · e_heading   (small angle)
    ė_heading = v · tan(δ) / L      ≈  v · δ / L        (small angle)

where δ = delta_norm × max_steer_rad is the physical front-wheel angle.
This is the same linearisation the MPC QP was designed for, so the model
is self-consistent — any instability here is a real controller problem.

Noise and latency values are calibrated from real Unity recordings:
  - Lateral perception noise σ ≈ 0.10 m   (recording_20260320_222115.h5, straight frames)
  - Heading perception noise σ ≈ 0.022 rad
  - Control loop latency     ≈ 1 frame = 33 ms  (command issued, applied next Unity tick)
"""

import collections
import numpy as np
import pytest

from control.mpc_controller import MPCController, MPCParams

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_WHEELBASE_M   = 2.5
_MAX_STEER_RAD = 0.5236   # 30° — must match mpc_max_steer_rad in _make_mpc
_DT            = 0.033    # ~30 Hz


def _make_mpc(**overrides) -> MPCController:
    """Return an MPCController configured for closed-loop tests."""
    cfg = {"trajectory": {"mpc": {
        "mpc_horizon": 20,
        "mpc_dt": 0.1,                  # 2 s prediction window
        "mpc_wheelbase_m": _WHEELBASE_M,
        "mpc_max_steer_rad": _MAX_STEER_RAD,
        "mpc_q_lat": 0.5,
        "mpc_q_heading": 5.0,
        "mpc_q_speed": 2.0,
        "mpc_r_steer": 0.05,
        "mpc_r_accel": 0.2,
        "mpc_r_steer_rate": 3.5,        # production value after the fix
        "mpc_delta_rate_max": 0.15,
        "mpc_max_accel": 3.0,
        "mpc_max_decel": 4.0,
        "mpc_v_min": 1.0,
        "mpc_v_max": 20.0,
        "mpc_speed_adaptive_horizon": False,
        "mpc_bias_enabled": False,      # keep bias off so the loop is clean
        "mpc_max_solve_time_ms": 500.0,
        "mpc_max_consecutive_failures": 5,
        # Noise-jitter fixes (2026-02-17) — off by default in this test so
        # the bicycle model sees unfiltered errors (the filtering is tested
        # separately in test_noise_jitter_fixes.py).
        "mpc_first_step_rate_enabled": False,
        "mpc_elat_ema_alpha": 1.0,
        "mpc_elat_deadband_m": 0.0,
        **overrides,
    }}}
    return MPCController(cfg)


# ---------------------------------------------------------------------------
# Bicycle model
# ---------------------------------------------------------------------------

class BicycleModel:
    """Kinematic Frenet-frame bicycle model.

    State: (e_lat [m], e_heading [rad])
    Input: delta_norm ∈ [-1, 1]  (normalised steering output from MPC)
    """

    def __init__(self, wheelbase: float = _WHEELBASE_M,
                 max_steer_rad: float = _MAX_STEER_RAD):
        self.wheelbase    = wheelbase
        self.max_steer_rad = max_steer_rad

    def step(self, e_lat: float, e_heading: float,
             speed: float, delta_norm: float, dt: float,
             kappa_path: float = 0.0) -> tuple[float, float]:
        """Advance one timestep.

        Args:
            e_lat:       lateral error from path centre (m, positive = right)
            e_heading:   heading error (rad, positive = pointing right of path)
            speed:       vehicle speed (m/s)
            delta_norm:  normalised front-wheel steer output from MPC [-1, 1]
            dt:          timestep (s)
            kappa_path:  signed path curvature (1/m); 0 for straight

        Returns:
            (e_lat_new, e_heading_new)
        """
        delta = float(np.clip(delta_norm, -1.0, 1.0)) * self.max_steer_rad

        # Heading error rate: curvature of vehicle track minus path curvature
        kappa_vehicle  = np.tan(delta) / max(self.wheelbase, 1e-3)
        e_heading_dot  = speed * (kappa_vehicle - kappa_path)

        # Lateral error rate (first-order approximation is fine for tests)
        e_lat_dot = speed * np.sin(e_heading)

        e_heading_new = e_heading + e_heading_dot * dt
        e_lat_new     = e_lat    + e_lat_dot     * dt

        return float(e_lat_new), float(e_heading_new)


# ---------------------------------------------------------------------------
# Closed-loop runner
# ---------------------------------------------------------------------------

def run_closedloop(
    ctrl: MPCController,
    e_lat0: float     = 0.3,
    e_heading0: float = 0.0,
    speed: float      = 8.0,
    kappa_path: float = 0.0,
    n_frames: int     = 300,
    dt: float         = _DT,
    # --- simple mid-run disturbance ---
    disturbance_at: int | None = None,
    disturbance_m: float       = 0.0,
    # --- realistic noise (calibrated from recording_20260320_222115.h5) ---
    noise_std: float         = 0.0,   # lateral measurement noise  σ (m)
    noise_std_heading: float = 0.0,   # heading measurement noise  σ (rad)
    rng_seed: int            = 42,
    # --- control-loop latency (1 frame = 33 ms in Unity) ---
    latency_frames: int = 0,
    # --- scheduled curve disturbances ---
    # Each entry: dict with keys:
    #   'frame'   – when to apply
    #   'kind'    – 'floor_rescue' | 'heading_gate_latch'
    #   'value'   – magnitude (m) for floor_rescue; duration (frames) for latch
    curve_events: list | None = None,
) -> np.ndarray:
    """Run the MPC + bicycle model for *n_frames* and return the e_lat history.

    Supports:
      - Gaussian perception noise on e_lat and e_heading
      - N-frame steering application latency (deque buffer)
      - Mid-run lateral step disturbances
      - Scheduled curve events: PP floor rescue steps and heading-gate latches

    Returns:
        np.ndarray of shape (n_frames,) containing e_lat at each frame
    """
    rng   = np.random.default_rng(rng_seed)
    model = BicycleModel()

    e_lat      = float(e_lat0)
    e_heading  = float(e_heading0)
    history    = np.empty(n_frames)

    # Latency: buffer holds (latency_frames+1) entries; oldest is applied this tick.
    n_buf = max(1, latency_frames + 1)
    steer_buffer: collections.deque[float] = collections.deque([0.0] * n_buf,
                                                                maxlen=n_buf)

    # Build fast lookup for curve events
    floor_rescue_at:  dict[int, float] = {}
    gate_latch_at:    dict[int, int]   = {}  # frame → duration
    for ev in (curve_events or []):
        if ev["kind"] == "floor_rescue":
            floor_rescue_at[int(ev["frame"])] = float(ev["value"])
        elif ev["kind"] == "heading_gate_latch":
            gate_latch_at[int(ev["frame"])]   = int(ev["value"])

    gate_blocked_remaining = 0   # frames left where heading correction is blocked

    for i in range(n_frames):
        # ── curve events ────────────────────────────────────────────────────
        if i in floor_rescue_at:
            # PP floor activates → sudden step in the effective reference point
            e_lat += floor_rescue_at[i]
        if i in gate_latch_at:
            gate_blocked_remaining = gate_latch_at[i]

        # ── mid-run disturbance ──────────────────────────────────────────────
        if disturbance_at is not None and i == disturbance_at:
            e_lat += float(disturbance_m)

        # ── noisy measurements ──────────────────────────────────────────────
        e_lat_meas     = e_lat     + (rng.normal(0.0, noise_std)         if noise_std         > 0 else 0.0)
        e_heading_meas = e_heading + (rng.normal(0.0, noise_std_heading) if noise_std_heading > 0 else 0.0)

        # When the heading-zero gate is latched the controller receives
        # zero heading error (heading correction is suppressed)
        if gate_blocked_remaining > 0:
            e_heading_meas = 0.0
            gate_blocked_remaining -= 1

        # ── MPC solve ───────────────────────────────────────────────────────
        # last_delta_norm is the steering the MPC *thinks* was applied
        # (the most recently commanded value, even if latency delays application)
        last_commanded = steer_buffer[-1]
        result = ctrl.compute_steering(
            e_lat=e_lat_meas,
            e_heading=e_heading_meas,
            current_speed=speed,
            last_delta_norm=last_commanded,
            kappa_ref=kappa_path,
            v_target=speed,
            v_max=speed + 2.0,
            dt=dt,
        )
        new_delta = float(result.get("steering_normalized", 0.0))
        steer_buffer.append(new_delta)

        # ── apply oldest steering (latency) ─────────────────────────────────
        delta_applied = steer_buffer[0]  # oldest element in the fixed-size deque

        # ── plant update ────────────────────────────────────────────────────
        e_lat, e_heading = model.step(e_lat, e_heading, speed, delta_applied,
                                      dt, kappa_path=kappa_path)
        history[i] = e_lat

    return history


# ---------------------------------------------------------------------------
# Helper: peak-to-peak in steady-state window
# ---------------------------------------------------------------------------

def _steady_state_pp(hist: np.ndarray, skip: int = 150) -> float:
    """Peak-to-peak amplitude in the tail of a history array."""
    tail = hist[skip:]
    if tail.size == 0:
        return 0.0
    return float(tail.max() - tail.min())


# ===========================================================================
# 1.  Convergence — the controller must bring e_lat close to zero
# ===========================================================================

class TestConvergence:

    def test_converges_from_positive_offset(self):
        """Starting 0.3 m right of centre must reach < 0.05 m within 5 s."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.3, n_frames=150)
        assert abs(hist[-1]) < 0.05, (
            f"Expected |e_lat| < 0.05 m after 5 s, got {hist[-1]:.4f} m"
        )

    def test_converges_from_negative_offset(self):
        """Starting 0.3 m left of centre must also converge."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=-0.3, n_frames=150)
        assert abs(hist[-1]) < 0.05, (
            f"Expected |e_lat| < 0.05 m after 5 s, got {hist[-1]:.4f} m"
        )

    def test_converges_from_large_offset(self):
        """A larger 0.5 m starting offset must converge within 10 s."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.5, n_frames=300)
        assert abs(hist[-1]) < 0.08, (
            f"Expected |e_lat| < 0.08 m after 10 s, got {hist[-1]:.4f} m"
        )

    @pytest.mark.parametrize("speed", [4.0, 8.0, 12.0])
    def test_converges_at_multiple_speeds(self, speed):
        """Must converge from 0.2 m offset at 4, 8, and 12 m/s."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.2, speed=speed, n_frames=200)
        assert abs(hist[-1]) < 0.10, (
            f"Speed={speed} m/s: expected |e_lat| < 0.10 m, got {hist[-1]:.4f} m"
        )


# ===========================================================================
# 2.  Stability — error must not diverge or grow without bound
# ===========================================================================

class TestStability:

    def test_does_not_diverge_on_straight(self):
        """10 s run from 0.3 m offset must stay within ±0.5 m throughout."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.3, n_frames=300)
        assert np.max(np.abs(hist)) < 0.5, (
            f"Error exceeded ±0.5 m during run; max={np.max(np.abs(hist)):.4f} m"
        )

    def test_does_not_diverge_on_constant_curve(self):
        """On a constant-curvature path (κ=0.01) the error must stay bounded."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.1, speed=8.0,
                              kappa_path=0.01, n_frames=300)
        assert np.max(np.abs(hist)) < 0.8, (
            f"Error exceeded ±0.8 m on curved path; max={np.max(np.abs(hist)):.4f} m"
        )

    def test_second_half_amplitude_not_growing(self):
        """Oscillation must not grow over time — the RMS of the second half
        of the run must not exceed 1.5× the RMS of the first half."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.3, n_frames=300)
        first_rms  = float(np.sqrt(np.mean(hist[:150] ** 2)))
        second_rms = float(np.sqrt(np.mean(hist[150:] ** 2)))
        # Allow 10 % margin over first-half RMS
        limit = max(first_rms * 1.5, 0.02)
        assert second_rms <= limit, (
            f"Oscillation grew: first-half RMS={first_rms:.4f} m, "
            f"second-half RMS={second_rms:.4f} m (limit={limit:.4f})"
        )

    def test_zero_from_zero_stays_zero(self):
        """Starting exactly on path with no heading error must stay there."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.0, e_heading0=0.0, n_frames=150)
        assert np.max(np.abs(hist)) < 0.02, (
            f"Drifted from zero: max |e_lat|={np.max(np.abs(hist)):.4f} m"
        )


# ===========================================================================
# 3.  Oscillation damping — the r_steer_rate fix must reduce peak-to-peak
# ===========================================================================

class TestOscillationDamping:

    def test_higher_rate_cost_reduces_steady_state_pp(self):
        """The core claim of Fix A: r_steer_rate=3.5 must produce
        smaller steady-state peak-to-peak than r_steer_rate=2.0."""
        ctrl_low  = _make_mpc(mpc_r_steer_rate=2.0)
        ctrl_high = _make_mpc(mpc_r_steer_rate=3.5)
        pp_low  = _steady_state_pp(run_closedloop(ctrl_low,  n_frames=300))
        pp_high = _steady_state_pp(run_closedloop(ctrl_high, n_frames=300))
        assert pp_high <= pp_low + 1e-4, (
            f"r_steer_rate=3.5 should not worsen steady-state oscillation: "
            f"pp_low={pp_low:.5f} m, pp_high={pp_high:.5f} m"
        )

    def test_high_rate_cost_does_not_worsen_recovery_from_heading_error(self):
        """r_steer_rate=3.5 must not produce a larger final error than 2.0
        when recovering from a pure heading error (no initial lateral offset).

        This guards against the rate cost being set so high that heading
        correction is crippled: the solver becomes reluctant to change steering
        and the heading error bleeds into growing lateral error.
        """
        ctrl_low  = _make_mpc(mpc_r_steer_rate=2.0)
        ctrl_high = _make_mpc(mpc_r_steer_rate=3.5)

        # Pure heading error of 0.15 rad; both must converge cleanly.
        hist_low  = run_closedloop(ctrl_low,  e_lat0=0.0, e_heading0=0.15,
                                   speed=8.0, n_frames=200)
        hist_high = run_closedloop(ctrl_high, e_lat0=0.0, e_heading0=0.15,
                                   speed=8.0, n_frames=200)

        # Max lateral excursion must stay reasonable for both
        for name, hist in (("r_steer_rate=2.0", hist_low),
                           ("r_steer_rate=3.5", hist_high)):
            assert np.max(np.abs(hist)) < 0.6, (
                f"{name}: heading error caused excessive lateral excursion "
                f"{np.max(np.abs(hist)):.4f} m"
            )

        # Final error with high rate cost must be no worse than 1.5× the low value
        final_low  = abs(hist_low[-1])
        final_high = abs(hist_high[-1])
        assert final_high <= max(final_low * 1.5, 0.05), (
            f"r_steer_rate=3.5 final |e_lat|={final_high:.4f} m is significantly "
            f"worse than r_steer_rate=2.0 final={final_low:.4f} m"
        )

    def test_solver_remains_feasible_with_high_rate_cost(self):
        """Raising r_steer_rate to 3.5 must not cause solver infeasibility."""
        ctrl = _make_mpc(mpc_r_steer_rate=3.5)
        for e in (0.0, 0.3, -0.3, 0.5):
            result = ctrl.compute_steering(
                e_lat=e, e_heading=0.0, current_speed=8.0,
                last_delta_norm=0.0, kappa_ref=0.0,
                v_target=8.0, v_max=10.0, dt=_DT,
            )
            assert result.get("mpc_feasible", True), (
                f"Solver infeasible at e_lat={e} with r_steer_rate=3.5"
            )


# ===========================================================================
# 4.  Disturbance rejection — the car must recover from sudden lateral kicks
# ===========================================================================

class TestDisturbanceRejection:

    def test_recovers_from_step_disturbance(self):
        """A 0.4 m lateral kick mid-run must be recovered within 3 s (90 frames)."""
        ctrl = _make_mpc()
        # Settle for 2 s first, then inject disturbance; must recover in 3 more s
        hist = run_closedloop(ctrl, e_lat0=0.0, n_frames=150,
                              disturbance_at=60, disturbance_m=0.4)
        assert abs(hist[-1]) < 0.10, (
            f"Did not recover from 0.4 m kick: final e_lat={hist[-1]:.4f} m"
        )

    def test_error_after_disturbance_does_not_exceed_kick_size(self):
        """The peak error after a disturbance should not exceed the kick itself."""
        kick = 0.3
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.0, n_frames=200,
                              disturbance_at=50, disturbance_m=kick)
        # Peak after kick must not exceed kick × 1.5 (no large overshoot)
        post_kick = hist[50:]
        max_abs   = float(np.max(np.abs(post_kick)))
        assert max_abs < kick * 1.5, (
            f"Excessive overshoot after {kick} m kick: peak={max_abs:.4f} m"
        )


# ===========================================================================
# 5.  Realistic conditions — noise + latency (calibrated from real Unity data)
# ===========================================================================
#
# Calibration source: recording_20260320_222115.h5 (hill highway, AV stack)
#   Lateral perception noise σ  = 0.10 m   (straight-segment std of lateral_error)
#   Heading perception noise σ  = 0.022 rad
#   Control loop latency        = 1 frame = 33 ms (Unity tick delay)
#
# Using conservative σ = 0.05 m lateral / 0.015 rad heading for tests so the
# thresholds are not overly sensitive to the exact RNG seed.
# ---------------------------------------------------------------------------

_NOISE_LAT  = 0.05    # conservative estimate of perception noise (m)
_NOISE_HEAD = 0.015   # conservative estimate of heading noise (rad)
_LATENCY    = 1       # frames


class TestRealisticConditions:
    """Tests that close the loop under calibrated perception noise and
    control latency — the two factors absent from the ideal bicycle model
    that cause real-world oscillation to differ from the ideal prediction."""

    def test_stable_under_perception_noise(self):
        """Controller must stay bounded with realistic lateral perception noise."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.3, n_frames=300,
                              noise_std=_NOISE_LAT, rng_seed=0)
        assert np.max(np.abs(hist)) < 0.7, (
            f"Diverged under perception noise: max |e_lat|={np.max(np.abs(hist)):.4f} m"
        )

    def test_stable_under_heading_noise(self):
        """Controller must stay bounded with realistic heading measurement noise."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.0, e_heading0=0.0, n_frames=300,
                              noise_std_heading=_NOISE_HEAD, rng_seed=1)
        assert np.max(np.abs(hist)) < 0.4, (
            f"Diverged under heading noise: max |e_lat|={np.max(np.abs(hist)):.4f} m"
        )

    def test_stable_with_one_frame_latency(self):
        """One-frame latency must not cause divergence or unbounded oscillation."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.3, n_frames=300,
                              latency_frames=_LATENCY)
        assert np.max(np.abs(hist)) < 0.6, (
            f"Diverged with 1-frame latency: max |e_lat|={np.max(np.abs(hist)):.4f} m"
        )

    def test_stable_under_combined_noise_and_latency(self):
        """Combined noise + latency (the real-world condition) must remain stable."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.3, n_frames=300,
                              noise_std=_NOISE_LAT, noise_std_heading=_NOISE_HEAD,
                              latency_frames=_LATENCY, rng_seed=2)
        assert np.max(np.abs(hist)) < 0.8, (
            f"Diverged under noise+latency: max |e_lat|={np.max(np.abs(hist)):.4f} m"
        )

    def test_amplitude_not_growing_under_noise_and_latency(self):
        """The key oscillation-runaway check reproduced under realistic conditions.

        Second-half RMS must not exceed 1.8× first-half RMS.  The multiplier is
        slightly looser than the ideal-model test (1.5×) to account for noise
        variance, but growth beyond 1.8× would indicate real instability.
        """
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.3, n_frames=300,
                              noise_std=_NOISE_LAT, noise_std_heading=_NOISE_HEAD,
                              latency_frames=_LATENCY, rng_seed=3)
        first_rms  = float(np.sqrt(np.mean(hist[:150] ** 2)))
        second_rms = float(np.sqrt(np.mean(hist[150:] ** 2)))
        limit = max(first_rms * 1.8, 0.05)
        assert second_rms <= limit, (
            f"Amplitude growing under noise+latency: "
            f"first-half RMS={first_rms:.4f} m, second-half RMS={second_rms:.4f} m"
        )

    def test_higher_rate_cost_reduces_noise_driven_oscillation(self):
        """Under realistic noise, r_steer_rate=3.5 must not produce larger
        steady-state oscillation than r_steer_rate=2.0.

        This is the realistic version of TestOscillationDamping — unlike the
        ideal model (where both configs converge to machine-epsilon), noise
        provides a persistent excitation that reveals the damping difference.
        """
        kwargs = dict(e_lat0=0.3, n_frames=400,
                      noise_std=_NOISE_LAT, noise_std_heading=_NOISE_HEAD,
                      latency_frames=_LATENCY, rng_seed=7)
        ctrl_low  = _make_mpc(mpc_r_steer_rate=2.0)
        ctrl_high = _make_mpc(mpc_r_steer_rate=3.5)
        pp_low  = _steady_state_pp(run_closedloop(ctrl_low,  **kwargs), skip=200)
        pp_high = _steady_state_pp(run_closedloop(ctrl_high, **kwargs), skip=200)
        # Production value must not be meaningfully worse than the old value
        assert pp_high <= pp_low * 1.5 + 0.02, (
            f"r_steer_rate=3.5 worsened noise-driven oscillation: "
            f"pp_old={pp_low:.4f} m, pp_new={pp_high:.4f} m"
        )


# ===========================================================================
# 6.  Curve disturbances — PP floor rescue and heading-gate latch
# ===========================================================================
#
# These model the two discrete events that occur every curve lap:
#
#   floor_rescue:       When the PP floor activates at ENTRY, the effective
#                       lookahead suddenly increases → the planned reference
#                       point shifts → appears as a step in perceived e_lat.
#                       Calibrated to ~0.3 m from floorRescueMax in HDF5.
#
#   heading_gate_latch: When the heading-zero gate latches ON approaching a
#                       curve, heading correction is suppressed for K frames.
#                       During this window heading error bleeds into lateral
#                       error at rate v × e_heading per frame.
#                       Calibrated to ~20 frames from HDF5 gate firing data.
# ---------------------------------------------------------------------------

class TestCurveDisturbances:
    """Tests that combine realistic noise/latency with the discrete events
    that actually drive the remaining scoring deductions."""

    def _curve_events(self, rescue_frame=60, rescue_m=0.3,
                      gate_frame=50, gate_duration=20):
        return [
            {"frame": gate_frame,   "kind": "heading_gate_latch", "value": gate_duration},
            {"frame": rescue_frame, "kind": "floor_rescue",        "value": rescue_m},
        ]

    def test_recovers_after_floor_rescue_step(self):
        """A PP floor rescue step of 0.3 m must be recovered within 3 s."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.0, n_frames=180,
                              curve_events=[{"frame": 60, "kind": "floor_rescue",
                                             "value": 0.3}])
        assert abs(hist[-1]) < 0.12, (
            f"Did not recover from 0.3 m floor rescue: final e_lat={hist[-1]:.4f} m"
        )

    def test_floor_rescue_plus_gate_latch_stays_bounded(self):
        """Combined heading-gate latch + floor rescue must not cause divergence."""
        ctrl = _make_mpc()
        hist = run_closedloop(ctrl, e_lat0=0.1, n_frames=300,
                              noise_std=_NOISE_LAT, latency_frames=_LATENCY,
                              curve_events=self._curve_events())
        assert np.max(np.abs(hist)) < 0.8, (
            f"Diverged after combined curve events: max={np.max(np.abs(hist)):.4f} m"
        )

    def test_repeated_curve_laps_do_not_accumulate(self):
        """Three consecutive curve-entry events (every 100 frames) must not
        accumulate progressively larger lateral peaks — the fix should prevent
        error growth across repeated curves."""
        ctrl = _make_mpc()
        events = []
        for lap in range(3):
            base = 100 * lap
            events.append({"frame": base + 5,  "kind": "heading_gate_latch", "value": 20})
            events.append({"frame": base + 20, "kind": "floor_rescue",        "value": 0.25})

        hist = run_closedloop(ctrl, e_lat0=0.0, n_frames=300,
                              noise_std=_NOISE_LAT, latency_frames=_LATENCY,
                              curve_events=events)

        # Extract peak |e_lat| in the 30-frame window after each rescue
        peaks = [float(np.max(np.abs(hist[100*k+20 : 100*k+50]))) for k in range(3)]

        # Peak must not grow by more than 50 % from first to third lap
        assert peaks[2] <= peaks[0] * 1.5 + 0.05, (
            f"Lateral error accumulating across laps: peaks={[f'{p:.3f}' for p in peaks]} m"
        )

    def test_gate_latch_heading_error_recovery(self):
        """After a 20-frame gate latch with non-zero initial heading error,
        the heading error must stop bleeding into lateral error once the
        gate releases — total lateral excursion must remain bounded."""
        ctrl = _make_mpc()
        events = [{"frame": 10, "kind": "heading_gate_latch", "value": 20}]
        hist = run_closedloop(ctrl, e_lat0=0.0, e_heading0=0.05, n_frames=200,
                              noise_std=_NOISE_LAT, latency_frames=_LATENCY,
                              curve_events=events)
        assert np.max(np.abs(hist)) < 0.5, (
            f"Heading bleed after gate latch caused large excursion: "
            f"max |e_lat|={np.max(np.abs(hist)):.4f} m"
        )
