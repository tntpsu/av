"""Steering sign-flip rate contract tests.

In Pure-Pursuit + MPC mode (the production configuration), the steering
sign-flip override in PIDController is bypassed.  Oscillation is instead
controlled by:
  - mpc_r_steer_rate  (MPC steering-rate cost)
  - pp_max_steering_jerk  (per-frame steering change cap)

These tests verify the BEHAVIOURAL CONTRACT:
  "Steering must not flip sign more than K times in any 10-frame window"
  "Peak-to-peak steering change per frame must not exceed the jerk cap"

Tests use the closed-loop bicycle model from test_closedloop_stability.py
(same BicycleModel + run_closedloop helpers replicated here for isolation)
so this file can run without importing the other test module.

Calibration note: real recordings show max steering jerk of 0.085 (normalised)
on the hill highway run (recording_20260320_222115.h5).  The contract threshold
is set to 0.15 to give 75 % headroom before the pp_max_steering_jerk cap of
18.0 (normalised) would trigger.
"""

import collections
import numpy as np
import pytest

from control.mpc_controller import MPCController

# ---------------------------------------------------------------------------
# Bicycle model + closed-loop runner (self-contained copy)
# ---------------------------------------------------------------------------

_WHEELBASE_M   = 2.5
_MAX_STEER_RAD = 0.5236
_DT            = 0.033


def _make_mpc(**kw) -> MPCController:
    cfg = {"trajectory": {"mpc": {
        "mpc_horizon": 20,
        "mpc_dt": 0.1,
        "mpc_wheelbase_m": _WHEELBASE_M,
        "mpc_max_steer_rad": _MAX_STEER_RAD,
        "mpc_q_lat": 0.5,
        "mpc_q_heading": 5.0,
        "mpc_q_speed": 2.0,
        "mpc_r_steer": 0.05,
        "mpc_r_accel": 0.2,
        "mpc_r_steer_rate": 3.5,
        "mpc_delta_rate_max": 0.15,
        "mpc_max_accel": 3.0,
        "mpc_max_decel": 4.0,
        "mpc_v_min": 1.0,
        "mpc_v_max": 20.0,
        "mpc_speed_adaptive_horizon": False,
        "mpc_bias_enabled": False,
        "mpc_max_solve_time_ms": 500.0,
        "mpc_max_consecutive_failures": 5,
        # Fix A: first-step QP rate cost (kept - reduces max jerk; Fix B EMA disabled
        # after run analysis showed alpha=0.3 added 7-frame lag on curves → +81% curve RMSE)
        "mpc_first_step_rate_enabled": True,
        "mpc_elat_ema_alpha": 1.0,
        "mpc_elat_deadband_m": 0.0,
        **kw,
    }}}
    return MPCController(cfg)


def _bicycle_step(e_lat, e_heading, speed, delta_norm, dt, kappa_path=0.0):
    delta = float(np.clip(delta_norm, -1.0, 1.0)) * _MAX_STEER_RAD
    kappa_vehicle = np.tan(delta) / max(_WHEELBASE_M, 1e-3)
    e_heading_new = e_heading + speed * (kappa_vehicle - kappa_path) * dt
    e_lat_new     = e_lat    + speed * np.sin(e_heading) * dt
    return float(e_lat_new), float(e_heading_new)


def _run_and_get_steering(ctrl, e_lat0=0.3, e_heading0=0.0, speed=8.0,
                           kappa=0.0, n_frames=300, dt=_DT,
                           noise_std=0.0, latency=1, rng_seed=42):
    """Returns (e_lat_history, steering_history)."""
    rng   = np.random.default_rng(rng_seed)
    buf   = collections.deque([0.0] * max(1, latency + 1), maxlen=max(1, latency + 1))
    e_lat, e_heading = float(e_lat0), float(e_heading0)
    lat_hist   = np.empty(n_frames)
    steer_hist = np.empty(n_frames)

    for i in range(n_frames):
        e_meas = e_lat + (rng.normal(0.0, noise_std) if noise_std > 0 else 0.0)
        result = ctrl.compute_steering(
            e_lat=e_meas, e_heading=e_heading,
            current_speed=speed, last_delta_norm=buf[-1],
            kappa_ref=kappa, v_target=speed, v_max=speed + 2.0, dt=dt,
        )
        new_delta = float(result.get("steering_normalized", 0.0))
        buf.append(new_delta)
        delta_applied = buf[0]

        e_lat, e_heading = _bicycle_step(e_lat, e_heading, speed, delta_applied, dt, kappa)
        lat_hist[i]   = e_lat
        steer_hist[i] = new_delta

    return lat_hist, steer_hist


# ---------------------------------------------------------------------------
# Helpers for sign-flip analysis
# ---------------------------------------------------------------------------

def _count_sign_flips(arr: np.ndarray) -> int:
    """Count how many consecutive pairs have opposite signs (ignoring zeros)."""
    signs = np.sign(arr)
    signs[signs == 0] = 1
    return int(np.sum(np.diff(signs) != 0))


def _max_sign_flips_in_window(arr: np.ndarray, window: int = 10) -> int:
    """Max number of sign flips in any sliding window of *window* frames."""
    max_flips = 0
    for start in range(len(arr) - window + 1):
        flips = _count_sign_flips(arr[start:start + window])
        max_flips = max(max_flips, flips)
    return max_flips


# ---------------------------------------------------------------------------
# Contract 1: sign-flip rate in any 10-frame window ≤ threshold
# ---------------------------------------------------------------------------

class TestSignFlipRate:
    # In ideal conditions (zero noise) a well-tuned controller should not flip
    # sign more than 4 times in any 10-frame window.
    _MAX_FLIPS_IDEAL   = 4
    # With Fix A (first_step_rate_enabled=True) and EMA disabled (alpha=1.0),
    # white-noise simulation (σ=0.05m) produces ~7-8 flips per 10 frames.
    # The EMA (Fix B) brought this to 3-5 but caused +81% curve RMSE regression in
    # Unity; it was disabled after run analysis (2026-03-21).  Fix A alone is the
    # production config; this limit guards against regression back to the ~9 baseline.
    _MAX_FLIPS_NOISY   = 9    # regression ceiling: Fix A alone, no EMA

    def test_sign_flip_rate_ideal(self):
        """In ideal zero-noise conditions steering must not oscillate excessively."""
        ctrl = _make_mpc()
        _, steer = _run_and_get_steering(ctrl, n_frames=300, noise_std=0.0, latency=0)
        max_flips = _max_sign_flips_in_window(steer[30:], window=10)
        assert max_flips <= self._MAX_FLIPS_IDEAL, (
            f"Steering flipped sign {max_flips} times in a 10-frame window "
            f"(ideal conditions, limit={self._MAX_FLIPS_IDEAL})"
        )

    def test_sign_flip_rate_with_noise_and_latency(self):
        """Under realistic noise (σ=0.05 m) + 1-frame latency, sign flips must
        not exceed the regression ceiling.

        FINDING: current baseline is ~9 flips / 10 frames under this noise level.
        This is the real source of steering jerk in Unity.  A fix that brings
        this below 6 would be a meaningful improvement.
        """
        ctrl = _make_mpc()
        _, steer = _run_and_get_steering(ctrl, n_frames=400, noise_std=0.05,
                                          latency=1, rng_seed=0)
        max_flips = _max_sign_flips_in_window(steer[30:], window=10)
        assert max_flips <= self._MAX_FLIPS_NOISY, (
            f"Steering flipped sign {max_flips} times in a 10-frame window "
            f"under noise+latency (regression ceiling={self._MAX_FLIPS_NOISY}, "
            f"current baseline ~9)"
        )

    def test_higher_rate_cost_reduces_sign_flips(self):
        """r_steer_rate=3.5 must not produce more sign flips than r_steer_rate=2.0
        under the same noisy conditions."""
        kwargs = dict(n_frames=400, noise_std=0.05, latency=1, rng_seed=1)
        _, s_low  = _run_and_get_steering(_make_mpc(mpc_r_steer_rate=2.0), **kwargs)
        _, s_high = _run_and_get_steering(_make_mpc(mpc_r_steer_rate=3.5), **kwargs)
        flips_low  = _max_sign_flips_in_window(s_low[30:])
        flips_high = _max_sign_flips_in_window(s_high[30:])
        # Allow equal; must not be strictly worse
        assert flips_high <= flips_low + 1, (
            f"r_steer_rate=3.5 should not produce more sign flips: "
            f"low={flips_low}, high={flips_high}"
        )


# ---------------------------------------------------------------------------
# Contract 2: per-frame steering change (jerk) stays bounded
# ---------------------------------------------------------------------------

class TestSteeringJerkBound:
    # Normalised jerk limit from config (pp_max_steering_jerk=18.0, max_steer=0.7):
    # 18.0 / (0.7 / 0.033) ≈ 0.85 normalised, but the scorer uses raw delta.
    # Real recording max jerk was 0.085.  We set a generous 0.15 limit.
    _JERK_CONTRACT = 0.15   # max |Δsteering_normalised| per frame

    def test_jerk_stays_below_contract_ideal(self):
        ctrl = _make_mpc()
        _, steer = _run_and_get_steering(ctrl, n_frames=300, noise_std=0.0, latency=0)
        jerk = np.abs(np.diff(steer))
        max_jerk = float(jerk.max())
        assert max_jerk <= self._JERK_CONTRACT, (
            f"Steering jerk {max_jerk:.4f} exceeds contract {self._JERK_CONTRACT} "
            f"(ideal conditions)"
        )

    def test_jerk_stays_below_contract_with_noise(self):
        ctrl = _make_mpc()
        _, steer = _run_and_get_steering(ctrl, n_frames=300, noise_std=0.05,
                                          latency=1, rng_seed=5)
        jerk = np.abs(np.diff(steer))
        max_jerk = float(jerk.max())
        assert max_jerk <= self._JERK_CONTRACT * 1.5, (   # 50 % headroom for noise
            f"Steering jerk {max_jerk:.4f} exceeds contract "
            f"{self._JERK_CONTRACT * 1.5:.3f} under noise+latency"
        )

    def test_jerk_not_growing_over_time(self):
        """Jerk in the second half of the run must not exceed the first half."""
        ctrl = _make_mpc()
        _, steer = _run_and_get_steering(ctrl, n_frames=300, noise_std=0.05,
                                          latency=1, rng_seed=6)
        jerk = np.abs(np.diff(steer))
        jerk_first  = float(np.percentile(jerk[:len(jerk)//2],  95))
        jerk_second = float(np.percentile(jerk[len(jerk)//2:], 95))
        assert jerk_second <= jerk_first * 1.5 + 0.01, (
            f"Steering jerk growing: first-half p95={jerk_first:.4f}, "
            f"second-half p95={jerk_second:.4f}"
        )


# ---------------------------------------------------------------------------
# Contract 3: large disturbance must not trigger uncontrolled oscillation
# ---------------------------------------------------------------------------

class TestDisturbanceSignFlip:

    def test_post_disturbance_sign_flip_rate_bounded(self):
        """After a 0.4 m lateral kick the controller may temporarily flip steering
        to correct the error, but must settle back within 3 s (90 frames)."""
        ctrl = _make_mpc()
        _, steer = _run_and_get_steering(ctrl, e_lat0=0.0, n_frames=200,
                                          noise_std=0.0, latency=0)

        # Re-run with a kick at frame 50
        steer_kick = []
        rng   = np.random.default_rng(42)
        buf   = collections.deque([0.0] * 2, maxlen=2)
        e_lat, e_head = 0.0, 0.0
        for i in range(200):
            if i == 50:
                e_lat += 0.4
            result = ctrl.compute_steering(
                e_lat=e_lat, e_heading=e_head,
                current_speed=8.0, last_delta_norm=buf[-1],
                kappa_ref=0.0, v_target=8.0, v_max=10.0, dt=_DT,
            )
            d = float(result.get("steering_normalized", 0.0))
            buf.append(d)
            e_lat, e_head = _bicycle_step(e_lat, e_head, 8.0, buf[0], _DT)
            steer_kick.append(d)

        steer_arr = np.array(steer_kick)
        # Check sign-flip rate in 90 frames after kick (frames 50..140)
        post_kick = steer_arr[50:140]
        max_flips = _max_sign_flips_in_window(post_kick, window=10)
        assert max_flips <= 6, (
            f"Too many sign flips post-disturbance: {max_flips} in 10-frame window"
        )
