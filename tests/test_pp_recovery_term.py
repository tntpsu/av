"""Unit tests for the Pure Pursuit lateral-error recovery term.

Design reference: project_recovery_mode_post_limiter.md

The recovery term replaces the orchestrator post-limiter steering multiplier
(1.2x at |e_lat|>1.0m, 1.5x at |e_lat|>2.0m). It is injected upstream of the
rate/jerk limiters in the PP pipeline, is continuous in |e_lat| via a
smoothstep, and is LPF-smoothed so the limiters see a well-behaved signal.

Note on API: `LateralController.compute_steering(current_heading, ref, ...)`
derives the lateral error from the reference point geometry (specifically,
`ref['x']` maps to `lateral_error` under the PP pipeline). Tests therefore
inject the desired signed lateral error via `ref['x']`, and the GT at-car
signal via `ref['gt_cross_track_road_frame_at_car_m']`.
"""

import numpy as np
import pytest

from control.pid_controller import LateralController


def _pp_controller(**kwargs) -> LateralController:
    defaults = dict(
        kp=1.0,
        ki=0.0,
        kd=0.1,
        control_mode="pure_pursuit",
        pp_feedback_gain=0.0,
        max_steering=1.0,
        pp_max_steering_rate=5.0,
        pp_max_steering_jerk=0.0,
        # Map-FF disabled so steering_before_limits is geometric PP only.
        pp_map_ff_enabled=False,
        # Recovery term explicitly enabled — default is False.
        pp_recovery_term_enabled=True,
        pp_recovery_term_shadow_mode=False,
        pp_recovery_e_lat_lo_m=1.0,
        pp_recovery_e_lat_hi_m=2.0,
        pp_recovery_term_beta=0.5,
        pp_recovery_term_lpf_tau_s=0.0,  # LPF off for analytic tests
        pp_recovery_signal_use_at_car=True,
    )
    defaults.update(kwargs)
    return LateralController(**defaults)


def _ref(lat_err: float = 1.0, at_car: float = 0.0, curv: float = 0.02) -> dict:
    """Reference point that maps to a specific lateral error via ref.x.

    The non-zero curvature ensures PP produces non-zero steering_before_limits;
    the recovery term is proportional to |steering_before_limits|.
    """
    return {
        "x": lat_err,
        "y": 8.0,
        "heading": 0.02 * np.sign(lat_err) if lat_err != 0.0 else 0.0,
        "velocity": 10.0,
        "curvature": curv,
        "curvature_primary_abs": abs(curv),
        "curvature_source": "primary_contract",
        "curvature_primary_source": "map_track",
        "gt_cross_track_road_frame_at_car_m": at_car,
    }


def _compute(ctrl: LateralController, lat_err: float, at_car: float = 0.0,
             dt: float = 0.033) -> dict:
    return ctrl.compute_steering(
        0.0,
        _ref(lat_err=lat_err, at_car=at_car),
        current_speed=10.0,
        dt=dt,
        return_metadata=True,
    )


class TestPPRecoveryTerm:
    def test_term_zero_below_lo_threshold(self) -> None:
        ctrl = _pp_controller()
        meta = _compute(ctrl, lat_err=0.5, at_car=0.5)
        assert meta["lateral_error_recovery_smoothstep_weight"] == pytest.approx(0.0, abs=1e-9)
        assert meta["lateral_error_recovery_term_applied_rad"] == pytest.approx(0.0, abs=1e-9)

    def test_term_full_above_hi_threshold(self) -> None:
        ctrl = _pp_controller()
        meta = _compute(ctrl, lat_err=2.5, at_car=2.5)
        assert meta["lateral_error_recovery_smoothstep_weight"] == pytest.approx(1.0)
        # term_raw = beta * weight * |steering_before_limits| * sign(e_lat)
        expected = 0.5 * 1.0 * abs(meta["pp_geometric_steering"])
        assert meta["lateral_error_recovery_term_applied_rad"] == pytest.approx(expected, rel=1e-6)

    def test_term_continuous_at_lo_crossing(self) -> None:
        # Smoothstep is (e - lo)/(hi - lo); crossing lo produces a C0-continuous
        # ramp (weight jumps from 0 at e=1.00 to 0.01 at e=1.01).
        meta_below = _compute(_pp_controller(), lat_err=0.99, at_car=0.99)
        meta_above = _compute(_pp_controller(), lat_err=1.01, at_car=1.01)
        assert meta_below["lateral_error_recovery_smoothstep_weight"] == pytest.approx(0.0)
        assert meta_above["lateral_error_recovery_smoothstep_weight"] == pytest.approx(0.01, abs=1e-6)

    def test_term_continuous_at_hi_crossing(self) -> None:
        meta_below = _compute(_pp_controller(), lat_err=1.99, at_car=1.99)
        meta_above = _compute(_pp_controller(), lat_err=2.01, at_car=2.01)
        assert meta_below["lateral_error_recovery_smoothstep_weight"] == pytest.approx(0.99, abs=1e-6)
        assert meta_above["lateral_error_recovery_smoothstep_weight"] == pytest.approx(1.0)

    def test_term_sign_follows_error(self) -> None:
        # Positive e_lat → positive term; negative → negative; zero e_lat with
        # small at_car → term=0 because weight=0.
        meta_pos = _compute(_pp_controller(), lat_err=1.5, at_car=0.0)
        assert meta_pos["lateral_error_recovery_term_applied_rad"] > 0.0

        meta_neg = _compute(_pp_controller(), lat_err=-1.5, at_car=0.0)
        assert meta_neg["lateral_error_recovery_term_applied_rad"] < 0.0

        meta_zero = _compute(_pp_controller(), lat_err=0.0, at_car=0.5)
        assert meta_zero["lateral_error_recovery_term_applied_rad"] == pytest.approx(0.0, abs=1e-9)

    def test_term_sign_flip_during_recovery_with_lpf(self) -> None:
        # LPF should absorb a hard sign flip: slew over one frame must be a
        # small fraction of the instantaneous flip magnitude.
        ctrl = _pp_controller(pp_recovery_term_lpf_tau_s=0.20)
        # Drive LPF state to steady-state positive.
        for _ in range(30):
            _compute(ctrl, lat_err=2.5, at_car=2.5, dt=0.033)
        term_before = ctrl._recovery_term_last
        assert term_before > 0.0

        # Flip sign in one frame.
        meta_after = _compute(ctrl, lat_err=-2.5, at_car=2.5, dt=0.033)
        term_after = meta_after["lateral_error_recovery_term_applied_rad"]
        instantaneous_flip = 2.0 * abs(term_before)  # raw step if LPF were off
        slew = abs(term_after - term_before)
        assert slew < instantaneous_flip * 0.25, (
            f"LPF should absorb the step; slew={slew:.4f}, instantaneous={instantaneous_flip:.4f}"
        )

    def test_shadow_mode_does_not_modify_steering(self) -> None:
        # Shadow: term computed, metadata populated, steering NOT adjusted.
        meta_shadow = _compute(
            _pp_controller(pp_recovery_term_shadow_mode=True),
            lat_err=1.5, at_car=1.5,
        )
        # Active: same inputs, term applied to steering_before_limits.
        meta_active = _compute(
            _pp_controller(pp_recovery_term_shadow_mode=False),
            lat_err=1.5, at_car=1.5,
        )

        # Both compute the same term value in telemetry.
        assert meta_shadow["lateral_error_recovery_term_applied_rad"] == pytest.approx(
            meta_active["lateral_error_recovery_term_applied_rad"], rel=1e-6
        )
        assert meta_shadow["lateral_error_recovery_shadow_mode"] == 1
        assert meta_active["lateral_error_recovery_shadow_mode"] == 0
        # Final steering differs: active adds the term upstream of limiters.
        assert meta_active["steering"] != pytest.approx(meta_shadow["steering"], abs=1e-9)

    def test_at_car_signal_used_when_larger(self) -> None:
        # Case A: at_car (1.5) > |lookahead| (0.2) → effective=1.5, source='at_car'.
        meta_a = _compute(_pp_controller(), lat_err=0.2, at_car=1.5)
        assert meta_a["lateral_error_recovery_e_lat_source"] == "at_car"
        assert meta_a["lateral_error_recovery_smoothstep_weight"] == pytest.approx(0.5, abs=1e-6)

        # Case B: |lookahead| (1.5) > at_car (0.2) → effective=1.5, source='lookahead'.
        meta_b = _compute(_pp_controller(), lat_err=1.5, at_car=0.2)
        assert meta_b["lateral_error_recovery_e_lat_source"] == "lookahead"
        assert meta_b["lateral_error_recovery_smoothstep_weight"] == pytest.approx(0.5, abs=1e-6)

    def test_lpf_smooths_abrupt_smoothstep_transition(self) -> None:
        # Stabilize at weight=0.8 (e=1.8), then drop to weight=0.0 (e=0.9).
        # LPF should retain most of the prior state for one frame.
        ctrl = _pp_controller(pp_recovery_term_lpf_tau_s=0.20)
        for _ in range(30):
            meta_hi = _compute(ctrl, lat_err=1.8, at_car=1.8, dt=0.033)
        term_hi = meta_hi["lateral_error_recovery_term_applied_rad"]
        assert term_hi > 0.0

        meta_step = _compute(ctrl, lat_err=0.9, at_car=0.9, dt=0.033)
        term_step = meta_step["lateral_error_recovery_term_applied_rad"]
        # alpha = 0.033/(0.20+0.033) ≈ 0.1417. Qualitative check: the LPF retains
        # most of the prior state (>70% of term_hi), not a single-frame collapse
        # to the new raw target. Exact numerical match is brittle because the
        # PP pipeline has its own lateral-error EMA upstream.
        retained_fraction = term_step / term_hi
        assert 0.70 < retained_fraction < 0.95, (
            f"LPF retained_fraction={retained_fraction:.3f} outside expected [0.70, 0.95]"
        )

    def test_disabled_flag(self) -> None:
        meta = _compute(
            _pp_controller(pp_recovery_term_enabled=False),
            lat_err=1.8, at_car=1.8,
        )
        assert meta["lateral_error_recovery_term_applied_rad"] == pytest.approx(0.0, abs=1e-9)
        assert meta["lateral_error_recovery_smoothstep_weight"] == pytest.approx(0.0, abs=1e-9)
        assert meta["lateral_error_recovery_e_lat_source"] == "none"

    def test_reset_zeros_lpf_state(self) -> None:
        ctrl = _pp_controller(pp_recovery_term_lpf_tau_s=0.20)
        for _ in range(20):
            _compute(ctrl, lat_err=2.5, at_car=2.5, dt=0.033)
        assert ctrl._recovery_term_last != pytest.approx(0.0, abs=1e-6)
        ctrl.reset()
        assert ctrl._recovery_term_last == pytest.approx(0.0, abs=1e-12)

    def test_lpf_starts_at_zero_no_initial_step(self) -> None:
        # Simulates MPC→PP handoff: fresh controller, high e_lat, must not
        # emit a full-magnitude term on frame 1 — LPF starts at 0 and ramps
        # over tau.
        ctrl = _pp_controller(pp_recovery_term_lpf_tau_s=0.20)
        meta_first = _compute(ctrl, lat_err=1.5, at_car=1.5, dt=0.033)
        raw_expected = 0.5 * 0.5 * abs(meta_first["pp_geometric_steering"])
        applied = abs(meta_first["lateral_error_recovery_term_applied_rad"])
        assert applied < raw_expected * 0.25, (
            f"First-frame LPF output should be small; applied={applied:.4f}, raw={raw_expected:.4f}"
        )
