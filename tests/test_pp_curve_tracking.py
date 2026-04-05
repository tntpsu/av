"""
Tests for Pure Pursuit curve tracking with and without map feedforward.

These tests would have caught the s_loop regression where pp_map_ff_enabled=false
left PP with zero curvature feedforward, causing oscillation and understeer when
live seg-model perception replaced frozen stale data.

Test categories:
  1. PP without map FF produces insufficient steering on realistic curves
  2. PP with map FF produces adequate curve steering
  3. Closed-loop: PP tracks a curve with dynamically shifting ref_x (live perception)
"""

import numpy as np
import pytest

from control.pid_controller import LateralController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pp_controller(**overrides) -> LateralController:
    """Build a PP controller with production-like defaults."""
    defaults = dict(
        kp=1.0,
        ki=0.002,
        kd=0.5,
        control_mode="pure_pursuit",
        max_steering=1.0,
        pp_feedback_gain=0.075,
        pp_min_lookahead=0.5,
        pp_max_steering_rate=5.0,        # high to avoid rate-limiting in tests
        pp_max_steering_jerk=0.0,        # disable jerk limit in tests
        pp_stale_decay=0.98,
        pp_ref_jump_clamp=0.5,
        pp_map_ff_enabled=False,         # default off; tests enable explicitly
        pp_map_ff_gain=0.8,
        pp_map_ff_wheelbase_m=2.5,
        pp_map_ff_curvature_min=0.005,
        pp_map_ff_curvature_max_clip=0.08,
        pp_curve_local_lookahead_floor_enabled=False,  # disable floor for determinism
    )
    defaults.update(overrides)
    return LateralController(**defaults)


def _curve_ref(ref_x: float, ref_y: float = 6.0,
               curvature: float = 0.0,
               heading: float = 0.0,
               with_primary_curvature: bool = False) -> dict:
    """Build a reference point simulating a curve scenario."""
    ref = {
        "x": ref_x,
        "y": ref_y,
        "heading": heading,
        "velocity": 10.0,
        "curvature": curvature,
    }
    if with_primary_curvature and abs(curvature) > 0:
        ref["curvature_primary_abs"] = abs(curvature)
        ref["curvature_source"] = "primary_contract"
        ref["curvature_primary_source"] = "map_track"
    return ref


def _ackermann_steering(curvature: float, wheelbase: float = 2.5) -> float:
    """Ideal Ackermann steering for a given curvature (normalised to [-1,1])."""
    steer_rad = np.arctan(wheelbase * curvature)
    return steer_rad / np.radians(30.0)


# ---------------------------------------------------------------------------
# 1. PP without map FF understeers on curves (BUG DETECTOR)
# ---------------------------------------------------------------------------
# This test verifies that PP geometric steering alone (from ref_x) is
# insufficient for moderate-to-tight curves.  It would have flagged the
# s_loop overlay (pp_map_ff_enabled: false) as risky.

class TestPPWithoutMapFF:
    """PP with map FF disabled is fragile: steering depends entirely on perception."""

    def test_no_map_ff_applied_when_disabled(self) -> None:
        ctrl = _pp_controller(pp_map_ff_enabled=False)
        ref = _curve_ref(0.5, curvature=0.025, with_primary_curvature=True)
        meta = ctrl.compute_steering(
            0.0, ref, current_speed=10.0, dt=0.033, return_metadata=True
        )
        assert meta["pp_map_ff_applied"] == pytest.approx(0.0, abs=1e-8)

    def test_perception_noise_causes_steering_variance(self) -> None:
        """
        Without map FF, steering variance is 100% driven by ref_x noise.
        With map FF, the constant FF component provides a stable floor.
        This is the key reason map FF matters — it decouples curve-following
        from perception accuracy.
        """
        curvature = 0.025
        R = 1.0 / curvature
        d = 6.0
        theta = d / R

        # Nominal arc reference
        ref_x_nominal = R * (1 - np.cos(theta))
        ref_y_nominal = R * np.sin(theta)

        # Simulate 5 frames with ±0.1m perception noise on ref_x
        noise_offsets = [-0.10, -0.05, 0.0, 0.05, 0.10]
        steers_no_ff = []
        steers_with_ff = []

        for noise in noise_offsets:
            ref = _curve_ref(
                ref_x=ref_x_nominal + noise,
                ref_y=ref_y_nominal,
                curvature=curvature,
                heading=theta,
                with_primary_curvature=True,
            )

            ctrl_no_ff = _pp_controller(pp_map_ff_enabled=False, pp_feedback_gain=0.0)
            meta_no_ff = ctrl_no_ff.compute_steering(
                0.0, ref, current_speed=10.0, dt=0.033, return_metadata=True
            )
            steers_no_ff.append(meta_no_ff["steering"])

            ctrl_ff = _pp_controller(pp_map_ff_enabled=True, pp_map_ff_gain=0.8, pp_feedback_gain=0.0)
            meta_ff = ctrl_ff.compute_steering(
                0.0, ref, current_speed=10.0, dt=0.033, return_metadata=True
            )
            steers_with_ff.append(meta_ff["steering"])

        range_no_ff = max(steers_no_ff) - min(steers_no_ff)
        range_with_ff = max(steers_with_ff) - min(steers_with_ff)

        # Both have the same steering RANGE (noise affects geometric equally),
        # but with FF the MEAN is higher and the fraction of steering that
        # comes from the stable FF component is significant.
        mean_no_ff = np.mean(steers_no_ff)
        mean_with_ff = np.mean(steers_with_ff)
        ff_fraction = (mean_with_ff - mean_no_ff) / mean_with_ff

        # Map FF should contribute at least 20% of total steering on this curve
        assert ff_fraction >= 0.20, (
            f"Map FF contributes only {ff_fraction:.1%} of total steering at κ={curvature}. "
            f"mean_no_ff={mean_no_ff:.4f}, mean_with_ff={mean_with_ff:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. PP with map FF provides adequate curve steering
# ---------------------------------------------------------------------------

class TestPPWithMapFF:
    """PP with map FF enabled should reach or exceed ideal Ackermann."""

    @pytest.mark.parametrize("curvature", [0.015, 0.025, 0.040])
    def test_map_ff_closes_understeer_gap(self, curvature: float) -> None:
        """With map FF, total steering should be >= ideal Ackermann."""
        ctrl = _pp_controller(
            pp_map_ff_enabled=True,
            pp_map_ff_gain=0.8,
            pp_feedback_gain=0.0,  # isolate geometric + FF
        )

        R = 1.0 / curvature
        s = 6.0
        theta = s / R
        ref_x = R * (1 - np.cos(theta))
        ref_y = R * np.sin(theta)

        ref = _curve_ref(
            ref_x=ref_x,
            ref_y=ref_y,
            curvature=curvature,
            heading=theta,
            with_primary_curvature=True,
        )

        meta = ctrl.compute_steering(
            current_heading=0.0,
            reference_point=ref,
            current_speed=10.0,
            dt=0.033,
            return_metadata=True,
        )

        ideal = abs(_ackermann_steering(curvature))
        actual = abs(meta["steering"])
        ff = abs(meta["pp_map_ff_applied"])

        assert ff > 0.0, "Map FF should be non-zero for this curvature"
        # With FF, total steering should be at least 80% of ideal
        # (exact match not expected due to PP geometry + gain < 1.0)
        assert actual >= ideal * 0.80, (
            f"κ={curvature}: PP+FF ({actual:.4f}) still below 80% of "
            f"ideal Ackermann ({ideal:.4f}).  ff_applied={ff:.4f}"
        )

    def test_map_ff_sign_matches_curve_direction(self) -> None:
        """FF steering sign should match the curve direction."""
        ctrl = _pp_controller(pp_map_ff_enabled=True, pp_feedback_gain=0.0)

        for sign in [1.0, -1.0]:
            curv = sign * 0.025
            ref = _curve_ref(
                ref_x=sign * 0.3,
                ref_y=6.0,
                curvature=curv,
                heading=sign * 0.05,
                with_primary_curvature=True,
            )
            meta = ctrl.compute_steering(
                0.0, ref, current_speed=10.0, dt=0.033, return_metadata=True
            )
            assert np.sign(meta["pp_map_ff_applied"]) == sign, (
                f"FF sign wrong for curvature={curv}"
            )


# ---------------------------------------------------------------------------
# 3. Closed-loop: PP tracks curve with shifting ref_x (live perception sim)
# ---------------------------------------------------------------------------

class TestPPClosedLoopCurveTracking:
    """
    Simulate a simplified closed-loop where the perceived lane center
    shifts dynamically as the car follows a constant-curvature curve.
    This models the live seg-model perception scenario.
    """

    @staticmethod
    def _simulate_curve(
        ctrl: LateralController,
        curvature: float,
        speed: float = 10.0,
        duration_s: float = 4.0,
        dt: float = 0.033,
        perception_smoothing_alpha: float = 0.2,
    ) -> dict:
        """
        Run a simplified closed-loop curve-following simulation.

        The car drives along a constant-curvature arc.  Each frame, the
        "perceived" lane center at the lookahead distance is computed from
        the car's current position/heading error relative to the arc,
        then passed through an EMA filter to model the real perception
        pipeline's smoothing (lane_position_smoothing_alpha = 0.2).

        This is critical: with perfect perception, PP geometric alone
        matches Ackermann and needs no map FF.  The value of map FF
        is resilience to the delay that perception smoothing introduces.

        Returns dict with lateral_errors, steering_history, etc.
        """
        R = 1.0 / curvature
        lookahead = 6.0
        wheelbase = 2.5
        n_frames = int(duration_s / dt)
        alpha = perception_smoothing_alpha

        # State: lateral offset from arc center-line, heading error
        e_lat = 0.0     # positive = car is to the right of the arc
        e_heading = 0.0  # positive = car pointing right of tangent
        smoothed_ref_x = None  # EMA state

        lateral_errors = []
        steering_history = []

        for i in range(n_frames):
            # Perceived lane center at lookahead in *vehicle* frame.
            # Arc center at distance d has road-frame offset (d²κ/2, d).
            # Car is at (e_lat, 0) with heading error e_heading.
            # Rotating (arc - car) by -e_heading into vehicle frame:
            #   ref_x ≈ d²κ/2 - e_lat - d·e_heading   (small angle)
            d = lookahead
            raw_ref_x = (d ** 2) * curvature / 2.0 - e_lat - d * e_heading

            # EMA smoothing (models real perception pipeline delay)
            if smoothed_ref_x is None:
                smoothed_ref_x = raw_ref_x
            else:
                smoothed_ref_x = alpha * raw_ref_x + (1 - alpha) * smoothed_ref_x

            ref_x = smoothed_ref_x
            ref_y = d

            heading_at_ref = -e_heading + d * curvature

            ref = _curve_ref(
                ref_x=ref_x,
                ref_y=ref_y,
                curvature=curvature,
                heading=heading_at_ref,
                with_primary_curvature=True,
            )

            meta = ctrl.compute_steering(
                current_heading=0.0,
                reference_point=ref,
                current_speed=speed,
                dt=dt,
                return_metadata=True,
            )

            steer_norm = meta["steering"]
            steer_rad = steer_norm * np.radians(30.0)
            kappa_cmd = np.tan(steer_rad) / wheelbase

            # Update vehicle state (bicycle model kinematics)
            ds = speed * dt
            e_lat += speed * e_heading * dt
            e_heading += (kappa_cmd - curvature) * ds

            lateral_errors.append(e_lat)
            steering_history.append(steer_norm)

        return {
            "lateral_errors": np.array(lateral_errors),
            "steering_history": np.array(steering_history),
            "rmse": float(np.sqrt(np.mean(np.array(lateral_errors) ** 2))),
            "max_abs_error": float(np.max(np.abs(lateral_errors))),
        }

    def test_with_map_ff_tracks_curve(self) -> None:
        """PP + map FF should track a moderate curve with RMSE < 0.40m."""
        ctrl = _pp_controller(
            pp_map_ff_enabled=True,
            pp_map_ff_gain=0.8,
            pp_feedback_gain=0.075,
        )
        result = self._simulate_curve(ctrl, curvature=0.020, duration_s=5.0)
        assert result["rmse"] < 0.40, (
            f"PP+FF closed-loop RMSE={result['rmse']:.3f}m > 0.40m gate on κ=0.020"
        )

    def test_ff_provides_immediate_steering_on_curve_entry(self) -> None:
        """
        On the first frame of a curve, map FF provides immediate steering
        while PP geometric alone must wait for ref_x to build up through
        the smoothing pipeline.  This tests the curve-entry transient where
        FF matters most.
        """
        curvature = 0.025

        # First frame: car on a straight, suddenly enters a curve.
        # ref_x is still ~0 (EMA hasn't responded yet), but curvature is known.
        ref = _curve_ref(
            ref_x=0.0,  # perception hasn't caught up yet
            ref_y=6.0,
            curvature=curvature,
            heading=0.0,
            with_primary_curvature=True,
        )

        ctrl_ff = _pp_controller(pp_map_ff_enabled=True, pp_map_ff_gain=0.8, pp_feedback_gain=0.0)
        ctrl_no_ff = _pp_controller(pp_map_ff_enabled=False, pp_feedback_gain=0.0)

        meta_ff = ctrl_ff.compute_steering(0.0, ref, current_speed=10.0, dt=0.033, return_metadata=True)
        meta_no_ff = ctrl_no_ff.compute_steering(0.0, ref, current_speed=10.0, dt=0.033, return_metadata=True)

        # Without FF: ref_x=0 → near-zero steering
        assert abs(meta_no_ff["steering"]) < 0.01, (
            f"Without FF, steering should be ~0 when ref_x=0, got {meta_no_ff['steering']:.4f}"
        )
        # With FF: map curvature provides immediate steering
        assert abs(meta_ff["steering"]) > 0.03, (
            f"With FF, steering should be >0.03 from map curvature alone, got {meta_ff['steering']:.4f}"
        )
        # FF provides 100% of steering in this scenario
        assert abs(meta_ff["pp_map_ff_applied"]) > 0.03

    def test_preview_curvature_triggers_ff_before_primary(self) -> None:
        """
        BUG DETECTOR: This test would have caught the s_loop late turn-in.

        When the car is approaching a curve, curvature_preview_abs > 0 but
        curvature_primary_abs is still 0 (car hasn't reached the arc).
        Map FF should activate from the preview curvature, not wait for primary.
        """
        ctrl = _pp_controller(
            pp_map_ff_enabled=True,
            pp_map_ff_gain=0.8,
            pp_feedback_gain=0.0,
        )

        # Approaching curve: preview sees κ=0.025 ahead, primary is zero
        ref = {
            "x": 0.0,
            "y": 6.0,
            "heading": 0.05,  # slight heading toward curve
            "velocity": 10.0,
            "curvature": 0.0,  # no curvature at current position
            # Primary absent — car hasn't reached the arc
            "curvature_preview_abs": 0.025,  # 30m lookahead sees the curve
        }

        meta = ctrl.compute_steering(
            0.0, ref, current_speed=10.0, dt=0.033, return_metadata=True
        )

        # FF should fire from preview curvature
        assert abs(meta["pp_map_ff_applied"]) > 0.03, (
            f"Preview curvature should trigger FF, got ff={meta['pp_map_ff_applied']:.4f}"
        )

    def test_primary_curvature_takes_precedence_over_preview(self) -> None:
        """When primary curvature is available, it should be used over preview."""
        ctrl = _pp_controller(
            pp_map_ff_enabled=True,
            pp_map_ff_gain=0.8,
            pp_feedback_gain=0.0,
        )

        ref = _curve_ref(
            ref_x=0.3,
            ref_y=6.0,
            curvature=0.020,
            heading=0.05,
            with_primary_curvature=True,
        )
        ref["curvature_preview_abs"] = 0.040  # preview is higher

        meta = ctrl.compute_steering(
            0.0, ref, current_speed=10.0, dt=0.033, return_metadata=True
        )

        # FF should use primary (0.020), not preview (0.040)
        # Ackermann at κ=0.020: arctan(2.5*0.020)/30° ≈ 0.095 normalized
        # With gain 0.8: ~0.076. At κ=0.040 it would be ~0.152.
        ff = abs(meta["pp_map_ff_applied"])
        assert ff > 0.05, f"FF should be active, got {ff:.4f}"
        assert ff < 0.12, (
            f"FF={ff:.4f} too high — preview (κ=0.040) may have overridden primary (κ=0.020)"
        )

    @pytest.mark.parametrize("curvature", [0.010, 0.020, 0.030])
    def test_map_ff_rmse_under_gate_across_curvatures(self, curvature: float) -> None:
        """PP + map FF should stay under 0.50m RMSE across moderate curvatures."""
        ctrl = _pp_controller(
            pp_map_ff_enabled=True,
            pp_map_ff_gain=0.8,
            pp_feedback_gain=0.075,
        )
        result = self._simulate_curve(ctrl, curvature=curvature, duration_s=5.0)
        assert result["rmse"] < 0.50, (
            f"κ={curvature}: PP+FF RMSE={result['rmse']:.3f}m exceeds 0.50m gate"
        )

    def test_oscillation_bounded(self) -> None:
        """
        Steering should not oscillate excessively during curve tracking.
        Count sign changes in steering rate — excessive sign changes indicate
        the perception-feedback loop is oscillating.
        """
        ctrl = _pp_controller(
            pp_map_ff_enabled=True,
            pp_map_ff_gain=0.8,
            pp_feedback_gain=0.075,
        )
        result = self._simulate_curve(ctrl, curvature=0.025, duration_s=5.0)
        steer = result["steering_history"]

        # Compute steering rate (frame-to-frame delta)
        steer_rate = np.diff(steer)
        # Count sign changes in steering rate (ignoring tiny changes < 0.001)
        significant = steer_rate[np.abs(steer_rate) > 0.001]
        if len(significant) < 2:
            return  # not enough data
        sign_changes = np.sum(np.diff(np.sign(significant)) != 0)
        # Allow at most 1 sign change per 8 frames on average (settle-in transient ok)
        max_sign_changes = len(significant) // 8 + 5  # +5 for initial transient
        assert sign_changes <= max_sign_changes, (
            f"Steering oscillation: {sign_changes} sign changes in {len(significant)} "
            f"frames (max allowed: {max_sign_changes})"
        )
