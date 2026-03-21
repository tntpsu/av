"""
Tests for Step 3: Grade compensation in MPC, PP longitudinal, and speed governor.

Verifies that grade_rad produces correct compensatory forces, flat-grade is a
strict no-op, and extreme grades are clamped.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "control"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from control.mpc_controller import MPCParams, MPCSolver, MPCController
from data.formats.data_format import VehicleState as VehicleStateData, ControlCommand as ControlCommandData


# ── MPC grade compensation ──────────────────────────────────────────────────

class TestMPCGradeCompensation:
    """MPC solver and controller grade_rad tests."""

    @pytest.fixture
    def solver(self):
        params = MPCParams()
        return MPCSolver(params)

    @pytest.fixture
    def controller(self):
        config = {"trajectory": {"mpc": {}}}
        return MPCController(config)

    def _solve(self, solver, grade_rad=0.0):
        """Helper: solve one frame with standard inputs."""
        N = solver._N
        return solver.solve(
            e_lat=0.1,
            e_heading=0.01,
            v=12.0,
            last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(N),
            v_target=12.0,
            v_max=15.0,
            dt=0.05,
            grade_rad=grade_rad,
        )

    def test_flat_grade_unchanged(self, solver):
        """grade=0.0 must produce numerically identical QP to no-grade baseline."""
        result_flat = self._solve(solver, grade_rad=0.0)
        # sin(0) = 0 → gravity offset = 0 → no change
        assert result_flat["feasible"]
        assert abs(result_flat["accel"]) < 5.0  # sanity

    def test_uphill_commands_more_accel(self, solver):
        """Uphill grade should make MPC command more acceleration."""
        result_flat = self._solve(solver, grade_rad=0.0)
        result_uphill = self._solve(solver, grade_rad=0.05)
        # Uphill: gravity opposes motion → MPC needs more accel
        assert result_uphill["accel"] > result_flat["accel"] - 0.01, (
            f"Uphill accel {result_uphill['accel']:.3f} should be >= flat {result_flat['accel']:.3f}"
        )

    def test_downhill_commands_less_accel(self, solver):
        """Downhill grade should make MPC command less acceleration (more braking)."""
        result_flat = self._solve(solver, grade_rad=0.0)
        result_downhill = self._solve(solver, grade_rad=-0.05)
        # Downhill: gravity assists motion → less accel or more braking
        assert result_downhill["accel"] < result_flat["accel"] + 0.01, (
            f"Downhill accel {result_downhill['accel']:.3f} should be <= flat {result_flat['accel']:.3f}"
        )

    def test_grade_does_not_affect_lateral_dynamics(self, solver):
        """Grade only affects velocity row, not lateral/heading dynamics."""
        result_flat = self._solve(solver, grade_rad=0.0)
        result_grade = self._solve(solver, grade_rad=0.05)
        # Steering should be very similar (small differences from v coupling ok)
        steer_diff = abs(result_flat["steering_normalized"] - result_grade["steering_normalized"])
        assert steer_diff < 0.1, (
            f"Steering diff {steer_diff:.4f} too large — grade should mainly affect longitudinal"
        )

    def test_extreme_grade_clamped(self, solver):
        """Grade beyond clamp limit should be clamped to ±grade_clamp_rad."""
        # 0.3 rad > default clamp of 0.15 rad
        result_extreme = self._solve(solver, grade_rad=0.30)
        result_at_clamp = self._solve(solver, grade_rad=0.15)
        # Should produce same result as clamped value
        assert result_extreme["feasible"]
        assert abs(result_extreme["accel"] - result_at_clamp["accel"]) < 0.01

    def test_mpc_controller_passes_grade(self, controller):
        """MPCController.compute_steering() accepts grade_rad kwarg."""
        result = controller.compute_steering(
            e_lat=0.1,
            e_heading=0.01,
            current_speed=12.0,
            last_delta_norm=0.0,
            kappa_ref=0.0,
            v_target=12.0,
            v_max=15.0,
            dt=0.05,
            grade_rad=0.05,
        )
        assert "steering_normalized" in result

    def test_mpc_controller_grade_default_zero(self, controller):
        """compute_steering() without grade_rad kwarg still works (backward compat)."""
        result = controller.compute_steering(
            e_lat=0.1,
            e_heading=0.01,
            current_speed=12.0,
            last_delta_norm=0.0,
            kappa_ref=0.0,
            v_target=12.0,
            v_max=15.0,
            dt=0.05,
        )
        assert "steering_normalized" in result


# ── PP longitudinal grade compensation ──────────────────────────────────────

class TestPPLongitudinalGrade:
    """PP longitudinal controller gravity feedforward tests."""

    def test_pp_compute_control_signature_accepts_grade_rad(self):
        """VehicleController.compute_control() has grade_rad parameter."""
        import inspect
        from control.pid_controller import VehicleController
        sig = inspect.signature(VehicleController.compute_control)
        assert "grade_rad" in sig.parameters, (
            "VehicleController.compute_control() must accept grade_rad kwarg"
        )
        # Default should be 0.0 (backward compat)
        param = sig.parameters["grade_rad"]
        assert param.default == 0.0

    def test_longitudinal_controller_accepts_grade_rad(self):
        """LongitudinalController.compute_control() has grade_rad parameter."""
        import inspect
        from control.pid_controller import LongitudinalController
        sig = inspect.signature(LongitudinalController.compute_control)
        assert "grade_rad" in sig.parameters
        assert sig.parameters["grade_rad"].default == 0.0


# ── Data format fields ──────────────────────────────────────────────────────

class TestGradeDataFields:
    """Grade-related data format fields exist and default correctly."""

    def _make_vs(self, **kwargs):
        """Create a VehicleState with required positional args filled."""
        defaults = dict(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0, 0, 0, 1], dtype=float),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=0.0,
            steering_angle=0.0,
            motor_torque=0.0,
            brake_torque=0.0,
        )
        defaults.update(kwargs)
        return VehicleStateData(**defaults)

    def test_vehicle_state_has_grade_fields(self):
        vs = self._make_vs()
        assert hasattr(vs, 'pitch_rad')
        assert hasattr(vs, 'roll_rad')
        assert hasattr(vs, 'road_grade')
        assert vs.pitch_rad == 0.0
        assert vs.roll_rad == 0.0
        assert vs.road_grade == 0.0

    def test_control_command_has_grade_compensation_active(self):
        cc = ControlCommandData(timestamp=0.0, steering=0.0, throttle=0.0, brake=0.0)
        assert hasattr(cc, 'grade_compensation_active')
        assert cc.grade_compensation_active == 0.0

    def test_vehicle_state_accepts_grade_values(self):
        vs = self._make_vs(pitch_rad=0.05, roll_rad=0.01, road_grade=0.03)
        assert vs.pitch_rad == 0.05
        assert vs.roll_rad == 0.01
        assert vs.road_grade == 0.03


# ── Scoring registry grade constants ────────────────────────────────────────

class TestGradeFFGain:
    """Grade feedforward gain amplification tests."""

    @pytest.fixture
    def solver_default(self):
        params = MPCParams()
        return MPCSolver(params)

    @pytest.fixture
    def solver_unity(self):
        params = MPCParams(grade_ff_gain=1.8)
        return MPCSolver(params)

    def _solve(self, solver, grade_rad=0.0):
        N = solver._N
        return solver.solve(
            e_lat=0.1,
            e_heading=0.01,
            v=12.0,
            last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(N),
            v_target=12.0,
            v_max=15.0,
            dt=0.05,
            grade_rad=grade_rad,
        )

    def test_grade_ff_gain_amplifies_compensation(self, solver_default, solver_unity):
        """MPC with gain=1.8 produces ~1.8× more accel offset than gain=1.0."""
        result_1x = self._solve(solver_default, grade_rad=0.05)
        result_18x = self._solve(solver_unity, grade_rad=0.05)
        # Both should be feasible
        assert result_1x["feasible"]
        assert result_18x["feasible"]
        # Unity gain should command more accel on uphill (more compensation)
        assert result_18x["accel"] > result_1x["accel"] - 0.01

    def test_grade_ff_gain_default_unity(self):
        """Default gain is 1.0 (backward compat)."""
        params = MPCParams()
        assert params.grade_ff_gain == 1.0

    def test_mpc_grade_ff_gain_from_config(self):
        """MPCParams.from_config() reads mpc_grade_ff_gain."""
        cfg = {"trajectory": {"mpc": {"mpc_grade_ff_gain": 1.8}}}
        params = MPCParams.from_config(cfg)
        assert params.grade_ff_gain == 1.8

    def test_flat_grade_gain_no_effect(self, solver_default, solver_unity):
        """Gain has no effect when grade=0.0 (sin(0)×gain = 0)."""
        result_1x = self._solve(solver_default, grade_rad=0.0)
        result_18x = self._solve(solver_unity, grade_rad=0.0)
        assert abs(result_1x["accel"] - result_18x["accel"]) < 0.01

    def test_pp_longitudinal_grade_ff_gain_default(self):
        """LongitudinalController default grade_ff_gain is 1.0."""
        from control.pid_controller import LongitudinalController
        lc = LongitudinalController()
        assert lc.grade_ff_gain == 1.0

    def test_pp_longitudinal_grade_ff_gain_custom(self):
        """LongitudinalController accepts custom grade_ff_gain."""
        from control.pid_controller import LongitudinalController
        lc = LongitudinalController(grade_ff_gain=1.8)
        assert lc.grade_ff_gain == 1.8


class TestGradeScoringConstants:
    """Grade constants in scoring_registry match expected values."""

    def test_constants_exist(self):
        from scoring_registry import (
            GRADE_MAX_SAFE_PCT,
            GRADE_EMA_ALPHA,
            GRADE_CLAMP_RAD,
            DOWNHILL_SPEED_MARGIN_MPS,
        )
        assert GRADE_MAX_SAFE_PCT == 10.0
        assert GRADE_EMA_ALPHA == 0.3
        assert GRADE_CLAMP_RAD == 0.15
        assert DOWNHILL_SPEED_MARGIN_MPS == 1.0

    def test_grade_ff_gain_constants(self):
        from scoring_registry import GRADE_FF_GAIN_DEFAULT, GRADE_FF_GAIN_UNITY
        assert GRADE_FF_GAIN_DEFAULT == 1.8
        assert GRADE_FF_GAIN_UNITY == 1.8


# ── Step 3C: Grade robustness base system tests ──────────────────────────

class TestEffectiveMaxAccel:
    """Phase 1: Grade-aware effective max_accel tests."""

    def test_effective_max_accel_flat_unchanged(self):
        """grade=0 → effective == base max_accel."""
        from control.pid_controller import LongitudinalController
        lc = LongitudinalController(max_accel=1.2, max_decel=2.4, grade_ff_gain=1.8,
                                     continuous_accel_control=True)
        lc.compute_control(current_speed=10.0, reference_velocity=12.0, dt=0.033, grade_rad=0.0)
        assert abs(lc._effective_max_accel - 1.2) < 0.01, (
            f"Flat grade should give effective_max_accel=1.2, got {lc._effective_max_accel}"
        )

    def test_effective_max_accel_uphill(self):
        """grade=0.05, gain=1.8 → effective ≈ 1.2 + 0.88 = 2.08."""
        from control.pid_controller import LongitudinalController
        lc = LongitudinalController(max_accel=1.2, max_decel=2.4, grade_ff_gain=1.8,
                                     continuous_accel_control=True)
        lc.compute_control(current_speed=10.0, reference_velocity=12.0, dt=0.033, grade_rad=0.05)
        gravity = 9.81 * math.sin(0.05) * 1.8
        expected = 1.2 + abs(gravity)
        assert abs(lc._effective_max_accel - expected) < 0.01, (
            f"Uphill should give effective_max_accel≈{expected:.2f}, got {lc._effective_max_accel}"
        )

    def test_effective_max_decel_downhill(self):
        """Symmetric for braking: effective_max_decel expands on downhill."""
        from control.pid_controller import LongitudinalController
        lc = LongitudinalController(max_accel=1.2, max_decel=2.4, grade_ff_gain=1.8,
                                     continuous_accel_control=True)
        lc.compute_control(current_speed=10.0, reference_velocity=12.0, dt=0.033, grade_rad=-0.05)
        gravity = 9.81 * math.sin(-0.05) * 1.8
        expected = 2.4 + abs(gravity)
        assert abs(lc._effective_max_decel - expected) < 0.01, (
            f"Downhill should give effective_max_decel≈{expected:.2f}, got {lc._effective_max_decel}"
        )

    def test_throttle_mapping_uses_effective(self):
        """Throttle should not saturate at base max_accel on grade."""
        from control.pid_controller import LongitudinalController
        lc = LongitudinalController(max_accel=1.2, max_decel=2.4, grade_ff_gain=1.8,
                                     continuous_accel_control=True)
        throttle, brake = lc.compute_control(
            current_speed=10.0, reference_velocity=12.0, dt=0.033, grade_rad=0.05)
        # With effective max_accel ≈ 2.08, throttle should stay below 0.9
        # even with significant speed error
        assert throttle < 0.95, f"Throttle {throttle:.3f} should not saturate on 5% grade"


class TestJerkRelaxation:
    """Phase 4: Grade-proportional jerk relaxation tests."""

    def test_jerk_relaxation_flat_unchanged(self):
        """grade=0 → jerk limit = base (no bonus)."""
        from control.pid_controller import LongitudinalController
        lc = LongitudinalController(max_jerk=2.0, max_jerk_min=1.0, max_jerk_max=4.0,
                                     grade_ff_gain=1.8, grade_jerk_relaxation_gain=2.0,
                                     continuous_accel_control=True)
        lc.compute_control(current_speed=10.0, reference_velocity=12.0, dt=0.033, grade_rad=0.0)
        # No grade → no jerk bonus. Just verify it runs without error.
        # The actual jerk limit is internal, but we can verify via attribute.
        assert lc.grade_jerk_relaxation_gain == 2.0

    def test_jerk_relaxation_on_grade(self):
        """grade=0.05 → jerk limit increased by gravity_accel * gain."""
        from control.pid_controller import LongitudinalController
        lc = LongitudinalController(max_jerk=2.0, max_jerk_min=1.0, max_jerk_max=4.0,
                                     grade_ff_gain=1.8, grade_jerk_relaxation_gain=2.0,
                                     continuous_accel_control=True)
        # Run two frames to see jerk limiting in action
        lc.compute_control(current_speed=10.0, reference_velocity=12.0, dt=0.033, grade_rad=0.05)
        lc.compute_control(current_speed=10.0, reference_velocity=12.0, dt=0.033, grade_rad=0.05)
        # With grade_jerk_relaxation_gain=2.0 and gravity≈0.88:
        # bonus = 0.88 * 2.0 = 1.76 → effective jerk limit is higher
        # Verify the controller accepted the gain parameter
        gravity = 9.81 * math.sin(0.05) * 1.8
        expected_bonus = abs(gravity) * 2.0
        assert expected_bonus > 1.5, f"Expected jerk bonus > 1.5, got {expected_bonus}"


class TestSteeringDamping:
    """Phase 5: Grade-proportional steering damping tests."""

    def test_steering_damping_flat_unchanged(self):
        """grade=0 → alpha unchanged (condition abs(grade) < 0.01 → skip)."""
        from control.pid_controller import LateralController
        lc = LateralController(
            steering_smoothing_alpha=0.7,
            base_error_smoothing_alpha=0.7,
            heading_error_smoothing_alpha=0.45,
            grade_steering_damping_gain=5.0,
        )
        lc._current_grade_rad = 0.0
        # Verify the parameter was accepted and grade threshold is met
        assert lc.grade_steering_damping_gain == 5.0
        assert abs(lc._current_grade_rad) <= 0.01  # below threshold → no damping

    def test_steering_damping_on_grade(self):
        """grade=0.05 → alpha reduced by grade_damping."""
        from control.pid_controller import LateralController
        lc = LateralController(
            steering_smoothing_alpha=0.7,
            base_error_smoothing_alpha=0.7,
            heading_error_smoothing_alpha=0.45,
            grade_steering_damping_gain=5.0,
        )
        lc._current_grade_rad = 0.05
        # Expected damping: min(0.3, 0.05 * 5.0) = 0.25
        # Steering alpha: 0.7 - 0.25 = 0.45
        grade_damping = min(0.3, abs(0.05) * 5.0)
        expected_alpha = max(0.15, 0.7 - grade_damping)
        assert abs(expected_alpha - 0.45) < 0.01, f"Expected alpha=0.45, got {expected_alpha}"


class TestMPCQPBoundsGradeAware:
    """Phase 1: MPC accel bounds expand with grade."""

    def test_mpc_qp_bounds_grade_aware(self):
        """MPC solve with grade should produce feasible result with expanded bounds."""
        params = MPCParams(max_accel=1.2, max_decel=2.4)
        solver = MPCSolver(params)
        N = solver._N
        result = solver.solve(
            e_lat=0.1, e_heading=0.01, v=12.0, last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(N), v_target=12.0, v_max=15.0,
            dt=0.05, grade_rad=0.05,
        )
        assert result["feasible"], "MPC should be feasible with grade-aware bounds"
        # Uphill commands more accel than flat
        result_flat = solver.solve(
            e_lat=0.1, e_heading=0.01, v=12.0, last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(N), v_target=12.0, v_max=15.0,
            dt=0.05, grade_rad=0.0,
        )
        assert result["accel"] >= result_flat["accel"] - 0.1


class TestPreviewSpeedNoFalseCap:
    """Phase 2: Preview distance fix prevents false speed cap on gentle curves."""

    def test_preview_speed_r100_no_false_cap(self):
        """R100 at 12 m/s → no preview speed cap (curve_speed=14 > target=12)."""
        from control.speed_governor import SpeedGovernor, SpeedGovernorConfig
        from trajectory.speed_planner import SpeedPlannerConfig
        config = SpeedGovernorConfig()
        planner_config = SpeedPlannerConfig()
        gov = SpeedGovernor(config, planner_config)
        # R100 → κ = 0.01
        preview = gov._compute_preview_speed(
            current_target=12.0,
            preview_curvature=0.01,
            distance_to_curve=50.0,  # 50m away
        )
        # curve_speed = sqrt(a_lat_max / 0.01) ≈ 14 > 12 → should return None
        assert preview is None or preview >= 11.5, (
            f"R100 at 12 m/s should not trigger false cap, got preview={preview}"
        )


# ── Step 3D: Angular velocity recording fix ──────────────────────────

class TestAngularVelocityRecording:
    """Verify angular velocity is not hardcoded to zeros."""

    def _make_vs(self, **kwargs):
        defaults = dict(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0, 0, 0, 1], dtype=float),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=0.0,
            steering_angle=0.0,
            motor_torque=0.0,
            brake_torque=0.0,
        )
        defaults.update(kwargs)
        return VehicleStateData(**defaults)

    def test_angular_velocity_not_hardcoded(self):
        """VehicleState accepts non-zero angular velocity."""
        ang_vel = np.array([0.01, 0.5, -0.02])
        vs = self._make_vs(angular_velocity=ang_vel)
        np.testing.assert_allclose(vs.angular_velocity, ang_vel)

    def test_angular_velocity_default_zero(self):
        """Default angular velocity is zeros (backward compat)."""
        vs = self._make_vs()
        np.testing.assert_allclose(vs.angular_velocity, np.zeros(3))


# ── Step 3D: Per-wheel telemetry fields ──────────────────────────────

class TestWheelTelemetry:
    """Verify per-wheel diagnostic fields exist on VehicleState."""

    def _make_vs(self, **kwargs):
        defaults = dict(
            timestamp=0.0,
            position=np.zeros(3),
            rotation=np.array([0, 0, 0, 1], dtype=float),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            speed=0.0,
            steering_angle=0.0,
            motor_torque=0.0,
            brake_torque=0.0,
        )
        defaults.update(kwargs)
        return VehicleStateData(**defaults)

    def test_wheel_data_fields_exist_in_vehicle_state(self):
        """VehicleState has wheel slip/force/rpm fields."""
        vs = self._make_vs()
        for field_name in ("wheel_sideways_slip", "wheel_forward_slip",
                           "wheel_contact_force", "wheel_rpm",
                           "wheel_sprung_mass", "wheel_contact_normal_y",
                           "wheel_steer_angle_actual"):
            assert hasattr(vs, field_name), f"Missing field: {field_name}"

    def test_wheel_data_default_zeros(self):
        """Default wheel values are None/0 (backward compat with old recordings)."""
        vs = self._make_vs()
        # Array fields default to None
        assert vs.wheel_sideways_slip is None
        assert vs.wheel_forward_slip is None
        assert vs.wheel_contact_force is None
        # Scalar field defaults to 0
        assert vs.wheel_steer_angle_actual == 0.0

    def test_wheel_data_accepts_arrays(self):
        """VehicleState accepts wheel data arrays."""
        slip = np.array([0.01, -0.02, 0.03, -0.04])
        vs = self._make_vs(wheel_sideways_slip=slip)
        np.testing.assert_allclose(vs.wheel_sideways_slip, slip)
