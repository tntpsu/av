"""
Guard tests for tools/scoring_registry.py — T-073.

Hard-asserts every constant value to catch accidental edits.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from scoring_registry import (
    ACCEL_P95_GATE_MPS2,
    JERK_P95_GATE_MPS3,
    LATERAL_P95_GATE_M,
    CENTERED_PCT_MIN,
    CENTERED_BAND_M,
    STEERING_JERK_GATE,
    OUT_OF_LANE_THRESHOLD_M,
    CATASTROPHIC_ERROR_M,
    MIN_CONSECUTIVE_OOL,
    CURVATURE_FLOOR_COEFF,
    STEERING_JERK_PENALTY_CAP,
    HEADING_PENALTY_FLOOR_DEG,
    PERCEPTION_CONF_FLOOR,
    TRAJ_REF_ERROR_MAX_M,
    CTRL_LATERAL_ERROR_MAX,
    CTRL_JERK_GATE,
    LOOKAHEAD_MIN_M,
    LOOKAHEAD_MAX_M,
    JERK_ALERT_MPS3,
    CTRL_JERK_ALERT,
    LOOKAHEAD_CONCERN_M,
    MPC_SOLVE_TIME_BUDGET_MS,
    MPC_SOLVE_TIME_ALERT_MS,
    MPC_SOLVE_TIME_P95_MS_GATE,
    MPC_SOLVE_TIME_MARGINAL_MS,
    MPC_INFEASIBILITY_RATE_GATE,
    MPC_FALLBACK_RATE_GATE,
    MPC_REGIME_CHATTER_PER_MIN,
    MPC_STEERING_OSC_RATE_GATE,
    MPC_HEADING_ERROR_P95_RAD,
    MPC_MIN_SPEED_DEFAULT_MPS,
    MPC_MAX_CURVATURE_DEFAULT,
    BENIGN_STALE_REASONS,
    COMFORT_GATES,
    GRADE_MAX_SAFE_PCT,
    GRADE_EMA_ALPHA,
    GRADE_CLAMP_RAD,
    DOWNHILL_SPEED_MARGIN_MPS,
    GRADE_FF_GAIN_DEFAULT,
    GRADE_FF_GAIN_UNITY,
    GRADE_THROTTLE_BUDGET_MIN_RATIO,
    GRADE_THROTTLE_SATURATION_RATE,
    GRADE_JERK_RELAXATION_GAIN,
    GRADE_STEERING_DAMPING_GAIN,
    GRADE_TRANSITION_BLEND_M,
)


# ── Value guard tests ────────────────────────────────────────────────────────

class TestComfortGateConstants:
    def test_accel_p95_gate(self):
        assert ACCEL_P95_GATE_MPS2 == 3.0

    def test_jerk_p95_gate(self):
        assert JERK_P95_GATE_MPS3 == 6.0

    def test_lateral_p95_gate(self):
        assert LATERAL_P95_GATE_M == 0.40

    def test_centered_pct_min(self):
        assert CENTERED_PCT_MIN == 70.0

    def test_centered_band(self):
        assert CENTERED_BAND_M == 0.5

    def test_steering_jerk_gate(self):
        assert STEERING_JERK_GATE == 20.0


class TestSafetyConstants:
    def test_out_of_lane_threshold(self):
        assert OUT_OF_LANE_THRESHOLD_M == 0.5

    def test_catastrophic_error(self):
        assert CATASTROPHIC_ERROR_M == 2.0

    def test_min_consecutive_ool(self):
        assert MIN_CONSECUTIVE_OOL == 10


class TestScoringPenaltyConstants:
    def test_curvature_floor_coeff(self):
        assert CURVATURE_FLOOR_COEFF == 3.0

    def test_steering_jerk_penalty_cap(self):
        assert STEERING_JERK_PENALTY_CAP == 18.0

    def test_heading_penalty_floor(self):
        assert HEADING_PENALTY_FLOOR_DEG == 10.0


class TestLayerHealthConstants:
    def test_perception_conf_floor(self):
        assert PERCEPTION_CONF_FLOOR == 0.1

    def test_traj_ref_error_max(self):
        assert TRAJ_REF_ERROR_MAX_M == 2.0

    def test_ctrl_lateral_error_max(self):
        assert CTRL_LATERAL_ERROR_MAX == 2.0

    def test_ctrl_jerk_gate(self):
        assert CTRL_JERK_GATE == 6.0

    def test_lookahead_min(self):
        assert LOOKAHEAD_MIN_M == 5.0

    def test_lookahead_max(self):
        assert LOOKAHEAD_MAX_M == 25.0


class TestDerivedConstants:
    def test_jerk_alert_is_75pct_of_gate(self):
        assert JERK_ALERT_MPS3 == JERK_P95_GATE_MPS3 * 0.75

    def test_ctrl_jerk_alert_alias(self):
        assert CTRL_JERK_ALERT == JERK_ALERT_MPS3

    def test_lookahead_concern(self):
        assert LOOKAHEAD_CONCERN_M == 8.0


class TestMPCConstants:
    def test_solve_time_budget(self):
        assert MPC_SOLVE_TIME_BUDGET_MS == 5.0

    def test_solve_time_alert(self):
        assert MPC_SOLVE_TIME_ALERT_MS == 8.0

    def test_solve_time_p95_gate(self):
        assert MPC_SOLVE_TIME_P95_MS_GATE == 5.0

    def test_solve_time_marginal(self):
        assert MPC_SOLVE_TIME_MARGINAL_MS == 3.0

    def test_infeasibility_rate_gate(self):
        assert MPC_INFEASIBILITY_RATE_GATE == 0.005

    def test_fallback_rate_gate(self):
        assert MPC_FALLBACK_RATE_GATE == 0.005

    def test_regime_chatter_per_min(self):
        assert MPC_REGIME_CHATTER_PER_MIN == 6.0

    def test_steering_osc_rate_gate(self):
        assert MPC_STEERING_OSC_RATE_GATE == 0.30

    def test_heading_error_p95_rad(self):
        assert MPC_HEADING_ERROR_P95_RAD == 0.25

    def test_min_speed_default(self):
        assert MPC_MIN_SPEED_DEFAULT_MPS == 3.0

    def test_max_curvature_default(self):
        assert MPC_MAX_CURVATURE_DEFAULT == 0.020


class TestBenignStaleReasons:
    def test_is_frozenset(self):
        assert isinstance(BENIGN_STALE_REASONS, frozenset)

    def test_contains_left_lane_low_vis(self):
        assert "left_lane_low_visibility" in BENIGN_STALE_REASONS


class TestComfortGatesDict:
    def test_keys(self):
        expected_keys = {
            "accel_p95_filtered_max",
            "commanded_jerk_p95_max",
            "lateral_p95_max",
            "centered_pct_min",
            "out_of_lane_events_max",
            "emergency_stops_max",
            "steering_jerk_max_max",
        }
        assert set(COMFORT_GATES.keys()) == expected_keys

    def test_values_match_constants(self):
        assert COMFORT_GATES["accel_p95_filtered_max"] == ACCEL_P95_GATE_MPS2
        assert COMFORT_GATES["commanded_jerk_p95_max"] == JERK_P95_GATE_MPS3
        assert COMFORT_GATES["lateral_p95_max"] == LATERAL_P95_GATE_M
        assert COMFORT_GATES["centered_pct_min"] == CENTERED_PCT_MIN
        assert COMFORT_GATES["out_of_lane_events_max"] == 0
        assert COMFORT_GATES["emergency_stops_max"] == 0
        assert COMFORT_GATES["steering_jerk_max_max"] == STEERING_JERK_GATE


class TestGradeConstants:
    def test_grade_max_safe_pct(self):
        assert GRADE_MAX_SAFE_PCT == 10.0

    def test_grade_ema_alpha(self):
        assert GRADE_EMA_ALPHA == 0.3

    def test_grade_clamp_rad(self):
        assert GRADE_CLAMP_RAD == 0.15

    def test_downhill_speed_margin(self):
        assert DOWNHILL_SPEED_MARGIN_MPS == 1.0

    def test_grade_ff_gain_default(self):
        assert GRADE_FF_GAIN_DEFAULT == 1.8

    def test_grade_ff_gain_unity(self):
        assert GRADE_FF_GAIN_UNITY == 1.8

    def test_grade_throttle_budget_min_ratio(self):
        from scoring_registry import GRADE_THROTTLE_BUDGET_MIN_RATIO
        assert GRADE_THROTTLE_BUDGET_MIN_RATIO == 0.5

    def test_grade_throttle_saturation_rate(self):
        from scoring_registry import GRADE_THROTTLE_SATURATION_RATE
        assert GRADE_THROTTLE_SATURATION_RATE == 0.10

    def test_grade_jerk_relaxation_gain(self):
        from scoring_registry import GRADE_JERK_RELAXATION_GAIN
        assert GRADE_JERK_RELAXATION_GAIN == 2.0

    def test_grade_steering_damping_gain(self):
        from scoring_registry import GRADE_STEERING_DAMPING_GAIN
        assert GRADE_STEERING_DAMPING_GAIN == 5.0

    def test_grade_transition_blend_m(self):
        from scoring_registry import GRADE_TRANSITION_BLEND_M
        assert GRADE_TRANSITION_BLEND_M == 3.0
