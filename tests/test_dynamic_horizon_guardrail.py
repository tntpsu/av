import numpy as np

from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner
from trajectory.utils import apply_speed_horizon_guardrail


def test_speed_horizon_guardrail_reduces_speed_when_margin_negative():
    result = apply_speed_horizon_guardrail(
        target_speed_mps=10.0,
        effective_horizon_m=12.0,
        dynamic_horizon_applied=True,
        config={
            "speed_horizon_guardrail_enabled": True,
            "speed_horizon_guardrail_time_headway_s": 1.8,
            "speed_horizon_guardrail_margin_m": 1.0,
            "speed_horizon_guardrail_min_speed_mps": 3.0,
            "speed_horizon_guardrail_gain": 1.0,
        },
    )

    assert result["diag_speed_horizon_guardrail_active"] == 1.0
    assert result["diag_speed_horizon_guardrail_margin_m"] < 0.0
    assert result["diag_speed_horizon_guardrail_target_speed_after_mps"] < 10.0


def test_speed_horizon_guardrail_no_effect_when_disabled():
    result = apply_speed_horizon_guardrail(
        target_speed_mps=10.0,
        effective_horizon_m=12.0,
        dynamic_horizon_applied=True,
        config={"speed_horizon_guardrail_enabled": False},
    )

    assert result["diag_speed_horizon_guardrail_active"] == 0.0
    assert np.isclose(result["diag_speed_horizon_guardrail_target_speed_after_mps"], 10.0)


def test_far_band_cap_reports_limiting_diagnostics():
    planner = RuleBasedTrajectoryPlanner(
        lookahead_distance=20.0,
        point_spacing=1.0,
        image_width=640.0,
        image_height=480.0,
        dynamic_effective_horizon_farfield_scale_min=0.5,
        far_band_contribution_cap_enabled=True,
        far_band_contribution_cap_start_m=0.0,
        far_band_contribution_cap_gain=0.8,
    )

    left_lane = np.array([0.0, 0.0, 280.0], dtype=np.float64)
    right_lane = np.array([0.0, 0.0, 520.0], dtype=np.float64)

    planner.plan(
        [left_lane, right_lane],
        vehicle_state={
            "dynamic_effective_horizon_applied": 1.0,
            "dynamic_effective_horizon_m": 8.0,
        },
    )
    diag = planner.get_last_generation_diagnostics()

    assert diag["diag_far_band_contribution_limited_active"] == 1.0
    assert np.isfinite(diag["diag_far_band_contribution_scale_mean_12_20m"])
    assert np.isfinite(diag["diag_far_band_contribution_limited_frac_12_20m"])
