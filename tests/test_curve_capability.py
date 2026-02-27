import math

from control.curve_capability import compute_turn_feasibility, curve_intent_state_weight


def test_compute_turn_feasibility_active_values() -> None:
    result = compute_turn_feasibility(
        speed_mps=8.0,
        curvature_abs=0.01,
        comfort_limit_g=0.20,
        peak_limit_g=0.26,
        use_peak_bound=True,
        guardband_g=0.015,
        curvature_min=0.002,
        active_when=True,
    )
    assert result.active is True
    assert result.required_lat_accel_g > 0.0
    assert result.selected_limit_g == 0.26
    assert result.speed_limit_mps > 0.0
    assert math.isclose(result.speed_delta_mps, max(0.0, 8.0 - result.speed_limit_mps), rel_tol=1e-6)


def test_compute_turn_feasibility_inactive_below_curvature_min() -> None:
    result = compute_turn_feasibility(
        speed_mps=8.0,
        curvature_abs=0.0001,
        comfort_limit_g=0.20,
        peak_limit_g=0.26,
        curvature_min=0.002,
        active_when=True,
    )
    assert result.active is False
    assert result.speed_limit_mps == 0.0
    assert result.speed_delta_mps == 0.0


def test_curve_intent_state_weight_map() -> None:
    assert curve_intent_state_weight("STRAIGHT") == 0.0
    assert curve_intent_state_weight("ENTRY") > 0.5
    assert curve_intent_state_weight("COMMIT") == 1.0
