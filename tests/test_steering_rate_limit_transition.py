import numpy as np

from control.pid_controller import LateralController


def _ref(x: float, *, curvature: float = 0.0) -> dict:
    return {"x": x, "y": 10.0, "heading": 0.0, "curvature": curvature}


def test_steering_rate_limit_transition_telemetry_and_smoothing() -> None:
    ctrl = LateralController(
        steering_rate_limit_transition_enabled=True,
        steering_rate_limit_transition_alpha=0.8,
        steering_rate_limit_transition_hysteresis=0.0,
    )
    m0 = ctrl.compute_steering(
        current_heading=0.0,
        reference_point=_ref(0.02),
        vehicle_position=np.array([0.0, 0.0]),
        current_speed=8.0,
        dt=0.05,
        return_metadata=True,
    )
    m1 = ctrl.compute_steering(
        current_heading=0.0,
        reference_point=_ref(1.2, curvature=0.02),
        vehicle_position=np.array([0.0, 0.0]),
        current_speed=8.0,
        dt=0.05,
        return_metadata=True,
    )

    assert "steering_rate_limit_effective_raw" in m1
    assert "steering_rate_limit_effective_smoothed" in m1
    assert "steering_rate_limit_transition_active" in m1
    assert float(m1["steering_rate_limit_effective_raw"]) != float(
        m0["steering_rate_limit_effective_raw"]
    )
    assert float(m1["steering_rate_limit_effective_smoothed"]) != float(
        m1["steering_rate_limit_effective_raw"]
    )
    assert bool(m1["steering_rate_limit_transition_active"]) is True


def test_steering_rate_limit_transition_disabled_is_passthrough() -> None:
    ctrl = LateralController(
        steering_rate_limit_transition_enabled=False,
        steering_rate_limit_transition_alpha=0.8,
        steering_rate_limit_transition_hysteresis=0.0,
    )
    m = ctrl.compute_steering(
        current_heading=0.0,
        reference_point=_ref(0.8, curvature=0.01),
        vehicle_position=np.array([0.0, 0.0]),
        current_speed=8.0,
        dt=0.05,
        return_metadata=True,
    )
    assert float(m["steering_rate_limit_effective_smoothed"]) == float(
        m["steering_rate_limit_effective_raw"]
    )
    assert bool(m["steering_rate_limit_transition_active"]) is False
