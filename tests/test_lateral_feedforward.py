"""
Tests for lateral controller feedforward curvature handling.
"""

import numpy as np

from control.pid_controller import LateralController


def _run_controller(controller: LateralController, reference_point: dict, steps: int = 1) -> dict:
    result = None
    for _ in range(steps):
        result = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point,
            vehicle_position=np.array([0.0, 0.0]),
            return_metadata=True,
        )
    return result


def test_feedforward_curvature_clamp_and_sign():
    controller = LateralController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        max_steering=1.0,
        steering_smoothing_alpha=1.0,
        curve_feedforward_gain=1.0,
        curve_feedforward_threshold=0.0,
        curve_feedforward_gain_min=1.0,
        curve_feedforward_gain_max=1.0,
        curve_feedforward_curvature_min=0.005,
        curve_feedforward_curvature_max=0.03,
        curve_feedforward_curvature_clamp=0.02,
        straight_curvature_threshold=0.01,
    )

    reference_point = {
        "x": 0.0,
        "y": 10.0,
        "heading": -0.1,
        "velocity": 8.0,
        "curvature": 0.05,
    }

    result = _run_controller(controller, reference_point, steps=1)
    assert np.isclose(result["path_curvature_input"], -0.02)


def test_feedforward_gain_schedule():
    controller = LateralController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        max_steering=1.0,
        steering_smoothing_alpha=1.0,
        curve_feedforward_gain=1.0,
        curve_feedforward_threshold=0.0,
        curve_feedforward_gain_min=0.8,
        curve_feedforward_gain_max=1.2,
        curve_feedforward_curvature_min=0.01,
        curve_feedforward_curvature_max=0.03,
        curve_feedforward_curvature_clamp=0.05,
        straight_curvature_threshold=0.01,
    )

    reference_point = {
        "x": 0.0,
        "y": 10.0,
        "heading": 0.1,
        "velocity": 8.0,
        "curvature": 0.02,
    }

    result = _run_controller(controller, reference_point, steps=1)
    assert np.isclose(result["curve_feedforward_gain_scheduled"], 1.0)
