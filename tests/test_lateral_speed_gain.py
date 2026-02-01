import pytest

from control.pid_controller import LateralController


def test_speed_gain_applied_on_low_curvature_high_speed() -> None:
    controller = LateralController(
        curve_feedforward_curvature_min=0.001,
        curve_feedforward_curvature_max=0.02,
        curve_feedforward_threshold=0.0,
        speed_gain_min_speed=4.0,
        speed_gain_max_speed=10.0,
        speed_gain_min=1.0,
        speed_gain_max=1.5,
        speed_gain_curvature_min=0.002,
        speed_gain_curvature_max=0.02,
        control_mode="pid",
    )

    reference_point = {
        "x": 0.5,
        "y": 10.0,
        "heading": 0.0,
        "curvature": 0.003,
    }

    result = controller.compute_steering(
        0.0,
        reference_point,
        current_speed=10.0,
        return_metadata=True,
    )

    expected_scale = 1.0 - ((0.003 - 0.002) / (0.02 - 0.002))
    expected_gain = 1.0 + (1.5 - 1.0) * expected_scale
    assert result["speed_gain_scale"] == pytest.approx(expected_scale, rel=1e-3)
    assert result["speed_gain_final"] == pytest.approx(expected_gain, rel=1e-3)
    assert result["speed_gain_applied"]


def test_speed_gain_disabled_on_high_curvature() -> None:
    controller = LateralController(
        curve_feedforward_curvature_min=0.001,
        curve_feedforward_curvature_max=0.02,
        curve_feedforward_threshold=0.0,
        speed_gain_min_speed=4.0,
        speed_gain_max_speed=10.0,
        speed_gain_min=1.0,
        speed_gain_max=1.5,
        speed_gain_curvature_min=0.002,
        speed_gain_curvature_max=0.02,
        control_mode="pid",
    )

    reference_point = {
        "x": 0.5,
        "y": 10.0,
        "heading": 0.0,
        "curvature": 0.03,
    }

    result = controller.compute_steering(
        0.0,
        reference_point,
        current_speed=10.0,
        return_metadata=True,
    )

    assert result["speed_gain_scale"] == pytest.approx(0.0, rel=1e-3)
    assert result["speed_gain_final"] == pytest.approx(1.0, rel=1e-3)
    assert result["speed_gain_applied"] is False
