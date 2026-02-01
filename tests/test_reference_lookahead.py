import pytest

from control.pid_controller import LateralController
from trajectory.utils import compute_reference_lookahead


def test_reference_lookahead_dynamic_scaling() -> None:
    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_min": 4.0,
        "reference_lookahead_scale_min": 0.7,
        "reference_lookahead_speed_min": 4.0,
        "reference_lookahead_speed_max": 10.0,
        "reference_lookahead_curvature_min": 0.002,
        "reference_lookahead_curvature_max": 0.015,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        config=config,
    )

    assert lookahead == pytest.approx(6.3, rel=1e-3)


def test_reference_lookahead_no_scaling_when_disabled() -> None:
    config = {
        "dynamic_reference_lookahead": False,
    }

    lookahead = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=10.0,
        path_curvature=0.001,
        config=config,
    )

    assert lookahead == pytest.approx(9.0, rel=1e-6)


def test_stanley_mode_uses_heading_and_crosstrack() -> None:
    controller = LateralController(
        control_mode="stanley",
        stanley_k=1.5,
        stanley_soft_speed=2.0,
        stanley_heading_weight=1.0,
    )

    reference_point = {
        "x": 0.5,
        "y": 10.0,
        "heading": 0.1,
        "curvature": 0.0,
    }

    result = controller.compute_steering(
        0.0,
        reference_point,
        current_speed=8.0,
        return_metadata=True,
    )

    assert result["control_mode"] == "stanley"
    assert result["stanley_heading_term"] != 0.0
    assert result["stanley_crosstrack_term"] != 0.0
