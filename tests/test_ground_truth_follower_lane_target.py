import math

from tools.ground_truth_follower import (
    compute_target_lane_center,
    resolve_lateral_correction_gain,
)


def test_compute_target_lane_center_right_left_center():
    left = -3.6
    right = 3.6
    fallback = 0.0

    assert math.isclose(
        compute_target_lane_center(left, right, fallback, "right"),
        1.8,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        compute_target_lane_center(left, right, fallback, "left"),
        -1.8,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        compute_target_lane_center(left, right, fallback, "center"),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_compute_target_lane_center_single_lane_width():
    left = 0.0
    right = 3.6
    fallback = 0.0
    assert math.isclose(
        compute_target_lane_center(left, right, fallback, "right"),
        1.8,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        compute_target_lane_center(left, right, fallback, "left"),
        1.8,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_compute_target_lane_center_fallback():
    assert compute_target_lane_center(None, 3.6, 0.5, "right") == 0.5
    assert compute_target_lane_center(-3.6, None, -0.25, "left") == -0.25


def test_compute_target_lane_center_invalid_width():
    assert compute_target_lane_center(2.0, 2.0, 0.0, "right") == 0.0


def test_resolve_lateral_correction_gain_defaults():
    assert resolve_lateral_correction_gain("center", None) == 0.2
    assert resolve_lateral_correction_gain("right", None) == 0.8
    assert resolve_lateral_correction_gain("left", None) == 0.8


def test_resolve_lateral_correction_gain_override():
    assert resolve_lateral_correction_gain("right", 0.5) == 0.5
