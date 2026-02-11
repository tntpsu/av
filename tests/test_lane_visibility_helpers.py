from av_stack import is_lane_low_visibility


def test_lane_low_visibility_empty() -> None:
    assert is_lane_low_visibility([], 640, "right") is True


def test_lane_low_visibility_right_edge() -> None:
    points = [[630.0, 200.0]] * 8
    assert is_lane_low_visibility(points, 640, "right") is True


def test_lane_low_visibility_left_edge() -> None:
    points = [[5.0, 200.0]] * 8
    assert is_lane_low_visibility(points, 640, "left") is True


def test_lane_low_visibility_ok() -> None:
    points = [[320.0, 200.0]] * 8
    assert is_lane_low_visibility(points, 640, "right") is False
