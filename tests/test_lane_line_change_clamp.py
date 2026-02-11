from av_stack import clamp_lane_line_deltas


def test_clamp_lane_line_deltas_no_clamp() -> None:
    left, right, clamped = clamp_lane_line_deltas(
        current_left=-1.2,
        current_right=1.2,
        previous_left=-1.1,
        previous_right=1.1,
        max_delta=0.5,
    )
    assert clamped is False
    assert left == -1.2
    assert right == 1.2


def test_clamp_lane_line_deltas_clamps_both_sides() -> None:
    left, right, clamped = clamp_lane_line_deltas(
        current_left=-2.0,
        current_right=2.1,
        previous_left=-1.0,
        previous_right=1.0,
        max_delta=0.3,
    )
    assert clamped is True
    assert left == -1.3
    assert right == 1.3


def test_clamp_lane_line_deltas_disabled() -> None:
    left, right, clamped = clamp_lane_line_deltas(
        current_left=-2.0,
        current_right=2.1,
        previous_left=-1.0,
        previous_right=1.0,
        max_delta=0.0,
    )
    assert clamped is False
    assert left == -2.0
    assert right == 2.1
