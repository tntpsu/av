from av_stack import clamp_lane_center_and_width


def test_clamp_lane_center_and_width_no_change() -> None:
    center, width, clamped = clamp_lane_center_and_width(
        current_center=0.5,
        current_width=7.0,
        previous_center=0.4,
        previous_width=7.1,
        max_center_delta=0.5,
        max_width_delta=1.0,
    )
    assert center == 0.5
    assert width == 7.0
    assert not clamped


def test_clamp_lane_center_and_width_center_only() -> None:
    center, width, clamped = clamp_lane_center_and_width(
        current_center=2.0,
        current_width=7.0,
        previous_center=0.0,
        previous_width=7.0,
        max_center_delta=0.4,
        max_width_delta=1.0,
    )
    assert center == 0.4
    assert width == 7.0
    assert clamped


def test_clamp_lane_center_and_width_width_only() -> None:
    center, width, clamped = clamp_lane_center_and_width(
        current_center=0.0,
        current_width=9.0,
        previous_center=0.0,
        previous_width=7.0,
        max_center_delta=0.4,
        max_width_delta=0.5,
    )
    assert center == 0.0
    assert width == 7.5
    assert clamped
