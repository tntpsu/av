from av_stack import estimate_single_lane_pair


def test_single_lane_fallback_uses_last_width() -> None:
    left, right = estimate_single_lane_pair(
        single_x_vehicle=-2.0,
        is_left_lane=True,
        last_width=6.5,
        default_width=7.0,
        width_min=1.0,
        width_max=10.0,
    )
    assert left == -2.0
    assert right == 4.5


def test_single_lane_fallback_uses_default_when_out_of_bounds() -> None:
    left, right = estimate_single_lane_pair(
        single_x_vehicle=2.0,
        is_left_lane=False,
        last_width=12.0,
        default_width=7.0,
        width_min=1.0,
        width_max=10.0,
    )
    assert left == -5.0
    assert right == 2.0
