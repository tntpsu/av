import math

import av_stack as av_module


def test_slew_limit_value_caps_change() -> None:
    assert av_module._slew_limit_value(10.0, 5.0, max_rate=1.0, dt=1.0) == 9.0
    assert av_module._slew_limit_value(5.0, 10.0, max_rate=2.0, dt=0.5) == 6.0


def test_slew_limit_value_no_rate_or_dt_returns_target() -> None:
    assert av_module._slew_limit_value(10.0, 5.0, max_rate=0.0, dt=1.0) == 5.0
    assert av_module._slew_limit_value(10.0, 5.0, max_rate=1.0, dt=0.0) == 5.0


def test_apply_speed_limit_preview_caps_speed() -> None:
    capped = av_module._apply_speed_limit_preview(
        base_speed=12.0,
        preview_limit=8.0,
        preview_distance=20.0,
        max_decel=1.0,
    )
    expected = math.sqrt((8.0 ** 2) + (2.0 * 1.0 * 20.0))
    assert capped == expected


def test_apply_speed_limit_preview_noop_when_far_from_brake_point() -> None:
    assert av_module._apply_speed_limit_preview(
        base_speed=12.0,
        preview_limit=8.0,
        preview_distance=20.0,
        max_decel=1.0,
    ) == math.sqrt((8.0 ** 2) + (2.0 * 1.0 * 20.0))


def test_apply_speed_limit_preview_noop_when_invalid() -> None:
    assert av_module._apply_speed_limit_preview(10.0, 0.0, 10.0, 1.0) == 10.0
    assert av_module._apply_speed_limit_preview(10.0, 8.0, 0.0, 1.0) == 10.0
    assert av_module._apply_speed_limit_preview(10.0, 8.0, 10.0, 0.0) == 10.0


def test_preview_min_distance_allows_release() -> None:
    assert av_module._preview_min_distance_allows_release(12.0, 12.0, 0.5) is False
    assert av_module._preview_min_distance_allows_release(12.0, 11.4, 0.5) is False
    assert av_module._preview_min_distance_allows_release(12.0, 0.4, 0.5) is True
    assert av_module._preview_min_distance_allows_release(None, 11.0, 0.5) is True


def test_apply_target_speed_slew_asymmetric() -> None:
    assert av_module._apply_target_speed_slew(5.0, 10.0, 2.0, 3.0, 1.0) == 7.0
    assert av_module._apply_target_speed_slew(10.0, 5.0, 2.0, 3.0, 1.0) == 7.0


def test_apply_restart_ramp_limits_target() -> None:
    speed, start, active = av_module._apply_restart_ramp(
        desired_speed=10.0,
        current_speed=0.1,
        ramp_start_time=None,
        timestamp=5.0,
        ramp_seconds=2.0,
        stop_threshold=0.5,
    )
    assert start == 5.0
    assert speed == 0.0
    assert active is True

    speed, start, active = av_module._apply_restart_ramp(
        desired_speed=10.0,
        current_speed=0.1,
        ramp_start_time=start,
        timestamp=6.0,
        ramp_seconds=2.0,
        stop_threshold=0.5,
    )
    assert start == 5.0
    assert speed == 5.0
    assert active is True

    speed, start, active = av_module._apply_restart_ramp(
        desired_speed=10.0,
        current_speed=1.0,
        ramp_start_time=start,
        timestamp=7.0,
        ramp_seconds=2.0,
        stop_threshold=0.5,
    )
    assert start is None
    assert speed == 10.0
    assert active is False
