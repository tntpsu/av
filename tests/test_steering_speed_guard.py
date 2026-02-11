from av_stack import _apply_steering_speed_guard


def test_speed_guard_inactive_without_steering():
    speed, active = _apply_steering_speed_guard(8.0, None, 0.5, 0.7, 3.0)
    assert speed == 8.0
    assert active is False


def test_speed_guard_inactive_below_threshold():
    speed, active = _apply_steering_speed_guard(8.0, 0.4, 0.5, 0.7, 3.0)
    assert speed == 8.0
    assert active is False


def test_speed_guard_caps_speed():
    speed, active = _apply_steering_speed_guard(8.0, 0.7, 0.5, 0.7, 3.0)
    assert speed == 5.6
    assert active is True


def test_speed_guard_respects_min_speed():
    speed, active = _apply_steering_speed_guard(4.0, 0.7, 0.5, 0.4, 3.0)
    assert speed == 3.0
    assert active is True
