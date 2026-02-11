from av_stack import _apply_curve_speed_preview


def test_curve_speed_preview_noop_when_disabled_inputs():
    speed, limit = _apply_curve_speed_preview(8.0, 0.0, 1.2, 3.0, 12.0, 2.0)
    assert speed == 8.0
    assert limit is None


def test_curve_speed_preview_caps_speed():
    # Curvature -> curve speed ~ sqrt(1.2/0.02) ≈ 7.746
    speed, limit = _apply_curve_speed_preview(12.0, 0.02, 1.2, 3.0, 2.0, 1.0)
    assert limit is not None
    assert speed < 12.0
    assert speed <= 12.0


def test_curve_speed_preview_respects_min_curve_speed():
    speed, limit = _apply_curve_speed_preview(9.0, 0.05, 1.2, 4.0, 2.0, 1.0)
    assert limit is not None
    assert limit >= 4.0
    assert speed <= 9.0
