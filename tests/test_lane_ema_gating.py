from av_stack import apply_lane_ema_gating


def test_lane_ema_initializes_when_missing():
    center, width, ema_center, ema_width, events, gated = apply_lane_ema_gating(
        lane_center=0.5,
        lane_width=7.0,
        lane_center_ema=None,
        lane_width_ema=None,
        center_gate_m=0.5,
        width_gate_m=0.5,
        center_alpha=0.5,
        width_alpha=0.5,
    )
    assert center == 0.5
    assert width == 7.0
    assert ema_center == 0.5
    assert ema_width == 7.0
    assert events == []
    assert gated is False


def test_lane_ema_gates_large_jump():
    center, width, ema_center, ema_width, events, gated = apply_lane_ema_gating(
        lane_center=2.0,
        lane_width=9.0,
        lane_center_ema=0.0,
        lane_width_ema=7.0,
        center_gate_m=0.5,
        width_gate_m=0.5,
        center_alpha=0.5,
        width_alpha=0.5,
    )
    assert gated is True
    assert "lane_center_gate_reject" in events
    assert "lane_width_gate_reject" in events
    assert center == 0.0
    assert width == 7.0
    assert ema_center == 0.0
    assert ema_width == 7.0


def test_lane_ema_updates_on_small_change():
    center, width, ema_center, ema_width, events, gated = apply_lane_ema_gating(
        lane_center=0.4,
        lane_width=7.2,
        lane_center_ema=0.2,
        lane_width_ema=7.0,
        center_gate_m=1.0,
        width_gate_m=1.0,
        center_alpha=0.5,
        width_alpha=0.5,
    )
    assert gated is False
    assert events == []
    assert abs(center - 0.3) < 1e-6
    assert abs(width - 7.1) < 1e-6
    assert abs(ema_center - 0.3) < 1e-6
    assert abs(ema_width - 7.1) < 1e-6
