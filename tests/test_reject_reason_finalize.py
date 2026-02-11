from av_stack import finalize_reject_reason


def test_finalize_reject_reason_prefers_reject():
    assert finalize_reject_reason("bad_width", "missing_lane_positions", ["lane_center_width_gate"]) == "bad_width"


def test_finalize_reject_reason_falls_back_to_stale_reason():
    assert finalize_reject_reason(None, "missing_lane_positions", ["lane_center_width_gate"]) == "missing_lane_positions"


def test_finalize_reject_reason_falls_back_to_clamp_events():
    assert finalize_reject_reason(None, None, ["lane_center_width_gate"]) == "lane_center_width_gate"


def test_finalize_reject_reason_none_when_empty():
    assert finalize_reject_reason(None, None, []) is None
