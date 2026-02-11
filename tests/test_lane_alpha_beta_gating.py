import pytest

from av_stack import apply_lane_alpha_beta_gating


def test_alpha_beta_updates_and_tracks_velocity():
    lane_center = 0.0
    lane_width = 3.6
    state_center = None
    state_center_vel = None
    state_width = None
    state_width_vel = None

    # First update initializes state.
    (
        lane_center_out,
        lane_width_out,
        state_center,
        state_center_vel,
        state_width,
        state_width_vel,
        events,
        gated,
    ) = apply_lane_alpha_beta_gating(
        lane_center,
        lane_width,
        state_center,
        state_center_vel,
        state_width,
        state_width_vel,
        center_gate_m=0.5,
        width_gate_m=0.5,
        center_alpha=0.5,
        center_beta=0.1,
        width_alpha=0.5,
        width_beta=0.1,
        dt=0.1,
    )

    assert gated is False
    assert events == []
    assert lane_center_out == pytest.approx(0.0)
    assert lane_width_out == pytest.approx(3.6)
    assert state_center_vel == pytest.approx(0.0)

    # Move center right by 0.2m; expect velocity to increase.
    (
        lane_center_out,
        lane_width_out,
        state_center,
        state_center_vel,
        state_width,
        state_width_vel,
        events,
        gated,
    ) = apply_lane_alpha_beta_gating(
        lane_center=0.2,
        lane_width=3.6,
        state_center=state_center,
        state_center_vel=state_center_vel,
        state_width=state_width,
        state_width_vel=state_width_vel,
        center_gate_m=0.5,
        width_gate_m=0.5,
        center_alpha=0.5,
        center_beta=0.1,
        width_alpha=0.5,
        width_beta=0.1,
        dt=0.1,
    )

    assert gated is False
    assert events == []
    assert lane_center_out > 0.0
    assert state_center_vel > 0.0


def test_alpha_beta_gates_outlier():
    (
        lane_center_out,
        lane_width_out,
        state_center,
        state_center_vel,
        state_width,
        state_width_vel,
        events,
        gated,
    ) = apply_lane_alpha_beta_gating(
        lane_center=0.0,
        lane_width=3.6,
        state_center=0.0,
        state_center_vel=0.0,
        state_width=3.6,
        state_width_vel=0.0,
        center_gate_m=0.1,
        width_gate_m=0.1,
        center_alpha=0.5,
        center_beta=0.1,
        width_alpha=0.5,
        width_beta=0.1,
        dt=0.1,
    )

    assert gated is False
    assert events == []

    (
        lane_center_out,
        lane_width_out,
        state_center,
        state_center_vel,
        state_width,
        state_width_vel,
        events,
        gated,
    ) = apply_lane_alpha_beta_gating(
        lane_center=1.0,
        lane_width=6.0,
        state_center=state_center,
        state_center_vel=state_center_vel,
        state_width=state_width,
        state_width_vel=state_width_vel,
        center_gate_m=0.1,
        width_gate_m=0.1,
        center_alpha=0.5,
        center_beta=0.1,
        width_alpha=0.5,
        width_beta=0.1,
        dt=0.1,
    )

    assert gated is True
    assert "lane_center_gate_reject" in events
    assert "lane_width_gate_reject" in events
