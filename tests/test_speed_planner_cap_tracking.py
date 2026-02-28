from __future__ import annotations

import pytest

from trajectory.speed_planner import SpeedPlanner, SpeedPlannerConfig


_DEF_DT = 1.0 / 30.0


def _make_planner(**overrides) -> SpeedPlanner:
    cfg_kwargs = dict(
        max_accel=1.2,
        max_decel=2.0,
        max_jerk=0.8,
        default_dt=_DEF_DT,
        cap_tracking_enabled=True,
        cap_tracking_error_on_mps=0.35,
        cap_tracking_error_off_mps=0.12,
        cap_tracking_hold_frames=3,
        cap_tracking_decel_gain=2.2,
        cap_tracking_jerk_gain=1.4,
        cap_tracking_max_decel_mps2=3.5,
        cap_tracking_max_jerk_mps3=3.0,
        cap_tracking_hard_ceiling_epsilon_mps=0.05,
    )
    cfg_kwargs.update(overrides)
    cfg = SpeedPlannerConfig(**cfg_kwargs)
    return SpeedPlanner(cfg)


def test_cap_tracking_enters_mode_and_exits_with_hysteresis() -> None:
    planner = _make_planner()
    t = 0.0
    planner.step(10.0, current_speed=10.0, timestamp=t)

    t += _DEF_DT
    planner.step(
        8.0,
        current_speed=10.0,
        timestamp=t,
        hard_upper_speed=8.0,
        cap_active=True,
        cap_reason="entry",
    )
    assert planner.last_cap_tracking_active is True
    assert planner.last_cap_tracking_mode in {"catch_up", "hold"}

    mode_seen = {planner.last_cap_tracking_mode}
    speed = 10.0
    for _ in range(40):
        t += _DEF_DT
        planned_speed, _, _, _ = planner.step(
            8.0,
            current_speed=speed,
            timestamp=t,
            hard_upper_speed=8.0,
            cap_active=True,
            cap_reason="entry",
        )
        speed = planned_speed
        mode_seen.add(planner.last_cap_tracking_mode)

    assert "catch_up" in mode_seen
    assert "hold" in mode_seen or planner.last_cap_tracking_mode == "inactive"


def test_cap_tracking_hold_frames_behavior() -> None:
    planner = _make_planner(cap_tracking_hold_frames=2)
    t = 0.0
    speed = 9.0
    planner.step(9.0, current_speed=speed, timestamp=t)

    modes = []
    for _ in range(30):
        t += _DEF_DT
        speed, _, _, _ = planner.step(
            7.5,
            current_speed=speed,
            timestamp=t,
            hard_upper_speed=7.5,
            cap_active=True,
            cap_reason="entry",
        )
        modes.append(planner.last_cap_tracking_mode)

    # Hold mode should be reachable with non-zero hold frame config.
    assert "hold" in modes


def test_no_positive_accel_while_cap_tracking_above_cap() -> None:
    planner = _make_planner()
    t = 0.0
    planner.step(10.0, current_speed=10.0, timestamp=t)

    t += _DEF_DT
    _, planned_accel, _, _ = planner.step(
        8.0,
        current_speed=10.0,
        timestamp=t,
        hard_upper_speed=8.0,
        cap_active=True,
        cap_reason="entry",
    )
    assert planned_accel <= 1e-6


def test_planned_speed_respects_cap_ceiling_with_epsilon() -> None:
    planner = _make_planner(cap_tracking_hard_ceiling_epsilon_mps=0.04)
    t = 0.0
    planner.step(10.0, current_speed=10.0, timestamp=t)

    t += _DEF_DT
    planned_speed, _, _, _ = planner.step(
        9.5,
        current_speed=10.0,
        timestamp=t,
        hard_upper_speed=8.0,
        cap_active=True,
        cap_reason="entry",
    )
    assert planned_speed <= 8.040001


def test_disabled_mode_preserves_baseline_when_no_cap_context() -> None:
    cfg = SpeedPlannerConfig(
        max_accel=1.2,
        max_decel=2.0,
        max_jerk=0.8,
        default_dt=_DEF_DT,
        cap_tracking_enabled=False,
    )
    planner = SpeedPlanner(cfg)
    t = 0.0
    s0, a0, j0, active0 = planner.step(8.0, current_speed=6.0, timestamp=t)

    t += _DEF_DT
    s1, a1, j1, active1 = planner.step(8.0, current_speed=s0, timestamp=t)

    planner2 = SpeedPlanner(cfg)
    t = 0.0
    ss0, aa0, jj0, aa_flag0 = planner2.step(8.0, current_speed=6.0, timestamp=t)
    t += _DEF_DT
    ss1, aa1, jj1, aa_flag1 = planner2.step(8.0, current_speed=ss0, timestamp=t)

    assert s0 == pytest.approx(ss0, rel=1e-6)
    assert a0 == pytest.approx(aa0, rel=1e-6)
    assert j0 == pytest.approx(jj0, rel=1e-6)
    assert active0 == aa_flag0
    assert s1 == pytest.approx(ss1, rel=1e-6)
    assert a1 == pytest.approx(aa1, rel=1e-6)
    assert j1 == pytest.approx(jj1, rel=1e-6)
    assert active1 == aa_flag1


def test_accel_not_negative_on_cap_tracking_exit() -> None:
    """After cap-tracking fires with large overspeed (hard ceiling), cap exits next
    frame; planned_accel must be non-negative (accel memory clamped on exit).

    Without the fix, the hard ceiling on frame 1 stores _last_accel ≈ -100 m/s²;
    on frame 2 the jerk limit can only nudge it by ~0.03 m/s² per frame, so the
    planner keeps decelerating for ~30 s after the cap lifts.
    With the fix, _last_accel is clamped to 0 on cap exit and recovery is immediate.
    """
    planner = _make_planner()
    t = 0.0
    # Prime at high speed.
    planner.step(12.0, current_speed=12.0, timestamp=t)

    # Single cap-active frame: 12 m/s vs cap 8.5 m/s triggers hard ceiling on
    # planned_speed, storing a very negative _last_accel.
    t += _DEF_DT
    planner.step(
        12.0,
        current_speed=12.0,
        timestamp=t,
        hard_upper_speed=8.5,
        cap_active=True,
        cap_reason="entry",
    )
    assert planner.last_cap_tracking_mode in {"catch_up", "hold"}

    # Cap lifts on the very next frame (short curve / quick exit).
    t += _DEF_DT
    _, planned_accel, _, _ = planner.step(
        12.0,
        current_speed=8.55,
        timestamp=t,
        hard_upper_speed=None,
        cap_active=False,
    )
    assert planned_accel >= 0.0, (
        f"planned_accel={planned_accel:.3f}: negative accel from catch_up should be "
        "clamped to 0 when cap-tracking exits and speed recovery is needed"
    )


def test_accel_not_negative_on_subthreshold_cap_exit() -> None:
    """Sub-threshold cap event: vehicle only slightly above cap (error < activation
    threshold) so cap_tracking mode never enters catch_up.  The governor's pre-planner
    min(target, cap_speed) still drives the planner to decelerate, accumulating a
    negative _last_accel.  When cap_active flips False the accel memory must be
    clamped to 0 so the car doesn't keep decelerating after the cap lifts.

    Mirrors the S-loop scenario: vehicle at 3.076 m/s, cap at 3.000 m/s (error=0.076,
    below 0.35 threshold).  Cap fires for one frame then lifts; without the fix the
    planner decelerates for ~40 frames, causing dip-to-zero.
    """
    planner = _make_planner()
    t = 0.0
    # Prime: vehicle and planner both at 3.1 m/s
    planner.step(12.0, current_speed=3.1, timestamp=t)

    # Sub-threshold cap: vehicle at 3.1, cap at 3.0 (error=0.1 < 0.35 threshold)
    # cap_tracking mode stays inactive; but governor clamps desired to ~3.0
    t += _DEF_DT
    planner.step(
        3.0,  # governor already clamped desired to cap
        current_speed=3.1,
        timestamp=t,
        hard_upper_speed=3.0,
        cap_active=True,
        cap_reason="subthreshold",
    )
    assert planner.last_cap_tracking_mode == "inactive", (
        "sub-threshold event should NOT enter catch_up"
    )

    # Cap lifts; desired returns to full target
    t += _DEF_DT
    _, planned_accel, _, _ = planner.step(
        12.0,
        current_speed=3.05,
        timestamp=t,
        hard_upper_speed=None,
        cap_active=False,
    )
    assert planned_accel >= 0.0, (
        f"planned_accel={planned_accel:.3f}: negative accel from sub-threshold cap should "
        "be clamped to 0 on cap_active True→False transition when acceleration is needed"
    )


def test_hard_ceiling_preserves_pre_clip_accel() -> None:
    """Hard ceiling fires while in catch_up (large overspeed): the ceiling is
    an external position clamp, not a commanded decel.  Recomputing
    planned_accel as (ceiling - _last_speed)/dt produces -9+ m/s² which traps
    the planner for 100+ frames.  Instead, the pre-clip planned_accel must be
    preserved clamped to ≤ 0, so _last_accel stays near 0 and recovery is
    immediate on the next frame.

    Before the ceiling fires, the jerk-limited planned_accel is small (≥ -max_decel
    from natural path).  After clip it must be ≤ 0 but NOT the raw
    (ceiling - _last_speed)/dt value.
    """
    max_decel_cap = 3.4
    planner = _make_planner(cap_tracking_max_decel_mps2=max_decel_cap)
    t = 0.0
    # Prime: planner and vehicle both at 3.884 m/s
    planner.step(12.0, current_speed=3.884, timestamp=t)

    # catch_up fires: vehicle 0.384 m/s above cap (> 0.35 threshold)
    # hard ceiling will enforce planned_speed ≤ cap + epsilon
    t += _DEF_DT
    planned_speed, planned_accel, _, _ = planner.step(
        12.0,
        current_speed=3.884,
        timestamp=t,
        hard_upper_speed=3.5,
        cap_active=True,
        cap_reason="curve",
    )
    assert planner.last_cap_tracking_mode == "catch_up"
    # planned_accel must be clamped to ≤ 0 (ceiling, don't accelerate past it)
    # but must NOT be the raw (ceiling - _last_speed)/dt ≈ -9+ m/s²
    assert planned_accel <= 0.0, (
        f"planned_accel={planned_accel:.3f}: must not accelerate past ceiling"
    )
    assert planned_accel >= -(max_decel_cap + 1e-6), (
        f"planned_accel={planned_accel:.3f}: ceiling clip must not store large decel; "
        f"pre-clip jerk-limited accel (≥ -{max_decel_cap}) should be preserved"
    )


def test_reset_gap_clears_stale_cap_tracking_state() -> None:
    planner = _make_planner(reset_gap_seconds=0.3)
    t = 0.0
    planner.step(10.0, current_speed=10.0, timestamp=t)
    t += _DEF_DT
    planner.step(
        8.0,
        current_speed=10.0,
        timestamp=t,
        hard_upper_speed=8.0,
        cap_active=True,
        cap_reason="entry",
    )
    assert planner.last_cap_tracking_mode in {"catch_up", "hold"}

    # Timestamp jump + no active cap should clear mode and avoid stale active flag.
    t += 0.6
    planner.step(
        9.0,
        current_speed=9.0,
        timestamp=t,
        hard_upper_speed=None,
        cap_active=False,
    )
    assert planner.last_cap_tracking_active is False
    assert planner.last_cap_tracking_mode == "inactive"
