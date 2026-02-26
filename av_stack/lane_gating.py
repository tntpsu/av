import numpy as np
from typing import Optional, Tuple


def clamp_lane_center_and_width(
    current_center: float,
    current_width: float,
    previous_center: float,
    previous_width: float,
    max_center_delta: float,
    max_width_delta: float,
) -> Tuple[float, float, bool]:
    """Clamp sudden lane center/width changes to avoid reference point jumps."""
    clamped_center = current_center
    clamped_width = current_width

    if max_center_delta > 0.0:
        center_delta = current_center - previous_center
        if abs(center_delta) > max_center_delta:
            clamped_center = previous_center + np.sign(center_delta) * max_center_delta

    if max_width_delta > 0.0:
        width_delta = current_width - previous_width
        if abs(width_delta) > max_width_delta:
            clamped_width = previous_width + np.sign(width_delta) * max_width_delta

    was_clamped = (clamped_center != current_center) or (clamped_width != current_width)
    return clamped_center, clamped_width, was_clamped


def clamp_lane_line_deltas(
    current_left: float,
    current_right: float,
    previous_left: float,
    previous_right: float,
    max_delta: float,
) -> Tuple[float, float, bool]:
    """Clamp sudden per-lane x changes to reduce perception jitter."""
    if max_delta <= 0.0:
        return current_left, current_right, False
    clamped_left = previous_left + np.clip(current_left - previous_left, -max_delta, max_delta)
    clamped_right = previous_right + np.clip(current_right - previous_right, -max_delta, max_delta)
    was_clamped = (clamped_left != current_left) or (clamped_right != current_right)
    return clamped_left, clamped_right, was_clamped


def apply_lane_ema_gating(
    lane_center: float,
    lane_width: float,
    lane_center_ema: float | None,
    lane_width_ema: float | None,
    center_gate_m: float,
    width_gate_m: float,
    center_alpha: float,
    width_alpha: float,
) -> Tuple[float, float, float, float, list, bool]:
    """Apply EMA smoothing with measurement gating for lane center/width."""
    gate_events: list = []
    gated = False
    center_alpha = min(max(center_alpha, 0.0), 1.0)
    width_alpha = min(max(width_alpha, 0.0), 1.0)

    if lane_center_ema is None or lane_width_ema is None:
        return lane_center, lane_width, lane_center, lane_width, gate_events, gated

    center_delta = abs(lane_center - lane_center_ema)
    width_delta = abs(lane_width - lane_width_ema)
    if center_gate_m > 0.0 and center_delta > center_gate_m:
        gate_events.append("lane_center_gate_reject")
        gated = True
    if width_gate_m > 0.0 and width_delta > width_gate_m:
        gate_events.append("lane_width_gate_reject")
        gated = True
    if gated:
        return lane_center_ema, lane_width_ema, lane_center_ema, lane_width_ema, gate_events, gated

    lane_center_ema = (center_alpha * lane_center) + ((1.0 - center_alpha) * lane_center_ema)
    lane_width_ema = (width_alpha * lane_width) + ((1.0 - width_alpha) * lane_width_ema)
    return lane_center_ema, lane_width_ema, lane_center_ema, lane_width_ema, gate_events, gated


def apply_lane_alpha_beta_gating(
    lane_center: float,
    lane_width: float,
    state_center: float | None,
    state_center_vel: float | None,
    state_width: float | None,
    state_width_vel: float | None,
    center_gate_m: float,
    width_gate_m: float,
    center_alpha: float,
    center_beta: float,
    width_alpha: float,
    width_beta: float,
    dt: float,
) -> Tuple[float, float, float, float, float, float, list, bool]:
    """Apply alpha-beta filter with measurement gating for lane center/width."""
    gate_events: list = []
    gated = False
    center_alpha = min(max(center_alpha, 0.0), 1.0)
    width_alpha = min(max(width_alpha, 0.0), 1.0)
    center_beta = max(center_beta, 0.0)
    width_beta = max(width_beta, 0.0)
    dt = max(dt, 1e-3)

    if state_center is None or state_center_vel is None:
        state_center = lane_center
        state_center_vel = 0.0
    if state_width is None or state_width_vel is None:
        state_width = lane_width
        state_width_vel = 0.0

    # Predict
    pred_center = state_center + state_center_vel * dt
    pred_width = state_width + state_width_vel * dt

    center_residual = lane_center - pred_center
    width_residual = lane_width - pred_width

    if center_gate_m > 0.0 and abs(center_residual) > center_gate_m:
        gate_events.append("lane_center_gate_reject")
        gated = True
    if width_gate_m > 0.0 and abs(width_residual) > width_gate_m:
        gate_events.append("lane_width_gate_reject")
        gated = True

    if gated:
        return (
            pred_center,
            pred_width,
            pred_center,
            state_center_vel,
            pred_width,
            state_width_vel,
            gate_events,
            gated,
        )

    # Update
    updated_center = pred_center + center_alpha * center_residual
    updated_center_vel = state_center_vel + (center_beta * center_residual / dt)
    updated_width = pred_width + width_alpha * width_residual
    updated_width_vel = state_width_vel + (width_beta * width_residual / dt)

    return (
        updated_center,
        updated_width,
        updated_center,
        updated_center_vel,
        updated_width,
        updated_width_vel,
        gate_events,
        gated,
    )


def finalize_reject_reason(
    reject_reason: Optional[str],
    stale_data_reason: Optional[str],
    clamp_events: Optional[list],
) -> Optional[str]:
    """Pick a human-readable rejection reason for diagnostics."""
    if reject_reason:
        return reject_reason
    if stale_data_reason:
        return stale_data_reason
    if clamp_events:
        return clamp_events[0]
    return None


def is_lane_low_visibility(
    points: Optional[list],
    image_width: float,
    side: str,
    min_points: int = 6,
    edge_margin: int = 12,
) -> bool:
    """Return True when lane points are too few or hugging the image edge."""
    if not points or len(points) < min_points:
        return True
    xs = [p[0] for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs:
        return True
    if side == "left":
        return min(xs) < edge_margin
    return max(xs) > (image_width - edge_margin)


def is_lane_low_visibility_at_lookahead(
    points: Optional[list],
    image_width: float,
    side: str,
    y_lookahead: float,
    min_points_in_band: int = 4,
    y_band_half_height: float = 18.0,
    edge_margin: int = 12,
) -> bool:
    """
    Return True when lane support is weak specifically around the lookahead row.

    This catches dashed-line gaps at lookahead that global fit-point counts can miss.
    """
    if points is None:
        return True
    valid = [p for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not valid:
        return True
    band_points = [p for p in valid if abs(float(p[1]) - float(y_lookahead)) <= y_band_half_height]
    if len(band_points) < min_points_in_band:
        return True
    xs = [float(p[0]) for p in band_points]
    if side == "left":
        return min(xs) < edge_margin
    return max(xs) > (image_width - edge_margin)


def estimate_single_lane_pair(
    single_x_vehicle: float,
    is_left_lane: bool,
    last_width: Optional[float],
    default_width: float,
    width_min: float,
    width_max: float,
) -> Tuple[float, float]:
    """Estimate missing lane boundary using last known width when available."""
    width = default_width
    if last_width is not None and width_min <= last_width <= width_max:
        width = last_width
    if is_left_lane:
        left_lane_line_x = float(single_x_vehicle)
        right_lane_line_x = float(single_x_vehicle + width)
    else:
        left_lane_line_x = float(single_x_vehicle - width)
        right_lane_line_x = float(single_x_vehicle)
    return left_lane_line_x, right_lane_line_x


def blend_lane_pair_with_previous(
    estimated_left_lane_x: float,
    estimated_right_lane_x: float,
    previous_left_lane_x: float,
    previous_right_lane_x: float,
    blend_alpha: float,
    center_shift_cap_m: float,
) -> Tuple[float, float]:
    """
    Blend fallback-estimated lanes with previous lanes and cap center shift.

    This keeps fallback bounded when dashed-line gaps are transient so
    synthetic lane reconstruction does not cause large centerline jumps.
    """
    alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    left_lane_x = (1.0 - alpha) * float(previous_left_lane_x) + alpha * float(
        estimated_left_lane_x
    )
    right_lane_x = (1.0 - alpha) * float(previous_right_lane_x) + alpha * float(
        estimated_right_lane_x
    )
    prev_center = 0.5 * (float(previous_left_lane_x) + float(previous_right_lane_x))
    center = 0.5 * (left_lane_x + right_lane_x)
    center_shift = center - prev_center
    cap = max(0.0, float(center_shift_cap_m))
    if cap > 0.0 and abs(center_shift) > cap:
        correction = np.sign(center_shift) * (abs(center_shift) - cap)
        left_lane_x -= correction
        right_lane_x -= correction
    return float(left_lane_x), float(right_lane_x)
