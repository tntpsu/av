from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence, Tuple


CURVE_SCHEDULER_MODE_BINARY = "binary"
CURVE_SCHEDULER_MODE_PHASE_SHADOW = "phase_shadow"
CURVE_SCHEDULER_MODE_PHASE_ACTIVE = "phase_active"
_CURVE_SCHEDULER_MODES = {
    CURVE_SCHEDULER_MODE_BINARY,
    CURVE_SCHEDULER_MODE_PHASE_SHADOW,
    CURVE_SCHEDULER_MODE_PHASE_ACTIVE,
}
LOCAL_CURVE_REFERENCE_MODE_OFF = "off"
LOCAL_CURVE_REFERENCE_MODE_SHADOW = "shadow"
LOCAL_CURVE_REFERENCE_MODE_ACTIVE = "active"
LOCAL_CURVE_REFERENCE_MODE_BOUNDED = "bounded"
_LOCAL_CURVE_REFERENCE_MODES = {
    LOCAL_CURVE_REFERENCE_MODE_OFF,
    LOCAL_CURVE_REFERENCE_MODE_SHADOW,
    LOCAL_CURVE_REFERENCE_MODE_ACTIVE,
    LOCAL_CURVE_REFERENCE_MODE_BOUNDED,
}


def _parse_reference_lookahead_speed_table(
    config: Mapping[str, object],
    key: str = "reference_lookahead_speed_table",
) -> list[tuple[float, float]]:
    """Parse optional speed-indexed lookahead table from config."""
    table_raw = config.get(key)
    if not isinstance(table_raw, Sequence) or isinstance(table_raw, (str, bytes)):
        return []

    parsed: list[tuple[float, float]] = []
    for item in table_raw:
        if not isinstance(item, Mapping):
            continue
        lookahead_raw = item.get("lookahead_m")
        if lookahead_raw is None:
            continue
        speed_raw = item.get("speed_mps")
        if speed_raw is None and "speed_mph" in item:
            try:
                speed_raw = float(item.get("speed_mph")) * 0.44704
            except (TypeError, ValueError):
                speed_raw = None
        if speed_raw is None:
            continue
        try:
            speed = float(speed_raw)
            lookahead = float(lookahead_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(speed) or not math.isfinite(lookahead):
            continue
        parsed.append((max(0.0, speed), max(0.1, lookahead)))

    if not parsed:
        return []

    parsed.sort(key=lambda entry: entry[0])
    deduped: list[tuple[float, float]] = []
    for speed, lookahead in parsed:
        if deduped and abs(speed - deduped[-1][0]) < 1e-6:
            deduped[-1] = (speed, lookahead)
        else:
            deduped.append((speed, lookahead))
    return deduped


def _interpolate_reference_lookahead_for_speed(
    speed_mps: float,
    speed_table: Sequence[tuple[float, float]],
) -> float:
    """Linearly interpolate lookahead from a sorted speed table."""
    speed = max(0.0, float(speed_mps))
    if speed <= speed_table[0][0]:
        return speed_table[0][1]
    if speed >= speed_table[-1][0]:
        return speed_table[-1][1]
    for idx in range(1, len(speed_table)):
        left_speed, left_lookahead = speed_table[idx - 1]
        right_speed, right_lookahead = speed_table[idx]
        if speed <= right_speed:
            denom = max(1e-6, right_speed - left_speed)
            ratio = (speed - left_speed) / denom
            return left_lookahead + ratio * (right_lookahead - left_lookahead)
    return speed_table[-1][1]


def _parse_local_curve_reference_distance_table(
    config: Mapping[str, object],
    key: str = "local_curve_reference_target_distance_table",
) -> list[tuple[float, float]]:
    """Parse optional speed-indexed local-arc target distance table."""
    table_raw = config.get(key)
    if not isinstance(table_raw, Sequence) or isinstance(table_raw, (str, bytes)):
        return []

    parsed: list[tuple[float, float]] = []
    for item in table_raw:
        if not isinstance(item, Mapping):
            continue
        speed_raw = item.get("speed_mps")
        if speed_raw is None and "speed_mph" in item:
            try:
                speed_raw = float(item.get("speed_mph")) * 0.44704
            except (TypeError, ValueError):
                speed_raw = None
        distance_raw = item.get("target_distance_m")
        if distance_raw is None:
            distance_raw = item.get("lookahead_m")
        if speed_raw is None or distance_raw is None:
            continue
        try:
            speed = float(speed_raw)
            distance = float(distance_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(speed) or not math.isfinite(distance):
            continue
        parsed.append((max(0.0, speed), max(0.1, distance)))

    if not parsed:
        return []

    parsed.sort(key=lambda entry: entry[0])
    deduped: list[tuple[float, float]] = []
    for speed, distance in parsed:
        if deduped and abs(speed - deduped[-1][0]) < 1e-6:
            deduped[-1] = (speed, distance)
        else:
            deduped.append((speed, distance))
    return deduped


def _smoothstep_01(value: float) -> float:
    """Smoothstep over [0, 1] with clamping."""
    ratio = max(0.0, min(1.0, float(value)))
    return ratio * ratio * (3.0 - 2.0 * ratio)


def _lerp_clamped(start: float, end: float, weight: float) -> float:
    """Clamp weight to [0, 1] and linearly interpolate."""
    ratio = max(0.0, min(1.0, float(weight)))
    return float(start) + ((float(end) - float(start)) * ratio)


def _normalize_curve_anticipation_term(
    value: float,
    value_min: float,
    value_max: float,
) -> float:
    """Normalize a curve-anticipation signal into [0, 1] with smoothstep."""
    low = float(value_min)
    high = float(value_max)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return 0.0
    raw = float(value)
    if not math.isfinite(raw) or raw <= low:
        return 0.0
    if raw >= high:
        return 1.0
    ratio = (raw - low) / max(1e-6, high - low)
    return _smoothstep_01(ratio)


def compute_curve_anticipation_score(
    *,
    path_curvature_abs: float,
    preview_curvature_abs: float,
    heading_delta_abs: float,
    far_geometry_rise_abs: float,
    config: Mapping[str, object],
) -> dict[str, float]:
    """
    Compute map-free curve-anticipation score from trajectory/preview evidence.

    The score is intentionally policy-light and bounded [0, 1]. It is built from
    three normalized terms:
    - `term_curvature`: near-term curvature evidence.
    - `term_heading`: far-vs-base heading divergence.
    - `term_far_rise`: rise in far-horizon geometry magnitude.

    The raw score is the max of terms to avoid suppressing a strong single cue.
    """
    curvature_metric = max(
        abs(float(path_curvature_abs)),
        abs(float(preview_curvature_abs)),
    )
    term_curvature = _normalize_curve_anticipation_term(
        value=curvature_metric,
        value_min=float(config.get("curve_anticipation_curvature_min", 0.0015)),
        value_max=float(config.get("curve_anticipation_curvature_max", 0.010)),
    )
    term_heading = _normalize_curve_anticipation_term(
        value=abs(float(heading_delta_abs)),
        value_min=float(config.get("curve_anticipation_heading_delta_min_rad", 0.10)),
        value_max=float(config.get("curve_anticipation_heading_delta_max_rad", 0.30)),
    )
    term_far_rise = _normalize_curve_anticipation_term(
        value=abs(float(far_geometry_rise_abs)),
        value_min=float(config.get("curve_anticipation_far_rise_min", 0.20)),
        value_max=float(config.get("curve_anticipation_far_rise_max", 0.70)),
    )
    score_raw = max(term_curvature, term_heading, term_far_rise)
    return {
        "term_curvature": float(term_curvature),
        "term_heading": float(term_heading),
        "term_far_rise": float(term_far_rise),
        "score_raw": float(score_raw),
    }


def _compute_curvature_blend_weight(
    abs_curvature: float,
    config: Mapping[str, object],
) -> float:
    """
    Compute smooth blend weight for curve-conditioned lookahead.

    Returns 0.0 for straight-table behavior and 1.0 for curve-table behavior.
    """
    curv_min = float(config.get("reference_lookahead_curve_blend_curvature_min", 0.0))
    curv_max = float(config.get("reference_lookahead_curve_blend_curvature_max", 0.0))
    if curv_max <= curv_min:
        return 0.0
    abs_curv = abs(float(abs_curvature))
    if abs_curv <= curv_min:
        return 0.0
    if abs_curv >= curv_max:
        return 1.0
    ratio = (abs_curv - curv_min) / max(1e-6, curv_max - curv_min)
    return _smoothstep_01(ratio)


def _compute_entry_preview_weight(
    abs_preview_curvature: float,
    config: Mapping[str, object],
) -> float:
    """Entry activation from previewed curvature."""
    curv_min = float(config.get("reference_lookahead_entry_preview_curvature_min", 0.0))
    curv_max = float(config.get("reference_lookahead_entry_preview_curvature_max", 0.0))
    # Optional fallback: if preview min/max are not configured, use hysteresis thresholds as window.
    if curv_max <= curv_min:
        curv_off = float(config.get("reference_lookahead_entry_hysteresis_off", 0.0))
        curv_on = float(config.get("reference_lookahead_entry_hysteresis_on", 0.0))
        if curv_on > curv_off > 0.0:
            curv_min = curv_off
            curv_max = curv_on
    if curv_max <= curv_min:
        return 0.0
    abs_curv = abs(float(abs_preview_curvature))
    if abs_curv <= curv_min:
        return 0.0
    if abs_curv >= curv_max:
        return 1.0
    ratio = (abs_curv - curv_min) / max(1e-6, curv_max - curv_min)
    return _smoothstep_01(ratio)


def _compute_entry_distance_weight(
    distance_to_curve_start_m: float | None,
    config: Mapping[str, object],
) -> float:
    """Entry activation from distance to upcoming curve start."""
    start_m = float(config.get("reference_lookahead_entry_distance_start_m", 0.0))
    end_m = float(config.get("reference_lookahead_entry_distance_end_m", 0.0))
    if start_m <= end_m:
        return 0.0
    if distance_to_curve_start_m is None:
        return 0.0
    distance_m = float(distance_to_curve_start_m)
    if not math.isfinite(distance_m):
        return 0.0
    if distance_m <= end_m:
        return 1.0
    if distance_m >= start_m:
        return 0.0
    ratio = (start_m - distance_m) / max(1e-6, start_m - end_m)
    return _smoothstep_01(ratio)


def _compute_reverse_distance_progress(
    distance_to_curve_start_m: float | None,
    *,
    start_m: float,
    end_m: float,
) -> float:
    """Increase progress as distance-to-curve shrinks from start_m to end_m."""
    start = float(start_m)
    end = float(end_m)
    if not math.isfinite(start) or not math.isfinite(end) or start <= end:
        return 0.0
    if distance_to_curve_start_m is None:
        return 0.0
    try:
        distance_m = float(distance_to_curve_start_m)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(distance_m):
        return 0.0
    if distance_m >= start:
        return 0.0
    if distance_m <= end:
        return 1.0
    ratio = (start - distance_m) / max(1e-6, start - end)
    return _smoothstep_01(ratio)


def build_local_curve_reference(
    *,
    current_speed_mps: float,
    curve_local_state: str,
    curve_local_phase: float,
    curve_local_entry_severity: float,
    distance_to_curve_start_m: float | None,
    current_curve_progress_ratio: float | None,
    road_curvature_abs: float | None,
    preview_curvature_abs: float | None,
    base_reference_point: Mapping[str, float],
    config: Mapping[str, object],
    road_curvature_signed: float | None = None,
    preview_curvature_signed: float | None = None,
) -> dict[str, float | bool | str]:
    """Build a local map-backed curve reference target for shadow/active evaluation."""
    mode_raw = str(config.get("local_curve_reference_mode", LOCAL_CURVE_REFERENCE_MODE_OFF) or "")
    mode = mode_raw.strip().lower()
    if mode not in _LOCAL_CURVE_REFERENCE_MODES:
        mode = LOCAL_CURVE_REFERENCE_MODE_OFF

    base_x = float(base_reference_point.get("x", 0.0) or 0.0)
    base_y = float(base_reference_point.get("y", 0.0) or 0.0)
    base_heading = float(base_reference_point.get("heading", 0.0) or 0.0)

    result: dict[str, float | bool | str] = {
        "local_curve_reference_mode": mode,
        "local_curve_reference_active": False,
        "local_curve_reference_shadow_only": mode == LOCAL_CURVE_REFERENCE_MODE_SHADOW,
        "local_curve_reference_valid": False,
        "local_curve_reference_source": "inactive",
        "local_curve_reference_fallback_active": False,
        "local_curve_reference_fallback_reason": (
            "mode_off" if mode == LOCAL_CURVE_REFERENCE_MODE_OFF else ""
        ),
        "local_curve_reference_blend_weight": 0.0,
        "local_curve_reference_progress_weight": 0.0,
        "local_curve_reference_arc_curvature_abs": 0.0,
        "local_curve_reference_target_x": base_x,
        "local_curve_reference_target_y": base_y,
        "local_curve_reference_target_heading": base_heading,
        "local_curve_reference_target_distance_m": 0.0,
        "local_curve_reference_vs_planner_delta_m": 0.0,
        "local_curve_reference_raw_delta_m": 0.0,
        "local_curve_reference_capped_delta_m": 0.0,
        "local_curve_reference_cap_active": False,
        "local_curve_reference_curve_direction_sign": 0.0,
    }
    if mode == LOCAL_CURVE_REFERENCE_MODE_OFF:
        return result

    state = str(curve_local_state or "").strip().upper()
    if state not in {"ENTRY", "COMMIT"}:
        result["local_curve_reference_fallback_reason"] = "state_inactive"
        return result

    distance_table = _parse_local_curve_reference_distance_table(config)
    if not distance_table:
        result["local_curve_reference_fallback_active"] = True
        result["local_curve_reference_fallback_reason"] = "distance_table_missing"
        return result

    curvature_source = "invalid"
    curvature_abs = None
    selected_signed_value = None
    for source_name, signed_value, abs_value in (
        ("road_curvature", road_curvature_signed, road_curvature_abs),
        ("preview_curvature", preview_curvature_signed, preview_curvature_abs),
        ("reference_curvature", base_reference_point.get("curvature"), None),
    ):
        try:
            if signed_value is not None:
                raw_signed = float(signed_value)
                if math.isfinite(raw_signed) and abs(raw_signed) > 1e-6:
                    curvature_abs = abs(raw_signed)
                    curvature_source = source_name
                    selected_signed_value = raw_signed
                    break
            if abs_value is None:
                continue
            value = abs(float(abs_value))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 1e-6:
            curvature_abs = value
            curvature_source = source_name
            break
    if curvature_abs is None:
        result["local_curve_reference_fallback_active"] = True
        result["local_curve_reference_fallback_reason"] = "curvature_invalid"
        return result

    curvature_min = float(config.get("local_curve_reference_curvature_min", 0.003) or 0.003)
    curvature_max = float(config.get("local_curve_reference_curvature_max", 0.015) or 0.015)
    curvature_cap = float(
        config.get("local_curve_reference_max_abs_curvature", curvature_max) or curvature_max
    )
    curvature_cap = max(curvature_min, curvature_cap)
    curvature_abs = min(curvature_abs, curvature_cap)
    if curvature_abs < max(1e-6, curvature_min * 0.5):
        result["local_curve_reference_fallback_active"] = True
        result["local_curve_reference_fallback_reason"] = "curvature_below_min"
        return result

    signed_metric = 0.0
    if selected_signed_value is not None:
        signed_metric = float(selected_signed_value)
    else:
        for candidate in (
            base_reference_point.get("curvature"),
            base_reference_point.get("heading"),
            base_reference_point.get("x"),
        ):
            try:
                signed_metric = float(candidate or 0.0)
            except (TypeError, ValueError):
                signed_metric = 0.0
            if math.isfinite(signed_metric) and abs(signed_metric) > 1e-6:
                break
    direction_sign = 1.0 if signed_metric >= 0.0 else -1.0

    speed_target_distance = _interpolate_reference_lookahead_for_speed(
        max(0.0, float(current_speed_mps)),
        distance_table,
    )
    severity = 0.0
    try:
        severity = float(curve_local_entry_severity or 0.0)
    except (TypeError, ValueError):
        severity = 0.0
    if not math.isfinite(severity):
        severity = 0.0
    severity = max(0.0, min(1.0, severity))

    curvature_weight = 0.0
    if curvature_max > curvature_min:
        if curvature_abs <= curvature_min:
            curvature_weight = 0.0
        elif curvature_abs >= curvature_max:
            curvature_weight = 1.0
        else:
            curvature_weight = _smoothstep_01(
                (curvature_abs - curvature_min) / max(1e-6, curvature_max - curvature_min)
            )
    tight_scale_min = float(
        config.get("local_curve_reference_tight_distance_scale_min", 0.78) or 0.78
    )
    tight_scale_min = max(0.1, min(1.0, tight_scale_min))
    target_distance = speed_target_distance * _lerp_clamped(1.0, tight_scale_min, max(severity, curvature_weight))

    phase = 0.0
    try:
        phase = float(curve_local_phase or 0.0)
    except (TypeError, ValueError):
        phase = 0.0
    if not math.isfinite(phase):
        phase = 0.0
    phase = max(0.0, min(1.0, phase))

    entry_phase_on = float(config.get("local_curve_reference_entry_phase_on", 0.30) or 0.30)
    entry_phase_full = float(config.get("local_curve_reference_entry_phase_full", 0.65) or 0.65)
    commit_phase_on = float(config.get("local_curve_reference_commit_phase_on", 0.60) or 0.60)
    commit_phase_full = float(config.get("local_curve_reference_commit_phase_full", 0.90) or 0.90)
    unwind_progress_start = float(
        config.get("local_curve_reference_unwind_progress_start", 0.35) or 0.35
    )
    unwind_progress_end = float(
        config.get("local_curve_reference_unwind_progress_end", 0.70) or 0.70
    )

    def _progress_between(value: float, start: float, end: float) -> float:
        if end <= start:
            return 1.0 if value >= start else 0.0
        if value <= start:
            return 0.0
        if value >= end:
            return 1.0
        return _smoothstep_01((value - start) / max(1e-6, end - start))

    entry_progress = _progress_between(phase, entry_phase_on, entry_phase_full)
    commit_progress = _progress_between(phase, commit_phase_on, commit_phase_full)
    progress_weight = entry_progress if state == "ENTRY" else max(entry_progress, commit_progress)

    unwind_weight = 0.0
    if current_curve_progress_ratio is not None:
        try:
            curve_progress = float(current_curve_progress_ratio)
        except (TypeError, ValueError):
            curve_progress = float("nan")
        if math.isfinite(curve_progress):
            curve_progress = max(0.0, min(1.0, curve_progress))
            unwind_weight = _progress_between(
                curve_progress,
                unwind_progress_start,
                unwind_progress_end,
            )

    blend_weight = progress_weight
    if state == "COMMIT":
        blend_weight *= (1.0 - unwind_weight)
    blend_weight = max(0.0, min(1.0, blend_weight))

    signed_curvature = direction_sign * curvature_abs
    theta = curvature_abs * target_distance
    theta = max(0.0, min(theta, math.pi * 0.75))
    if curvature_abs <= 1e-6:
        arc_x = 0.0
        arc_y = target_distance
    else:
        arc_x = direction_sign * ((1.0 - math.cos(theta)) / curvature_abs)
        arc_y = math.sin(theta) / curvature_abs
    arc_heading = direction_sign * theta
    if not (math.isfinite(arc_x) and math.isfinite(arc_y) and math.isfinite(arc_heading)):
        result["local_curve_reference_fallback_active"] = True
        result["local_curve_reference_fallback_reason"] = "arc_invalid"
        return result

    # -- Bounded correction mode: entry-only gating + safety cap --
    raw_delta_m = math.hypot(arc_x - base_x, arc_y - base_y)
    cap_active = False

    if mode == LOCAL_CURVE_REFERENCE_MODE_BOUNDED:
        # 1. Blend ceiling (safety)
        max_blend = float(config.get("local_curve_reference_max_blend", 0.7) or 0.7)
        max_blend = max(0.0, min(1.0, max_blend))
        blend_weight = min(blend_weight, max_blend)

        # 2. Entry-only fade: unwind correction once car enters curve body
        #    During ENTRY: full correction (entry_fade = 1.0)
        #    During COMMIT: fade from 1.0 → 0.0 based on curve_progress_ratio
        bounded_unwind_start = float(
            config.get("local_curve_reference_bounded_unwind_start", 0.0) or 0.0
        )
        bounded_unwind_end = float(
            config.get("local_curve_reference_bounded_unwind_end", 0.25) or 0.25
        )
        if state == "COMMIT" and current_curve_progress_ratio is not None:
            try:
                cpr = float(current_curve_progress_ratio)
            except (TypeError, ValueError):
                cpr = float("nan")
            if math.isfinite(cpr):
                cpr = max(0.0, min(1.0, cpr))
                entry_fade = 1.0 - _progress_between(
                    cpr, bounded_unwind_start, bounded_unwind_end
                )
                blend_weight *= entry_fade

    # Compute blended correction
    dx = blend_weight * (arc_x - base_x)
    dy = blend_weight * (arc_y - base_y)
    dh = blend_weight * (arc_heading - base_heading)
    applied_delta = math.hypot(dx, dy)

    # 3. Safety delta cap (high ceiling — not the primary limiter)
    if mode == LOCAL_CURVE_REFERENCE_MODE_BOUNDED:
        delta_cap_m = float(config.get("local_curve_reference_delta_cap_m", 1.5) or 1.5)
        delta_cap_m = max(0.01, delta_cap_m)
        if applied_delta > delta_cap_m:
            scale = delta_cap_m / applied_delta
            dx *= scale
            dy *= scale
            dh *= scale
            applied_delta = delta_cap_m
            cap_active = True

    target_x = base_x + dx
    target_y = base_y + dy
    target_heading = base_heading + dh
    capped_delta_m = math.hypot(target_x - base_x, target_y - base_y)
    planner_delta = capped_delta_m

    result.update(
        {
            "local_curve_reference_active": blend_weight > 1e-3,
            "local_curve_reference_valid": True,
            "local_curve_reference_source": curvature_source,
            "local_curve_reference_fallback_active": False,
            "local_curve_reference_fallback_reason": "",
            "local_curve_reference_blend_weight": float(blend_weight),
            "local_curve_reference_progress_weight": float(progress_weight),
            "local_curve_reference_arc_curvature_abs": float(curvature_abs),
            "local_curve_reference_target_x": float(target_x),
            "local_curve_reference_target_y": float(target_y),
            "local_curve_reference_target_heading": float(target_heading),
            "local_curve_reference_target_distance_m": float(target_distance),
            "local_curve_reference_vs_planner_delta_m": float(planner_delta),
            "local_curve_reference_raw_delta_m": float(raw_delta_m),
            "local_curve_reference_capped_delta_m": float(capped_delta_m),
            "local_curve_reference_cap_active": bool(cap_active),
            "local_curve_reference_curve_direction_sign": float(direction_sign),
            "local_curve_reference_signed_curvature": float(signed_curvature),
        }
    )
    if distance_to_curve_start_m is not None:
        try:
            distance_value = float(distance_to_curve_start_m)
        except (TypeError, ValueError):
            distance_value = float("nan")
        if math.isfinite(distance_value):
            result["local_curve_reference_distance_to_curve_start_m"] = float(distance_value)
    if current_curve_progress_ratio is not None:
        try:
            progress_value = float(current_curve_progress_ratio)
        except (TypeError, ValueError):
            progress_value = float("nan")
        if math.isfinite(progress_value):
            result["local_curve_reference_curve_progress_ratio"] = float(
                max(0.0, min(1.0, progress_value))
            )
    return result


def _compute_entry_phase_weight(
    preview_curvature: float | None,
    distance_to_curve_start_m: float | None,
    config: Mapping[str, object],
) -> float:
    """
    Entry-phase blend weight from preview curvature and distance-to-curve.

    Weight rises only when both signals indicate a near upcoming curve.
    """
    if preview_curvature is None:
        return 0.0
    preview_weight = _compute_entry_preview_weight(abs(float(preview_curvature)), config)
    if distance_to_curve_start_m is None:
        if bool(config.get("reference_lookahead_entry_preview_only_fallback", False)):
            return preview_weight
        return 0.0
    distance_weight = _compute_entry_distance_weight(distance_to_curve_start_m, config)
    return preview_weight * distance_weight


def _compute_curve_local_entry_severity(
    *,
    preview_curvature_abs: float,
    curvature_rise_abs: float,
    config: Mapping[str, object],
) -> float:
    """
    Compute a bounded local-entry severity signal.

    This remains policy-light: it does not arm the scheduler by itself. It only
    shapes effective local-entry horizon/threshold so tighter upcoming curves can
    arm earlier than gentler ones.
    """
    severity_preview = _normalize_curve_anticipation_term(
        value=abs(float(preview_curvature_abs)),
        value_min=float(
            config.get(
                "curve_local_entry_severity_curvature_min",
                config.get("curve_phase_preview_curvature_min", 0.003),
            )
        ),
        value_max=float(
            config.get(
                "curve_local_entry_severity_curvature_max",
                config.get("curve_phase_preview_curvature_max", 0.015),
            )
        ),
    )
    severity_rise = _normalize_curve_anticipation_term(
        value=max(0.0, float(curvature_rise_abs)),
        value_min=float(
            config.get(
                "curve_local_entry_severity_rise_min",
                config.get("curve_phase_rise_min", 0.0005),
            )
        ),
        value_max=float(
            config.get(
                "curve_local_entry_severity_rise_max",
                config.get("curve_phase_rise_max", 0.006),
            )
        ),
    )
    return float(max(severity_preview, severity_rise))


def _compute_local_curve_relevance_terms(
    *,
    distance_to_curve_start_m: float | None,
    time_to_curve_s: float | None,
    path_curvature_abs: float,
    config: Mapping[str, object],
    distance_start_m: float | None = None,
    distance_end_m: float | None = None,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
) -> dict[str, float]:
    """Return the local gate terms that determine steering relevance."""
    path_weight = _normalize_curve_anticipation_term(
        value=abs(float(path_curvature_abs)),
        value_min=float(
            config.get(
                "curve_local_phase_path_curvature_min",
                config.get("curve_phase_path_curvature_min", 0.003),
            )
        ),
        value_max=float(
            config.get(
                "curve_local_phase_path_curvature_max",
                config.get("curve_phase_path_curvature_max", 0.015),
            )
        ),
    )

    distance_weight = 0.0
    has_distance = (
        distance_to_curve_start_m is not None
        and math.isfinite(float(distance_to_curve_start_m))
    )
    if has_distance:
        start_m = float(
            distance_start_m
            if distance_start_m is not None
            else config.get("curve_local_phase_distance_start_m", 8.0)
        )
        end_m = float(
            distance_end_m
            if distance_end_m is not None
            else config.get("curve_local_phase_distance_end_m", 1.5)
        )
        if start_m > end_m:
            distance_m = float(distance_to_curve_start_m)
            if distance_m <= end_m:
                distance_weight = 1.0
            elif distance_m >= start_m:
                distance_weight = 0.0
            else:
                ratio = (start_m - distance_m) / max(1e-6, start_m - end_m)
                distance_weight = _smoothstep_01(ratio)

    time_weight = 0.0
    has_time = time_to_curve_s is not None and math.isfinite(float(time_to_curve_s))
    if has_time:
        start_s = float(
            time_start_s
            if time_start_s is not None
            else config.get("curve_local_phase_time_start_s", 1.2)
        )
        end_s = float(
            time_end_s
            if time_end_s is not None
            else config.get("curve_local_phase_time_end_s", 0.25)
        )
        if start_s > end_s:
            time_s = max(0.0, float(time_to_curve_s))
            if time_s <= end_s:
                time_weight = 1.0
            elif time_s >= start_s:
                time_weight = 0.0
            else:
                ratio = (start_s - time_s) / max(1e-6, start_s - end_s)
                time_weight = _smoothstep_01(ratio)

    return {
        "path": float(path_weight),
        "distance": float(distance_weight),
        "time": float(time_weight),
    }


def _compute_local_curve_relevance_weight(
    *,
    distance_to_curve_start_m: float | None,
    time_to_curve_s: float | None,
    path_curvature_abs: float,
    config: Mapping[str, object],
    distance_start_m: float | None = None,
    distance_end_m: float | None = None,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
) -> float:
    """
    Compute how locally relevant the upcoming curve is for steering lookahead.

    This is intentionally separate from far preview. Preview may stay active
    around short-straight loops, while local relevance should rise only when the
    curve is close in distance/time or already under the vehicle.
    """
    has_distance = (
        distance_to_curve_start_m is not None
        and math.isfinite(float(distance_to_curve_start_m))
    )
    has_time = time_to_curve_s is not None and math.isfinite(float(time_to_curve_s))
    if not has_distance and not has_time:
        # Compatibility fallback for older paths/recordings with no local-relevance telemetry.
        return 1.0

    terms = _compute_local_curve_relevance_terms(
        distance_to_curve_start_m=distance_to_curve_start_m,
        time_to_curve_s=time_to_curve_s,
        path_curvature_abs=path_curvature_abs,
        config=config,
        distance_start_m=distance_start_m,
        distance_end_m=distance_end_m,
        time_start_s=time_start_s,
        time_end_s=time_end_s,
    )
    return float(max(terms.values()))


def _compute_curve_anticipation_entry_weight(
    *,
    anticipation_score: float | None,
    anticipation_active: bool | None,
    config: Mapping[str, object],
) -> float:
    """
    Convert curve-anticipation score into entry blend weight for single-owner mode.

    This path is active only when curve anticipation is enabled, non-shadow, and
    single-owner mode is explicitly turned on.
    """
    if not bool(config.get("curve_anticipation_enabled", False)):
        return 0.0
    if bool(config.get("curve_anticipation_shadow_only", True)):
        return 0.0
    if not bool(config.get("curve_anticipation_single_owner_mode", False)):
        return 0.0

    score = float(anticipation_score) if anticipation_score is not None else 0.0
    if not math.isfinite(score):
        score = 0.0
    score = max(0.0, min(1.0, score))

    on = float(
        config.get(
            "curve_anticipation_entry_weight_on",
            config.get("curve_anticipation_score_on", 0.55),
        )
    )
    full = float(config.get("curve_anticipation_entry_weight_full", 0.90))
    max_weight = float(config.get("curve_anticipation_entry_weight_max", 1.0))
    min_active = float(config.get("curve_anticipation_entry_weight_min_active", 0.0))
    max_weight = max(0.0, min(1.0, max_weight))
    min_active = max(0.0, min(max_weight, min_active))

    if full <= on:
        weight = max_weight if score >= on else 0.0
    elif score <= on:
        weight = 0.0
    elif score >= full:
        weight = max_weight
    else:
        ratio = (score - on) / max(1e-6, full - on)
        weight = max_weight * _smoothstep_01(ratio)

    if bool(anticipation_active):
        weight = max(weight, min_active)
    return max(0.0, min(max_weight, weight))


def resolve_curve_scheduler_mode(
    *,
    config: Mapping[str, object],
    explicit_mode: str | None = None,
) -> str:
    """Return a valid curve scheduler mode with safe fallback."""
    candidate = explicit_mode if explicit_mode is not None else config.get("curve_scheduler_mode", "binary")
    mode = str(candidate or "binary").strip().lower()
    if mode not in _CURVE_SCHEDULER_MODES:
        return CURVE_SCHEDULER_MODE_BINARY
    return mode


def compute_curve_phase_scheduler(
    *,
    preview_curvature_abs: float,
    path_curvature_abs: float,
    curvature_rise_abs: float,
    distance_to_curve_start_m: float | None = None,
    time_to_curve_s: float | None = None,
    preview_confidence: float | None = None,
    previous_phase: float,
    previous_state: str,
    previous_entry_frames: int,
    previous_rearm_hold_frames: int,
    config: Mapping[str, object],
) -> dict[str, float | str | bool | int]:
    """
    Compute unified curve phase (0..1) with internal ENTRY/COMMIT/REARM state.

    This is the single-owner trajectory scheduler primitive used for migration:
    - `phase_shadow`: telemetry only.
    - `phase_active`: drives lookahead blending.
    """
    term_preview = _normalize_curve_anticipation_term(
        value=abs(float(preview_curvature_abs)),
        value_min=float(config.get("curve_phase_preview_curvature_min", 0.003)),
        value_max=float(config.get("curve_phase_preview_curvature_max", 0.015)),
    )
    term_path = _normalize_curve_anticipation_term(
        value=abs(float(path_curvature_abs)),
        value_min=float(config.get("curve_phase_path_curvature_min", 0.003)),
        value_max=float(config.get("curve_phase_path_curvature_max", 0.015)),
    )
    term_rise = _normalize_curve_anticipation_term(
        value=max(0.0, float(curvature_rise_abs)),
        value_min=float(config.get("curve_phase_rise_min", 0.0005)),
        value_max=float(config.get("curve_phase_rise_max", 0.006)),
    )
    term_time = 0.0
    if time_to_curve_s is not None and math.isfinite(float(time_to_curve_s)):
        time_min = max(0.05, float(config.get("curve_phase_time_to_curve_min_s", 0.7)))
        time_max = max(time_min + 1e-3, float(config.get("curve_phase_time_to_curve_max_s", 3.0)))
        time_val = max(0.0, float(time_to_curve_s))
        term_time = _normalize_curve_anticipation_term(
            value=time_max - time_val,
            value_min=0.0,
            value_max=max(1e-3, time_max - time_min),
        )

    confidence_scale = 1.0
    if preview_confidence is not None and math.isfinite(float(preview_confidence)):
        c_val = max(0.0, min(1.0, float(preview_confidence)))
        c_min = float(config.get("curve_phase_confidence_min", 0.35))
        c_max = float(config.get("curve_phase_confidence_max", 0.80))
        c_floor = max(0.0, min(1.0, float(config.get("curve_phase_confidence_floor_scale", 0.6))))
        if c_max <= c_min:
            confidence_scale = 1.0 if c_val >= c_max else c_floor
        else:
            c_ratio = max(0.0, min(1.0, (c_val - c_min) / (c_max - c_min)))
            confidence_scale = c_floor + ((1.0 - c_floor) * _smoothstep_01(c_ratio))

    entry_severity = _compute_curve_local_entry_severity(
        preview_curvature_abs=preview_curvature_abs,
        curvature_rise_abs=curvature_rise_abs,
        config=config,
    )
    base_distance_start_m = float(config.get("curve_local_phase_distance_start_m", 8.0))
    distance_end_m = float(config.get("curve_local_phase_distance_end_m", 1.5))
    tight_distance_start_m = float(
        config.get("curve_local_phase_distance_start_tight_m", base_distance_start_m)
    )
    effective_distance_start_m = max(
        distance_end_m + 1e-3,
        _lerp_clamped(base_distance_start_m, tight_distance_start_m, entry_severity),
    )
    base_time_start_s = float(config.get("curve_local_phase_time_start_s", 1.2))
    time_end_s = float(config.get("curve_local_phase_time_end_s", 0.25))
    tight_time_start_s = float(
        config.get("curve_local_phase_time_start_tight_s", base_time_start_s)
    )
    effective_time_start_s = max(
        time_end_s + 1e-3,
        _lerp_clamped(base_time_start_s, tight_time_start_s, entry_severity),
    )
    entry_on_base = float(config.get("curve_local_entry_on_base", config.get("curve_phase_on", 0.45)))
    entry_on_tight = float(config.get("curve_local_entry_on_tight", entry_on_base))
    entry_on_effective = _lerp_clamped(entry_on_base, entry_on_tight, entry_severity)

    far_preview_phase = max(term_preview, term_time) * confidence_scale
    local_gate_terms = _compute_local_curve_relevance_terms(
        distance_to_curve_start_m=distance_to_curve_start_m,
        time_to_curve_s=time_to_curve_s,
        path_curvature_abs=path_curvature_abs,
        config=config,
        distance_start_m=effective_distance_start_m,
        distance_end_m=distance_end_m,
        time_start_s=effective_time_start_s,
        time_end_s=time_end_s,
    )
    local_distance_term = float(local_gate_terms.get("distance", 0.0))
    local_time_gate_term = float(local_gate_terms.get("time", 0.0))
    local_path_gate_term = float(local_gate_terms.get("path", 0.0))
    in_curve_distance_eps_m = max(
        0.0, float(config.get("curve_local_in_curve_distance_eps_m", 0.25))
    )
    in_curve_time_eps_s = max(
        0.0, float(config.get("curve_local_in_curve_time_eps_s", 0.05))
    )
    in_curve_path_min = max(
        0.0, min(1.0, float(config.get("curve_local_in_curve_path_min", 0.70)))
    )
    local_time_ready = bool(
        time_to_curve_s is not None
        and math.isfinite(float(time_to_curve_s))
        and local_time_gate_term > 1e-6
    )
    local_distance_gate_ready = bool(local_distance_term > 1e-6)
    local_in_curve_now = False
    if distance_to_curve_start_m is not None and math.isfinite(float(distance_to_curve_start_m)):
        local_in_curve_now = float(distance_to_curve_start_m) <= in_curve_distance_eps_m
    elif time_to_curve_s is not None and math.isfinite(float(time_to_curve_s)):
        local_in_curve_now = max(0.0, float(time_to_curve_s)) <= in_curve_time_eps_s
    else:
        local_in_curve_now = local_path_gate_term >= in_curve_path_min

    local_arm_ready = bool(local_distance_gate_ready or local_time_ready or local_in_curve_now)
    commit_distance_ready_m = max(
        in_curve_distance_eps_m,
        float(config.get("curve_local_commit_distance_ready_m", 3.0)),
    )
    commit_time_ready_s = max(
        in_curve_time_eps_s,
        float(config.get("curve_local_commit_time_ready_s", 0.60)),
    )
    local_commit_distance_ready = bool(
        distance_to_curve_start_m is not None
        and math.isfinite(float(distance_to_curve_start_m))
        and float(distance_to_curve_start_m) <= commit_distance_ready_m
    )
    local_commit_time_ready = bool(
        time_to_curve_s is not None
        and math.isfinite(float(time_to_curve_s))
        and max(0.0, float(time_to_curve_s)) <= commit_time_ready_s
    )
    local_commit_ready = bool(
        local_in_curve_now
        or local_commit_distance_ready
        or local_commit_time_ready
    )
    local_proximity_weight = float(
        max(
            local_distance_term,
            local_time_gate_term,
            1.0 if local_in_curve_now else 0.0,
        )
    )
    local_gate_weight = float(max(0.0, min(1.0, local_proximity_weight)))
    local_preview_term = max(term_preview, term_rise) * local_gate_weight
    local_time_term = 0.0
    if bool(config.get("curve_local_phase_use_time_term", False)):
        local_time_term = term_time * local_gate_weight
    local_arm_phase_raw = max(
        local_preview_term,
        local_time_term,
        term_path if local_in_curve_now else 0.0,
    ) * confidence_scale
    if local_commit_ready:
        # Locally committed (vehicle within commit_distance_ready_m of curve start, or
        # physically in-curve): sustain with all terms, including far preview.
        local_sustain_phase_raw = max(term_path, local_preview_term, local_time_term) * confidence_scale
    else:
        # Not locally committed: only actual road curvature at the vehicle sustains phase.
        # Far preview (term_preview / local_preview_term) is suppressed — it represents
        # awareness of a future curve, not a reason to hold COMMIT/ENTRY on a straight.
        # This prevents the state machine from latching in COMMIT on looped tracks (e.g.
        # s_loop) where the next curve is always visible in the preview window.
        local_sustain_phase_raw = term_path * confidence_scale

    prev_state_norm = str(previous_state or "STRAIGHT").strip().upper()
    if prev_state_norm not in {"STRAIGHT", "ENTRY", "COMMIT", "REARM"}:
        prev_state_norm = "STRAIGHT"
    alpha = float(config.get("curve_phase_ema_alpha", 0.45))
    alpha = max(0.0, min(1.0, alpha))
    prev_phase = float(previous_phase) if math.isfinite(float(previous_phase)) else 0.0
    phase_raw = local_sustain_phase_raw if prev_state_norm in {"ENTRY", "COMMIT"} else local_arm_phase_raw
    phase = (alpha * phase_raw) + ((1.0 - alpha) * prev_phase)

    on = float(entry_on_effective)
    off = float(config.get("curve_phase_off", 0.30))
    if off > on:
        off = on
    commit_on = float(config.get("curve_phase_commit_on", 0.55))
    commit_on = max(on, commit_on)
    commit_min_frames = max(1, int(config.get("curve_phase_commit_min_frames", 4)))
    entry_floor = max(0.0, min(1.0, float(config.get("curve_phase_entry_floor", 0.15))))
    commit_floor = max(entry_floor, min(1.0, float(config.get("curve_phase_commit_floor", 0.30))))
    rearm_hold_cfg = max(0, int(config.get("curve_phase_rearm_hold_frames", 4)))
    reentry_gate_min = max(0.0, min(1.0, float(config.get("curve_local_phase_reentry_gate_min", 0.10))))
    reentry_path_min = max(0.0, min(1.0, float(config.get("curve_local_phase_reentry_path_min", 0.15))))

    entry_frames = max(0, int(previous_entry_frames))
    rearm_hold_frames = max(0, int(previous_rearm_hold_frames))

    state = prev_state_norm
    rearm_event = False
    local_distance_ready = bool(local_commit_ready)
    local_reentry_ready = bool(
        local_arm_ready and (
            local_gate_weight >= reentry_gate_min
            or (local_in_curve_now and term_path >= reentry_path_min)
        )
    )
    if prev_state_norm == "COMMIT":
        if phase < off:
            state = "REARM"
            entry_frames = 0
            rearm_hold_frames = rearm_hold_cfg
        elif not local_commit_ready:
            if local_arm_ready and phase >= on:
                state = "ENTRY"
                entry_frames = max(1, entry_frames)
            else:
                state = "REARM"
                entry_frames = 0
                rearm_hold_frames = rearm_hold_cfg
        else:
            state = "COMMIT"
            entry_frames = max(1, entry_frames)
    elif prev_state_norm == "ENTRY":
        if phase < off or not local_arm_ready:
            state = "REARM"
            entry_frames = 0
            rearm_hold_frames = rearm_hold_cfg
        else:
            entry_frames = max(1, entry_frames + 1)
            if (
                local_commit_ready
                and phase >= commit_on
                and entry_frames >= commit_min_frames
            ):
                state = "COMMIT"
            else:
                state = "ENTRY"
    elif prev_state_norm == "REARM":
        if local_arm_phase_raw >= on and local_reentry_ready:
            state = "ENTRY"
            rearm_event = True
            entry_frames = 1
            rearm_hold_frames = 0
        else:
            rearm_hold_frames = max(0, rearm_hold_frames - 1)
            if rearm_hold_frames == 0 and phase < off:
                state = "STRAIGHT"
                entry_frames = 0
            else:
                state = "REARM"
    else:  # STRAIGHT
        if local_arm_phase_raw >= on and local_reentry_ready:
            state = "ENTRY"
            entry_frames = 1
        else:
            state = "STRAIGHT"
            entry_frames = 0
            rearm_hold_frames = 0

    if state == "ENTRY":
        phase = max(phase, entry_floor)
    elif state == "COMMIT":
        phase = max(phase, commit_floor)

    local_driver_terms = {
        "path": float(term_path),
        "local_preview": float(local_preview_term),
        "local_time": float(local_time_term),
    }
    active_driver_terms = [
        name for name, value in local_driver_terms.items()
        if value > 1e-6 and abs(value - max(local_driver_terms.values())) <= 1e-6
    ]
    if not active_driver_terms:
        local_phase_source = "none"
    elif len(active_driver_terms) == 1:
        local_phase_source = active_driver_terms[0]
    else:
        local_phase_source = "mixed"

    local_entry_driver_terms = {
        "distance": float(local_distance_term),
        "time": float(local_time_gate_term),
        "in_curve": 1.0 if local_in_curve_now else 0.0,
    }
    local_entry_driver_active = [
        name for name, value in local_entry_driver_terms.items()
        if value > 1e-6 and abs(value - max(local_entry_driver_terms.values())) <= 1e-6
    ]
    if not local_entry_driver_active:
        local_entry_driver = "none"
    elif len(local_entry_driver_active) == 1:
        local_entry_driver = local_entry_driver_active[0]
    else:
        local_entry_driver = "mixed"

    local_commit_driver_terms = {
        "distance": 1.0 if local_commit_distance_ready else 0.0,
        "time": 1.0 if local_commit_time_ready else 0.0,
        "in_curve": 1.0 if local_in_curve_now else 0.0,
    }
    local_commit_driver_active = [
        name for name, value in local_commit_driver_terms.items()
        if value > 1e-6 and abs(value - max(local_commit_driver_terms.values())) <= 1e-6
    ]
    if not local_commit_driver_active:
        local_commit_driver = "none"
    elif len(local_commit_driver_active) == 1:
        local_commit_driver = local_commit_driver_active[0]
    else:
        local_commit_driver = "mixed"

    phase = max(0.0, min(1.0, phase))
    return {
        "curve_phase_raw": float(max(0.0, min(1.0, phase_raw))),
        "curve_phase": float(phase),
        "curve_phase_state": state,
        "curve_phase_rearm_event": bool(rearm_event),
        "curve_phase_entry_frames": int(entry_frames),
        "curve_phase_rearm_hold_frames": int(rearm_hold_frames),
        "curve_phase_term_preview": float(term_preview),
        "curve_phase_term_path": float(term_path),
        "curve_phase_term_rise": float(term_rise),
        "curve_phase_term_time": float(term_time),
        "curve_phase_confidence_scale": float(confidence_scale),
        "curve_preview_far_upcoming": bool(
            far_preview_phase >= float(config.get("curve_preview_far_on", 0.10))
        ),
        "curve_preview_far_phase": float(max(0.0, min(1.0, far_preview_phase))),
        "curve_local_entry_severity": float(max(0.0, min(1.0, entry_severity))),
        "curve_local_entry_on_effective": float(max(0.0, min(1.0, entry_on_effective))),
        "curve_local_phase_distance_start_effective_m": float(effective_distance_start_m),
        "curve_local_phase_time_start_effective_s": float(effective_time_start_s),
        "curve_local_entry_driver": str(local_entry_driver),
        "curve_local_gate_weight": float(max(0.0, min(1.0, local_gate_weight))),
        "curve_local_arm_ready": bool(local_arm_ready),
        "curve_local_time_ready": bool(local_time_ready),
        "curve_local_in_curve_now": bool(local_in_curve_now),
        "curve_local_commit_ready": bool(local_commit_ready),
        "curve_local_commit_driver": str(local_commit_driver),
        "curve_local_arm_phase_raw": float(max(0.0, min(1.0, local_arm_phase_raw))),
        "curve_local_sustain_phase_raw": float(max(0.0, min(1.0, local_sustain_phase_raw))),
        "curve_local_path_sustain_active": bool(
            term_path > 1e-6 and term_path >= max(local_preview_term, local_time_term)
        ),
        "curve_local_distance_ready": bool(local_distance_ready),
        "curve_local_distance_horizon_m": float(effective_distance_start_m),
        "curve_local_time_horizon_s": float(effective_time_start_s),
        "curve_local_reentry_ready": bool(local_reentry_ready),
        "curve_local_phase_raw": float(max(0.0, min(1.0, phase_raw))),
        "curve_local_phase": float(phase),
        "curve_local_state": state,
        "curve_local_phase_source": str(local_phase_source),
    }


def _apply_lookahead_slew_limit(
    lookahead_m: float,
    previous_lookahead_m: float | None,
    config: Mapping[str, object],
) -> float:
    """
    Limit frame-to-frame lookahead change for stability.

    The limit is in meters per frame to avoid requiring loop timing assumptions.
    """
    legacy_slew = float(config.get("reference_lookahead_slew_rate_m_per_frame", 0.0))
    shorten_slew = float(config.get("reference_lookahead_slew_rate_shorten_m_per_frame", 0.0))
    lengthen_slew = float(config.get("reference_lookahead_slew_rate_lengthen_m_per_frame", 0.0))
    if previous_lookahead_m is None:
        return float(lookahead_m)
    prev = float(previous_lookahead_m)
    if not math.isfinite(prev):
        return float(lookahead_m)
    delta = float(lookahead_m) - prev
    if shorten_slew > 0.0 or lengthen_slew > 0.0:
        if delta >= 0.0:
            max_delta = lengthen_slew if lengthen_slew > 0.0 else legacy_slew
        else:
            max_delta = shorten_slew if shorten_slew > 0.0 else legacy_slew
    else:
        max_delta = legacy_slew
    if max_delta <= 0.0:
        return float(lookahead_m)
    if delta > max_delta:
        return prev + max_delta
    if delta < -max_delta:
        return prev - max_delta
    return float(lookahead_m)


def _curve_state_rank(state: str | None) -> int:
    state_key = str(state or "STRAIGHT").strip().upper()
    if state_key == "COMMIT":
        return 3
    if state_key == "ENTRY":
        return 2
    if state_key == "REARM":
        return 1
    return 0


def _apply_entry_local_shorten_guard(
    lookahead_m: float,
    previous_lookahead_m: float | None,
    *,
    curve_local_state: str | None,
    local_gate_weight: float | None,
    distance_to_curve_start_m: float | None,
    config: Mapping[str, object],
) -> tuple[float, bool, float]:
    """
    Guard local entry-related lookahead shortening before PP floor rescue.

    This shapes planner-side contraction during pre-entry so the PP-local floor
    remains a backstop instead of a large late correction.
    """
    target = float(lookahead_m)
    guard_slew = float(
        config.get("reference_lookahead_entry_shorten_slew_m_per_frame", 0.0) or 0.0
    )
    if guard_slew <= 0.0:
        return target, False, 0.0

    if previous_lookahead_m is None:
        return target, False, 0.0
    prev = float(previous_lookahead_m)
    if not math.isfinite(prev):
        return target, False, 0.0
    if target >= prev:
        return target, False, 0.0

    state_min = str(
        config.get("reference_lookahead_entry_shorten_state_min", "ENTRY") or "ENTRY"
    ).strip().upper()
    state_max = str(
        config.get("reference_lookahead_entry_shorten_state_max", "ENTRY") or "ENTRY"
    ).strip().upper()
    if _curve_state_rank(curve_local_state) < _curve_state_rank(state_min):
        return target, False, 0.0
    if _curve_state_rank(curve_local_state) > _curve_state_rank(state_max):
        return target, False, 0.0

    gate_min = max(
        0.0,
        min(
            1.0,
            float(config.get("reference_lookahead_entry_shorten_gate_min", 0.20) or 0.20),
        ),
    )
    distance_max = max(
        0.0,
        float(config.get("reference_lookahead_entry_shorten_distance_max_m", 8.0) or 8.0),
    )
    distance_min = max(
        0.0,
        float(config.get("reference_lookahead_entry_shorten_distance_min_m", 0.01) or 0.01),
    )
    gate_ready = False
    if local_gate_weight is not None:
        try:
            gate_ready = math.isfinite(float(local_gate_weight)) and float(local_gate_weight) >= gate_min
        except (TypeError, ValueError):
            gate_ready = False
    distance_ready = False
    if distance_to_curve_start_m is not None:
        try:
            distance_value = float(distance_to_curve_start_m)
            if math.isfinite(distance_value) and distance_value <= distance_min:
                return target, False, 0.0
            distance_ready = (
                math.isfinite(distance_value)
                and distance_value > distance_min
                and distance_value <= distance_max
            )
        except (TypeError, ValueError):
            distance_ready = False
    if not (gate_ready or distance_ready):
        return target, False, 0.0

    min_allowed = prev - guard_slew
    if target < min_allowed:
        return float(min_allowed), True, float(min_allowed - target)
    return target, False, 0.0


def _compute_tight_curve_scale(abs_curvature: float, config: Mapping[str, object]) -> float:
    """
    Compute tight-curve lookahead scale with optional smooth blending near threshold.

    When reference_lookahead_tight_blend_band > 0, the tight-curve scale is blended
    with a smoothstep transition around reference_lookahead_tight_curvature_threshold.
    """
    threshold = float(config.get("reference_lookahead_tight_curvature_threshold", 0.0))
    if threshold <= 0.0:
        return 1.0

    tight_scale = float(config.get("reference_lookahead_tight_scale", 1.0))
    tight_scale = max(0.1, min(1.0, tight_scale))
    blend_band = max(0.0, float(config.get("reference_lookahead_tight_blend_band", 0.0)))

    abs_curv = abs(float(abs_curvature))
    if blend_band <= 1e-6:
        return tight_scale if abs_curv >= threshold else 1.0

    half_band = 0.5 * blend_band
    lower = max(0.0, threshold - half_band)
    upper = threshold + half_band
    if abs_curv <= lower:
        return 1.0
    if abs_curv >= upper:
        return tight_scale

    ratio = (abs_curv - lower) / max(1e-6, upper - lower)
    blend = _smoothstep_01(ratio)
    return 1.0 + (tight_scale - 1.0) * blend


def compute_reference_lookahead(
    base_lookahead: float,
    current_speed: float,
    path_curvature: float,
    config: dict,
    path_curvature_preview: float | None = None,
    distance_to_curve_start_m: float | None = None,
    previous_lookahead: float | None = None,
    anticipation_score: float | None = None,
    anticipation_active: bool | None = None,
    curve_scheduler_mode: str | None = None,
    curve_phase: float | None = None,
    curve_local_state: str | None = None,
    current_curve_progress_ratio: float | None = None,
    curve_local_entry_severity: float | None = None,
    local_gate_weight: float | None = None,
    return_diagnostics: bool = False,
) -> float | dict[str, float | str]:
    """
    Compute a dynamic reference lookahead based on speed and curvature.

    If dual speed tables are configured, lookahead is blended by curvature.
    If only single speed table is configured, it remains the source of truth.
    Otherwise, legacy scale-based dynamic lookahead is used.
    """
    scheduler_mode = resolve_curve_scheduler_mode(
        config=config,
        explicit_mode=curve_scheduler_mode,
    )
    if not config.get("dynamic_reference_lookahead", False):
        if return_diagnostics:
            return {
                "lookahead": float(base_lookahead),
                "lookahead_target": float(base_lookahead),
                "reference_lookahead_target_pre_entry_guard": float(base_lookahead),
                "lookahead_after_slew": float(base_lookahead),
                "curve_scheduler_mode": scheduler_mode,
                "curve_phase": 0.0,
                "reference_lookahead_owner_mode": "static",
                "reference_lookahead_entry_weight_source": "base_lookahead",
                "reference_lookahead_fallback_active": False,
                "reference_lookahead_local_gate_weight": 0.0,
                "reference_lookahead_entry_shorten_guard_active": False,
                "reference_lookahead_entry_shorten_guard_delta_m": 0.0,
            }
        return base_lookahead

    min_lookahead = float(config.get("reference_lookahead_min", 4.0))
    min_lookahead = max(0.1, min_lookahead)
    owner_nominal_target = float(base_lookahead)
    owner_commit_band_target = 0.0
    owner_entry_progress = 0.0
    owner_commit_distance_progress = 0.0
    owner_commit_phase_progress = 0.0
    owner_commit_progress = 0.0
    owner_commit_hold_progress = 1.0
    owner_commit_distance_start_effective_m = 0.0
    owner_commit_band_clamp_active = False
    owner_commit_band_clamp_delta_m = 0.0

    def _finalize(
        lookahead_target: float,
        *,
        mode_used: str,
        phase_used: float,
        owner_mode: str,
        entry_weight_source: str,
        fallback_active: bool,
    ) -> float | dict[str, float | str]:
        target = float(lookahead_target)
        target *= _compute_tight_curve_scale(abs(path_curvature), config)
        target = max(min_lookahead, target)
        after_global_slew = _apply_lookahead_slew_limit(target, previous_lookahead, config)
        after_entry_guard, entry_guard_active, entry_guard_delta = _apply_entry_local_shorten_guard(
            after_global_slew,
            previous_lookahead,
            curve_local_state=curve_local_state,
            local_gate_weight=local_gate_weight,
            distance_to_curve_start_m=distance_to_curve_start_m,
            config=config,
        )
        if return_diagnostics:
            return {
                "lookahead": float(after_entry_guard),
                "lookahead_target": float(target),
                "reference_lookahead_target_pre_entry_guard": float(after_global_slew),
                "lookahead_after_slew": float(after_entry_guard),
                "curve_scheduler_mode": str(mode_used),
                "curve_phase": float(max(0.0, min(1.0, phase_used))),
                "reference_lookahead_owner_mode": str(owner_mode),
                "reference_lookahead_owner_nominal_target": float(owner_nominal_target),
                "reference_lookahead_owner_commit_band_target": float(
                    owner_commit_band_target
                ),
                "reference_lookahead_owner_entry_progress": float(owner_entry_progress),
                "reference_lookahead_owner_commit_distance_progress": float(
                    owner_commit_distance_progress
                ),
                "reference_lookahead_owner_commit_phase_progress": float(
                    owner_commit_phase_progress
                ),
                "reference_lookahead_owner_commit_progress": float(owner_commit_progress),
                "reference_lookahead_owner_commit_hold_progress": float(
                    owner_commit_hold_progress
                ),
                "reference_lookahead_owner_commit_distance_start_effective_m": float(
                    owner_commit_distance_start_effective_m
                ),
                "reference_lookahead_owner_commit_band_clamp_active": bool(
                    owner_commit_band_clamp_active
                ),
                "reference_lookahead_owner_commit_band_clamp_delta_m": float(
                    owner_commit_band_clamp_delta_m
                ),
                "reference_lookahead_entry_weight_source": str(entry_weight_source),
                "reference_lookahead_fallback_active": bool(fallback_active),
                "reference_lookahead_local_gate_weight": float(
                    max(
                        0.0,
                        min(
                            1.0,
                            float(local_gate_weight if local_gate_weight is not None else phase_used),
                        ),
                    )
                ),
                "reference_lookahead_entry_shorten_guard_active": bool(entry_guard_active),
                "reference_lookahead_entry_shorten_guard_delta_m": float(entry_guard_delta),
            }
        return float(after_entry_guard)

    speed_table = _parse_reference_lookahead_speed_table(config)
    entry_speed_table = _parse_reference_lookahead_speed_table(
        config,
        key="reference_lookahead_speed_table_entry",
    )
    commit_speed_table = _parse_reference_lookahead_speed_table(
        config,
        key="reference_lookahead_speed_table_commit",
    )
    if speed_table and entry_speed_table:
        lookahead_straight = _interpolate_reference_lookahead_for_speed(
            current_speed,
            speed_table,
        )
        preview_curvature = (
            float(path_curvature_preview)
            if path_curvature_preview is not None
            else float(path_curvature)
        )
        phase_used = 0.0
        owner_mode = CURVE_SCHEDULER_MODE_BINARY
        entry_weight_source = "preview_distance"
        fallback_active = False

        def _compute_binary_entry_weight() -> tuple[float, str]:
            binary_weight = _compute_entry_phase_weight(
                preview_curvature=preview_curvature,
                distance_to_curve_start_m=distance_to_curve_start_m,
                config=config,
            )
            source_used = "preview_distance"
            anticipation_weight = _compute_curve_anticipation_entry_weight(
                anticipation_score=anticipation_score,
                anticipation_active=anticipation_active,
                config=config,
            )
            if anticipation_weight > 0.0:
                if bool(config.get("curve_anticipation_entry_weight_merge_with_preview", False)):
                    if anticipation_weight > binary_weight + 1e-6:
                        source_used = (
                            "preview_distance+curve_anticipation_max"
                            if binary_weight > 0.0
                            else "curve_anticipation"
                        )
                    binary_weight = max(binary_weight, anticipation_weight)
                else:
                    binary_weight = anticipation_weight
                    source_used = "curve_anticipation"
            return float(binary_weight), source_used

        if scheduler_mode == CURVE_SCHEDULER_MODE_PHASE_ACTIVE:
            phase_value = 0.0
            if curve_phase is not None:
                try:
                    phase_value = float(curve_phase)
                except (TypeError, ValueError):
                    phase_value = 0.0
            if math.isfinite(phase_value):
                phase_used = max(0.0, min(1.0, phase_value))
                entry_weight = phase_used
                owner_mode = CURVE_SCHEDULER_MODE_PHASE_ACTIVE
                entry_weight_source = "curve_local_phase"
            else:
                entry_weight, entry_weight_source = _compute_binary_entry_weight()
                phase_used = float(entry_weight)
                owner_mode = CURVE_SCHEDULER_MODE_PHASE_ACTIVE
                entry_weight_source = f"{entry_weight_source}_fallback"
                fallback_active = True
        else:
            entry_weight, entry_weight_source = _compute_binary_entry_weight()
            phase_used = float(entry_weight)
        lookahead_entry = _interpolate_reference_lookahead_for_speed(
            current_speed,
            entry_speed_table,
        )
        owner_entry_progress = float(max(0.0, min(1.0, entry_weight)))
        lookahead_target = (
            lookahead_straight * (1.0 - entry_weight)
            + lookahead_entry * entry_weight
        )
        curve_local_state_normalized = str(curve_local_state or "").strip().upper()
        severity_val = 0.0
        if curve_local_entry_severity is not None:
            try:
                severity_val = float(curve_local_entry_severity)
            except (TypeError, ValueError):
                severity_val = 0.0
        if not math.isfinite(severity_val):
            severity_val = 0.0
        severity_val = max(0.0, min(1.0, severity_val))
        if (
            scheduler_mode == CURVE_SCHEDULER_MODE_PHASE_ACTIVE
            and commit_speed_table
            and curve_local_state_normalized == "COMMIT"
        ):
            phase_on = float(
                config.get("reference_lookahead_phase_active_commit_phase_on", 0.75)
            )
            phase_full = float(
                config.get("reference_lookahead_phase_active_commit_phase_full", 0.95)
            )
            if phase_full <= phase_on:
                owner_commit_phase_progress = 1.0 if phase_used >= phase_on else 0.0
            elif phase_used <= phase_on:
                owner_commit_phase_progress = 0.0
            elif phase_used >= phase_full:
                owner_commit_phase_progress = 1.0
            else:
                owner_commit_phase_progress = _smoothstep_01(
                    (phase_used - phase_on) / max(1e-6, phase_full - phase_on)
                )
            owner_commit_progress = float(owner_commit_phase_progress)
            lookahead_commit = _interpolate_reference_lookahead_for_speed(
                current_speed,
                commit_speed_table,
            )
            owner_commit_band_target = float(lookahead_commit)
            lookahead_target = (
                lookahead_target * (1.0 - owner_commit_progress)
                + lookahead_commit * owner_commit_progress
            )
        if (
            scheduler_mode == CURVE_SCHEDULER_MODE_PHASE_ACTIVE
            and curve_local_state_normalized == "ENTRY"
        ):
            scale_min = float(
                config.get("reference_lookahead_phase_active_entry_scale_min", 1.0)
            )
            scale_min = max(0.1, min(1.0, scale_min))
            sev_on = float(
                config.get("reference_lookahead_phase_active_entry_scale_severity_on", 0.35)
            )
            sev_full = float(
                config.get("reference_lookahead_phase_active_entry_scale_severity_full", 0.85)
            )
            if scale_min < 0.999:
                if sev_full <= sev_on:
                    scale_weight = 1.0 if severity_val >= sev_on else 0.0
                elif severity_val <= sev_on:
                    scale_weight = 0.0
                elif severity_val >= sev_full:
                    scale_weight = 1.0
                else:
                    scale_weight = _smoothstep_01(
                        (severity_val - sev_on) / max(1e-6, sev_full - sev_on)
                    )
                owner_scale = 1.0 + ((scale_min - 1.0) * scale_weight)
                lookahead_target *= owner_scale
        owner_nominal_target = float(lookahead_target)
        return _finalize(
            lookahead_target,
            mode_used=scheduler_mode if scheduler_mode != CURVE_SCHEDULER_MODE_PHASE_SHADOW else CURVE_SCHEDULER_MODE_BINARY,
            phase_used=phase_used,
            owner_mode=owner_mode,
            entry_weight_source=entry_weight_source,
            fallback_active=fallback_active,
        )

    straight_speed_table = _parse_reference_lookahead_speed_table(
        config,
        key="reference_lookahead_speed_table_straight",
    )
    curve_speed_table = _parse_reference_lookahead_speed_table(
        config,
        key="reference_lookahead_speed_table_curve",
    )
    if straight_speed_table and curve_speed_table:
        lookahead_straight = _interpolate_reference_lookahead_for_speed(
            current_speed,
            straight_speed_table,
        )
        lookahead_curve = _interpolate_reference_lookahead_for_speed(
            current_speed,
            curve_speed_table,
        )
        curve_blend_weight = _compute_curvature_blend_weight(abs(path_curvature), config)
        lookahead_target = (
            lookahead_straight * (1.0 - curve_blend_weight)
            + lookahead_curve * curve_blend_weight
        )
        return _finalize(
            lookahead_target,
            mode_used="dual_table_curvature_blend",
            phase_used=float(curve_blend_weight),
            owner_mode="dual_table_curvature_blend",
            entry_weight_source="curvature_blend",
            fallback_active=False,
        )

    # Compatibility path: preserve prior single-table behavior when dual tables are absent.
    if speed_table:
        lookahead_target = _interpolate_reference_lookahead_for_speed(current_speed, speed_table)
        return _finalize(
            lookahead_target,
            mode_used="single_table_compat",
            phase_used=0.0,
            owner_mode="single_table_compat",
            entry_weight_source="single_table",
            fallback_active=False,
        )

    scale_min = float(config.get("reference_lookahead_scale_min", 0.7))
    scale_min = max(0.1, min(1.0, scale_min))
    scale = 1.0

    speed_min = float(config.get("reference_lookahead_speed_min", 4.0))
    speed_max = float(config.get("reference_lookahead_speed_max", 10.0))
    if speed_max > speed_min:
        if current_speed <= speed_min:
            speed_scale = 1.0
        elif current_speed >= speed_max:
            speed_scale = scale_min
        else:
            ratio = (current_speed - speed_min) / (speed_max - speed_min)
            speed_scale = 1.0 - ratio * (1.0 - scale_min)
        scale = min(scale, speed_scale)

    curv_min = float(config.get("reference_lookahead_curvature_min", 0.002))
    curv_max = float(config.get("reference_lookahead_curvature_max", 0.015))
    abs_curv = abs(path_curvature)
    if curv_max > curv_min:
        if abs_curv <= curv_min:
            curv_scale = scale_min
        elif abs_curv >= curv_max:
            curv_scale = 1.0
        else:
            ratio = (abs_curv - curv_min) / (curv_max - curv_min)
            curv_scale = scale_min + ratio * (1.0 - scale_min)
        scale = min(scale, curv_scale)

    lookahead_target = max(min_lookahead, base_lookahead * scale)
    return _finalize(
        lookahead_target,
        mode_used="legacy_scale",
        phase_used=0.0,
        owner_mode="legacy_scale",
        entry_weight_source="legacy_scale",
        fallback_active=False,
    )


def curvature_smoothing_alpha(distance_m: float, window_m: float) -> float:
    """Compute EMA alpha for a distance-based smoothing window."""
    if window_m <= 0.0:
        return 1.0
    distance_m = max(0.0, float(distance_m))
    return float(1.0 - math.exp(-distance_m / window_m))


def smooth_curvature_distance(
    curvature: Iterable[float],
    speed: Iterable[float],
    timestamps: Iterable[float],
    window_m: float,
    min_speed: float = 0.0,
) -> list[float]:
    """Smooth curvature using a distance-based EMA over a spatial window."""
    curvatures = list(curvature)
    speeds = list(speed)
    times = list(timestamps)
    if not curvatures or len(curvatures) != len(speeds) or len(curvatures) != len(times):
        return curvatures
    if window_m <= 0.0:
        return curvatures
    min_speed = max(0.0, float(min_speed))
    smoothed = [float(curvatures[0])]
    for i in range(1, len(curvatures)):
        dt = float(times[i]) - float(times[i - 1])
        if dt <= 0.0:
            dt = 1e-3
        distance = max(float(speeds[i]), min_speed) * dt
        alpha = curvature_smoothing_alpha(distance, window_m)
        smoothed_value = alpha * float(curvatures[i]) + (1.0 - alpha) * smoothed[-1]
        smoothed.append(smoothed_value)
    return smoothed


def select_curvature_bin_limits(
    abs_curvature: float,
    bins: Sequence[Mapping[str, float]] | None,
    default_max_lateral_accel: float,
    default_min_curve_speed: float,
) -> Tuple[float, float]:
    """Select max lateral accel and min curve speed based on curvature bins."""
    max_lateral_accel = float(default_max_lateral_accel)
    min_curve_speed = float(default_min_curve_speed)
    if not bins:
        return max_lateral_accel, min_curve_speed
    if not isinstance(abs_curvature, (int, float)):
        return max_lateral_accel, min_curve_speed

    curvature_val = float(abs_curvature)
    for bin_cfg in bins:
        if not isinstance(bin_cfg, Mapping):
            continue
        min_curv = float(bin_cfg.get("min_curvature", 0.0))
        max_curv = float(bin_cfg.get("max_curvature", float("inf")))
        if curvature_val < min_curv or curvature_val >= max_curv:
            continue
        if "max_lateral_accel" in bin_cfg:
            max_lateral_accel = float(bin_cfg["max_lateral_accel"])
        if "min_curve_speed" in bin_cfg:
            min_curve_speed = float(bin_cfg["min_curve_speed"])
        break
    return max_lateral_accel, min_curve_speed


def compute_dynamic_effective_horizon(
    base_horizon_m: float,
    current_speed_mps: float,
    path_curvature: float,
    confidence: float | None,
    config: Mapping[str, object],
) -> dict[str, float]:
    """
    Compute diagnostics for a dynamic effective horizon policy.

    Phase 1 is diagnostics-only: this function reports what the policy would do,
    but does not change downstream behavior unless a future phase explicitly applies it.
    """
    base = max(0.1, float(base_horizon_m))
    min_m = max(0.1, float(config.get("dynamic_effective_horizon_min_m", 6.0)))
    max_m = max(min_m, float(config.get("dynamic_effective_horizon_max_m", base)))

    speed_min = float(config.get("reference_lookahead_speed_min", 4.0))
    speed_max = float(config.get("reference_lookahead_speed_max", 10.0))
    curv_min = float(config.get("reference_lookahead_curvature_min", 0.002))
    curv_max = float(config.get("reference_lookahead_curvature_max", 0.015))

    speed_gain = max(0.0, min(1.0, float(config.get("dynamic_effective_horizon_speed_scale", 0.0))))
    curvature_gain = max(
        0.0,
        min(1.0, float(config.get("dynamic_effective_horizon_curvature_scale", 0.0))),
    )
    confidence_floor = max(
        0.1,
        min(1.0, float(config.get("dynamic_effective_horizon_confidence_floor", 0.6))),
    )
    confidence_fallback = max(
        0.1,
        min(1.0, float(config.get("dynamic_effective_horizon_confidence_fallback", 1.0))),
    )

    speed_norm = 0.0
    if speed_max > speed_min:
        speed_norm = (float(current_speed_mps) - speed_min) / (speed_max - speed_min)
        speed_norm = max(0.0, min(1.0, speed_norm))
    speed_scale = 1.0 - (speed_gain * speed_norm)
    speed_scale = max(0.1, min(1.0, speed_scale))

    abs_curv = abs(float(path_curvature))
    curvature_norm = 0.0
    if curv_max > curv_min:
        curvature_norm = (abs_curv - curv_min) / (curv_max - curv_min)
        curvature_norm = max(0.0, min(1.0, curvature_norm))
    curvature_scale = 1.0 - (curvature_gain * curvature_norm)
    curvature_scale = max(0.1, min(1.0, curvature_scale))

    conf_used = (
        confidence_fallback
        if confidence is None or not math.isfinite(float(confidence))
        else max(0.0, min(1.0, float(confidence)))
    )
    confidence_scale = max(confidence_floor, conf_used)
    confidence_scale = max(0.1, min(1.0, confidence_scale))

    final_scale = min(speed_scale, curvature_scale, confidence_scale)
    unclamped_horizon = base * final_scale
    effective_horizon = max(min_m, min(max_m, unclamped_horizon))

    # limiter_code: 0=none, 1=speed, 2=curvature, 3=confidence
    limiter_code = 0.0
    limiter_scale = final_scale
    if limiter_scale < 0.999:
        candidates = [
            (speed_scale, 1.0),
            (curvature_scale, 2.0),
            (confidence_scale, 3.0),
        ]
        limiter_code = min(candidates, key=lambda item: item[0])[1]

    # Phase 2: when enabled, downstream trajectory generation may apply this horizon.
    applied = 1.0 if bool(config.get("dynamic_effective_horizon_enabled", False)) else 0.0

    return {
        "diag_dynamic_effective_horizon_m": float(effective_horizon),
        "diag_dynamic_effective_horizon_base_m": float(base),
        "diag_dynamic_effective_horizon_min_m": float(min_m),
        "diag_dynamic_effective_horizon_max_m": float(max_m),
        "diag_dynamic_effective_horizon_speed_scale": float(speed_scale),
        "diag_dynamic_effective_horizon_curvature_scale": float(curvature_scale),
        "diag_dynamic_effective_horizon_confidence_scale": float(confidence_scale),
        "diag_dynamic_effective_horizon_final_scale": float(final_scale),
        "diag_dynamic_effective_horizon_speed_mps": float(current_speed_mps),
        "diag_dynamic_effective_horizon_curvature_abs": float(abs_curv),
        "diag_dynamic_effective_horizon_confidence_used": float(conf_used),
        "diag_dynamic_effective_horizon_limiter_code": float(limiter_code),
        "diag_dynamic_effective_horizon_applied": float(applied),
    }


def apply_speed_horizon_guardrail(
    target_speed_mps: float,
    effective_horizon_m: float,
    dynamic_horizon_applied: bool,
    config: Mapping[str, object],
) -> dict[str, float]:
    """
    Apply a lightweight speed-horizon consistency guardrail.

    The guardrail enforces a minimum planning-distance budget:
      required_horizon = speed * time_headway + margin_buffer

    If the dynamic horizon is applied and falls below that budget, speed is
    conservatively reduced toward the allowed speed implied by the same budget.
    """
    target_before = max(0.0, float(target_speed_mps))
    horizon = max(0.1, float(effective_horizon_m))
    applied = 1.0 if bool(dynamic_horizon_applied) else 0.0

    enabled = bool(config.get("speed_horizon_guardrail_enabled", False))
    time_headway = max(0.1, float(config.get("speed_horizon_guardrail_time_headway_s", 1.8)))
    margin_buffer = max(0.0, float(config.get("speed_horizon_guardrail_margin_m", 1.0)))
    min_speed = max(0.0, float(config.get("speed_horizon_guardrail_min_speed_mps", 3.0)))
    gain = max(0.0, min(1.0, float(config.get("speed_horizon_guardrail_gain", 1.0))))

    required_horizon = target_before * time_headway + margin_buffer
    horizon_margin = horizon - required_horizon
    allowed_speed = max(0.0, (horizon - margin_buffer) / time_headway)

    guardrail_active = 0.0
    target_after = target_before
    if enabled and applied > 0.5 and horizon_margin < 0.0:
        guardrail_active = 1.0
        capped_speed = min(target_before, allowed_speed)
        target_after = max(min_speed, target_before + (capped_speed - target_before) * gain)

    return {
        "target_speed_mps": float(target_after),
        "diag_speed_horizon_guardrail_active": float(guardrail_active),
        "diag_speed_horizon_guardrail_margin_m": float(horizon_margin),
        "diag_speed_horizon_guardrail_horizon_m": float(horizon),
        "diag_speed_horizon_guardrail_time_headway_s": float(time_headway),
        "diag_speed_horizon_guardrail_margin_buffer_m": float(margin_buffer),
        "diag_speed_horizon_guardrail_allowed_speed_mps": float(allowed_speed),
        "diag_speed_horizon_guardrail_target_speed_before_mps": float(target_before),
        "diag_speed_horizon_guardrail_target_speed_after_mps": float(target_after),
    }
