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


def _smoothstep_01(value: float) -> float:
    """Smoothstep over [0, 1] with clamping."""
    ratio = max(0.0, min(1.0, float(value)))
    return ratio * ratio * (3.0 - 2.0 * ratio)


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

    phase_raw = max(term_preview, term_path, term_rise, term_time) * confidence_scale

    alpha = float(config.get("curve_phase_ema_alpha", 0.45))
    alpha = max(0.0, min(1.0, alpha))
    prev_phase = float(previous_phase) if math.isfinite(float(previous_phase)) else 0.0
    phase = (alpha * phase_raw) + ((1.0 - alpha) * prev_phase)

    on = float(config.get("curve_phase_on", 0.45))
    off = float(config.get("curve_phase_off", 0.30))
    if off > on:
        off = on
    commit_on = float(config.get("curve_phase_commit_on", 0.55))
    commit_on = max(on, commit_on)
    commit_min_frames = max(1, int(config.get("curve_phase_commit_min_frames", 4)))
    entry_floor = max(0.0, min(1.0, float(config.get("curve_phase_entry_floor", 0.15))))
    commit_floor = max(entry_floor, min(1.0, float(config.get("curve_phase_commit_floor", 0.30))))
    rearm_hold_cfg = max(0, int(config.get("curve_phase_rearm_hold_frames", 4)))

    prev_state_norm = str(previous_state or "STRAIGHT").strip().upper()
    if prev_state_norm not in {"STRAIGHT", "ENTRY", "COMMIT", "REARM"}:
        prev_state_norm = "STRAIGHT"
    entry_frames = max(0, int(previous_entry_frames))
    rearm_hold_frames = max(0, int(previous_rearm_hold_frames))

    state = prev_state_norm
    rearm_event = False
    if prev_state_norm == "COMMIT":
        if phase < off:
            state = "REARM"
            entry_frames = 0
            rearm_hold_frames = rearm_hold_cfg
        else:
            state = "COMMIT"
            entry_frames = max(1, entry_frames)
    elif prev_state_norm == "ENTRY":
        if phase < off:
            state = "REARM"
            entry_frames = 0
            rearm_hold_frames = rearm_hold_cfg
        else:
            entry_frames = max(1, entry_frames + 1)
            if phase >= commit_on and entry_frames >= commit_min_frames:
                state = "COMMIT"
            else:
                state = "ENTRY"
    elif prev_state_norm == "REARM":
        if phase >= on:
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
        if phase >= on:
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
                "lookahead_after_slew": float(base_lookahead),
                "curve_scheduler_mode": scheduler_mode,
                "curve_phase": 0.0,
            }
        return base_lookahead

    min_lookahead = float(config.get("reference_lookahead_min", 4.0))
    min_lookahead = max(0.1, min_lookahead)

    def _finalize(
        lookahead_target: float,
        *,
        mode_used: str,
        phase_used: float,
    ) -> float | dict[str, float | str]:
        target = float(lookahead_target)
        target *= _compute_tight_curve_scale(abs(path_curvature), config)
        target = max(min_lookahead, target)
        after_slew = _apply_lookahead_slew_limit(target, previous_lookahead, config)
        if return_diagnostics:
            return {
                "lookahead": float(after_slew),
                "lookahead_target": float(target),
                "lookahead_after_slew": float(after_slew),
                "curve_scheduler_mode": str(mode_used),
                "curve_phase": float(max(0.0, min(1.0, phase_used))),
            }
        return float(after_slew)

    speed_table = _parse_reference_lookahead_speed_table(config)
    entry_speed_table = _parse_reference_lookahead_speed_table(
        config,
        key="reference_lookahead_speed_table_entry",
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
        if scheduler_mode == CURVE_SCHEDULER_MODE_PHASE_ACTIVE:
            phase_used = float(curve_phase) if curve_phase is not None else 0.0
            entry_weight = max(0.0, min(1.0, phase_used))
        else:
            entry_weight = _compute_entry_phase_weight(
                preview_curvature=preview_curvature,
                distance_to_curve_start_m=distance_to_curve_start_m,
                config=config,
            )
            anticipation_weight = _compute_curve_anticipation_entry_weight(
                anticipation_score=anticipation_score,
                anticipation_active=anticipation_active,
                config=config,
            )
            if anticipation_weight > 0.0:
                if bool(config.get("curve_anticipation_entry_weight_merge_with_preview", False)):
                    entry_weight = max(entry_weight, anticipation_weight)
                else:
                    entry_weight = anticipation_weight
            phase_used = float(entry_weight)
        lookahead_entry = _interpolate_reference_lookahead_for_speed(
            current_speed,
            entry_speed_table,
        )
        lookahead_target = (
            lookahead_straight * (1.0 - entry_weight)
            + lookahead_entry * entry_weight
        )
        return _finalize(
            lookahead_target,
            mode_used=scheduler_mode if scheduler_mode != CURVE_SCHEDULER_MODE_PHASE_SHADOW else CURVE_SCHEDULER_MODE_BINARY,
            phase_used=phase_used,
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
        )

    # Compatibility path: preserve prior single-table behavior when dual tables are absent.
    if speed_table:
        lookahead_target = _interpolate_reference_lookahead_for_speed(current_speed, speed_table)
        return _finalize(
            lookahead_target,
            mode_used="single_table_compat",
            phase_used=0.0,
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
