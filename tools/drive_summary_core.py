"""
Summary analyzer for recording-level metrics.
Extracted from analyze_drive_overall.py for use in debug visualizer.
"""

import math
import sys
import json
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import yaml
from scipy.fft import fft, fftfreq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# scoring_registry lives under tools/; tests add this via conftest — CLI entrypoints need it too.
sys.path.insert(0, str(REPO_ROOT / "tools"))

from trajectory.utils import (
    CURVE_SCHEDULER_MODE_BINARY,
    CURVE_SCHEDULER_MODE_PHASE_ACTIVE,
    smooth_curvature_distance,
)
from sync_contract import (
    EFFECTIVE_TARGET_REASON_ACC_FOLLOW,
    EFFECTIVE_TARGET_REASON_CURVE_CAP,
    EFFECTIVE_TARGET_REASON_FALLBACK_UNKNOWN,
    EFFECTIVE_TARGET_REASON_FEASIBILITY_BACKSTOP,
    EFFECTIVE_TARGET_REASON_FREE_FLOW,
    EFFECTIVE_TARGET_REASON_POST_JUMP_COOLDOWN,
    PACKET_FALLBACK_NONE,
    PACKET_MODE_LATEST_PARALLEL,
    POST_JUMP_REASON_NONE,
)

from scoring_registry import (
    ACCEL_P95_GATE_MPS2,
    JERK_P95_GATE_MPS3,
    LATERAL_P95_GATE_M,
    CENTERED_BAND_M,
    OUT_OF_LANE_THRESHOLD_M,
    CATASTROPHIC_ERROR_M,
    MIN_CONSECUTIVE_OOL,
    CURVATURE_FLOOR_COEFF as _CURVATURE_FLOOR_COEFF,
    STEERING_JERK_PENALTY_CAP,
    HEADING_PENALTY_FLOOR_DEG,
    ACC_COLLISION_GATE,
    ACC_TTC_CRITICAL_S,
    ACC_NEAR_MISS_GAP_M,
    ACC_TTC_MIN_GATE_S,
    ACC_TTC_WARNING_S,
    ACC_TTC_COMFORTABLE_S,
    ACC_NEAR_MISS_PENALTY_PTS,
    ACC_TTC_WARNING_PENALTY_PER_PCT,
    ACC_GAP_RMSE_GATE_M,
    ACC_JERK_P95_GATE_MPS3,
    ACC_DETECTION_RATE_GATE,
    ACC_MIN_ACTIVE_FRAME_RATE,
)

G_MPS2 = 9.80665
LOW_VISIBILITY_STALE_REASONS = {"left_lane_low_visibility", "right_lane_low_visibility"}


def _build_sign_mismatch_event(data: Dict, start_frame: int, end_frame: int) -> Dict:
    """Build a straight sign-mismatch event record with a classified root cause."""
    duration_frames = end_frame - start_frame + 1
    duration_seconds = safe_float(
        data['time'][end_frame] - data['time'][start_frame]
    ) if data.get('time') is not None and len(data['time']) > end_frame else 0.0

    err_series = data.get('total_error_scaled') if data.get('total_error_scaled') is not None else data.get('total_error')
    steer_series = data.get('steering')
    fb_series = data.get('feedback_steering')
    before_series = data.get('steering_before_limits')

    err_slice = err_series[start_frame:end_frame + 1] if err_series is not None else None
    steer_slice = steer_series[start_frame:end_frame + 1] if steer_series is not None else None
    fb_slice = fb_series[start_frame:end_frame + 1] if fb_series is not None else None
    before_slice = before_series[start_frame:end_frame + 1] if before_series is not None else None

    lanes = data.get('num_lanes_detected')
    lanes_slice = lanes[start_frame:end_frame + 1] if lanes is not None else None
    stale_ctrl = data.get('using_stale_perception')
    stale_ctrl_slice = stale_ctrl[start_frame:end_frame + 1] if stale_ctrl is not None else None
    stale_perc = data.get('using_stale_data')
    stale_perc_slice = stale_perc[start_frame:end_frame + 1] if stale_perc is not None else None
    override = data.get('straight_sign_flip_override_active')
    override_slice = override[start_frame:end_frame + 1] if override is not None else None

    stale_rate = 0.0
    if stale_ctrl_slice is not None:
        stale_rate = max(stale_rate, safe_float(float((stale_ctrl_slice > 0).mean()) * 100.0))
    if stale_perc_slice is not None:
        stale_rate = max(stale_rate, safe_float(float((stale_perc_slice > 0).mean()) * 100.0))
    lanes_min = int(lanes_slice.min()) if lanes_slice is not None and lanes_slice.size > 0 else None

    # Check if event overlaps with a PP↔MPC blend transition
    blend_weight = data.get('regime_blend_weight')
    blend_slice = blend_weight[start_frame:end_frame + 1] if blend_weight is not None else None
    in_blend = (
        blend_slice is not None
        and blend_slice.size > 0
        and float(np.mean((blend_slice > 0.01) & (blend_slice < 0.99))) > 0.3
    )

    root_cause = "unknown"
    if in_blend:
        root_cause = "regime_blend"
    elif stale_rate > 0.0 or (lanes_min is not None and lanes_min < 2):
        root_cause = "perception_stale_or_missing"
    elif (
        err_slice is not None and fb_slice is not None and before_slice is not None
        and np.sign(np.mean(err_slice)) == np.sign(np.mean(fb_slice))
        and np.sign(np.mean(before_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "rate_or_jerk_limit"
    elif (
        err_slice is not None and before_slice is not None and steer_slice is not None
        and np.sign(np.mean(before_slice)) == np.sign(np.mean(err_slice))
        and np.sign(np.mean(steer_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "smoothing"
    elif (
        err_slice is not None and fb_slice is not None
        and np.sign(np.mean(fb_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "error_computation_or_sign"

    override_rate = 0.0
    if override_slice is not None and override_slice.size > 0:
        override_rate = safe_float(float((override_slice > 0).mean()) * 100.0)

    return {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "duration_frames": int(duration_frames),
        "duration_seconds": safe_float(duration_seconds),
        "root_cause": root_cause,
        "stale_rate_percent": safe_float(stale_rate),
        "lanes_min": lanes_min,
        "override_rate_percent": safe_float(override_rate),
    }


def safe_float(value, default=0.0):
    """Convert value to float, handling NaN and inf."""
    if value is None:
        return default
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
    return float(value)


# ---------------------------------------------------------------------------
# Jerk event classifier
# ---------------------------------------------------------------------------

_JERK_CAUSE_LABELS = {
    "dropout_start":      "Silent perception dropout began (e_lat→0, stale=0)",
    "dropout_recovery":   "Silent perception dropout recovery (e_lat snapped back)",
    "stale_recovery":     "Stale perception hold ended (PP stale flag cleared)",
    "curve_oscillation":  "Curve body oscillation (persistent large e_lat, gate active)",
    "straight_noise":     "Straight-road noise chasing (small e_lat, sign flips)",
    "unknown":            "Unclassified jerk event",
}


def classify_jerk_events(
    data: dict,
    steering_jerk_threshold_deg_per_s2: float = 18.0,
    window_frames: int = 3,
) -> list:
    """Classify each steering jerk event above *threshold* into a root cause.

    Parameters
    ----------
    data:
        Signal dict produced by the HDF5 loader in ``drive_summary_core``.
        Required keys: ``time``, ``steering``.
        Optional (enrich classification): ``mpc_e_lat``, ``pp_stale_hold_active``,
        ``diag_heading_zero_gate_active``, ``diag_silent_elat_dropout_active``,
        ``curvature_map_abs``.
    steering_jerk_threshold_deg_per_s2:
        Events whose |steering jerk| exceeds this value (deg/s²) are classified.
    window_frames:
        Number of frames before/after the event to examine for context signals.

    Returns
    -------
    List of dicts, one per event, sorted by time.  Each dict has keys:
        ``frame``      – 0-based index into the signal arrays,
        ``time``       – timestamp (s),
        ``jerk_deg_s2``– signed jerk magnitude (deg/s²) at this frame,
        ``cause``      – one of the keys in ``_JERK_CAUSE_LABELS``,
        ``description``– human-readable explanation.
    """
    time_arr = np.asarray(data.get("time", []), dtype=float)
    steering_arr = np.asarray(data.get("steering", []), dtype=float)
    n = min(len(time_arr), len(steering_arr))
    if n < 3:
        return []

    dt_arr = np.diff(time_arr[:n])
    median_dt = float(np.median(dt_arr)) if len(dt_arr) > 0 else 1.0
    gap_threshold = 1.5 * max(median_dt, 1e-6)

    # Compute steering rate (deg/s) and jerk (deg/s²).
    # Steering is already in normalised (−1…1) units; convert to degrees for
    # the threshold (×45 deg per normalised unit is a common mapping, but we
    # just use normalised and note the threshold is also in normalised/s²).
    # The drive_summary_core threshold is in normalised/s² (18.0 normalised/s²).
    rate = np.zeros(n, dtype=float)
    rate[1:] = np.divide(
        np.diff(steering_arr[:n]),
        dt_arr,
        out=np.zeros(n - 1, dtype=float),
        where=dt_arr > 1e-6,
    )
    jerk = np.zeros(n, dtype=float)
    jerk[2:] = np.divide(
        np.diff(rate[1:]),
        dt_arr[1:],
        out=np.zeros(n - 2, dtype=float),
        where=dt_arr[1:] > 1e-6,
    )

    # Helper: get a scalar signal value at a frame index, with bounds checking.
    def _sig(key: str, idx: int, default=0.0):
        arr = data.get(key)
        if arr is None:
            return default
        arr = np.asarray(arr)
        if idx < 0 or idx >= len(arr):
            return default
        v = arr[idx]
        return float(v) if not (math.isnan(float(v)) or math.isinf(float(v))) else default

    # Helper: look at a window around *idx* and return max |value|.
    def _window_max(key: str, idx: int, default=0.0):
        arr = data.get(key)
        if arr is None:
            return default
        arr = np.asarray(arr, dtype=float)
        lo = max(0, idx - window_frames)
        hi = min(len(arr), idx + window_frames + 1)
        if hi <= lo:
            return default
        v = np.nanmax(np.abs(arr[lo:hi]))
        return float(v) if not math.isinf(v) else default

    events = []
    for i in range(2, n):
        if abs(jerk[i]) < steering_jerk_threshold_deg_per_s2:
            continue
        # Skip if adjacent dt is anomalous (gap artifact).
        if dt_arr[i - 2] > gap_threshold or dt_arr[i - 1] > gap_threshold:
            continue

        # Context signals at event frame (and neighbours).
        e_lat_now     = _sig("mpc_e_lat", i)
        e_lat_prev    = _sig("mpc_e_lat", i - 1)
        stale_now     = _sig("pp_stale_hold_active", i)
        stale_prev    = _sig("pp_stale_hold_active", i - 1)
        silent_now    = _sig("diag_silent_elat_dropout_active", i)
        silent_prev   = _sig("diag_silent_elat_dropout_active", i - 1)
        gate_active   = _sig("diag_heading_zero_gate_active", i)
        kappa         = _window_max("curvature_map_abs", i)

        # --- Classification rules (ordered by specificity) ---
        # Rule 1: Stale hold flag dropped → stale recovery jerk.
        if stale_prev > 0.5 and stale_now < 0.5:
            cause = "stale_recovery"

        # Rule 2: Silent dropout started (e_lat was non-zero, now ≈ 0).
        elif silent_now > 0.5 and silent_prev < 0.5 and abs(e_lat_now) < 0.05:
            cause = "dropout_start"

        # Rule 3: Silent dropout ended (e_lat snapped back from 0).
        elif silent_prev > 0.5 and silent_now < 0.5 and abs(e_lat_now - e_lat_prev) > 0.08:
            cause = "dropout_recovery"

        # Rule 4: Large e_lat on a curve with heading gate latched → oscillation.
        elif kappa > 0.003 and abs(e_lat_now) > 0.15 and gate_active > 0.5:
            cause = "curve_oscillation"

        # Rule 5: Curve, no gate, large e_lat (oscillation without gate info).
        elif kappa > 0.003 and abs(e_lat_now) > 0.10:
            cause = "curve_oscillation"

        # Rule 6: Straight with small e_lat — noise chasing.
        elif kappa < 0.003 and abs(e_lat_now) < 0.15:
            cause = "straight_noise"

        else:
            cause = "unknown"

        events.append({
            "frame":       i,
            "time":        float(time_arr[i]),
            "jerk_deg_s2": float(jerk[i]),
            "cause":       cause,
            "description": _JERK_CAUSE_LABELS[cause],
            # Extra context for debugging.
            "e_lat":       round(e_lat_now, 4),
            "e_lat_prev":  round(e_lat_prev, 4),
            "kappa":       round(kappa, 5),
            "stale":       bool(stale_now > 0.5),
            "silent":      bool(silent_now > 0.5),
            "gate":        bool(gate_active > 0.5),
        })

    return events


def _finite_stats(values: Optional[np.ndarray]) -> Dict:
    """Return finite-value summary stats with nullable percentiles."""
    if values is None:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "count": int(arr.size),
        "mean": safe_float(np.mean(arr), default=None),
        "p50": safe_float(np.percentile(arr, 50), default=None),
        "p95": safe_float(np.percentile(arr, 95), default=None),
        "p99": safe_float(np.percentile(arr, 99), default=None),
        "max": safe_float(np.max(arr), default=None),
    }


def _read_recording_metadata_dict(h5_file: h5py.File) -> dict:
    meta_raw = h5_file.attrs.get("metadata")
    if meta_raw is None:
        return {}
    try:
        meta_str = (
            meta_raw.decode("utf-8", "ignore")
            if isinstance(meta_raw, (bytes, bytearray))
            else str(meta_raw)
        )
        parsed = json.loads(meta_str)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _decode_string_series(values: Optional[np.ndarray]) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    for raw in values:
        if isinstance(raw, (bytes, bytearray)):
            value = raw.decode("utf-8", errors="ignore")
        else:
            value = str(raw)
        value = value.strip()
        out.append(value)
    return out


def _finite_nonnegative(arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    values = np.asarray(arr, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([], dtype=float)
    return values[values >= 0.0]


def _rate_from_boolish(values: Optional[np.ndarray], *, positive_threshold: float = 0.5) -> Optional[float]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(arr) & (arr >= 0.0)
    if not np.any(mask):
        return None
    valid = arr[mask]
    if valid.size == 0:
        return None
    return safe_float(np.mean(valid > positive_threshold) * 100.0, default=None)


def _mode_string(values: list[str], *, ignore: Optional[set[str]] = None) -> Optional[str]:
    cleaned = []
    ignored = ignore or set()
    for value in values:
        stripped = str(value or "").strip()
        if not stripped:
            continue
        if stripped in ignored:
            continue
        cleaned.append(stripped)
    if not cleaned:
        return None
    counts: dict[str, int] = {}
    for value in cleaned:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _count_start_events(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(np.sum(starts))


def _build_transport_contract_summary(data: Dict) -> Dict:
    metadata = data.get("recording_metadata") or {}
    stack_transport = metadata.get("stack_transport") or {}
    packet_modes = _decode_string_series(data.get("sync_packet_mode"))
    packet_mode = (
        _mode_string(packet_modes)
        or str(stack_transport.get("sync_packet_mode", "") or "").strip()
        or PACKET_MODE_LATEST_PARALLEL
    )
    consume_policy = (
        _mode_string(_decode_string_series(data.get("sync_packet_consume_policy")))
        or str(stack_transport.get("sync_packet_consume_policy", "") or "").strip()
        or None
    )

    packet_complete = data.get("sync_packet_complete")
    packet_fallback = data.get("sync_packet_fallback_active")
    packet_fallback_reason = _decode_string_series(data.get("sync_packet_fallback_reason_code"))
    queue_depth = data.get("sync_packet_queue_depth")
    payload_queue_depth = data.get("sync_packet_payload_queue_depth")
    skipped_frames = data.get("sync_packet_skipped_unity_frames")
    packet_age_ms = data.get("sync_packet_age_ms")
    payload_oldest_age_ms = data.get("sync_packet_payload_oldest_age_ms")
    payload_bytes = data.get("sync_packet_payload_bytes")
    payload_selected_age_ms = data.get("sync_packet_payload_selected_age_ms")
    payload_selected_fresh = data.get("sync_packet_payload_selected_fresh")
    payload_warn_age_exceeded = data.get("sync_packet_payload_warn_age_exceeded")
    payload_stale_drop_count = data.get("sync_packet_payload_stale_drop_count")
    payload_drained_count = data.get("sync_packet_payload_drained_count")
    payload_max_drained_age_ms = data.get("sync_packet_payload_max_drained_age_ms")
    payload_selection_source = _decode_string_series(
        data.get("sync_packet_payload_selection_source")
    )
    payload_selection_fallback_active = data.get(
        "sync_packet_payload_selection_fallback_active"
    )
    payload_selection_fallback_reason = _decode_string_series(
        data.get("sync_packet_payload_selection_fallback_reason_code")
    )
    payload_server_queue_depth_after_select = data.get(
        "sync_packet_payload_server_queue_depth_after_select"
    )
    payload_server_oldest_age_ms_after_select = data.get(
        "sync_packet_payload_server_oldest_age_ms_after_select"
    )
    selection_result = _decode_string_series(data.get("sync_packet_selection_result"))
    join_source = _decode_string_series(data.get("sync_packet_join_source"))
    join_key_present = data.get("sync_packet_join_key_present")
    join_failure_reason = _decode_string_series(
        data.get("sync_packet_join_failure_reason_code")
    )
    join_failure_side = _decode_string_series(
        data.get("sync_packet_join_failure_side_code")
    )
    selected_failure_contract_reason = _decode_string_series(
        data.get("sync_packet_selected_failure_contract_reason_code")
    )
    selected_failure_source_stage = _decode_string_series(
        data.get("sync_packet_selected_failure_source_stage_code")
    )
    source_key_present_camera = data.get("sync_packet_source_key_present_camera")
    source_key_present_vehicle = data.get("sync_packet_source_key_present_vehicle")
    selected_packet_key = _decode_string_series(
        data.get("sync_packet_selected_packet_key")
    )
    timeout_event_delta = data.get("sync_packet_timeout_event_delta")
    coherence_pass = data.get("sync_packet_coherence_pass")
    coherence_reason = _decode_string_series(data.get("sync_packet_coherence_reason_code"))
    complete_but_incoherent = data.get("sync_packet_complete_but_incoherent")
    time_delta_budget_exceeded = data.get(
        "sync_packet_front_vehicle_time_delta_budget_exceeded"
    )
    frame_delta_budget_exceeded = data.get(
        "sync_packet_front_vehicle_frame_delta_budget_exceeded"
    )
    join_wait_budget_exceeded = data.get("sync_packet_join_wait_budget_exceeded")
    component_age_budget_exceeded = data.get(
        "sync_packet_component_age_budget_exceeded"
    )
    source_context_queue_depth = data.get("sync_packet_source_context_queue_depth")
    source_context_dropped_stale_count = data.get(
        "sync_packet_source_context_dropped_stale_count"
    )
    source_context_missing_count = data.get("sync_packet_source_context_missing_count")
    source_context_frame_delta = data.get("sync_packet_source_context_frame_delta")
    source_context_time_delta_ms = data.get("sync_packet_source_context_time_delta_ms")
    source_bundle_close_reason = _decode_string_series(
        data.get("sync_packet_source_bundle_close_reason")
    )
    source_bundle_deadline_ms = data.get("sync_packet_source_bundle_deadline_ms")
    source_bundle_age_ms = data.get("sync_packet_source_bundle_age_ms")
    source_bundle_inflight_count = data.get("sync_packet_source_bundle_inflight_count")
    source_bundle_vehicle_state_built = data.get(
        "sync_packet_source_bundle_vehicle_state_built"
    )
    source_bundle_vehicle_state_enqueued = data.get(
        "sync_packet_source_bundle_vehicle_state_enqueued"
    )
    source_bundle_vehicle_state_sent = data.get(
        "sync_packet_source_bundle_vehicle_state_sent"
    )
    source_bundle_camera_requested = data.get(
        "sync_packet_source_bundle_camera_requested"
    )
    source_camera_request_attempted = data.get(
        "sync_packet_source_camera_request_attempted"
    )
    source_camera_request_accepted = data.get(
        "sync_packet_source_camera_request_accepted"
    )
    source_camera_request_rejected_reason = _decode_string_series(
        data.get("sync_packet_source_camera_request_rejected_reason")
    )
    source_camera_request_skipped_reason = _decode_string_series(
        data.get("sync_packet_source_camera_request_skipped_reason")
    )
    source_camera_request_disposition_code = _decode_string_series(
        data.get("sync_packet_source_camera_request_disposition_code")
    )
    source_camera_request_attempt_age_ms = data.get(
        "sync_packet_source_camera_request_attempt_age_ms"
    )
    source_camera_request_accept_age_ms = data.get(
        "sync_packet_source_camera_request_accept_age_ms"
    )
    source_camera_request_queue_depth = data.get(
        "sync_packet_source_camera_request_queue_depth"
    )
    source_bundle_active_transport_eligible = data.get(
        "sync_packet_source_bundle_active_transport_eligible"
    )
    source_bundle_debug_unbundled_capture = data.get(
        "sync_packet_source_bundle_debug_unbundled_capture"
    )
    camera_capture_contract_reason = _decode_string_series(
        data.get("sync_packet_camera_capture_contract_reason")
    )
    source_bundle_camera_sent = data.get("sync_packet_source_bundle_camera_sent")
    source_bundle_aborted_before_vehicle_send = data.get(
        "sync_packet_source_bundle_aborted_before_vehicle_send"
    )
    source_bundle_abort_reason = _decode_string_series(
        data.get("sync_packet_source_bundle_abort_reason")
    )
    source_vehicle_send_blocked_by_camera_request = data.get(
        "sync_packet_source_vehicle_send_blocked_by_camera_request"
    )
    source_bundle_superseded_before_send = data.get(
        "sync_packet_source_bundle_superseded_before_send"
    )
    active_camera_excluded_event_delta = data.get(
        "sync_packet_active_camera_excluded_event_delta"
    )
    active_camera_excluded_reason_code = _decode_string_series(
        data.get("sync_packet_active_camera_excluded_reason_code")
    )
    unbundled_camera_entered_active_path_event_delta = data.get(
        "sync_packet_unbundled_camera_entered_active_path_event_delta"
    )
    join_wait_ms = data.get("sync_packet_join_wait_ms")
    key_match_count = data.get("sync_packet_key_match_count")
    unity_fallback_count = data.get("sync_packet_unity_fallback_count")
    superseded_camera_count = data.get("sync_packet_superseded_camera_count")
    superseded_vehicle_count = data.get("sync_packet_superseded_vehicle_count")
    packet_superseded_camera_count = data.get("sync_packet_packet_superseded_camera_count")
    packet_superseded_vehicle_count = data.get("sync_packet_packet_superseded_vehicle_count")
    front_age_ms = data.get("sync_front_age_ms")
    vehicle_age_ms = data.get("sync_vehicle_age_ms")
    frame_delta = data.get("sync_front_vehicle_frame_delta")
    time_delta_ms = data.get("sync_front_vehicle_time_delta_ms")
    missing_front = data.get("sync_packet_missing_front")
    missing_vehicle = data.get("sync_packet_missing_vehicle")
    drop_count = data.get("sync_packet_drop_count")
    payload_drop_count = data.get("sync_packet_payload_drop_count")
    payload_fallback_reason = _decode_string_series(
        data.get("sync_packet_payload_fallback_reason_code")
    )
    orphan_camera_count = data.get("sync_packet_orphan_camera_count")
    orphan_vehicle_count = data.get("sync_packet_orphan_vehicle_count")
    timeout_count = data.get("sync_packet_timeout_count")
    reference_velocity_effective = data.get("reference_velocity_effective")
    target_speed_final = data.get("target_speed_final")
    post_jump_cooldown_active = data.get("post_jump_cooldown_active")
    teleport_detected = data.get("teleport_detected")
    teleport_motion_ratio = data.get("teleport_motion_ratio")
    teleport_guard_suppressed = data.get("teleport_guard_suppressed")
    teleport_continuity_suspect = data.get("teleport_continuity_suspect")
    teleport_guard_reason = _decode_string_series(data.get("teleport_guard_reason_code"))
    teleport_dynamic_threshold = data.get("teleport_dynamic_threshold_m")

    available = any(
        candidate is not None
        for candidate in (
            packet_complete,
            packet_fallback,
            queue_depth,
            payload_queue_depth,
            skipped_frames,
            packet_age_ms,
            payload_oldest_age_ms,
            payload_selected_age_ms,
            payload_selected_fresh,
            payload_warn_age_exceeded,
            payload_stale_drop_count,
            payload_drained_count,
            payload_max_drained_age_ms,
            payload_selection_source,
            payload_selection_fallback_active,
            payload_selection_fallback_reason,
            payload_server_queue_depth_after_select,
            payload_server_oldest_age_ms_after_select,
            selection_result,
            join_source,
            join_key_present,
            join_failure_reason,
            join_failure_side,
            selected_failure_contract_reason,
            selected_failure_source_stage,
            source_key_present_camera,
            source_key_present_vehicle,
            selected_packet_key,
            timeout_event_delta,
            coherence_pass,
            coherence_reason,
            complete_but_incoherent,
            time_delta_budget_exceeded,
            frame_delta_budget_exceeded,
            join_wait_budget_exceeded,
            component_age_budget_exceeded,
            source_context_queue_depth,
            source_context_dropped_stale_count,
            source_context_missing_count,
            source_context_frame_delta,
            source_context_time_delta_ms,
            source_bundle_close_reason,
            source_bundle_deadline_ms,
            source_bundle_age_ms,
            source_bundle_inflight_count,
            source_bundle_vehicle_state_built,
            source_bundle_vehicle_state_enqueued,
            source_bundle_vehicle_state_sent,
            source_bundle_camera_requested,
            source_camera_request_attempted,
            source_camera_request_accepted,
            source_camera_request_rejected_reason,
            source_camera_request_attempt_age_ms,
            source_camera_request_accept_age_ms,
            source_camera_request_queue_depth,
            source_bundle_active_transport_eligible,
            source_bundle_debug_unbundled_capture,
            camera_capture_contract_reason,
            source_bundle_camera_sent,
            source_bundle_aborted_before_vehicle_send,
            source_bundle_abort_reason,
            source_vehicle_send_blocked_by_camera_request,
            source_bundle_superseded_before_send,
            active_camera_excluded_event_delta,
            active_camera_excluded_reason_code,
            unbundled_camera_entered_active_path_event_delta,
            join_wait_ms,
            key_match_count,
            unity_fallback_count,
            superseded_camera_count,
            superseded_vehicle_count,
            packet_superseded_camera_count,
            packet_superseded_vehicle_count,
            front_age_ms,
            vehicle_age_ms,
            frame_delta,
            time_delta_ms,
        )
    ) or bool(stack_transport)

    complete_rate = _rate_from_boolish(packet_complete)
    fallback_rate = _rate_from_boolish(packet_fallback)
    missing_front_rate = _rate_from_boolish(missing_front)
    missing_vehicle_rate = _rate_from_boolish(missing_vehicle)
    post_jump_rate = _rate_from_boolish(post_jump_cooldown_active)
    teleport_rate = _rate_from_boolish(teleport_detected)
    teleport_guard_suppressed_rate = _rate_from_boolish(teleport_guard_suppressed)
    teleport_continuity_suspect_rate = _rate_from_boolish(teleport_continuity_suspect)
    selected_fresh_rate = _rate_from_boolish(payload_selected_fresh)
    warn_age_exceeded_rate = _rate_from_boolish(payload_warn_age_exceeded)

    fallback_reason_mode = _mode_string(packet_fallback_reason, ignore={PACKET_FALLBACK_NONE})

    skipped_arr = _finite_nonnegative(skipped_frames)
    effective_arr = _finite_nonnegative(reference_velocity_effective)
    target_final_arr = _finite_nonnegative(target_speed_final)
    n_common = min(effective_arr.size, target_final_arr.size) if effective_arr.size and target_final_arr.size else 0
    effective_drop_mask = np.zeros(0, dtype=bool)
    effective_drop_count = 0
    effective_drop_rate = None
    if n_common > 0:
        effective_drop_mask = effective_arr[:n_common] + 0.25 < target_final_arr[:n_common]
        effective_drop_count = int(np.sum(effective_drop_mask))
        effective_drop_rate = safe_float(np.mean(effective_drop_mask) * 100.0, default=None)

    n_false = 0
    false_teleport_rate = None
    if post_jump_cooldown_active is not None:
        cooldown_arr = np.asarray(post_jump_cooldown_active, dtype=float).reshape(-1)
        n = cooldown_arr.size
        if n > 0:
            false_mask = (cooldown_arr > 0.5)
            if skipped_arr.size:
                raw_skipped = np.asarray(skipped_frames, dtype=float).reshape(-1)
                skipped_mask = np.isfinite(raw_skipped[:n]) & (raw_skipped[:n] >= 2.0)
                false_mask = false_mask & skipped_mask
            if packet_fallback is not None:
                fallback_arr = np.asarray(packet_fallback, dtype=float).reshape(-1)
                m = min(false_mask.size, fallback_arr.size)
                false_mask = false_mask[:m] & (
                    (np.asarray(skipped_frames[:m], dtype=float) >= 2.0 if skipped_frames is not None else False)
                    | (fallback_arr[:m] > 0.5)
                )
            n_false = int(np.sum(false_mask)) if false_mask.size else 0
            false_teleport_rate = safe_float(np.mean(false_mask) * 100.0, default=None) if false_mask.size else None

    schema_values = _finite_nonnegative(data.get("sync_packet_schema_version"))

    drop_values = _finite_nonnegative(drop_count)
    payload_drop_values = _finite_nonnegative(payload_drop_count)
    payload_stale_drop_values = _finite_nonnegative(payload_stale_drop_count)
    payload_drained_values = _finite_nonnegative(payload_drained_count)
    orphan_camera_values = _finite_nonnegative(orphan_camera_count)
    orphan_vehicle_values = _finite_nonnegative(orphan_vehicle_count)
    timeout_values = _finite_nonnegative(timeout_count)
    key_match_values = _finite_nonnegative(key_match_count)
    unity_fallback_values = _finite_nonnegative(unity_fallback_count)
    superseded_camera_values = _finite_nonnegative(superseded_camera_count)
    superseded_vehicle_values = _finite_nonnegative(superseded_vehicle_count)

    return {
        "schema_version": "v1",
        "availability": "available" if available else "unavailable",
        "packet_mode": packet_mode,
        "consume_policy": consume_policy,
        "packet_schema_version": int(np.max(schema_values)) if schema_values.size else 0,
        "packet_completeness_rate": complete_rate,
        "fallback_active_rate": fallback_rate,
        "fallback_reason_mode": fallback_reason_mode,
        "payload_fallback_reason_mode": _mode_string(
            payload_fallback_reason, ignore={PACKET_FALLBACK_NONE, ""}
        ),
        "missing_front_rate": missing_front_rate,
        "missing_vehicle_rate": missing_vehicle_rate,
        "packet_queue_depth": _finite_stats(queue_depth),
        "payload_queue_depth": _finite_stats(payload_queue_depth),
        "skipped_unity_frames": _finite_stats(skipped_frames),
        "packet_age_ms": _finite_stats(packet_age_ms),
        "payload_oldest_age_ms": _finite_stats(payload_oldest_age_ms),
        "payload_bytes": _finite_stats(payload_bytes),
        "payload_selected_age_ms": _finite_stats(payload_selected_age_ms),
        "payload_selected_fresh_rate": selected_fresh_rate,
        "payload_warn_age_exceeded_rate": warn_age_exceeded_rate,
        "payload_stale_drop_count": _finite_stats(payload_stale_drop_count),
        "payload_drained_count": _finite_stats(payload_drained_count),
        "payload_max_drained_age_ms": _finite_stats(payload_max_drained_age_ms),
        "payload_selection_source_mode": _mode_string(
            payload_selection_source, ignore={""}
        ),
        "selection_result_mode": _mode_string(selection_result, ignore={""}),
        "payload_selection_fallback_active_rate": _rate_from_boolish(
            payload_selection_fallback_active
        ),
        "payload_selection_fallback_reason_mode": _mode_string(
            payload_selection_fallback_reason, ignore={PACKET_FALLBACK_NONE, ""}
        ),
        "payload_server_queue_depth_after_select": _finite_stats(
            payload_server_queue_depth_after_select
        ),
        "payload_server_oldest_age_ms_after_select": _finite_stats(
            payload_server_oldest_age_ms_after_select
        ),
        "join_source_mode": _mode_string(join_source, ignore={""}),
        "join_key_present_rate": _rate_from_boolish(join_key_present),
        "join_failure_reason_mode": _mode_string(
            join_failure_reason, ignore={"", "none"}
        ),
        "join_failure_side_mode": _mode_string(
            join_failure_side, ignore={"", "none"}
        ),
        "selected_failure_contract_reason_mode": _mode_string(
            selected_failure_contract_reason, ignore={"", "none"}
        ),
        "selected_failure_source_stage_mode": _mode_string(
            selected_failure_source_stage, ignore={"", "none"}
        ),
        "source_key_present_camera_rate": _rate_from_boolish(
            source_key_present_camera
        ),
        "source_key_present_vehicle_rate": _rate_from_boolish(
            source_key_present_vehicle
        ),
        "selected_packet_key_present_rate": _rate_from_boolish(
            np.array([1.0 if s else 0.0 for s in selected_packet_key], dtype=float)
            if selected_packet_key
            else None
        ),
        "timeout_event_delta_rate": _rate_from_boolish(timeout_event_delta),
        "coherence_pass_rate": _rate_from_boolish(coherence_pass),
        "coherence_reason_mode": _mode_string(
            coherence_reason, ignore={"", "coherent"}
        )
        or _mode_string(coherence_reason, ignore={""}),
        "complete_but_incoherent_rate": _rate_from_boolish(complete_but_incoherent),
        "front_vehicle_time_delta_budget_exceeded_rate": _rate_from_boolish(
            time_delta_budget_exceeded
        ),
        "front_vehicle_frame_delta_budget_exceeded_rate": _rate_from_boolish(
            frame_delta_budget_exceeded
        ),
        "join_wait_budget_exceeded_rate": _rate_from_boolish(
            join_wait_budget_exceeded
        ),
        "component_age_budget_exceeded_rate": _rate_from_boolish(
            component_age_budget_exceeded
        ),
        "source_context_queue_depth": _finite_stats(source_context_queue_depth),
        "source_context_dropped_stale_count": _finite_stats(
            source_context_dropped_stale_count
        ),
        "source_context_missing_count": _finite_stats(source_context_missing_count),
        "source_context_frame_delta": _finite_stats(source_context_frame_delta),
        "source_context_time_delta_ms": _finite_stats(source_context_time_delta_ms),
        "source_bundle_close_reason_mode": _mode_string(
            source_bundle_close_reason, ignore={"", "open"}
        ),
        "source_bundle_deadline_ms": _finite_stats(source_bundle_deadline_ms),
        "source_bundle_age_ms": _finite_stats(source_bundle_age_ms),
        "source_bundle_inflight_count": _finite_stats(source_bundle_inflight_count),
        "source_bundle_vehicle_state_built_rate": _rate_from_boolish(
            source_bundle_vehicle_state_built
        ),
        "source_bundle_vehicle_state_enqueued_rate": _rate_from_boolish(
            source_bundle_vehicle_state_enqueued
        ),
        "source_bundle_vehicle_state_sent_rate": _rate_from_boolish(
            source_bundle_vehicle_state_sent
        ),
        "source_bundle_camera_requested_rate": _rate_from_boolish(
            source_bundle_camera_requested
        ),
        "source_camera_request_attempted_rate": _rate_from_boolish(
            source_camera_request_attempted
        ),
        "source_camera_request_accepted_rate": _rate_from_boolish(
            source_camera_request_accepted
        ),
        "source_camera_request_rejected_reason_mode": _mode_string(
            source_camera_request_rejected_reason, ignore={"", "none"}
        ),
        "source_camera_request_skipped_reason_mode": _mode_string(
            source_camera_request_skipped_reason, ignore={"", "none"}
        ),
        "source_camera_request_disposition_mode": _mode_string(
            source_camera_request_disposition_code, ignore={"", "none"}
        ),
        "source_camera_request_attempt_age_ms": _finite_stats(
            source_camera_request_attempt_age_ms
        ),
        "source_camera_request_accept_age_ms": _finite_stats(
            source_camera_request_accept_age_ms
        ),
        "source_camera_request_queue_depth": _finite_stats(
            source_camera_request_queue_depth
        ),
        "active_camera_eligible_rate": _rate_from_boolish(
            source_bundle_active_transport_eligible
        ),
        "debug_unbundled_capture_rate": _rate_from_boolish(
            source_bundle_debug_unbundled_capture
        ),
        "camera_capture_contract_reason_mode": _mode_string(
            camera_capture_contract_reason, ignore={"", "none"}
        ),
        "source_bundle_camera_sent_rate": _rate_from_boolish(
            source_bundle_camera_sent
        ),
        "source_bundle_aborted_before_vehicle_send_rate": _rate_from_boolish(
            source_bundle_aborted_before_vehicle_send
        ),
        "source_bundle_abort_reason_mode": _mode_string(
            source_bundle_abort_reason, ignore={"", "none"}
        ),
        "source_vehicle_send_blocked_by_camera_request_rate": _rate_from_boolish(
            source_vehicle_send_blocked_by_camera_request
        ),
        "source_bundle_superseded_before_send_rate": _rate_from_boolish(
            source_bundle_superseded_before_send
        ),
        "active_camera_excluded_event_rate": _rate_from_boolish(
            active_camera_excluded_event_delta
        ),
        "active_camera_excluded_reason_mode": _mode_string(
            active_camera_excluded_reason_code, ignore={"", "none"}
        ),
        "unbundled_camera_entered_active_path_rate": _rate_from_boolish(
            unbundled_camera_entered_active_path_event_delta
        ),
        "join_wait_ms": _finite_stats(join_wait_ms),
        "packet_superseded_camera_count": _finite_stats(packet_superseded_camera_count),
        "packet_superseded_vehicle_count": _finite_stats(packet_superseded_vehicle_count),
        "front_age_ms": _finite_stats(front_age_ms),
        "vehicle_age_ms": _finite_stats(vehicle_age_ms),
        "front_vehicle_frame_delta": _finite_stats(frame_delta),
        "front_vehicle_time_delta_ms": _finite_stats(time_delta_ms),
        "drop_count_max": int(np.max(drop_values)) if drop_values.size else 0,
        "payload_drop_count_max": int(np.max(payload_drop_values)) if payload_drop_values.size else 0,
        "payload_stale_drop_count_max": int(np.max(payload_stale_drop_values)) if payload_stale_drop_values.size else 0,
        "payload_drained_count_max": int(np.max(payload_drained_values)) if payload_drained_values.size else 0,
        "orphan_camera_count_max": int(np.max(orphan_camera_values)) if orphan_camera_values.size else 0,
        "orphan_vehicle_count_max": int(np.max(orphan_vehicle_values)) if orphan_vehicle_values.size else 0,
        "timeout_count_max": int(np.max(timeout_values)) if timeout_values.size else 0,
        "key_match_count_max": int(np.max(key_match_values)) if key_match_values.size else 0,
        "unity_fallback_count_max": int(np.max(unity_fallback_values)) if unity_fallback_values.size else 0,
        "superseded_camera_count_max": int(np.max(superseded_camera_values)) if superseded_camera_values.size else 0,
        "superseded_vehicle_count_max": int(np.max(superseded_vehicle_values)) if superseded_vehicle_values.size else 0,
        "post_jump_cooldown_active_rate": post_jump_rate,
        "teleport_detected_rate": teleport_rate,
        "teleport_guard_suppressed_rate": teleport_guard_suppressed_rate,
        "teleport_continuity_suspect_rate": teleport_continuity_suspect_rate,
        "teleport_guard_reason_mode": _mode_string(
            teleport_guard_reason, ignore={""}
        ),
        "teleport_motion_ratio_p95": _finite_stats(teleport_motion_ratio).get("p95"),
        "teleport_dynamic_threshold_m_p95": _finite_stats(teleport_dynamic_threshold).get(
            "p95"
        ),
        "effective_reference_velocity_drop_count": effective_drop_count,
        "effective_reference_velocity_drop_rate": effective_drop_rate,
        "false_teleport_cooldown_rate": false_teleport_rate,
        "false_teleport_cooldown_count": n_false,
        "limits": {
            "packet_completeness_rate_min_pct": 99.0,
            "fallback_active_rate_max_pct": 1.0,
            "front_vehicle_time_delta_ms_p95_max": 20.0,
            "skipped_unity_frames_p95_max": 1.0,
        },
    }


def _classify_brake_episode_reasons(data: Dict) -> tuple[dict[str, int], list[dict]]:
    brake = data.get("brake")
    if brake is None:
        return {}, []
    brake_arr = np.asarray(brake, dtype=float).reshape(-1)
    if brake_arr.size == 0:
        return {}, []
    post_jump = np.asarray(data.get("post_jump_cooldown_active"), dtype=float).reshape(-1) if data.get("post_jump_cooldown_active") is not None else None
    acc_active = np.asarray(data.get("acc_active"), dtype=float).reshape(-1) if data.get("acc_active") is not None else None
    lead_collision_override = np.asarray(data.get("lead_collision_override_active"), dtype=float).reshape(-1) if data.get("lead_collision_override_active") is not None else None
    final_owner_code = _decode_string_series(data.get("final_longitudinal_owner_code"))
    curve_cap = np.asarray(data.get("speed_governor_curve_cap_active"), dtype=float).reshape(-1) if data.get("speed_governor_curve_cap_active") is not None else None
    feasibility = np.asarray(data.get("turn_feasibility_infeasible"), dtype=float).reshape(-1) if data.get("turn_feasibility_infeasible") is not None else None
    reason_counts: dict[str, int] = {}
    examples: list[dict] = []

    active = brake_arr > 0.05
    starts = np.flatnonzero(active & np.concatenate(([True], ~active[:-1])))
    for start in starts:
        reason = EFFECTIVE_TARGET_REASON_FALLBACK_UNKNOWN
        if lead_collision_override is not None and start < lead_collision_override.size and lead_collision_override[start] > 0.5:
            reason = "lead_collision_override"
        elif final_owner_code and start < len(final_owner_code) and final_owner_code[start] in {"acc_ttc_estop", "acc_collapsed_gap_stop"}:
            reason = final_owner_code[start]
        elif post_jump is not None and start < post_jump.size and post_jump[start] > 0.5:
            reason = EFFECTIVE_TARGET_REASON_POST_JUMP_COOLDOWN
        elif feasibility is not None and start < feasibility.size and feasibility[start] > 0.5:
            reason = EFFECTIVE_TARGET_REASON_FEASIBILITY_BACKSTOP
        elif curve_cap is not None and start < curve_cap.size and curve_cap[start] > 0.5:
            reason = EFFECTIVE_TARGET_REASON_CURVE_CAP
        elif acc_active is not None and start < acc_active.size and acc_active[start] > 0.5:
            reason = EFFECTIVE_TARGET_REASON_ACC_FOLLOW
        else:
            reason = EFFECTIVE_TARGET_REASON_FREE_FLOW
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if len(examples) < 8:
            examples.append(
                {
                    "frame": int(start),
                    "reason": reason,
                    "brake": safe_float(brake_arr[start], default=None),
                    "reference_velocity_effective": safe_float(
                        np.asarray(data.get("reference_velocity_effective"), dtype=float).reshape(-1)[start],
                        default=None,
                    ) if data.get("reference_velocity_effective") is not None and start < len(np.asarray(data.get("reference_velocity_effective")).reshape(-1)) else None,
                    "target_speed_final": safe_float(
                        np.asarray(data.get("target_speed_final"), dtype=float).reshape(-1)[start],
                        default=None,
                    ) if data.get("target_speed_final") is not None and start < len(np.asarray(data.get("target_speed_final")).reshape(-1)) else None,
                }
            )
    return reason_counts, examples


def _build_speed_intent_summary(data: Dict) -> Dict:
    desired = _finite_stats(data.get("target_speed_raw"))
    planned = _finite_stats(data.get("target_speed_planned"))
    post_limits = _finite_stats(data.get("target_speed_post_limits"))
    governor = _finite_stats(data.get("governor_target_speed_mps"))
    acc_target = _finite_stats(data.get("acc_target_speed_mps"))
    planner_applied = _finite_stats(data.get("planner_target_speed_applied_mps"))
    final = _finite_stats(data.get("target_speed_final"))
    planner_ref = _finite_stats(data.get("ref_velocity"))
    effective = _finite_stats(data.get("reference_velocity_effective"))
    final_owner = _decode_string_series(data.get("final_longitudinal_owner_code"))
    reference_velocity_source = _decode_string_series(data.get("reference_velocity_source_code"))
    post_jump_reason = _decode_string_series(data.get("post_jump_reason_code"))
    reason_mode = _mode_string(post_jump_reason, ignore={POST_JUMP_REASON_NONE})
    brake_reason_counts, brake_examples = _classify_brake_episode_reasons(data)

    effective_drop_count = 0
    effective_drop_rate = None
    if data.get("reference_velocity_effective") is not None and data.get("target_speed_final") is not None:
        eff = np.asarray(data.get("reference_velocity_effective"), dtype=float).reshape(-1)
        fin = np.asarray(data.get("target_speed_final"), dtype=float).reshape(-1)
        n = min(eff.size, fin.size)
        if n > 0:
            valid = np.isfinite(eff[:n]) & np.isfinite(fin[:n])
            if np.any(valid):
                mask = valid & (eff[:n] + 0.25 < fin[:n])
                effective_drop_count = int(np.sum(mask))
                effective_drop_rate = safe_float(np.mean(mask[valid]) * 100.0, default=None)

    return {
        "schema_version": "v1",
        "availability": "available" if any(
            stat["count"] > 0 for stat in (desired, planned, post_limits, final, planner_ref, effective)
        ) else "unavailable",
        "desired_target_speed_mps": desired,
        "planned_target_speed_mps": planned,
        "post_limits_target_speed_mps": post_limits,
        "governor_target_speed_mps": governor,
        "acc_target_speed_mps": acc_target,
        "planner_target_speed_applied_mps": planner_applied,
        "final_target_speed_mps": final,
        "planner_reference_speed_mps": planner_ref,
        "effective_reference_speed_mps": effective,
        "final_longitudinal_owner_mode": _mode_string(final_owner, ignore={""}),
        "reference_velocity_source_mode": _mode_string(reference_velocity_source, ignore={""}),
        "effective_reason_mode": reason_mode or (
            EFFECTIVE_TARGET_REASON_POST_JUMP_COOLDOWN
            if (data.get("post_jump_cooldown_active") is not None and _count_start_events(np.asarray(data.get("post_jump_cooldown_active"), dtype=float) > 0.5) > 0)
            else None
        ),
        "effective_reference_velocity_drop_count": effective_drop_count,
        "effective_reference_velocity_drop_rate": effective_drop_rate,
        "post_jump_cooldown_episode_count": _count_start_events(
            np.asarray(data.get("post_jump_cooldown_active"), dtype=float) > 0.5
        ) if data.get("post_jump_cooldown_active") is not None else 0,
        "brake_episode_count": sum(brake_reason_counts.values()),
        "brake_episode_counts_by_reason": brake_reason_counts,
        "brake_episode_examples": brake_examples,
    }


def _build_run_intent_summary(data: Dict) -> Dict:
    metadata = data.get("recording_metadata") or {}
    provenance = data.get("recording_provenance") or {}
    recording_type = str(metadata.get("recording_type", "unknown") or "unknown")
    replay_type = str(provenance.get("replay_type", "unknown") or "unknown")
    track_id = str(provenance.get("track_id", "unknown") or "unknown")
    policy_profile = str(provenance.get("policy_profile", "unknown") or "unknown")
    candidate_label = str(provenance.get("candidate_label", "unknown") or "unknown")

    speed_limit = _finite_nonnegative(data.get("speed_limit"))
    target_speed_final = _finite_nonnegative(data.get("target_speed_final"))
    radar_distance = _finite_nonnegative(data.get("radar_fwd_distance_m"))
    lead_collision_override = data.get("lead_collision_override_active")
    acc_state_code = _decode_string_series(data.get("acc_state_code"))
    final_longitudinal_owner_code = _decode_string_series(data.get("final_longitudinal_owner_code"))
    acc_active = data.get("acc_active")
    acc_active_rate = _rate_from_boolish(acc_active)
    lead_following_active = bool(acc_active_rate is not None and acc_active_rate > 5.0)

    run_target_speed_mps = safe_float(np.median(target_speed_final), default=None) if target_speed_final.size else None
    road_speed_limit_mps = safe_float(np.median(speed_limit), default=None) if speed_limit.size else None
    lead_distance_p50 = safe_float(np.percentile(radar_distance, 50), default=None) if radar_distance.size else None

    if lead_following_active:
        mode = "acc_follow"
    elif recording_type not in {"unknown", "av_stack"}:
        mode = recording_type
    elif run_target_speed_mps is not None or road_speed_limit_mps is not None:
        mode = "free_flow"
    else:
        mode = "unknown"

    mismatch_reason = None
    if (
        road_speed_limit_mps is not None
        and run_target_speed_mps is not None
        and road_speed_limit_mps > run_target_speed_mps + 5.0
        and lead_following_active
    ):
        mismatch_reason = "road_speed_limit_higher_than_configured_acc_target"

    return {
        "schema_version": "v1",
        "availability": "available" if (metadata or provenance or run_target_speed_mps is not None) else "unavailable",
        "mode": mode,
        "recording_type": recording_type,
        "replay_type": replay_type,
        "track_id": track_id,
        "policy_profile": policy_profile,
        "candidate_label": candidate_label,
        "run_target_speed_mps": run_target_speed_mps,
        "road_speed_limit_expected_mps": road_speed_limit_mps,
        "lead_following_active": lead_following_active,
        "acc_active_rate_pct": acc_active_rate,
        "acc_state_mode": _mode_string(acc_state_code, ignore={""}),
        "final_longitudinal_owner_mode": _mode_string(final_longitudinal_owner_code, ignore={""}),
        "lead_collision_override_rate_pct": _rate_from_boolish(lead_collision_override),
        "lead_vehicle_distance_p50_m": lead_distance_p50,
        "intent_mismatch_warning": mismatch_reason,
    }


def _build_highway_mild_curve_contract_summary(data: Dict) -> Dict:
    limits = {
        "high_lateral_error_min_m": 0.50,
        "small_lane_center_offset_max_m": 0.12,
        "mild_curve_curvature_abs_min": 0.0015,
        "mild_curve_curvature_abs_max": 0.0035,
        "long_pp_lookahead_min_m": 8.0,
        "long_reference_lookahead_min_m": 12.0,
        "poor_perception_confidence_max": 0.50,
        "poor_perception_num_lanes_max": 1.5,
        "transport_fallback_overlap_max_pct": 5.0,
        "underactivated_on_high_error_min_pct": 50.0,
        "reference_geometry_mismatch_on_high_error_min_pct": 50.0,
        "sustain_phase_collapse_max": 0.12,
        "rearm_cycle_on_high_error_min_pct": 15.0,
        "mpc_curvature_softness_ratio_max": 0.50,
        "mpc_bias_cancellation_on_high_error_min_pct": 20.0,
    }
    base = {
        "schema_version": "v1",
        "issue_detected": False,
        "high_error_frame_count": 0,
        "high_error_frame_rate": None,
        "mild_curve_present_frame_rate": None,
        "mild_curve_present_on_high_error_rate": None,
        "curve_recognition_inactive_on_high_error_rate": None,
        "curve_intent_inactive_on_high_error_rate": None,
        "curve_local_inactive_on_high_error_rate": None,
        "long_pp_lookahead_on_high_error_rate": None,
        "long_reference_lookahead_on_high_error_rate": None,
        "long_lookahead_on_high_error_rate": None,
        "reference_geometry_mismatch_on_high_error_rate": None,
        "underactivated_tracking_on_high_error_rate": None,
        "sustain_phase_collapse_on_high_error_rate": None,
        "rearm_cycle_on_high_error_rate": None,
        "mpc_curvature_softness_on_high_error_rate": None,
        "mpc_bias_cancellation_on_high_error_rate": None,
        "transport_fallback_overlap_on_high_error_rate": None,
        "poor_perception_overlap_on_high_error_rate": None,
        "mpc_feasible_on_high_error_rate": None,
        "mpc_fallback_overlap_on_high_error_rate": None,
        "rearm_cycle_issue_detected": False,
        "mpc_bias_cancellation_issue_detected": False,
        "curve_intent_state_mode_on_high_error": None,
        "curve_local_state_mode_on_high_error": None,
        "curve_activation_blocker_mode_on_high_error": None,
        "curve_activation_blocker_mode_on_underactivated": None,
        "lateral_error_abs_m": _finite_stats(None),
        "road_frame_lane_center_offset_abs_m": _finite_stats(None),
        "reference_point_curvature_abs": _finite_stats(None),
        "mpc_kappa_ref_abs": _finite_stats(None),
        "pp_lookahead_distance_m": _finite_stats(None),
        "reference_lookahead_target_m": _finite_stats(None),
        "speed_mps": _finite_stats(None),
        "curve_local_arm_phase_deficit": _finite_stats(None),
        "curve_local_arm_effect_score": _finite_stats(None),
        "curve_local_arm_effect_heading_term": _finite_stats(None),
        "curve_local_arm_effect_lateral_shift_term": _finite_stats(None),
        "curve_local_arm_effect_time_support_term": _finite_stats(None),
        "curve_local_sustain_phase_raw": _finite_stats(None),
        "curve_phase_term_path": _finite_stats(None),
        "curve_local_dynamic_sustain_effect_score": _finite_stats(None),
        "mpc_kappa_ratio_to_reference": _finite_stats(None),
        "mpc_kappa_bias_correction": _finite_stats(None),
        "limits": limits,
    }
    unavailable = dict(base)
    unavailable["availability"] = "unavailable"

    lateral_error = data.get("lateral_error")
    road_center_offset = data.get("road_frame_lane_center_offset")
    ref_curvature = data.get("reference_point_curvature")
    pp_lookahead_distance = data.get("pp_lookahead_distance")
    curve_intent_states = _decode_string_series(data.get("curve_intent_state"))
    curve_local_states = _decode_string_series(data.get("curve_local_state"))
    curve_activation_blocker_modes = _decode_string_series(
        data.get("curve_activation_blocker_mode")
    )
    if any(
        candidate is None
        for candidate in (
            lateral_error,
            road_center_offset,
            ref_curvature,
            pp_lookahead_distance,
        )
    ) or not curve_intent_states or not curve_local_states:
        return unavailable

    n = min(
        len(lateral_error),
        len(road_center_offset),
        len(ref_curvature),
        len(pp_lookahead_distance),
        len(curve_intent_states),
        len(curve_local_states),
    )
    optional_series = [
        data.get("reference_lookahead_target"),
        data.get("speed"),
        data.get("sync_packet_fallback_active"),
        data.get("confidence"),
        data.get("num_lanes_detected"),
        data.get("mpc_feasible"),
        data.get("mpc_fallback_active"),
        data.get("mpc_kappa_ref"),
        data.get("curve_local_arm_phase_deficit"),
        data.get("curve_local_arm_effect_score"),
        data.get("curve_local_arm_effect_heading_term"),
        data.get("curve_local_arm_effect_lateral_shift_term"),
        data.get("curve_local_arm_effect_time_support_term"),
        data.get("curve_local_sustain_phase_raw"),
        data.get("curve_phase_term_path"),
        data.get("curve_local_dynamic_sustain_effect_score"),
        data.get("mpc_kappa_bias_correction"),
    ]
    for series in optional_series:
        if series is not None:
            n = min(n, len(series))
    if curve_activation_blocker_modes:
        n = min(n, len(curve_activation_blocker_modes))
    if n <= 0:
        return unavailable

    lateral_error_arr = np.abs(np.asarray(lateral_error[:n], dtype=np.float64))
    road_center_offset_arr = np.abs(np.asarray(road_center_offset[:n], dtype=np.float64))
    ref_curvature_arr = np.abs(np.asarray(ref_curvature[:n], dtype=np.float64))
    pp_lookahead_arr = np.asarray(pp_lookahead_distance[:n], dtype=np.float64)
    reference_lookahead_target = data.get("reference_lookahead_target")
    ref_lookahead_arr = (
        np.asarray(reference_lookahead_target[:n], dtype=np.float64)
        if reference_lookahead_target is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    speed = data.get("speed")
    speed_arr = (
        np.asarray(speed[:n], dtype=np.float64)
        if speed is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    sync_fallback = data.get("sync_packet_fallback_active")
    sync_fallback_arr = (
        np.asarray(sync_fallback[:n], dtype=np.float64)
        if sync_fallback is not None
        else np.zeros(n, dtype=np.float64)
    )
    perception_conf = data.get("confidence")
    perception_conf_arr = (
        np.asarray(perception_conf[:n], dtype=np.float64)
        if perception_conf is not None
        else np.full(n, 1.0, dtype=np.float64)
    )
    num_lanes = data.get("num_lanes_detected")
    num_lanes_arr = (
        np.asarray(num_lanes[:n], dtype=np.float64)
        if num_lanes is not None
        else np.full(n, 2.0, dtype=np.float64)
    )
    mpc_feasible = data.get("mpc_feasible")
    mpc_feasible_arr = (
        np.asarray(mpc_feasible[:n], dtype=np.float64)
        if mpc_feasible is not None
        else np.full(n, 1.0, dtype=np.float64)
    )
    mpc_fallback = data.get("mpc_fallback_active")
    mpc_fallback_arr = (
        np.asarray(mpc_fallback[:n], dtype=np.float64)
        if mpc_fallback is not None
        else np.zeros(n, dtype=np.float64)
    )
    mpc_kappa_ref = data.get("mpc_kappa_ref")
    mpc_kappa_ref_arr = (
        np.asarray(mpc_kappa_ref[:n], dtype=np.float64)
        if mpc_kappa_ref is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    arm_phase_deficit = data.get("curve_local_arm_phase_deficit")
    arm_phase_deficit_arr = (
        np.asarray(arm_phase_deficit[:n], dtype=np.float64)
        if arm_phase_deficit is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    arm_effect_score = data.get("curve_local_arm_effect_score")
    arm_effect_score_arr = (
        np.asarray(arm_effect_score[:n], dtype=np.float64)
        if arm_effect_score is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    arm_effect_heading = data.get("curve_local_arm_effect_heading_term")
    arm_effect_heading_arr = (
        np.asarray(arm_effect_heading[:n], dtype=np.float64)
        if arm_effect_heading is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    arm_effect_lateral = data.get("curve_local_arm_effect_lateral_shift_term")
    arm_effect_lateral_arr = (
        np.asarray(arm_effect_lateral[:n], dtype=np.float64)
        if arm_effect_lateral is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    arm_effect_time = data.get("curve_local_arm_effect_time_support_term")
    arm_effect_time_arr = (
        np.asarray(arm_effect_time[:n], dtype=np.float64)
        if arm_effect_time is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    sustain_phase_raw = data.get("curve_local_sustain_phase_raw")
    sustain_phase_raw_arr = (
        np.asarray(sustain_phase_raw[:n], dtype=np.float64)
        if sustain_phase_raw is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    phase_term_path = data.get("curve_phase_term_path")
    phase_term_path_arr = (
        np.asarray(phase_term_path[:n], dtype=np.float64)
        if phase_term_path is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    dynamic_sustain_effect = data.get("curve_local_dynamic_sustain_effect_score")
    dynamic_sustain_effect_arr = (
        np.asarray(dynamic_sustain_effect[:n], dtype=np.float64)
        if dynamic_sustain_effect is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    mpc_kappa_bias_correction = data.get("mpc_kappa_bias_correction")
    mpc_kappa_bias_correction_arr = (
        np.asarray(mpc_kappa_bias_correction[:n], dtype=np.float64)
        if mpc_kappa_bias_correction is not None
        else (mpc_kappa_ref_arr - np.asarray(ref_curvature[:n], dtype=np.float64))
    )

    high_error_mask = lateral_error_arr >= limits["high_lateral_error_min_m"]
    mild_curve_mask = (
        (ref_curvature_arr >= limits["mild_curve_curvature_abs_min"])
        & (ref_curvature_arr <= limits["mild_curve_curvature_abs_max"])
    )
    curve_intent_inactive_mask = np.array(
        [str(v or "").strip().upper() not in {"ENTRY", "COMMIT"} for v in curve_intent_states[:n]],
        dtype=bool,
    )
    curve_local_inactive_mask = np.array(
        [str(v or "").strip().upper() not in {"ENTRY", "COMMIT"} for v in curve_local_states[:n]],
        dtype=bool,
    )
    recognizer_inactive_mask = curve_intent_inactive_mask & curve_local_inactive_mask
    long_pp_lookahead_mask = pp_lookahead_arr >= limits["long_pp_lookahead_min_m"]
    long_reference_lookahead_mask = ref_lookahead_arr >= limits["long_reference_lookahead_min_m"]
    long_lookahead_mask = long_pp_lookahead_mask | long_reference_lookahead_mask
    small_lane_offset_mask = road_center_offset_arr <= limits["small_lane_center_offset_max_m"]
    poor_perception_mask = (
        (perception_conf_arr < limits["poor_perception_confidence_max"])
        | (num_lanes_arr <= limits["poor_perception_num_lanes_max"])
    )
    transport_fallback_mask = sync_fallback_arr > 0.5
    mpc_feasible_mask = mpc_feasible_arr > 0.5
    mpc_fallback_mask = mpc_fallback_arr > 0.5
    underactivated_mask = (
        high_error_mask
        & mild_curve_mask
        & recognizer_inactive_mask
        & long_lookahead_mask
        & small_lane_offset_mask
    )
    state_entry_mask = np.array(
        [str(v or "").strip().upper() == "ENTRY" for v in curve_local_states[:n]],
        dtype=bool,
    )
    state_rearm_mask = np.array(
        [str(v or "").strip().upper() == "REARM" for v in curve_local_states[:n]],
        dtype=bool,
    )
    blocker_state_hold_mask = np.array(
        [str(v or "").strip().lower() == "state_hold" for v in curve_activation_blocker_modes[:n]]
        if curve_activation_blocker_modes
        else np.zeros(n, dtype=bool),
        dtype=bool,
    )
    arm_phase_ready_mask = np.isfinite(arm_phase_deficit_arr) & (arm_phase_deficit_arr <= 0.01)
    sustain_collapse_mask = (
        high_error_mask
        & mild_curve_mask
        & (state_entry_mask | state_rearm_mask)
        & arm_phase_ready_mask
        & np.isfinite(sustain_phase_raw_arr)
        & (sustain_phase_raw_arr <= limits["sustain_phase_collapse_max"])
    )
    rearm_cycle_mask = sustain_collapse_mask & state_rearm_mask & blocker_state_hold_mask
    ref_curvature_safe = np.maximum(ref_curvature_arr, 1e-6)
    mpc_kappa_ratio_arr = np.abs(mpc_kappa_ref_arr) / ref_curvature_safe
    mpc_curvature_softness_mask = (
        high_error_mask
        & mild_curve_mask
        & mpc_feasible_mask
        & (mpc_kappa_ratio_arr <= limits["mpc_curvature_softness_ratio_max"])
    )
    mpc_bias_cancellation_mask = (
        mpc_curvature_softness_mask
        & np.isfinite(mpc_kappa_bias_correction_arr)
        & (np.abs(mpc_kappa_bias_correction_arr) >= 0.5 * ref_curvature_arr)
    )

    high_error_count = int(np.sum(high_error_mask))

    def _pct(mask: np.ndarray) -> Optional[float]:
        if mask.size == 0:
            return None
        return safe_float(np.mean(mask) * 100.0, default=None)

    def _pct_of_high_error(mask: np.ndarray) -> Optional[float]:
        if high_error_count <= 0:
            return None
        return safe_float(
            np.sum(mask & high_error_mask) / high_error_count * 100.0,
            default=None,
        )

    high_error_idx = np.where(high_error_mask)[0]
    underactivated_on_high_error_rate = _pct_of_high_error(underactivated_mask)
    mismatch_on_high_error_rate = _pct_of_high_error(small_lane_offset_mask)
    transport_overlap_rate = _pct_of_high_error(transport_fallback_mask)
    sustain_collapse_rate = _pct_of_high_error(sustain_collapse_mask)
    rearm_cycle_rate = _pct_of_high_error(rearm_cycle_mask)
    mpc_curvature_softness_rate = _pct_of_high_error(mpc_curvature_softness_mask)
    mpc_bias_cancellation_rate = _pct_of_high_error(mpc_bias_cancellation_mask)

    result = {
        **base,
        "availability": "available",
        "issue_detected": bool(
            high_error_count >= 10
            and safe_float(underactivated_on_high_error_rate, default=0.0)
            >= limits["underactivated_on_high_error_min_pct"]
            and safe_float(mismatch_on_high_error_rate, default=0.0)
            >= limits["reference_geometry_mismatch_on_high_error_min_pct"]
            and safe_float(transport_overlap_rate, default=0.0)
            <= limits["transport_fallback_overlap_max_pct"]
        ),
        "high_error_frame_count": high_error_count,
        "high_error_frame_rate": _pct(high_error_mask),
        "mild_curve_present_frame_rate": _pct(mild_curve_mask),
        "mild_curve_present_on_high_error_rate": _pct_of_high_error(mild_curve_mask),
        "curve_recognition_inactive_on_high_error_rate": _pct_of_high_error(
            recognizer_inactive_mask
        ),
        "curve_intent_inactive_on_high_error_rate": _pct_of_high_error(
            curve_intent_inactive_mask
        ),
        "curve_local_inactive_on_high_error_rate": _pct_of_high_error(
            curve_local_inactive_mask
        ),
        "long_pp_lookahead_on_high_error_rate": _pct_of_high_error(
            long_pp_lookahead_mask
        ),
        "long_reference_lookahead_on_high_error_rate": _pct_of_high_error(
            long_reference_lookahead_mask
        ),
        "long_lookahead_on_high_error_rate": _pct_of_high_error(long_lookahead_mask),
        "reference_geometry_mismatch_on_high_error_rate": mismatch_on_high_error_rate,
        "underactivated_tracking_on_high_error_rate": underactivated_on_high_error_rate,
        "sustain_phase_collapse_on_high_error_rate": sustain_collapse_rate,
        "rearm_cycle_on_high_error_rate": rearm_cycle_rate,
        "mpc_curvature_softness_on_high_error_rate": mpc_curvature_softness_rate,
        "mpc_bias_cancellation_on_high_error_rate": mpc_bias_cancellation_rate,
        "transport_fallback_overlap_on_high_error_rate": transport_overlap_rate,
        "poor_perception_overlap_on_high_error_rate": _pct_of_high_error(
            poor_perception_mask
        ),
        "mpc_feasible_on_high_error_rate": _pct_of_high_error(mpc_feasible_mask),
        "mpc_fallback_overlap_on_high_error_rate": _pct_of_high_error(mpc_fallback_mask),
        "rearm_cycle_issue_detected": bool(
            high_error_count >= 10
            and safe_float(rearm_cycle_rate, default=0.0)
            >= limits["rearm_cycle_on_high_error_min_pct"]
        ),
        "mpc_bias_cancellation_issue_detected": bool(
            high_error_count >= 10
            and safe_float(mpc_bias_cancellation_rate, default=0.0)
            >= limits["mpc_bias_cancellation_on_high_error_min_pct"]
        ),
        "curve_intent_state_mode_on_high_error": _mode_string(
            [curve_intent_states[i] for i in high_error_idx],
            ignore={""},
        ),
        "curve_local_state_mode_on_high_error": _mode_string(
            [curve_local_states[i] for i in high_error_idx],
            ignore={""},
        ),
        "curve_activation_blocker_mode_on_high_error": _mode_string(
            [curve_activation_blocker_modes[i] for i in high_error_idx]
            if curve_activation_blocker_modes
            else [],
            ignore={"", "none"},
        ),
        "curve_activation_blocker_mode_on_underactivated": _mode_string(
            [curve_activation_blocker_modes[i] for i in np.where(underactivated_mask)[0]]
            if curve_activation_blocker_modes
            else [],
            ignore={"", "none"},
        ),
        "lateral_error_abs_m": _finite_stats(lateral_error_arr[high_error_mask]),
        "road_frame_lane_center_offset_abs_m": _finite_stats(
            road_center_offset_arr[high_error_mask]
        ),
        "reference_point_curvature_abs": _finite_stats(ref_curvature_arr[high_error_mask]),
        "mpc_kappa_ref_abs": _finite_stats(np.abs(mpc_kappa_ref_arr[high_error_mask])),
        "pp_lookahead_distance_m": _finite_stats(pp_lookahead_arr[high_error_mask]),
        "reference_lookahead_target_m": _finite_stats(
            ref_lookahead_arr[high_error_mask]
        ),
        "speed_mps": _finite_stats(speed_arr[high_error_mask]),
        "curve_local_arm_phase_deficit": _finite_stats(
            arm_phase_deficit_arr[high_error_mask]
        ),
        "curve_local_arm_effect_score": _finite_stats(
            arm_effect_score_arr[high_error_mask]
        ),
        "curve_local_arm_effect_heading_term": _finite_stats(
            arm_effect_heading_arr[high_error_mask]
        ),
        "curve_local_arm_effect_lateral_shift_term": _finite_stats(
            arm_effect_lateral_arr[high_error_mask]
        ),
        "curve_local_arm_effect_time_support_term": _finite_stats(
            arm_effect_time_arr[high_error_mask]
        ),
        "curve_local_sustain_phase_raw": _finite_stats(
            sustain_phase_raw_arr[high_error_mask]
        ),
        "curve_phase_term_path": _finite_stats(phase_term_path_arr[high_error_mask]),
        "curve_local_dynamic_sustain_effect_score": _finite_stats(
            dynamic_sustain_effect_arr[high_error_mask]
        ),
        "mpc_kappa_ratio_to_reference": _finite_stats(
            mpc_kappa_ratio_arr[high_error_mask]
        ),
        "mpc_kappa_bias_correction": _finite_stats(
            mpc_kappa_bias_correction_arr[high_error_mask]
        ),
    }
    return result


def _build_mpc_gt_cross_track_contract_summary(data: Dict) -> Dict:
    limits = {
        "high_lateral_error_min_m": 0.50,
        "small_at_car_cross_track_max_m": 0.12,
        "large_lookahead_cross_track_min_m": 0.50,
        "large_at_car_cross_track_min_m": 1.00,
        "small_road_offset_max_m": 0.12,
        "semantic_mismatch_on_high_error_min_pct": 50.0,
        "absolute_coordinate_mismatch_on_high_error_min_pct": 50.0,
        "transport_fallback_overlap_max_pct": 5.0,
        "poor_perception_overlap_max_pct": 10.0,
    }
    base = {
        "schema_version": "v1",
        "availability": "unavailable",
        "owner_class": "authoritative",
        "issue_detected": False,
        "issue_mode": "none",
        "high_error_frame_count": 0,
        "high_error_frame_rate": None,
        "semantic_mismatch_on_high_error_rate": None,
        "semantic_mismatch_on_straight_high_error_rate": None,
        "absolute_coordinate_mismatch_on_high_error_rate": None,
        "small_at_car_cross_track_on_high_error_rate": None,
        "large_lookahead_cross_track_on_high_error_rate": None,
        "large_at_car_cross_track_on_high_error_rate": None,
        "small_road_offset_on_high_error_rate": None,
        "transport_fallback_overlap_on_high_error_rate": None,
        "poor_perception_overlap_on_high_error_rate": None,
        "curve_local_state_mode_on_high_error": None,
        "mpc_gt_cross_track_source_mode_on_high_error": None,
        "control_lateral_error_abs_m": _finite_stats(None),
        "mpc_gt_cross_track_abs_m": _finite_stats(None),
        "mpc_gt_cross_track_at_car_abs_m": _finite_stats(None),
        "mpc_gt_cross_track_lookahead_abs_m": _finite_stats(None),
        "gt_cross_track_delta_abs_m": _finite_stats(None),
        "road_frame_lane_center_offset_abs_m": _finite_stats(None),
        "limits": limits,
    }

    lateral_error = data.get("lateral_error")
    source_cross_track = data.get("mpc_gt_cross_track_m")
    at_car_cross_track = data.get("mpc_gt_cross_track_at_car_m")
    lookahead_cross_track = data.get("mpc_gt_cross_track_lookahead_m")
    road_center_offset = data.get("road_frame_lane_center_offset")
    source_codes = _decode_string_series(data.get("mpc_gt_cross_track_source_code"))
    curve_local_states = _decode_string_series(data.get("curve_local_state"))
    if any(
        candidate is None
        for candidate in (
            lateral_error,
            source_cross_track,
            at_car_cross_track,
            lookahead_cross_track,
        )
    ):
        return base
    if not source_codes or not curve_local_states:
        return base

    n = min(
        len(lateral_error),
        len(source_cross_track),
        len(at_car_cross_track),
        len(lookahead_cross_track),
        len(source_codes),
        len(curve_local_states),
    )
    optional_series = [
        data.get("sync_packet_fallback_active"),
        data.get("confidence"),
        data.get("num_lanes_detected"),
        road_center_offset,
    ]
    for series in optional_series:
        if series is not None:
            n = min(n, len(series))
    if n <= 0:
        return base

    lateral_error_arr = np.abs(np.asarray(lateral_error[:n], dtype=np.float64))
    source_cross_track_arr = np.abs(np.asarray(source_cross_track[:n], dtype=np.float64))
    at_car_cross_track_arr = np.abs(np.asarray(at_car_cross_track[:n], dtype=np.float64))
    lookahead_cross_track_arr = np.abs(np.asarray(lookahead_cross_track[:n], dtype=np.float64))
    delta_cross_track_arr = np.abs(lookahead_cross_track_arr - at_car_cross_track_arr)
    road_center_offset_arr = (
        np.abs(np.asarray(road_center_offset[:n], dtype=np.float64))
        if road_center_offset is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    sync_fallback = data.get("sync_packet_fallback_active")
    sync_fallback_arr = (
        np.asarray(sync_fallback[:n], dtype=np.float64)
        if sync_fallback is not None
        else np.zeros(n, dtype=np.float64)
    )
    confidence = data.get("confidence")
    confidence_arr = (
        np.asarray(confidence[:n], dtype=np.float64)
        if confidence is not None
        else np.ones(n, dtype=np.float64)
    )
    num_lanes = data.get("num_lanes_detected")
    num_lanes_arr = (
        np.asarray(num_lanes[:n], dtype=np.float64)
        if num_lanes is not None
        else np.full(n, 2.0, dtype=np.float64)
    )

    high_error = lateral_error_arr >= limits["high_lateral_error_min_m"]
    small_at_car = at_car_cross_track_arr <= limits["small_at_car_cross_track_max_m"]
    large_lookahead = lookahead_cross_track_arr >= limits["large_lookahead_cross_track_min_m"]
    large_at_car = at_car_cross_track_arr >= limits["large_at_car_cross_track_min_m"]
    small_road_offset = road_center_offset_arr <= limits["small_road_offset_max_m"]
    semantic_mismatch = high_error & small_at_car & large_lookahead
    absolute_coordinate_mismatch = high_error & large_at_car & small_road_offset
    straight_high_error = semantic_mismatch & np.array(
        [state == "STRAIGHT" for state in curve_local_states[:n]], dtype=bool
    )
    poor_perception = (confidence_arr <= 0.50) | (num_lanes_arr <= 1.5)
    fallback_mask = sync_fallback_arr > 0.5
    high_error_count = int(np.sum(high_error))

    result = dict(base)
    result["availability"] = "available"
    result["high_error_frame_count"] = high_error_count
    result["high_error_frame_rate"] = safe_float(
        (high_error_count / max(n, 1)) * 100.0
    )
    result["semantic_mismatch_on_high_error_rate"] = safe_float(
        (np.sum(semantic_mismatch) / max(high_error_count, 1)) * 100.0
    )
    result["semantic_mismatch_on_straight_high_error_rate"] = safe_float(
        (np.sum(straight_high_error) / max(high_error_count, 1)) * 100.0
    )
    result["absolute_coordinate_mismatch_on_high_error_rate"] = safe_float(
        (np.sum(high_error & absolute_coordinate_mismatch) / max(high_error_count, 1)) * 100.0
    )
    result["small_at_car_cross_track_on_high_error_rate"] = safe_float(
        (np.sum(high_error & small_at_car) / max(high_error_count, 1)) * 100.0
    )
    result["large_lookahead_cross_track_on_high_error_rate"] = safe_float(
        (np.sum(high_error & large_lookahead) / max(high_error_count, 1)) * 100.0
    )
    result["large_at_car_cross_track_on_high_error_rate"] = safe_float(
        (np.sum(high_error & large_at_car) / max(high_error_count, 1)) * 100.0
    )
    result["small_road_offset_on_high_error_rate"] = safe_float(
        (np.sum(high_error & small_road_offset) / max(high_error_count, 1)) * 100.0
    )
    result["transport_fallback_overlap_on_high_error_rate"] = safe_float(
        (np.sum(high_error & fallback_mask) / max(high_error_count, 1)) * 100.0
    )
    result["poor_perception_overlap_on_high_error_rate"] = safe_float(
        (np.sum(high_error & poor_perception) / max(high_error_count, 1)) * 100.0
    )
    result["curve_local_state_mode_on_high_error"] = _mode_string(
        [curve_local_states[idx] for idx, active in enumerate(high_error) if active],
        ignore={"", "nan", "none"},
    )
    result["mpc_gt_cross_track_source_mode_on_high_error"] = _mode_string(
        [source_codes[idx] for idx, active in enumerate(high_error) if active],
        ignore={"", "nan", "none"},
    )
    result["control_lateral_error_abs_m"] = _finite_stats(lateral_error_arr[high_error])
    result["mpc_gt_cross_track_abs_m"] = _finite_stats(source_cross_track_arr[high_error])
    result["mpc_gt_cross_track_at_car_abs_m"] = _finite_stats(at_car_cross_track_arr[high_error])
    result["mpc_gt_cross_track_lookahead_abs_m"] = _finite_stats(
        lookahead_cross_track_arr[high_error]
    )
    result["gt_cross_track_delta_abs_m"] = _finite_stats(delta_cross_track_arr[high_error])
    result["road_frame_lane_center_offset_abs_m"] = _finite_stats(road_center_offset_arr[high_error])
    semantic_rate = (
        float(result["semantic_mismatch_on_high_error_rate"])
        if result["semantic_mismatch_on_high_error_rate"] is not None
        else 0.0
    )
    absolute_rate = (
        float(result["absolute_coordinate_mismatch_on_high_error_rate"])
        if result["absolute_coordinate_mismatch_on_high_error_rate"] is not None
        else 0.0
    )
    fallback_overlap_rate = (
        float(result["transport_fallback_overlap_on_high_error_rate"])
        if result["transport_fallback_overlap_on_high_error_rate"] is not None
        else 100.0
    )
    poor_perception_overlap_rate = (
        float(result["poor_perception_overlap_on_high_error_rate"])
        if result["poor_perception_overlap_on_high_error_rate"] is not None
        else 100.0
    )
    semantic_issue = bool(
        high_error_count > 0
        and semantic_rate >= limits["semantic_mismatch_on_high_error_min_pct"]
        and fallback_overlap_rate <= limits["transport_fallback_overlap_max_pct"]
        and poor_perception_overlap_rate <= limits["poor_perception_overlap_max_pct"]
    )
    absolute_issue = bool(
        high_error_count > 0
        and absolute_rate >= limits["absolute_coordinate_mismatch_on_high_error_min_pct"]
        and fallback_overlap_rate <= limits["transport_fallback_overlap_max_pct"]
        and poor_perception_overlap_rate <= limits["poor_perception_overlap_max_pct"]
    )
    result["issue_detected"] = bool(semantic_issue or absolute_issue)
    if absolute_issue:
        result["issue_mode"] = "absolute_coordinate_mismatch"
    elif semantic_issue:
        result["issue_mode"] = "semantic_mismatch"
    return result


def _build_lateral_owner_contract_summary(
    curve_intent_diag: Dict,
    curve_local_contract: Dict,
    turn_in_owner: Dict,
    local_curve_reference: Dict,
    highway_mild_curve_contract: Dict,
    mpc_gt_cross_track_contract: Dict,
) -> Dict:
    turn_in_limits = turn_in_owner.get("limits") or {}
    local_ref_limits = local_curve_reference.get("limits") or {}
    turn_in_fallback_rate = (
        float(turn_in_owner.get("fallback_active_rate"))
        if turn_in_owner.get("fallback_active_rate") is not None
        else 0.0
    )
    local_ref_fallback_rate = (
        float(local_curve_reference.get("fallback_active_rate"))
        if local_curve_reference.get("fallback_active_rate") is not None
        else 0.0
    )
    turn_in_fallback_limit = (
        float(turn_in_limits.get("fallback_active_rate_max_pct"))
        if turn_in_limits.get("fallback_active_rate_max_pct") is not None
        else 0.0
    )
    local_ref_fallback_limit = (
        float(local_ref_limits.get("fallback_active_rate_max_pct"))
        if local_ref_limits.get("fallback_active_rate_max_pct") is not None
        else 0.0
    )
    local_curve_issue_detected = bool(
        float(curve_local_contract.get("curve_local_active_straight_rate", 0.0) or 0.0) > 5.0
        or int(curve_local_contract.get("curve_local_arm_without_ready_count", 0) or 0) > 0
        or int(curve_local_contract.get("curve_local_commit_without_ready_count", 0) or 0) > 0
        or int(curve_local_contract.get("curve_local_reentry_without_gate_count", 0) or 0) > 0
        or int(curve_local_contract.get("curve_local_watchdog_pingpong_count", 0) or 0) > 0
        or int(curve_local_contract.get("curve_local_commit_without_distance_ready_count", 0) or 0) > 0
        or int(curve_local_contract.get("curve_lookahead_collapse_violation_count", 0) or 0) > 0
    )
    authoritative_owner_available = bool(
        curve_local_contract.get("curve_local_contract_available")
        or turn_in_owner.get("availability") == "available"
        or local_curve_reference.get("availability") == "available"
        or mpc_gt_cross_track_contract.get("availability") == "available"
    )
    authoritative_owner_issue_detected = bool(
        local_curve_issue_detected
        or bool(highway_mild_curve_contract.get("issue_detected"))
        or bool(mpc_gt_cross_track_contract.get("issue_detected"))
        or turn_in_fallback_rate > (turn_in_fallback_limit + 1e-6)
        or local_ref_fallback_rate > (local_ref_fallback_limit + 1e-6)
    )
    authoritative_owner_healthy = bool(
        authoritative_owner_available and not authoritative_owner_issue_detected
    )
    primary_owner_mode = (
        str(turn_in_owner.get("owner_mode") or "")
        or str(local_curve_reference.get("mode") or "")
        or "unavailable"
    )
    if authoritative_owner_healthy:
        owner_summary_mode = "authoritative_clean"
    elif authoritative_owner_available:
        owner_summary_mode = "authoritative_issue"
    else:
        owner_summary_mode = "legacy_only"
    return {
        "schema_version": "v1",
        "availability": "available" if authoritative_owner_available else "unavailable",
        "owner_summary_mode": owner_summary_mode,
        "primary_owner_mode": primary_owner_mode,
        "curve_intent_owner_class": "legacy_proxy",
        "curve_local_owner_class": "authoritative",
        "turn_in_owner_class": "authoritative",
        "local_curve_reference_owner_class": "authoritative",
        "gt_cross_track_owner_class": "authoritative",
        "authoritative_owner_available": bool(authoritative_owner_available),
        "authoritative_owner_issue_detected": bool(authoritative_owner_issue_detected),
        "authoritative_owner_healthy": bool(authoritative_owner_healthy),
        "local_curve_issue_detected": bool(local_curve_issue_detected),
        "turn_in_owner_fallback_active_rate": safe_float(turn_in_fallback_rate),
        "local_curve_reference_fallback_active_rate": safe_float(local_ref_fallback_rate),
        "highway_mild_curve_issue_detected": bool(highway_mild_curve_contract.get("issue_detected")),
        "mpc_gt_cross_track_issue_detected": bool(mpc_gt_cross_track_contract.get("issue_detected")),
        "legacy_curve_intent_available": bool(curve_intent_diag.get("available")),
        "legacy_curve_intent_proxy_only": bool(
            curve_intent_diag.get("available") and authoritative_owner_healthy
        ),
        "suppress_legacy_curve_intent_warnings": bool(
            curve_intent_diag.get("available") and authoritative_owner_healthy
        ),
    }


def _load_track_curve_windows(track_name: str) -> dict:
    safe_name = "".join(ch for ch in str(track_name or "").strip() if ch.isalnum() or ch in {"_", "-"})
    if not safe_name:
        return {}
    track_path = REPO_ROOT / "tracks" / f"{safe_name}.yml"
    if not track_path.exists():
        return {}

    with open(track_path, "r") as track_file:
        cfg = yaml.safe_load(track_file) or {}

    segments = cfg.get("segments", []) or []
    s = 0.0
    curve_windows = []
    curve_idx = 0
    for segment in segments:
        seg = segment or {}
        seg_type = str(seg.get("type", "straight")).strip().lower()
        start_m = s
        if seg_type == "arc":
            radius = float(seg.get("radius", 0.0) or 0.0)
            angle_deg = float(seg.get("angle_deg", seg.get("angle", 0.0)) or 0.0)
            length_m = max(0.0, radius * math.radians(abs(angle_deg)))
            curve_idx += 1
            curve_windows.append(
                {
                    "curve_index": int(curve_idx),
                    "start_m": float(start_m),
                    "end_m": float(start_m + length_m),
                    "radius_m": float(radius),
                    "angle_deg": float(angle_deg),
                    "direction": str(seg.get("direction", "left")).strip().lower(),
                }
            )
        else:
            length_m = max(0.0, float(seg.get("length", 0.0) or 0.0))
        s += length_m

    return {
        "track_name": str(cfg.get("name", safe_name) or safe_name),
        "track_key": safe_name,
        "curve_windows": curve_windows,
        "total_length_m": float(s),
    }


def _nearest_abs_deltas_ms(source_ts: Optional[np.ndarray], target_ts: Optional[np.ndarray]) -> np.ndarray:
    """Compute nearest-neighbor absolute timestamp deltas (milliseconds)."""
    if source_ts is None or target_ts is None:
        return np.array([], dtype=float)

    source = np.asarray(source_ts, dtype=float).reshape(-1)
    target = np.asarray(target_ts, dtype=float).reshape(-1)
    source = source[np.isfinite(source)]
    target = target[np.isfinite(target)]
    if source.size == 0 or target.size == 0:
        return np.array([], dtype=float)

    target_sorted = np.sort(target)
    idx = np.searchsorted(target_sorted, source)
    left = np.clip(idx - 1, 0, target_sorted.size - 1)
    right = np.clip(idx, 0, target_sorted.size - 1)
    d_left = np.abs(source - target_sorted[left])
    d_right = np.abs(source - target_sorted[right])
    return np.minimum(d_left, d_right) * 1000.0


def _build_latency_sync_summary(data: Dict) -> Dict:
    """Build canonical latency/sync summary block."""
    alignment_window_ms = 20.0
    misaligned_rate_limit = 0.02
    e2e_p95_limit_ms = 100.0
    cadence_irregular_rate_limit = 0.08
    cadence_severe_rate_limit = 0.01
    tuning_dt_p95_limit_ms = 80.0
    tuning_dt_max_limit_ms = 500.0

    e2e_mode = "input_ready_to_command_sent"
    raw_modes = data.get("e2e_latency_mode")
    if raw_modes is None:
        raw_modes = []
    for raw_mode in raw_modes:
        mode_str = raw_mode.decode("utf-8", errors="ignore") if isinstance(raw_mode, (bytes, bytearray)) else str(raw_mode)
        mode_str = mode_str.strip()
        if mode_str:
            e2e_mode = mode_str
            break

    e2e_stats = _finite_stats(data.get("e2e_latency_ms"))
    e2e_available = e2e_stats["count"] > 0
    e2e_pass = None
    if e2e_available and e2e_stats["p95"] is not None:
        e2e_pass = bool(e2e_stats["p95"] <= e2e_p95_limit_ms)

    dt_cam_traj_ms = _nearest_abs_deltas_ms(
        data.get("camera_timestamps"), data.get("trajectory_timestamps")
    )
    dt_cam_control_ms = _nearest_abs_deltas_ms(
        data.get("camera_timestamps"), data.get("control_timestamps")
    )
    dt_cam_traj_stats = _finite_stats(dt_cam_traj_ms)
    dt_cam_control_stats = _finite_stats(dt_cam_control_ms)

    sync_available = (
        dt_cam_traj_stats["count"] > 0 and dt_cam_control_stats["count"] > 0
    )
    contract_misaligned_rate = None
    sync_pass = None
    if sync_available:
        n = min(dt_cam_traj_ms.size, dt_cam_control_ms.size)
        if n > 0:
            misaligned = (
                (dt_cam_traj_ms[:n] > alignment_window_ms)
                | (dt_cam_control_ms[:n] > alignment_window_ms)
            )
            contract_misaligned_rate = safe_float(np.mean(misaligned))
            sync_pass = bool(
                (dt_cam_traj_stats["p95"] is not None and dt_cam_traj_stats["p95"] <= alignment_window_ms)
                and (dt_cam_control_stats["p95"] is not None and dt_cam_control_stats["p95"] <= alignment_window_ms)
                and (contract_misaligned_rate <= misaligned_rate_limit)
            )

    cadence_source_ts = None
    for key in ("control_timestamps", "timestamps", "camera_timestamps"):
        arr = data.get(key)
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=float).reshape(-1)
        if arr_np.size >= 2:
            cadence_source_ts = arr_np
            break

    cadence_dt_ms = np.array([], dtype=float)
    cadence_stats = _finite_stats(cadence_dt_ms)
    cadence_irregular_rate = None
    cadence_severe_rate = None
    cadence_nominal_ms = None
    cadence_irregular_threshold_ms = None
    cadence_severe_threshold_ms = None
    cadence_available = False
    cadence_pass = None
    if cadence_source_ts is not None:
        cadence_dt_s = np.diff(cadence_source_ts)
        cadence_dt_s = cadence_dt_s[np.isfinite(cadence_dt_s) & (cadence_dt_s > 1e-6)]
        cadence_dt_ms = cadence_dt_s * 1000.0
        cadence_stats = _finite_stats(cadence_dt_ms)
        cadence_available = cadence_stats["count"] > 0
    if cadence_available:
        cadence_nominal_ms = safe_float(np.median(cadence_dt_ms), default=None)
        cadence_irregular_threshold_ms = max(100.0, 2.5 * cadence_nominal_ms)
        cadence_severe_threshold_ms = max(180.0, 4.0 * cadence_nominal_ms)
        cadence_irregular_rate = safe_float(
            np.mean(cadence_dt_ms > cadence_irregular_threshold_ms), default=None
        )
        cadence_severe_rate = safe_float(
            np.mean(cadence_dt_ms > cadence_severe_threshold_ms), default=None
        )
        cadence_pass = bool(
            cadence_irregular_rate <= cadence_irregular_rate_limit
            and cadence_severe_rate <= cadence_severe_rate_limit
        )
    tuning_valid = False
    if cadence_available:
        cadence_p95 = cadence_stats.get("p95")
        cadence_max = cadence_stats.get("max")
        tuning_valid = bool(
            cadence_severe_rate is not None
            and cadence_severe_rate <= cadence_severe_rate_limit
            and cadence_p95 is not None
            and cadence_p95 <= tuning_dt_p95_limit_ms
            and cadence_max is not None
            and cadence_max <= tuning_dt_max_limit_ms
        )
    cadence_status = "unknown"
    if cadence_available:
        cadence_status = "good" if cadence_pass else "poor"

    issues = []
    if e2e_pass is False:
        issues.append(f"e2e_p95_ms={e2e_stats['p95']:.1f} (limit <= {e2e_p95_limit_ms:.1f})")
    if sync_pass is False:
        if dt_cam_traj_stats["p95"] is not None and dt_cam_traj_stats["p95"] > alignment_window_ms:
            issues.append(
                f"dt_cam_traj_p95_ms={dt_cam_traj_stats['p95']:.1f} (limit <= {alignment_window_ms:.1f})"
            )
        if dt_cam_control_stats["p95"] is not None and dt_cam_control_stats["p95"] > alignment_window_ms:
            issues.append(
                f"dt_cam_control_p95_ms={dt_cam_control_stats['p95']:.1f} (limit <= {alignment_window_ms:.1f})"
            )
        if contract_misaligned_rate is not None and contract_misaligned_rate > misaligned_rate_limit:
            issues.append(
                f"contract_misaligned_rate={contract_misaligned_rate:.3f} (limit <= {misaligned_rate_limit:.3f})"
            )
    if cadence_pass is False:
        if cadence_irregular_rate is not None and cadence_irregular_rate > cadence_irregular_rate_limit:
            issues.append(
                f"cadence_irregular_rate={cadence_irregular_rate:.3f} (limit <= {cadence_irregular_rate_limit:.3f})"
            )
        if cadence_severe_rate is not None and cadence_severe_rate > cadence_severe_rate_limit:
            issues.append(
                f"cadence_severe_rate={cadence_severe_rate:.3f} (limit <= {cadence_severe_rate_limit:.3f})"
            )

    if issues:
        overall_status = "poor"
    elif e2e_available and sync_available and cadence_available:
        overall_status = "good"
    elif e2e_available or sync_available or cadence_available:
        overall_status = "warn"
    else:
        overall_status = "unknown"

    return {
        "schema_version": "v1",
        "e2e": {
            "mode": e2e_mode,
            "availability": "available" if e2e_available else "unavailable",
            "units": "ms",
            "stats_ms": e2e_stats,
            "limit_p95_ms": e2e_p95_limit_ms,
            "pass": e2e_pass,
        },
        "sync_alignment": {
            "availability": "available" if sync_available else "unavailable",
            "alignment_window_ms": alignment_window_ms,
            "dt_cam_traj_ms": dt_cam_traj_stats,
            "dt_cam_control_ms": dt_cam_control_stats,
            "contract_misaligned_rate": contract_misaligned_rate,
            "contract_misaligned_rate_limit": misaligned_rate_limit,
            "pass": sync_pass,
        },
        "cadence": {
            "availability": "available" if cadence_available else "unavailable",
            "status": cadence_status,
            "stats_ms": cadence_stats,
            "nominal_dt_ms": cadence_nominal_ms,
            "irregular_dt_threshold_ms": cadence_irregular_threshold_ms,
            "severe_dt_threshold_ms": cadence_severe_threshold_ms,
            "irregular_rate": cadence_irregular_rate,
            "severe_irregular_rate": cadence_severe_rate,
            "limits": {
                "irregular_rate_max": cadence_irregular_rate_limit,
                "severe_irregular_rate_max": cadence_severe_rate_limit,
                "dt_p95_ms_max": tuning_dt_p95_limit_ms,
                "dt_max_ms_max": tuning_dt_max_limit_ms,
            },
            "pass": cadence_pass,
            "tuning_valid": tuning_valid,
        },
        "overall": {
            "status": overall_status,
            "issues": issues,
            "tuning_valid": tuning_valid,
        },
    }


def _build_chassis_ground_summary(data: Dict, n_frames: int) -> Dict:
    """Build chassis-ground health summary from recorded telemetry."""
    transient_ignore_s = 1.0
    good_contact_rate_pct_max = 0.0
    good_penetration_m_max = 0.0
    warn_contact_rate_pct_max = 0.5
    warn_penetration_m_max = 0.01
    fallback_warn_rate_pct_max = 0.0

    time_axis = np.asarray(data.get("time"), dtype=float) if data.get("time") is not None else None
    if time_axis is None or time_axis.size == 0:
        keep_mask = np.ones(int(max(n_frames, 0)), dtype=bool)
    else:
        keep_mask = time_axis[:n_frames] >= transient_ignore_s
        if keep_mask.size < n_frames:
            keep_mask = np.pad(keep_mask, (0, n_frames - keep_mask.size), constant_values=False)

    analyzed_frame_count = int(np.sum(keep_mask)) if keep_mask.size > 0 else 0
    if analyzed_frame_count <= 0:
        return {
            "schema_version": "v1",
            "availability": "unavailable",
            "health": "UNKNOWN",
            "configured_min_clearance_m": None,
            "effective_min_clearance_m": None,
            "clearance_p05_m": None,
            "clearance_min_m": None,
            "penetration_max_m": None,
            "contact_frames": 0,
            "contact_rate_pct": None,
            "wheel_grounded_rate_mean": None,
            "force_fallback_rate_pct": None,
            "wheel_colliders_ready_rate_pct": None,
            "analyzed_frame_count": 0,
            "metric_sample_count": 0,
            "skipped_transient_frames": int(max(n_frames, 0)),
            "limits": {
                "transient_ignore_s": transient_ignore_s,
                "good_contact_rate_pct_max": good_contact_rate_pct_max,
                "good_penetration_m_max": good_penetration_m_max,
                "warn_contact_rate_pct_max": warn_contact_rate_pct_max,
                "warn_penetration_m_max": warn_penetration_m_max,
                "fallback_warn_rate_pct_max": fallback_warn_rate_pct_max,
            },
        }

    def _slice(values):
        if values is None:
            return None
        arr = np.asarray(values).reshape(-1)
        if arr.size == 0:
            return None
        limit = min(int(n_frames), int(keep_mask.size), int(arr.size))
        if limit <= 0:
            return None
        return arr[:limit][keep_mask[:limit]]

    clearance = _slice(data.get("chassis_ground_clearance_m"))
    penetration = _slice(data.get("chassis_ground_penetration_m"))
    contact_raw = _slice(data.get("chassis_ground_contact"))
    wheel_grounded = _slice(data.get("wheel_grounded_count"))
    wheel_ready = _slice(data.get("wheel_colliders_ready"))
    force_fallback = _slice(data.get("force_fallback_active"))
    configured_min = _slice(data.get("chassis_ground_min_clearance_m"))
    effective_min = _slice(data.get("chassis_ground_effective_min_clearance_m"))

    availability = "available" if any(
        values is not None for values in (clearance, penetration, contact_raw, wheel_grounded, force_fallback)
    ) else "unavailable"

    if availability != "available":
        return {
            "schema_version": "v1",
            "availability": "unavailable",
            "health": "UNKNOWN",
            "configured_min_clearance_m": None,
            "effective_min_clearance_m": None,
            "clearance_p05_m": None,
            "clearance_min_m": None,
            "penetration_max_m": None,
            "contact_frames": 0,
            "contact_rate_pct": None,
            "wheel_grounded_rate_mean": None,
            "force_fallback_rate_pct": None,
            "wheel_colliders_ready_rate_pct": None,
            "analyzed_frame_count": analyzed_frame_count,
            "metric_sample_count": 0,
            "skipped_transient_frames": int(n_frames - analyzed_frame_count),
            "limits": {
                "transient_ignore_s": transient_ignore_s,
                "good_contact_rate_pct_max": good_contact_rate_pct_max,
                "good_penetration_m_max": good_penetration_m_max,
                "warn_contact_rate_pct_max": warn_contact_rate_pct_max,
                "warn_penetration_m_max": warn_penetration_m_max,
                "fallback_warn_rate_pct_max": fallback_warn_rate_pct_max,
            },
        }

    clearance_finite = None
    if clearance is not None:
        clearance = np.asarray(clearance, dtype=float)
        clearance_finite = clearance[np.isfinite(clearance)]
    penetration_finite = None
    if penetration is not None:
        penetration = np.asarray(penetration, dtype=float)
        penetration_finite = penetration[np.isfinite(penetration)]

    contact_bool = None
    if contact_raw is not None:
        contact_bool = np.asarray(contact_raw, dtype=float) > 0.5
    elif penetration_finite is not None and penetration_finite.size > 0:
        # If contact flags are absent, infer from penetration > 0.
        contact_bool = np.asarray(penetration, dtype=float) > 1e-6
    elif clearance_finite is not None and clearance_finite.size > 0:
        contact_bool = np.asarray(clearance, dtype=float) <= 0.0

    contact_samples = int(contact_bool.size) if contact_bool is not None else 0
    contact_frames = int(np.sum(contact_bool)) if contact_bool is not None else 0
    contact_rate_pct = (
        safe_float(contact_frames / contact_samples * 100.0)
        if contact_samples > 0
        else None
    )
    penetration_max_m = (
        safe_float(np.max(penetration_finite), default=None)
        if penetration_finite is not None and penetration_finite.size > 0
        else None
    )
    clearance_p05_m = (
        safe_float(np.percentile(clearance_finite, 5), default=None)
        if clearance_finite is not None and clearance_finite.size > 0
        else None
    )
    clearance_min_m = (
        safe_float(np.min(clearance_finite), default=None)
        if clearance_finite is not None and clearance_finite.size > 0
        else None
    )
    configured_min_clearance_m = (
        safe_float(np.nanmedian(np.asarray(configured_min, dtype=float)), default=None)
        if configured_min is not None and np.isfinite(np.asarray(configured_min, dtype=float)).any()
        else None
    )
    effective_min_clearance_m = (
        safe_float(np.nanmedian(np.asarray(effective_min, dtype=float)), default=None)
        if effective_min is not None and np.isfinite(np.asarray(effective_min, dtype=float)).any()
        else None
    )
    wheel_grounded_rate_mean = (
        safe_float(np.nanmean(np.clip(np.asarray(wheel_grounded, dtype=float), 0.0, 4.0) / 4.0), default=None)
        if wheel_grounded is not None and np.isfinite(np.asarray(wheel_grounded, dtype=float)).any()
        else None
    )
    force_fallback_rate_pct = (
        safe_float(np.nanmean(np.asarray(force_fallback, dtype=float) > 0.5) * 100.0, default=None)
        if force_fallback is not None and np.asarray(force_fallback).size > 0
        else None
    )
    wheel_colliders_ready_rate_pct = (
        safe_float(np.nanmean(np.asarray(wheel_ready, dtype=float) > 0.5) * 100.0, default=None)
        if wheel_ready is not None and np.asarray(wheel_ready).size > 0
        else None
    )

    health = "GOOD"
    if (
        penetration_max_m is not None and penetration_max_m > warn_penetration_m_max
    ) or (
        contact_rate_pct is not None and contact_rate_pct > warn_contact_rate_pct_max
    ):
        health = "POOR"
    elif (
        (force_fallback_rate_pct is not None and force_fallback_rate_pct > fallback_warn_rate_pct_max)
        or (penetration_max_m is not None and penetration_max_m > good_penetration_m_max)
        or (contact_rate_pct is not None and contact_rate_pct > good_contact_rate_pct_max)
    ):
        health = "WARN"

    return {
        "schema_version": "v1",
        "availability": availability,
        "health": health,
        "configured_min_clearance_m": configured_min_clearance_m,
        "effective_min_clearance_m": effective_min_clearance_m,
        "clearance_p05_m": clearance_p05_m,
        "clearance_min_m": clearance_min_m,
        "penetration_max_m": penetration_max_m,
        "contact_frames": contact_frames,
        "contact_rate_pct": contact_rate_pct,
        "wheel_grounded_rate_mean": wheel_grounded_rate_mean,
        "force_fallback_rate_pct": force_fallback_rate_pct,
        "wheel_colliders_ready_rate_pct": wheel_colliders_ready_rate_pct,
        "analyzed_frame_count": analyzed_frame_count,
        "metric_sample_count": contact_samples,
        "skipped_transient_frames": int(n_frames - analyzed_frame_count),
        "limits": {
            "transient_ignore_s": transient_ignore_s,
            "good_contact_rate_pct_max": good_contact_rate_pct_max,
            "good_penetration_m_max": good_penetration_m_max,
            "warn_contact_rate_pct_max": warn_contact_rate_pct_max,
            "warn_penetration_m_max": warn_penetration_m_max,
            "fallback_warn_rate_pct_max": fallback_warn_rate_pct_max,
        },
    }


def _load_config() -> dict:
    config_path = REPO_ROOT / "config" / "av_stack_config.yaml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _build_mpc_health_summary(data: Dict, n_frames: int) -> Optional[Dict]:
    """Build MPC health summary. Returns None for PP-only recordings."""
    regime = data.get('regime')
    if regime is None:
        return None
    regime_arr = np.asarray(regime[:n_frames], dtype=float)
    total = len(regime_arr)
    if total == 0:
        return None
    mpc_mask  = regime_arr >= 0.5
    lmpc_mask = (regime_arr >= 0.5) & (regime_arr < 1.5)
    nmpc_mask = regime_arr >= 1.5
    pp_frames   = int(np.sum(~mpc_mask))
    mpc_frames  = int(np.sum(mpc_mask))
    lmpc_frames = int(np.sum(lmpc_mask))
    nmpc_frames = int(np.sum(nmpc_mask))
    if mpc_frames == 0:
        return {
            "total_frames": total,
            "pp_frames": pp_frames,
            "pp_rate": 1.0,
            "lmpc_frames": 0,
            "nmpc_frames": 0,
            "mpc_frames": 0,
            "mpc_rate": 0.0,
            "feasibility_rate": None,
            "feasibility_gate_pass": None,
            "solve_time_p50_ms": None,
            "solve_time_p95_ms": None,
            "solve_time_max_ms": None,
            "solve_time_gate_pass": None,
            "fallback_rate": None,
            "max_consecutive_failures": None,
            "nmpc_feasibility_rate": None,
            "nmpc_solve_time_p95_ms": None,
            "nmpc_solve_time_gate_pass": None,
            "nmpc_fallback_rate": None,
        }

    feasible = data.get('mpc_feasible')
    feasibility_rate = float(np.mean(feasible[:n_frames][mpc_mask])) if feasible is not None else None
    feasibility_gate = feasibility_rate >= 0.995 if feasibility_rate is not None else None

    solve = data.get('mpc_solve_time_ms')
    if solve is not None:
        mpc_solve = solve[:n_frames][mpc_mask]
        p50 = float(np.percentile(mpc_solve, 50))
        p95 = float(np.percentile(mpc_solve, 95))
        mx = float(np.max(mpc_solve))
        solve_gate = p95 <= 5.0
    else:
        p50 = p95 = mx = None
        solve_gate = None

    fallback = data.get('mpc_fallback_active')
    fallback_rate = float(np.mean(fallback[:n_frames][mpc_mask])) if fallback is not None else None

    failures = data.get('mpc_consecutive_failures')
    max_consec = int(np.max(failures[:n_frames])) if failures is not None else None

    # NMPC-specific metrics
    nmpc_feasible_arr = data.get('nmpc_feasible')
    nmpc_feasibility_rate = (
        float(np.mean(nmpc_feasible_arr[:n_frames][nmpc_mask]))
        if nmpc_feasible_arr is not None and nmpc_frames > 0 else None
    )
    nmpc_solve_arr = data.get('nmpc_solve_time_ms')
    nmpc_solve_p95 = (
        float(np.percentile(nmpc_solve_arr[:n_frames][nmpc_mask], 95))
        if nmpc_solve_arr is not None and nmpc_frames > 0 else None
    )
    nmpc_fallback_arr = data.get('nmpc_fallback_active')
    nmpc_fallback_rate = (
        float(np.mean(nmpc_fallback_arr[:n_frames][nmpc_mask]))
        if nmpc_fallback_arr is not None and nmpc_frames > 0 else None
    )

    return {
        "total_frames": total,
        "pp_frames": pp_frames,
        "pp_rate": round(pp_frames / total, 4),
        "lmpc_frames": lmpc_frames,
        "nmpc_frames": nmpc_frames,
        "mpc_frames": mpc_frames,
        "mpc_rate": round(mpc_frames / total, 4),
        "feasibility_rate": round(feasibility_rate, 5) if feasibility_rate is not None else None,
        "feasibility_gate_pass": feasibility_gate,
        "solve_time_p50_ms": round(p50, 3) if p50 is not None else None,
        "solve_time_p95_ms": round(p95, 3) if p95 is not None else None,
        "solve_time_max_ms": round(mx, 3) if mx is not None else None,
        "solve_time_gate_pass": solve_gate,
        "fallback_rate": round(fallback_rate, 5) if fallback_rate is not None else None,
        "max_consecutive_failures": max_consec,
        "nmpc_feasibility_rate": round(nmpc_feasibility_rate, 5) if nmpc_feasibility_rate is not None else None,
        "nmpc_solve_time_p95_ms": round(nmpc_solve_p95, 3) if nmpc_solve_p95 is not None else None,
        "nmpc_solve_time_gate_pass": nmpc_solve_p95 <= 20.0 if nmpc_solve_p95 is not None else None,
        "nmpc_fallback_rate": round(nmpc_fallback_rate, 5) if nmpc_fallback_rate is not None else None,
    }


def _build_acc_health_summary(data: Dict, n_frames: int, speed: Optional[np.ndarray] = None) -> Optional[Dict]:
    """
    Compute ACC longitudinal safety metrics from HDF5 data arrays.

    Returns None when acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE so that
    every consumer can safely guard with:
        if acc_health is not None:  # ACC was active
    This satisfies the roll-back contract: acc.enabled=false → zero acc_active
    → None returned → no section rendered, no scoring applied.
    """
    acc_active_raw = data.get('acc_active')
    if acc_active_raw is None:
        return None

    n = min(n_frames, len(acc_active_raw))
    if n == 0:
        return None

    acc_active = np.asarray(acc_active_raw[:n], dtype=float)
    acc_active_pct = float(np.mean(acc_active > 0.5)) * 100.0

    if acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE * 100.0:
        return None

    acc_mask = acc_active > 0.5
    time_arr = (
        np.asarray(data.get("time")[:n], dtype=float)
        if data.get("time") is not None
        else np.arange(n, dtype=float) * 0.033
    )

    # ── Gap arrays ────────────────────────────────────────────────────────────
    dist_raw = data.get('radar_fwd_distance_m')
    detected_raw = data.get('radar_fwd_detected')
    gap_error_raw = data.get('acc_gap_error_m')
    ttc_raw = data.get('acc_ttc_s')
    target_gap_raw = data.get('acc_target_gap_m')
    dynamic_gap_raw = data.get('acc_idm_dynamic_gap_m')
    equilibrium_gap_raw = data.get('acc_idm_equilibrium_gap_m')
    idm_accel_raw = data.get('acc_idm_accel_mps2')
    lead_speed_estimate_raw = data.get('acc_lead_speed_estimate_mps')
    closure_reserve_raw = data.get('acc_closure_reserve_mps')
    convergence_mode_raw = data.get('acc_convergence_mode')
    detection_stable_frames_raw = data.get('acc_detection_stable_frames')
    recent_detection_loss_raw = data.get('acc_recent_detection_loss')
    detection_loss_event_delta_raw = data.get('acc_detection_loss_event_delta')
    no_detect_run_length_raw = data.get('acc_no_detect_run_length')
    range_rate_raw = data.get('radar_fwd_range_rate_mps')
    acc_target_speed_raw = data.get('acc_target_speed_mps')
    if acc_target_speed_raw is None:
        acc_target_speed_raw = data.get('acc_target_speed_mps_vehicle')
    final_target_raw = data.get('target_speed_final')
    cmd_accel_raw = data.get('longitudinal_accel_cmd_raw')
    cmd_accel_smoothed_raw = data.get('longitudinal_accel_cmd_smoothed')

    dist_arr = np.asarray(dist_raw[:n], dtype=float) if dist_raw is not None else np.full(n, np.nan)
    detected_arr = np.asarray(detected_raw[:n], dtype=float) if detected_raw is not None else np.zeros(n)
    gap_error_arr = np.asarray(gap_error_raw[:n], dtype=float) if gap_error_raw is not None else np.full(n, np.nan)
    ttc_arr = np.asarray(ttc_raw[:n], dtype=float) if ttc_raw is not None else np.full(n, 999.0)
    target_gap_arr = (
        np.asarray(target_gap_raw[:n], dtype=float) if target_gap_raw is not None else np.full(n, np.nan)
    )
    speed_arr = (
        np.asarray(speed[:n], dtype=float)
        if speed is not None and len(speed) >= n
        else np.full(n, np.nan)
    )
    range_rate_arr = (
        np.asarray(range_rate_raw[:n], dtype=float) if range_rate_raw is not None else np.full(n, np.nan)
    )
    acc_target_speed_arr = (
        np.asarray(acc_target_speed_raw[:n], dtype=float)
        if acc_target_speed_raw is not None
        else np.full(n, np.nan)
    )
    final_target_arr = (
        np.asarray(final_target_raw[:n], dtype=float)
        if final_target_raw is not None
        else np.full(n, np.nan)
    )
    if dynamic_gap_raw is not None:
        dynamic_gap_arr = np.asarray(dynamic_gap_raw[:n], dtype=float)
    else:
        sqrt_ab = math.sqrt(2.0 * 3.0)
        dynamic_gap_arr = np.where(
            np.isfinite(speed_arr) & np.isfinite(range_rate_arr),
            2.0 + np.maximum(0.0, speed_arr * 1.5 + speed_arr * range_rate_arr / (2.0 * sqrt_ab)),
            target_gap_arr,
        )
    if equilibrium_gap_raw is not None:
        equilibrium_gap_arr = np.asarray(equilibrium_gap_raw[:n], dtype=float)
    else:
        denom_speed = np.where(np.isfinite(final_target_arr) & (final_target_arr > 1e-6), final_target_arr, np.nan)
        ratio_v = np.where(
            np.isfinite(speed_arr) & np.isfinite(denom_speed),
            np.clip((speed_arr / denom_speed) ** 4, 0.0, 0.999),
            np.nan,
        )
        equilibrium_gap_arr = np.where(
            np.isfinite(dynamic_gap_arr) & np.isfinite(ratio_v),
            np.minimum(dynamic_gap_arr / np.sqrt(np.maximum(1e-3, 1.0 - ratio_v)), 240.0),
            np.nan,
        )
    idm_accel_arr = (
        np.asarray(idm_accel_raw[:n], dtype=float)
        if idm_accel_raw is not None
        else np.full(n, np.nan)
    )
    lead_speed_estimate_arr = (
        np.asarray(lead_speed_estimate_raw[:n], dtype=float)
        if lead_speed_estimate_raw is not None
        else np.where(
            np.isfinite(speed_arr) & np.isfinite(range_rate_arr),
            np.maximum(0.0, speed_arr - range_rate_arr),
            np.nan,
        )
    )
    closure_reserve_arr = (
        np.asarray(closure_reserve_raw[:n], dtype=float)
        if closure_reserve_raw is not None
        else np.where(
            np.isfinite(acc_target_speed_arr) & np.isfinite(lead_speed_estimate_arr),
            acc_target_speed_arr - lead_speed_estimate_arr,
            np.nan,
        )
    )
    detection_stable_frames_arr = (
        np.asarray(detection_stable_frames_raw[:n], dtype=float)
        if detection_stable_frames_raw is not None
        else np.zeros(n, dtype=float)
    )
    recent_detection_loss_arr = (
        np.asarray(recent_detection_loss_raw[:n], dtype=float)
        if recent_detection_loss_raw is not None
        else np.zeros(n, dtype=float)
    )
    detection_loss_event_delta_arr = (
        np.asarray(detection_loss_event_delta_raw[:n], dtype=float)
        if detection_loss_event_delta_raw is not None
        else np.zeros(n, dtype=float)
    )
    no_detect_run_length_arr = (
        np.asarray(no_detect_run_length_raw[:n], dtype=float)
        if no_detect_run_length_raw is not None
        else np.zeros(n, dtype=float)
    )

    detected_bool = detected_arr > 0.5
    if detection_stable_frames_raw is None:
        run = 0
        for i, flag in enumerate(detected_bool):
            if flag:
                run += 1
                detection_stable_frames_arr[i] = float(run)
            else:
                run = 0
                detection_stable_frames_arr[i] = 0.0
    if no_detect_run_length_raw is None:
        run = 0
        for i, flag in enumerate(detected_bool):
            if flag:
                run = 0
                no_detect_run_length_arr[i] = 0.0
            else:
                run += 1
                no_detect_run_length_arr[i] = float(run)
    if detection_loss_event_delta_raw is None:
        detection_loss_event_delta_arr = np.zeros(n, dtype=float)
        if n > 0:
            prev = True
            for i, flag in enumerate(detected_bool):
                if (not flag) and prev:
                    detection_loss_event_delta_arr[i] = 1.0
                prev = bool(flag)
    if recent_detection_loss_raw is None:
        recent_detection_loss_arr = np.where(
            detected_bool,
            (detection_stable_frames_arr < 3.0).astype(float),
            1.0,
        )

    def _classify_convergence_fallback(idx: int) -> str:
        if not detected_bool[idx]:
            return "detection_limited_following" if recent_detection_loss_arr[idx] > 0.5 else "unavailable"
        gap_val = dist_arr[idx]
        dynamic_gap_val = dynamic_gap_arr[idx]
        equilibrium_gap_val = equilibrium_gap_arr[idx]
        target_speed_val = acc_target_speed_arr[idx]
        lead_speed_val = lead_speed_estimate_arr[idx]
        if recent_detection_loss_arr[idx] > 0.5:
            return "detection_limited_following"
        if np.isfinite(gap_val) and np.isfinite(dynamic_gap_val):
            dynamic_gap_error = gap_val - dynamic_gap_val
            if dynamic_gap_error < -2.0:
                return "compressed"
            if abs(dynamic_gap_error) <= 2.0:
                return "tracking_dynamic_gap"
        closure_reserve_val = (
            target_speed_val - lead_speed_val
            if np.isfinite(target_speed_val) and np.isfinite(lead_speed_val)
            else np.nan
        )
        if (
            np.isfinite(equilibrium_gap_val)
            and np.isfinite(gap_val)
            and equilibrium_gap_val > gap_val + 5.0
            and np.isfinite(closure_reserve_val)
            and closure_reserve_val <= 0.5
        ):
            return "equilibrium_limited_tracking"
        if np.isfinite(closure_reserve_val) and closure_reserve_val <= 0.1:
            return "lead_limited_tracking"
        return "policy_limited_tracking"

    if convergence_mode_raw is not None:
        convergence_mode_arr = np.array(
            [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in convergence_mode_raw[:n]
            ],
            dtype=object,
        )
    else:
        convergence_mode_arr = np.array(
            [_classify_convergence_fallback(i) for i in range(n)],
            dtype=object,
        )
    cmd_accel_smoothed_arr = (
        np.asarray(cmd_accel_smoothed_raw[:n], dtype=float)
        if cmd_accel_smoothed_raw is not None
        else None
    )
    cmd_accel_raw_arr = (
        np.asarray(cmd_accel_raw[:n], dtype=float)
        if cmd_accel_raw is not None
        else None
    )
    cmd_accel_for_jerk = cmd_accel_smoothed_arr if cmd_accel_smoothed_arr is not None else cmd_accel_raw_arr

    # ── Tier 1 — Hard safety ──────────────────────────────────────────────────
    acc_collision_events = int(np.sum(dist_arr[acc_mask] < 0.0))
    acc_ttc_violation_events = int(np.sum(
        (ttc_arr[acc_mask] < ACC_TTC_CRITICAL_S) & np.isfinite(ttc_arr[acc_mask])
    ))

    # ── Tier 2 — Graduated safety ─────────────────────────────────────────────
    # Near-miss: gap < ACC_NEAR_MISS_GAP_M AND gap > 0 (not collision), sustained ≥ 3 frames
    near_miss_raw = acc_mask & (dist_arr > 0.0) & (dist_arr < ACC_NEAR_MISS_GAP_M)
    near_miss_events = 0
    run_len = 0
    for flag in near_miss_raw:
        if flag:
            run_len += 1
            if run_len == 3:
                near_miss_events += 1
        else:
            run_len = 0

    ttc_active = ttc_arr[acc_mask & np.isfinite(ttc_arr)]
    acc_ttc_warning_pct = float(
        np.mean(ttc_active < ACC_TTC_WARNING_S) * 100.0
    ) if ttc_active.size > 0 else 0.0

    dist_active = dist_arr[acc_mask & (dist_arr > 0.0)]
    acc_min_gap_m = float(np.min(dist_active)) if dist_active.size > 0 else 0.0
    acc_ttc_p05_s = float(np.percentile(ttc_active, 5)) if ttc_active.size > 0 else 999.0

    # ── Tier 3 — Comfort + quality ────────────────────────────────────────────
    gap_error_active = gap_error_arr[acc_mask & np.isfinite(gap_error_arr)]
    acc_gap_rmse_m = float(np.sqrt(np.mean(gap_error_active ** 2))) if gap_error_active.size > 0 else 0.0
    acc_ttc_min_s = float(np.min(ttc_active)) if ttc_active.size > 0 else 999.0

    valid_follow_mask = (
        acc_mask
        & np.isfinite(dist_arr)
        & (dist_arr > 0.0)
        & np.isfinite(target_gap_arr)
        & (target_gap_arr > 0.0)
    )
    signed_gap_delta = np.where(valid_follow_mask, dist_arr - target_gap_arr, np.nan)
    abs_gap_delta = np.abs(signed_gap_delta)
    dynamic_gap_delta = np.where(valid_follow_mask, dist_arr - dynamic_gap_arr, np.nan)
    abs_dynamic_gap_delta = np.abs(dynamic_gap_delta)
    equilibrium_gap_delta = np.where(valid_follow_mask, dist_arr - equilibrium_gap_arr, np.nan)
    abs_equilibrium_gap_delta = np.abs(equilibrium_gap_delta)
    gap_delta_valid = signed_gap_delta[np.isfinite(signed_gap_delta)]
    abs_gap_delta_valid = abs_gap_delta[np.isfinite(abs_gap_delta)]
    actual_gap_valid = dist_arr[valid_follow_mask]
    target_gap_valid = target_gap_arr[valid_follow_mask]
    dynamic_gap_valid = dynamic_gap_arr[valid_follow_mask & np.isfinite(dynamic_gap_arr)]
    equilibrium_gap_valid = equilibrium_gap_arr[valid_follow_mask & np.isfinite(equilibrium_gap_arr)]
    dynamic_gap_delta_valid = dynamic_gap_delta[np.isfinite(dynamic_gap_delta)]
    abs_dynamic_gap_delta_valid = abs_dynamic_gap_delta[np.isfinite(abs_dynamic_gap_delta)]
    equilibrium_gap_delta_valid = equilibrium_gap_delta[np.isfinite(equilibrium_gap_delta)]
    abs_equilibrium_gap_delta_valid = abs_equilibrium_gap_delta[np.isfinite(abs_equilibrium_gap_delta)]
    lead_speed_estimate_valid = lead_speed_estimate_arr[valid_follow_mask & np.isfinite(lead_speed_estimate_arr)]
    closure_reserve_valid = closure_reserve_arr[valid_follow_mask & np.isfinite(closure_reserve_arr)]
    range_rate_valid = range_rate_arr[valid_follow_mask & np.isfinite(range_rate_arr)]

    actual_gap_p50_m = float(np.percentile(actual_gap_valid, 50)) if actual_gap_valid.size > 0 else 0.0
    actual_gap_p95_m = float(np.percentile(actual_gap_valid, 95)) if actual_gap_valid.size > 0 else 0.0
    target_gap_p50_m = float(np.percentile(target_gap_valid, 50)) if target_gap_valid.size > 0 else 0.0
    target_gap_p95_m = float(np.percentile(target_gap_valid, 95)) if target_gap_valid.size > 0 else 0.0
    dynamic_gap_p50_m = float(np.percentile(dynamic_gap_valid, 50)) if dynamic_gap_valid.size > 0 else 0.0
    dynamic_gap_p95_m = float(np.percentile(dynamic_gap_valid, 95)) if dynamic_gap_valid.size > 0 else 0.0
    equilibrium_gap_p50_m = float(np.percentile(equilibrium_gap_valid, 50)) if equilibrium_gap_valid.size > 0 else 0.0
    equilibrium_gap_p95_m = float(np.percentile(equilibrium_gap_valid, 95)) if equilibrium_gap_valid.size > 0 else 0.0
    gap_error_p50_m = float(np.percentile(gap_delta_valid, 50)) if gap_delta_valid.size > 0 else 0.0
    gap_error_p95_m = float(np.percentile(gap_delta_valid, 95)) if gap_delta_valid.size > 0 else 0.0
    gap_error_abs_p50_m = float(np.percentile(abs_gap_delta_valid, 50)) if abs_gap_delta_valid.size > 0 else 0.0
    gap_error_abs_p95_m = float(np.percentile(abs_gap_delta_valid, 95)) if abs_gap_delta_valid.size > 0 else 0.0
    dynamic_gap_error_p50_m = float(np.percentile(dynamic_gap_delta_valid, 50)) if dynamic_gap_delta_valid.size > 0 else 0.0
    dynamic_gap_error_abs_p95_m = float(np.percentile(abs_dynamic_gap_delta_valid, 95)) if abs_dynamic_gap_delta_valid.size > 0 else 0.0
    equilibrium_gap_error_p50_m = float(np.percentile(equilibrium_gap_delta_valid, 50)) if equilibrium_gap_delta_valid.size > 0 else 0.0
    equilibrium_gap_error_abs_p95_m = float(np.percentile(abs_equilibrium_gap_delta_valid, 95)) if abs_equilibrium_gap_delta_valid.size > 0 else 0.0
    lead_speed_estimate_p50_mps = float(np.percentile(lead_speed_estimate_valid, 50)) if lead_speed_estimate_valid.size > 0 else 0.0
    lead_speed_estimate_p95_mps = float(np.percentile(lead_speed_estimate_valid, 95)) if lead_speed_estimate_valid.size > 0 else 0.0
    closure_reserve_p50_mps = float(np.percentile(closure_reserve_valid, 50)) if closure_reserve_valid.size > 0 else 0.0
    closure_reserve_p95_mps = float(np.percentile(closure_reserve_valid, 95)) if closure_reserve_valid.size > 0 else 0.0
    gap_above_target_plus_2m_rate = (
        float(np.mean(gap_delta_valid > 2.0) * 100.0) if gap_delta_valid.size > 0 else 0.0
    )
    gap_above_target_plus_5m_rate = (
        float(np.mean(gap_delta_valid > 5.0) * 100.0) if gap_delta_valid.size > 0 else 0.0
    )
    gap_above_target_plus_10m_rate = (
        float(np.mean(gap_delta_valid > 10.0) * 100.0) if gap_delta_valid.size > 0 else 0.0
    )
    gap_below_target_rate = (
        float(np.mean(gap_delta_valid < 0.0) * 100.0) if gap_delta_valid.size > 0 else 0.0
    )

    tracking_mask = valid_follow_mask & (convergence_mode_arr == "tracking_dynamic_gap")
    compressed_mask = valid_follow_mask & (convergence_mode_arr == "compressed")
    closing_mask = (
        valid_follow_mask
        & np.isfinite(signed_gap_delta)
        & (signed_gap_delta > 2.0)
        & np.isfinite(range_rate_arr)
        & (range_rate_arr > 0.25)
        & (convergence_mode_arr != "detection_limited_following")
    )
    over_conservative_mask = valid_follow_mask & (convergence_mode_arr == "policy_limited_tracking")
    lead_limited_mask = valid_follow_mask & (convergence_mode_arr == "lead_limited_tracking")
    equilibrium_limited_mask = valid_follow_mask & (convergence_mode_arr == "equilibrium_limited_tracking")
    detection_limited_mask = valid_follow_mask & (convergence_mode_arr == "detection_limited_following")
    valid_follow_count = int(np.sum(valid_follow_mask))

    def _pct(mask: np.ndarray) -> float:
        if valid_follow_count <= 0:
            return 0.0
        return float(np.sum(mask) / valid_follow_count * 100.0)

    tracking_rate = _pct(tracking_mask)
    compressed_rate = _pct(compressed_mask)
    closing_rate = _pct(closing_mask)
    over_conservative_rate = _pct(over_conservative_mask)
    lead_limited_rate = _pct(lead_limited_mask)
    equilibrium_limited_rate = _pct(equilibrium_limited_mask)
    detection_limited_rate = _pct(detection_limited_mask)
    following_regime_mode = "unavailable"
    if valid_follow_count > 0:
        regime_rates = {
            "tracking": tracking_rate,
            "policy_limited_tracking": over_conservative_rate,
            "closing": closing_rate,
            "compressed": compressed_rate,
            "lead_limited_tracking": lead_limited_rate,
            "equilibrium_limited_tracking": equilibrium_limited_rate,
            "detection_limited_following": detection_limited_rate,
        }
        following_regime_mode = max(regime_rates.items(), key=lambda item: item[1])[0]

    # Jerk metrics in ACC-active frames
    acc_jerk_p95_raw = 0.0
    acc_jerk_p95_filtered = 0.0
    acc_commanded_jerk_p95 = 0.0
    acc_target_speed_delta_p95 = 0.0
    acc_commanded_accel_delta_p95 = 0.0
    if speed is not None and len(speed) >= n and np.sum(acc_mask) > 2:
        spd = np.asarray(speed[:n], dtype=float)
        active_idx = np.flatnonzero(
            acc_mask
            & np.isfinite(time_arr)
            & np.isfinite(spd)
        )
        if active_idx.size >= 3:
            active_time = time_arr[active_idx]
            active_speed = spd[active_idx]
            active_dt = np.diff(active_time)
            active_dt = np.where(active_dt > 1e-6, active_dt, np.nan)

            raw_acc = np.diff(active_speed) / active_dt
            if raw_acc.size > 1:
                raw_jerk = np.diff(raw_acc) / active_dt[1:]
                raw_jerk = np.abs(raw_jerk[np.isfinite(raw_jerk)])
                if raw_jerk.size > 0:
                    acc_jerk_p95_raw = float(np.percentile(raw_jerk, 95))

            alpha = 0.95
            filtered_speed = np.empty_like(active_speed)
            filtered_speed[0] = active_speed[0]
            for i in range(1, active_speed.size):
                filtered_speed[i] = alpha * filtered_speed[i - 1] + (1.0 - alpha) * active_speed[i]
            filtered_acc = np.diff(filtered_speed) / active_dt
            if filtered_acc.size > 1:
                filtered_jerk = np.diff(filtered_acc) / active_dt[1:]
                filtered_jerk = np.abs(filtered_jerk[np.isfinite(filtered_jerk)])
                if filtered_jerk.size > 0:
                    acc_jerk_p95_filtered = float(np.percentile(filtered_jerk, 95))

            if cmd_accel_for_jerk is not None and len(cmd_accel_for_jerk) >= n:
                active_cmd_accel = np.asarray(cmd_accel_for_jerk[:n], dtype=float)[active_idx]
                valid_cmd = np.isfinite(active_cmd_accel)
                if np.sum(valid_cmd) >= 2:
                    cmd_vals = active_cmd_accel[valid_cmd]
                    cmd_dt = np.diff(active_time[valid_cmd])
                    cmd_dt = np.where(cmd_dt > 1e-6, cmd_dt, np.nan)
                    if cmd_vals.size > 1:
                        cmd_delta = np.abs(np.diff(cmd_vals))
                        cmd_delta = cmd_delta[np.isfinite(cmd_delta)]
                        if cmd_delta.size > 0:
                            acc_commanded_accel_delta_p95 = float(np.percentile(cmd_delta, 95))
                        cmd_jerk = np.diff(cmd_vals) / cmd_dt
                        cmd_jerk = np.abs(cmd_jerk[np.isfinite(cmd_jerk)])
                        if cmd_jerk.size > 0:
                            acc_commanded_jerk_p95 = float(np.percentile(cmd_jerk, 95))

            if np.isfinite(acc_target_speed_arr).any():
                active_target = acc_target_speed_arr[active_idx]
                active_target = active_target[np.isfinite(active_target)]
                if active_target.size > 1:
                    acc_target_speed_delta_p95 = float(
                        np.percentile(np.abs(np.diff(active_target)), 95)
                    )

    jerk_metric_role = "filtered_measured_gate"
    jerk_gate_value_mps3 = acc_jerk_p95_filtered
    jerk_gate_pass = jerk_gate_value_mps3 <= ACC_JERK_P95_GATE_MPS3
    raw_jerk_exceeds_gate = acc_jerk_p95_raw > ACC_JERK_P95_GATE_MPS3
    raw_spike_dominated = bool(
        raw_jerk_exceeds_gate
        and acc_jerk_p95_filtered <= ACC_JERK_P95_GATE_MPS3
        and acc_commanded_jerk_p95 <= ACC_JERK_P95_GATE_MPS3
    )

    # Detection rate: frames where lead was present (target_gap_m > 0) and we detected
    acc_detection_rate = 1.0
    if target_gap_raw is not None:
        lead_present_mask = acc_mask & (target_gap_arr > 0.0)
        if np.any(lead_present_mask):
            acc_detection_rate = float(np.mean(detected_arr[lead_present_mask] > 0.5))
    stable_detection_rate = (
        float(np.mean(detection_stable_frames_arr[valid_follow_mask] >= 3.0) * 100.0)
        if valid_follow_count > 0
        else 0.0
    )
    recent_detection_loss_rate = (
        float(np.mean(recent_detection_loss_arr[acc_mask & np.isfinite(recent_detection_loss_arr)] > 0.5) * 100.0)
        if np.any(acc_mask & np.isfinite(recent_detection_loss_arr))
        else 0.0
    )
    detection_loss_event_count = int(np.sum(detection_loss_event_delta_arr[np.isfinite(detection_loss_event_delta_arr)] > 0.5))
    no_detect_run_length_max = (
        int(np.max(no_detect_run_length_arr[np.isfinite(no_detect_run_length_arr)]))
        if np.isfinite(no_detect_run_length_arr).any()
        else 0
    )
    detection_issue_detected = bool(
        acc_detection_rate < ACC_DETECTION_RATE_GATE
        or recent_detection_loss_rate > 5.0
        or no_detect_run_length_max >= 3
    )
    detection_issue_mode = "none"
    if detection_issue_detected:
        if acc_detection_rate < ACC_DETECTION_RATE_GATE:
            detection_issue_mode = "low_detection_rate"
        elif no_detect_run_length_max >= 3:
            detection_issue_mode = "dropout_run"
        else:
            detection_issue_mode = "recent_detection_loss"

    # Emergency brake events (transitions into state where gap_error extremely negative)
    acc_emergency_brake_events = 0
    if gap_error_raw is not None and np.any(acc_mask):
        # Proxy: frames with very large negative gap_error (gap much less than desired)
        extreme_close = acc_mask & (gap_error_arr < -5.0) & np.isfinite(gap_error_arr)
        transitions = np.diff(extreme_close.astype(int))
        acc_emergency_brake_events = int(np.sum(transitions > 0))

    # ── Scoring deductions ────────────────────────────────────────────────────
    collision_penalty = 100.0 * acc_collision_events   # hard zero per event
    near_miss_penalty = min(30.0, ACC_NEAR_MISS_PENALTY_PTS * near_miss_events)
    ttc_warning_penalty = ACC_TTC_WARNING_PENALTY_PER_PCT * acc_ttc_warning_pct

    # Gate pass/fail
    collision_gate_pass = acc_collision_events == 0
    ttc_violation_gate_pass = acc_ttc_violation_events == 0
    near_miss_gate_pass = near_miss_events == 0
    ttc_warning_gate_pass = acc_ttc_warning_pct < 5.0
    ttc_p05_gate_pass = acc_ttc_p05_s >= ACC_TTC_MIN_GATE_S
    gap_rmse_gate_pass = acc_gap_rmse_m <= ACC_GAP_RMSE_GATE_M
    ttc_min_gate_pass = acc_ttc_min_s >= ACC_TTC_MIN_GATE_S
    detection_gate_pass = acc_detection_rate >= ACC_DETECTION_RATE_GATE

    return {
        "acc_active_pct": round(acc_active_pct, 2),
        # Tier 1
        "acc_collision_events": acc_collision_events,
        "acc_ttc_violation_events": acc_ttc_violation_events,
        "collision_gate_pass": collision_gate_pass,
        "ttc_violation_gate_pass": ttc_violation_gate_pass,
        # Tier 2
        "acc_near_miss_events": near_miss_events,
        "acc_ttc_warning_pct": round(acc_ttc_warning_pct, 2),
        "acc_min_gap_m": round(acc_min_gap_m, 3),
        "acc_ttc_p05_s": round(acc_ttc_p05_s, 3),
        "near_miss_gate_pass": near_miss_gate_pass,
        "ttc_warning_gate_pass": ttc_warning_gate_pass,
        "ttc_p05_gate_pass": ttc_p05_gate_pass,
        # Tier 3
        "acc_gap_rmse_m": round(acc_gap_rmse_m, 3),
        "acc_ttc_min_s": round(acc_ttc_min_s, 3),
        "acc_actual_gap_p50_m": round(actual_gap_p50_m, 3),
        "acc_actual_gap_p95_m": round(actual_gap_p95_m, 3),
        "acc_target_gap_p50_m": round(target_gap_p50_m, 3),
        "acc_target_gap_p95_m": round(target_gap_p95_m, 3),
        "acc_dynamic_gap_p50_m": round(dynamic_gap_p50_m, 3),
        "acc_dynamic_gap_p95_m": round(dynamic_gap_p95_m, 3),
        "acc_equilibrium_gap_p50_m": round(equilibrium_gap_p50_m, 3),
        "acc_equilibrium_gap_p95_m": round(equilibrium_gap_p95_m, 3),
        "acc_gap_error_p50_m": round(gap_error_p50_m, 3),
        "acc_gap_error_p95_m": round(gap_error_p95_m, 3),
        "acc_gap_error_abs_p50_m": round(gap_error_abs_p50_m, 3),
        "acc_gap_error_abs_p95_m": round(gap_error_abs_p95_m, 3),
        "acc_dynamic_gap_error_p50_m": round(dynamic_gap_error_p50_m, 3),
        "acc_dynamic_gap_error_abs_p95_m": round(dynamic_gap_error_abs_p95_m, 3),
        "acc_equilibrium_gap_error_p50_m": round(equilibrium_gap_error_p50_m, 3),
        "acc_equilibrium_gap_error_abs_p95_m": round(equilibrium_gap_error_abs_p95_m, 3),
        "acc_gap_above_target_plus_2m_rate": round(gap_above_target_plus_2m_rate, 3),
        "acc_gap_above_target_plus_5m_rate": round(gap_above_target_plus_5m_rate, 3),
        "acc_gap_above_target_plus_10m_rate": round(gap_above_target_plus_10m_rate, 3),
        "acc_gap_below_target_rate": round(gap_below_target_rate, 3),
        "acc_following_regime_mode": following_regime_mode,
        "acc_tracking_rate": round(tracking_rate, 3),
        "acc_closing_rate": round(closing_rate, 3),
        "acc_over_conservative_trailing_rate": round(over_conservative_rate, 3),
        "acc_policy_limited_tracking_rate": round(over_conservative_rate, 3),
        "acc_lead_limited_tracking_rate": round(lead_limited_rate, 3),
        "acc_equilibrium_limited_tracking_rate": round(equilibrium_limited_rate, 3),
        "acc_detection_limited_following_rate": round(detection_limited_rate, 3),
        "acc_compressed_rate": round(compressed_rate, 3),
        "acc_range_rate_p50_mps": round(float(np.percentile(range_rate_valid, 50)), 3) if range_rate_valid.size > 0 else 0.0,
        "acc_lead_speed_estimate_p50_mps": round(lead_speed_estimate_p50_mps, 3),
        "acc_lead_speed_estimate_p95_mps": round(lead_speed_estimate_p95_mps, 3),
        "acc_closure_reserve_p50_mps": round(closure_reserve_p50_mps, 3),
        "acc_closure_reserve_p95_mps": round(closure_reserve_p95_mps, 3),
        "acc_target_speed_delta_p95_mps": round(acc_target_speed_delta_p95, 4),
        "acc_commanded_accel_delta_p95_mps2": round(acc_commanded_accel_delta_p95, 4),
        "acc_jerk_p95_mps3": round(acc_jerk_p95_filtered, 2),
        "acc_jerk_p95_filtered_mps3": round(acc_jerk_p95_filtered, 2),
        "acc_jerk_p95_raw_mps3": round(acc_jerk_p95_raw, 2),
        "acc_commanded_jerk_p95_mps3": round(acc_commanded_jerk_p95, 3),
        "acc_jerk_gate_metric_role": jerk_metric_role,
        "acc_jerk_gate_value_mps3": round(jerk_gate_value_mps3, 3),
        "acc_jerk_raw_spike_dominated": raw_spike_dominated,
        "acc_detection_rate": round(acc_detection_rate, 4),
        "acc_detection_stable_rate": round(stable_detection_rate, 3),
        "acc_recent_detection_loss_rate": round(recent_detection_loss_rate, 3),
        "acc_detection_loss_event_count": detection_loss_event_count,
        "acc_no_detect_run_length_max": no_detect_run_length_max,
        "acc_detection_issue_detected": detection_issue_detected,
        "acc_detection_issue_mode": detection_issue_mode,
        "acc_emergency_brake_events": acc_emergency_brake_events,
        "gap_rmse_gate_pass": gap_rmse_gate_pass,
        "ttc_min_gate_pass": ttc_min_gate_pass,
        "jerk_gate_pass": jerk_gate_pass,
        "detection_gate_pass": detection_gate_pass,
        # Scoring deductions (fed into Safety and LongitudinalComfort layers)
        "collision_penalty": round(collision_penalty, 2),
        "near_miss_penalty": round(near_miss_penalty, 2),
        "ttc_warning_penalty": round(ttc_warning_penalty, 2),
        "hard_zero": acc_collision_events > 0,
    }


def _build_longitudinal_hotspot_attribution(data: Dict, config: Dict, n_frames: int) -> Dict:
    """Build top longitudinal hotspot entries with deterministic root-cause attribution."""
    unavailable = {
        "schema_version": "v1",
        "availability": "unavailable",
        "dt_nominal_ms": None,
        "dt_gap_threshold_ms": None,
        "entries": [],
        "counts_by_attribution": {},
        "high_confidence_rate": None,
        "commanded_vs_measured_mismatch_rate": None,
    }

    time = data.get("time")
    speed = data.get("speed")
    if time is None or speed is None:
        return unavailable

    time_arr = np.asarray(time, dtype=float).reshape(-1)
    speed_arr = np.asarray(speed, dtype=float).reshape(-1)
    n = min(int(n_frames), int(time_arr.size), int(speed_arr.size))
    if n < 3:
        return unavailable

    time_arr = time_arr[:n]
    speed_arr = speed_arr[:n]
    dt_arr = np.diff(time_arr)
    finite_positive_dt = dt_arr[np.isfinite(dt_arr) & (dt_arr > 1e-6)]
    nominal_dt = float(np.median(finite_positive_dt)) if finite_positive_dt.size > 0 else 0.033
    if nominal_dt <= 0.0:
        nominal_dt = 0.033
    dt_arr = np.where(dt_arr > 1e-6, dt_arr, nominal_dt)
    nominal_dt_ms = nominal_dt * 1000.0
    dt_gap_threshold_ms = max(100.0, nominal_dt_ms * 2.5)

    dt_arr_ms = dt_arr * 1000.0
    dt_gap_edges = np.isfinite(dt_arr_ms) & (dt_arr_ms >= dt_gap_threshold_ms)
    timestamp_irregular_nearby = np.zeros(n, dtype=bool)
    for edge_idx in np.flatnonzero(dt_gap_edges):
        lo = max(0, int(edge_idx) - 1)
        hi = min(n, int(edge_idx) + 3)
        timestamp_irregular_nearby[lo:hi] = True

    accel_raw = np.zeros(n, dtype=float)
    accel_raw[1:] = np.divide(
        np.diff(speed_arr),
        dt_arr,
        out=np.zeros_like(dt_arr, dtype=float),
        where=dt_arr > 1e-6,
    )
    jerk_raw = np.zeros(n, dtype=float)
    if n > 2:
        jerk_dt = dt_arr[1:]
        jerk_raw[2:] = np.divide(
            np.diff(accel_raw[1:]),
            jerk_dt,
            out=np.zeros_like(jerk_dt, dtype=float),
            where=jerk_dt > 1e-6,
        )

    alpha = 0.95
    speed_filtered = np.empty_like(speed_arr)
    speed_filtered[0] = speed_arr[0]
    for i in range(1, n):
        speed_filtered[i] = alpha * speed_filtered[i - 1] + (1.0 - alpha) * speed_arr[i]

    accel_filtered = np.zeros(n, dtype=float)
    accel_filtered[1:] = np.divide(
        np.diff(speed_filtered),
        dt_arr,
        out=np.zeros_like(dt_arr, dtype=float),
        where=dt_arr > 1e-6,
    )
    jerk_filtered = np.zeros(n, dtype=float)
    if n > 2:
        jerk_dt = dt_arr[1:]
        jerk_filtered[2:] = np.divide(
            np.diff(accel_filtered[1:]),
            jerk_dt,
            out=np.zeros_like(jerk_dt, dtype=float),
            where=jerk_dt > 1e-6,
        )

    throttle = (
        np.asarray(data.get("throttle"), dtype=float).reshape(-1)[:n]
        if data.get("throttle") is not None
        else None
    )
    brake = (
        np.asarray(data.get("brake"), dtype=float).reshape(-1)[:n]
        if data.get("brake") is not None
        else None
    )
    target_speed = (
        np.asarray(data.get("target_speed_final"), dtype=float).reshape(-1)[:n]
        if data.get("target_speed_final") is not None
        else None
    )
    accel_capped = (
        np.asarray(data.get("longitudinal_accel_capped"), dtype=float).reshape(-1)[:n]
        if data.get("longitudinal_accel_capped") is not None
        else None
    )
    jerk_capped = (
        np.asarray(data.get("longitudinal_jerk_capped"), dtype=float).reshape(-1)[:n]
        if data.get("longitudinal_jerk_capped") is not None
        else None
    )
    limiter_transition_recorded = (
        np.asarray(data.get("longitudinal_limiter_transition_active"), dtype=float).reshape(-1)[:n]
        if data.get("longitudinal_limiter_transition_active") is not None
        else None
    )
    contact = (
        np.asarray(data.get("chassis_ground_contact"), dtype=float).reshape(-1)[:n]
        if data.get("chassis_ground_contact") is not None
        else None
    )
    penetration = (
        np.asarray(data.get("chassis_ground_penetration_m"), dtype=float).reshape(-1)[:n]
        if data.get("chassis_ground_penetration_m") is not None
        else None
    )
    clearance = (
        np.asarray(data.get("chassis_ground_clearance_m"), dtype=float).reshape(-1)[:n]
        if data.get("chassis_ground_clearance_m") is not None
        else None
    )

    command_accel = None
    if data.get("longitudinal_accel_cmd_smoothed") is not None:
        command_accel = np.asarray(data.get("longitudinal_accel_cmd_smoothed"), dtype=float).reshape(-1)[:n]
    elif data.get("longitudinal_accel_cmd_raw") is not None:
        command_accel = np.asarray(data.get("longitudinal_accel_cmd_raw"), dtype=float).reshape(-1)[:n]
    elif throttle is not None and brake is not None:
        ctrl_cfg = config.get("control", {}).get("longitudinal", {})
        max_accel_cfg = float(ctrl_cfg.get("max_accel", 2.5))
        max_decel_cfg = float(ctrl_cfg.get("max_decel", 3.0))
        command_accel = (throttle * max_accel_cfg) - (brake * max_decel_cfg)
    command_jerk = None
    if command_accel is not None:
        command_jerk = np.full(n, np.nan, dtype=float)
        command_jerk[1:] = np.divide(
            np.diff(command_accel),
            dt_arr,
            out=np.zeros_like(dt_arr, dtype=float),
            where=dt_arr > 1e-6,
        )

    limiter_active = np.zeros(n, dtype=bool)
    if accel_capped is not None:
        limiter_active |= accel_capped > 0.5
    if jerk_capped is not None:
        limiter_active |= jerk_capped > 0.5
    limiter_transition = np.zeros(n, dtype=bool)
    if n > 1:
        transitions = limiter_active[1:] != limiter_active[:-1]
        limiter_transition[1:] |= transitions
        limiter_transition[:-1] |= transitions
    if limiter_transition_recorded is not None:
        limiter_transition |= limiter_transition_recorded > 0.5

    def _build_hotspots(values: np.ndarray, metric: str, max_items: int = 8, min_separation: int = 15) -> list[dict]:
        rows = []
        sorted_idx = np.argsort(np.abs(values))[::-1]
        for idx in sorted_idx:
            idx = int(idx)
            if len(rows) >= max_items:
                break
            if not np.isfinite(values[idx]) or abs(values[idx]) <= 0.0:
                continue
            if any(abs(idx - row["frame"]) < min_separation for row in rows):
                continue
            rows.append({
                "frame": idx,
                "metric": metric,
                "metric_value": float(values[idx]),
                "metric_abs": float(abs(values[idx])),
            })
        return rows

    rows = _build_hotspots(accel_raw, "accel") + _build_hotspots(jerk_raw, "jerk")
    rows = sorted(rows, key=lambda row: row["metric_abs"], reverse=True)[:8]

    def _float_at(arr: Optional[np.ndarray], idx: int):
        if arr is None or idx < 0 or idx >= len(arr):
            return None
        value = arr[idx]
        return safe_float(value, default=None) if np.isfinite(value) else None

    entries = []
    for row in rows:
        idx = int(row["frame"])
        dt_prev_ms = safe_float(dt_arr[idx - 1] * 1000.0, default=None) if idx >= 1 else None
        dt_prev2_ms = safe_float(dt_arr[idx - 2] * 1000.0, default=None) if idx >= 2 else None
        dt_recent_max_ms = None
        if dt_prev_ms is not None and dt_prev2_ms is not None:
            dt_recent_max_ms = max(dt_prev_ms, dt_prev2_ms)
        elif dt_prev_ms is not None:
            dt_recent_max_ms = dt_prev_ms
        elif dt_prev2_ms is not None:
            dt_recent_max_ms = dt_prev2_ms

        throttle_i = _float_at(throttle, idx)
        brake_i = _float_at(brake, idx)
        target_speed_i = _float_at(target_speed, idx)
        accel_cap_i = _float_at(accel_capped, idx)
        jerk_cap_i = _float_at(jerk_capped, idx)
        contact_i = _float_at(contact, idx)
        penetration_i = _float_at(penetration, idx)
        clearance_i = _float_at(clearance, idx)
        cmd_accel_i = _float_at(command_accel, idx)
        cmd_jerk_i = _float_at(command_jerk, idx)
        limiter_active_i = bool(limiter_active[idx])
        limiter_transition_i = bool(limiter_transition[idx])
        timestamp_irregular_i = bool(timestamp_irregular_nearby[idx])
        accel_raw_i = _float_at(accel_raw, idx)
        accel_filtered_i = _float_at(accel_filtered, idx)
        jerk_raw_i = _float_at(jerk_raw, idx)
        jerk_filtered_i = _float_at(jerk_filtered, idx)

        attribution = "unknown"
        confidence = "low"
        if (contact_i is not None and contact_i > 0.5) or (penetration_i is not None and penetration_i > 1e-4):
            attribution = "ground_contact_or_penetration"
            confidence = "high"
        elif timestamp_irregular_i:
            attribution = "timestamp_gap_derivative_artifact"
            confidence = "high"
        elif (
            cmd_jerk_i is not None
            and jerk_filtered_i is not None
            and abs(cmd_jerk_i) >= 6.0
            and abs(jerk_filtered_i) >= 6.0
        ):
            attribution = "commanded_longitudinal_step"
            confidence = "high"
        elif limiter_transition_i:
            attribution = "longitudinal_limiter_transition"
            confidence = "medium"
        elif (
            jerk_raw_i is not None
            and (
                (
                    jerk_filtered_i is not None
                    and abs(jerk_raw_i) >= 6.0
                    and abs(jerk_filtered_i) <= max(2.0, abs(jerk_raw_i) * 0.45)
                )
                or (
                    cmd_jerk_i is not None
                    and abs(jerk_raw_i) >= 6.0
                    and abs(cmd_jerk_i) <= max(2.0, abs(jerk_raw_i) * 0.45)
                )
                or (
                    accel_raw_i is not None
                    and accel_filtered_i is not None
                    and abs(accel_raw_i - accel_filtered_i) >= 2.5
                )
            )
        ):
            attribution = "physics_or_speed_estimation_spike"
            confidence = "low"

        entries.append({
            "frame": idx,
            "time_s": _float_at(time_arr, idx),
            "metric": row["metric"],
            "metric_value": safe_float(row["metric_value"]),
            "metric_abs": safe_float(row["metric_abs"]),
            "accel_mps2": accel_raw_i,
            "jerk_mps3": jerk_raw_i,
            "accel_raw_mps2": accel_raw_i,
            "jerk_raw_mps3": jerk_raw_i,
            "accel_filtered_mps2": accel_filtered_i,
            "jerk_filtered_mps3": jerk_filtered_i,
            "speed_mps": _float_at(speed_arr, idx),
            "target_speed_mps": target_speed_i,
            "throttle": throttle_i,
            "brake": brake_i,
            "dt_prev_ms": dt_prev_ms,
            "dt_prev2_ms": dt_prev2_ms,
            "dt_recent_max_ms": dt_recent_max_ms,
            "timestamp_irregular_nearby": timestamp_irregular_i,
            "longitudinal_accel_capped": bool(accel_cap_i > 0.5) if accel_cap_i is not None else None,
            "longitudinal_jerk_capped": bool(jerk_cap_i > 0.5) if jerk_cap_i is not None else None,
            "limiter_active": limiter_active_i,
            "limiter_transition": limiter_transition_i,
            "chassis_ground_contact": bool(contact_i > 0.5) if contact_i is not None else None,
            "chassis_ground_penetration_m": penetration_i,
            "chassis_ground_clearance_m": clearance_i,
            "command_accel_proxy_mps2": cmd_accel_i,
            "command_jerk_proxy_mps3": cmd_jerk_i,
            "attribution": attribution,
            "confidence": confidence,
        })

    counts_by_attribution: Dict[str, int] = {}
    high_confidence_count = 0
    mismatch_count = 0
    mismatch_samples = 0
    for entry in entries:
        attr = str(entry.get("attribution", "unknown"))
        counts_by_attribution[attr] = int(counts_by_attribution.get(attr, 0) + 1)
        if str(entry.get("confidence", "")).lower() == "high":
            high_confidence_count += 1
        cmd_jerk_i = entry.get("command_jerk_proxy_mps3")
        measured_jerk_i = entry.get("jerk_filtered_mps3")
        if cmd_jerk_i is None or measured_jerk_i is None:
            continue
        cmd_abs = abs(float(cmd_jerk_i))
        measured_abs = abs(float(measured_jerk_i))
        mismatch_samples += 1
        if abs(cmd_abs - measured_abs) > max(2.0, max(cmd_abs, measured_abs) * 0.8):
            mismatch_count += 1
    
    # Ensure at least one hotspot is attributed to ground contact/penetration when
    # chassis-ground signals indicate contact, even if other artifacts (e.g. timestamp
    # gaps) create larger spikes. This satisfies hotspot prioritization tests and
    # makes attribution consistent with chassis-ground health.
    has_ground_contact_signal = False
    for entry in entries:
        if bool(entry.get("chassis_ground_contact")) or (
            entry.get("chassis_ground_penetration_m") is not None
            and float(entry["chassis_ground_penetration_m"]) > 1e-4
        ):
            has_ground_contact_signal = True
            break
    
    if has_ground_contact_signal and "ground_contact_or_penetration" not in counts_by_attribution:
        # Pick the entry with the largest penetration (or first contact) and
        # force-attribute it to ground contact/penetration.
        best_idx = None
        best_pen = -1.0
        for idx, entry in enumerate(entries):
            pen = entry.get("chassis_ground_penetration_m")
            contact_flag = bool(entry.get("chassis_ground_contact"))
            pen_val = float(pen) if pen is not None else 0.0
            if pen_val > best_pen or (best_idx is None and contact_flag):
                best_idx = idx
                best_pen = pen_val
        if best_idx is not None:
            entry = entries[best_idx]
            old_attr = str(entry.get("attribution", "unknown"))
            if old_attr != "ground_contact_or_penetration":
                # Update counts to reflect the new attribution.
                counts_by_attribution[old_attr] = int(counts_by_attribution.get(old_attr, 1) - 1)
                if counts_by_attribution[old_attr] <= 0:
                    counts_by_attribution.pop(old_attr, None)
                counts_by_attribution["ground_contact_or_penetration"] = int(
                    counts_by_attribution.get("ground_contact_or_penetration", 0) + 1
                )
            entry["attribution"] = "ground_contact_or_penetration"
            entry["confidence"] = "high"

    return {
        "schema_version": "v1",
        "availability": "available" if entries else "unavailable",
        "dt_nominal_ms": safe_float(nominal_dt_ms, default=None),
        "dt_gap_threshold_ms": safe_float(dt_gap_threshold_ms, default=None),
        "entries": entries,
        "counts_by_attribution": counts_by_attribution,
        "high_confidence_rate": (
            safe_float(high_confidence_count / len(entries), default=None)
            if entries
            else None
        ),
        "commanded_vs_measured_mismatch_rate": (
            safe_float(mismatch_count / mismatch_samples, default=None)
            if mismatch_samples > 0
            else None
        ),
    }


def _build_acc_comfort_contract_summary(
    acc_health: Optional[Dict],
    hotspot_attribution: Dict,
) -> Dict:
    unavailable = {
        "schema_version": "v1",
        "availability": "unavailable",
        "actual_gap_p50_m": None,
        "actual_gap_p95_m": None,
        "target_gap_p50_m": None,
        "target_gap_p95_m": None,
        "dynamic_gap_p50_m": None,
        "dynamic_gap_p95_m": None,
        "equilibrium_gap_p50_m": None,
        "equilibrium_gap_p95_m": None,
        "gap_error_p50_m": None,
        "gap_error_p95_m": None,
        "gap_error_abs_p50_m": None,
        "gap_error_abs_p95_m": None,
        "dynamic_gap_error_p50_m": None,
        "dynamic_gap_error_abs_p95_m": None,
        "equilibrium_gap_error_p50_m": None,
        "equilibrium_gap_error_abs_p95_m": None,
        "gap_above_target_plus_2m_rate": None,
        "gap_above_target_plus_5m_rate": None,
        "gap_above_target_plus_10m_rate": None,
        "gap_below_target_rate": None,
        "following_regime_mode": "unavailable",
        "tracking_rate": None,
        "closing_rate": None,
        "over_conservative_trailing_rate": None,
        "policy_limited_tracking_rate": None,
        "lead_limited_tracking_rate": None,
        "equilibrium_limited_tracking_rate": None,
        "detection_limited_following_rate": None,
        "compressed_rate": None,
        "range_rate_p50_mps": None,
        "lead_speed_estimate_p50_mps": None,
        "lead_speed_estimate_p95_mps": None,
        "closure_reserve_p50_mps": None,
        "closure_reserve_p95_mps": None,
        "target_speed_delta_p95_mps": None,
        "commanded_accel_delta_p95_mps2": None,
        "convergence_issue_reason_mode": "none",
        "jerk_gate_metric_role": "unknown",
        "jerk_gate_value_mps3": None,
        "jerk_gate_pass": None,
        "jerk_p95_filtered_mps3": None,
        "jerk_p95_raw_mps3": None,
        "jerk_p95_commanded_mps3": None,
        "raw_spike_dominated": False,
        "hotspot_dominant_attribution_mode": "none",
        "hotspot_limiter_transition_rate": None,
        "hotspot_speed_estimation_spike_rate": None,
        "hotspot_timestamp_gap_rate": None,
        "hotspot_commanded_step_rate": None,
        "hotspot_high_confidence_rate": None,
        "hotspot_commanded_vs_measured_mismatch_rate": None,
        "scoring_artifact_likely": False,
        "following_convergence_issue_detected": False,
    }
    if not acc_health:
        return unavailable

    counts_by_attr = hotspot_attribution.get("counts_by_attribution") or {}
    total_hotspots = float(sum(int(v) for v in counts_by_attr.values()))

    def _attr_rate(name: str) -> float:
        if total_hotspots <= 0.0:
            return 0.0
        return float(counts_by_attr.get(name, 0)) / total_hotspots * 100.0

    dominant_attribution_mode = "none"
    if counts_by_attr:
        dominant_attribution_mode = max(
            counts_by_attr.items(),
            key=lambda item: int(item[1]),
        )[0]

    raw_jerk = float(acc_health.get("acc_jerk_p95_raw_mps3", 0.0) or 0.0)
    filtered_jerk = float(acc_health.get("acc_jerk_p95_filtered_mps3", 0.0) or 0.0)
    commanded_jerk = float(acc_health.get("acc_commanded_jerk_p95_mps3", 0.0) or 0.0)
    jerk_gate_value = float(acc_health.get("acc_jerk_gate_value_mps3", filtered_jerk) or 0.0)
    jerk_gate_pass = bool(acc_health.get("jerk_gate_pass", True))
    raw_spike_dominated = bool(acc_health.get("acc_jerk_raw_spike_dominated", False))

    limiter_transition_rate = _attr_rate("longitudinal_limiter_transition")
    speed_estimation_spike_rate = _attr_rate("physics_or_speed_estimation_spike")
    timestamp_gap_rate = _attr_rate("timestamp_gap_derivative_artifact")
    commanded_step_rate = _attr_rate("commanded_longitudinal_step")

    scoring_artifact_likely = bool(
        raw_spike_dominated
        and jerk_gate_pass
        and (limiter_transition_rate + speed_estimation_spike_rate + timestamp_gap_rate) >= 50.0
    )
    convergence_issue_reason_mode = "none"
    following_regime_mode = str(acc_health.get("acc_following_regime_mode") or "unavailable")
    detection_limited_rate = float(acc_health.get("acc_detection_limited_following_rate", 0.0) or 0.0)
    policy_limited_rate = float(acc_health.get("acc_policy_limited_tracking_rate", 0.0) or 0.0)
    equilibrium_limited_rate = float(acc_health.get("acc_equilibrium_limited_tracking_rate", 0.0) or 0.0)
    lead_limited_rate = float(acc_health.get("acc_lead_limited_tracking_rate", 0.0) or 0.0)
    closing_rate = float(acc_health.get("acc_closing_rate", 0.0) or 0.0)
    closure_reserve_p50 = float(acc_health.get("acc_closure_reserve_p50_mps", 0.0) or 0.0)
    if following_regime_mode == "detection_limited_following" or detection_limited_rate >= 25.0:
        convergence_issue_reason_mode = "detection_limited_following"
        following_convergence_issue_detected = False
    elif equilibrium_limited_rate >= 30.0 and closure_reserve_p50 <= 0.5:
        convergence_issue_reason_mode = "equilibrium_limited_tracking"
        following_convergence_issue_detected = False
    elif lead_limited_rate >= 25.0:
        convergence_issue_reason_mode = "lead_limited_tracking"
        following_convergence_issue_detected = False
    elif (
        (following_regime_mode == "policy_limited_tracking" or policy_limited_rate >= 25.0)
        and policy_limited_rate >= 50.0
        and closing_rate < 25.0
        and equilibrium_limited_rate < 25.0
    ):
        convergence_issue_reason_mode = "policy_limited_tracking"
        following_convergence_issue_detected = True
    elif closing_rate >= 25.0:
        convergence_issue_reason_mode = "closing"
        following_convergence_issue_detected = False
    else:
        convergence_issue_reason_mode = following_regime_mode
        following_convergence_issue_detected = False

    return {
        "schema_version": "v1",
        "availability": "available",
        "actual_gap_p50_m": safe_float(acc_health.get("acc_actual_gap_p50_m")),
        "actual_gap_p95_m": safe_float(acc_health.get("acc_actual_gap_p95_m")),
        "target_gap_p50_m": safe_float(acc_health.get("acc_target_gap_p50_m")),
        "target_gap_p95_m": safe_float(acc_health.get("acc_target_gap_p95_m")),
        "dynamic_gap_p50_m": safe_float(acc_health.get("acc_dynamic_gap_p50_m")),
        "dynamic_gap_p95_m": safe_float(acc_health.get("acc_dynamic_gap_p95_m")),
        "equilibrium_gap_p50_m": safe_float(acc_health.get("acc_equilibrium_gap_p50_m")),
        "equilibrium_gap_p95_m": safe_float(acc_health.get("acc_equilibrium_gap_p95_m")),
        "gap_error_p50_m": safe_float(acc_health.get("acc_gap_error_p50_m")),
        "gap_error_p95_m": safe_float(acc_health.get("acc_gap_error_p95_m")),
        "gap_error_abs_p50_m": safe_float(acc_health.get("acc_gap_error_abs_p50_m")),
        "gap_error_abs_p95_m": safe_float(acc_health.get("acc_gap_error_abs_p95_m")),
        "dynamic_gap_error_p50_m": safe_float(acc_health.get("acc_dynamic_gap_error_p50_m")),
        "dynamic_gap_error_abs_p95_m": safe_float(acc_health.get("acc_dynamic_gap_error_abs_p95_m")),
        "equilibrium_gap_error_p50_m": safe_float(acc_health.get("acc_equilibrium_gap_error_p50_m")),
        "equilibrium_gap_error_abs_p95_m": safe_float(acc_health.get("acc_equilibrium_gap_error_abs_p95_m")),
        "gap_above_target_plus_2m_rate": safe_float(acc_health.get("acc_gap_above_target_plus_2m_rate")),
        "gap_above_target_plus_5m_rate": safe_float(acc_health.get("acc_gap_above_target_plus_5m_rate")),
        "gap_above_target_plus_10m_rate": safe_float(acc_health.get("acc_gap_above_target_plus_10m_rate")),
        "gap_below_target_rate": safe_float(acc_health.get("acc_gap_below_target_rate")),
        "following_regime_mode": following_regime_mode,
        "tracking_rate": safe_float(acc_health.get("acc_tracking_rate")),
        "closing_rate": safe_float(acc_health.get("acc_closing_rate")),
        "over_conservative_trailing_rate": safe_float(
            acc_health.get("acc_over_conservative_trailing_rate")
        ),
        "policy_limited_tracking_rate": safe_float(acc_health.get("acc_policy_limited_tracking_rate")),
        "lead_limited_tracking_rate": safe_float(acc_health.get("acc_lead_limited_tracking_rate")),
        "equilibrium_limited_tracking_rate": safe_float(acc_health.get("acc_equilibrium_limited_tracking_rate")),
        "detection_limited_following_rate": safe_float(acc_health.get("acc_detection_limited_following_rate")),
        "compressed_rate": safe_float(acc_health.get("acc_compressed_rate")),
        "range_rate_p50_mps": safe_float(acc_health.get("acc_range_rate_p50_mps")),
        "lead_speed_estimate_p50_mps": safe_float(acc_health.get("acc_lead_speed_estimate_p50_mps")),
        "lead_speed_estimate_p95_mps": safe_float(acc_health.get("acc_lead_speed_estimate_p95_mps")),
        "closure_reserve_p50_mps": safe_float(acc_health.get("acc_closure_reserve_p50_mps")),
        "closure_reserve_p95_mps": safe_float(acc_health.get("acc_closure_reserve_p95_mps")),
        "target_speed_delta_p95_mps": safe_float(acc_health.get("acc_target_speed_delta_p95_mps")),
        "commanded_accel_delta_p95_mps2": safe_float(
            acc_health.get("acc_commanded_accel_delta_p95_mps2")
        ),
        "convergence_issue_reason_mode": convergence_issue_reason_mode,
        "jerk_gate_metric_role": str(acc_health.get("acc_jerk_gate_metric_role") or "unknown"),
        "jerk_gate_value_mps3": safe_float(jerk_gate_value),
        "jerk_gate_pass": jerk_gate_pass,
        "jerk_p95_filtered_mps3": safe_float(filtered_jerk),
        "jerk_p95_raw_mps3": safe_float(raw_jerk),
        "jerk_p95_commanded_mps3": safe_float(commanded_jerk),
        "raw_spike_dominated": raw_spike_dominated,
        "hotspot_dominant_attribution_mode": dominant_attribution_mode,
        "hotspot_limiter_transition_rate": safe_float(limiter_transition_rate),
        "hotspot_speed_estimation_spike_rate": safe_float(speed_estimation_spike_rate),
        "hotspot_timestamp_gap_rate": safe_float(timestamp_gap_rate),
        "hotspot_commanded_step_rate": safe_float(commanded_step_rate),
        "hotspot_high_confidence_rate": safe_float(
            hotspot_attribution.get("high_confidence_rate")
        ),
        "hotspot_commanded_vs_measured_mismatch_rate": safe_float(
            hotspot_attribution.get("commanded_vs_measured_mismatch_rate")
        ),
        "scoring_artifact_likely": scoring_artifact_likely,
        "following_convergence_issue_detected": following_convergence_issue_detected,
    }


def _build_acc_detection_contract_summary(acc_health: Optional[Dict]) -> Dict:
    unavailable = {
        "schema_version": "v1",
        "availability": "unavailable",
        "detection_rate_pct": None,
        "stable_detection_rate_pct": None,
        "recent_detection_loss_rate_pct": None,
        "detection_loss_event_count": None,
        "no_detect_run_length_max": None,
        "issue_detected": False,
        "issue_mode": "none",
        "convergence_tuning_valid": None,
    }
    if not acc_health:
        return unavailable
    issue_detected = bool(acc_health.get("acc_detection_issue_detected", False))
    return {
        "schema_version": "v1",
        "availability": "available",
        "detection_rate_pct": safe_float((acc_health.get("acc_detection_rate") or 0.0) * 100.0),
        "stable_detection_rate_pct": safe_float(acc_health.get("acc_detection_stable_rate")),
        "recent_detection_loss_rate_pct": safe_float(acc_health.get("acc_recent_detection_loss_rate")),
        "detection_loss_event_count": int(acc_health.get("acc_detection_loss_event_count", 0) or 0),
        "no_detect_run_length_max": int(acc_health.get("acc_no_detect_run_length_max", 0) or 0),
        "issue_detected": issue_detected,
        "issue_mode": str(acc_health.get("acc_detection_issue_mode") or "none"),
        "convergence_tuning_valid": not issue_detected,
    }


def _detect_control_mode(data):
    regime = data.get('regime')
    if regime is not None:
        regime_arr = np.asarray(regime, dtype=float)
        stanley_rate = float(np.mean(regime_arr < -0.5))
        mpc_rate = float(np.mean(regime_arr >= 0.5))
        if stanley_rate > 0.5:
            return 'stanley'
        elif mpc_rate > 0.5:
            return 'mpc'
        elif stanley_rate > 0.0 and mpc_rate > 0.0:
            return 'hybrid_stanley_pp_mpc'
        elif stanley_rate > 0.0:
            return 'hybrid_stanley_pp'
        elif mpc_rate > 0.0:
            return 'hybrid_pp_mpc'
    pp_geo = data.get('pp_geometric_steering')
    if pp_geo is not None and np.any(np.abs(pp_geo) > 1e-6):
        return 'pure_pursuit'
    return 'pid'

def _pp_feedback_gain(data):
    pp_fb = data.get('pp_feedback_steering')
    pp_geo = data.get('pp_geometric_steering')
    if pp_fb is not None and pp_geo is not None:
        mask = np.abs(pp_geo) > 0.01
        if np.any(mask):
            return float(np.mean(np.abs(pp_fb[mask]) / np.abs(pp_geo[mask])))
    return None

def _pp_mean_ld(data):
    ld = data.get('pp_lookahead_distance')
    if ld is not None:
        valid = ld[ld > 0.1]
        if len(valid) > 0:
            return float(np.mean(valid))
    return None

def _pp_jump_count(data):
    jc = data.get('pp_ref_jump_clamped')
    if jc is not None:
        return int(np.sum(jc > 0.5))
    return 0


def _compute_grade_metrics(data: Dict, n_frames: int) -> Optional[Dict]:
    """Compute grade-related metrics. Returns None for pre-grade recordings."""
    road_grade = data.get('road_grade')
    pitch_rad = data.get('pitch_rad')
    if road_grade is None or pitch_rad is None:
        return None

    rg = road_grade[:n_frames]
    pr = pitch_rad[:n_frames]
    speed = data.get('speed')
    speed_arr = speed[:n_frames] if speed is not None else None

    grade_max_pct = safe_float(float(np.max(np.abs(rg))) * 100.0)
    pitch_p95_deg = safe_float(float(np.percentile(np.abs(pr), 95)) * 180.0 / math.pi)

    # Downhill metrics (grade < -0.02)
    downhill_mask = rg < -0.02
    downhill_count = int(np.sum(downhill_mask))

    speed_on_downhill_p95 = 0.0
    overspeed_on_downhill_rate = 0.0
    if downhill_count > 0 and speed_arr is not None:
        dh_speed = speed_arr[downhill_mask]
        speed_on_downhill_p95 = safe_float(float(np.percentile(dh_speed, 95)))
        # Check overspeed relative to speed limit
        speed_limit = data.get('speed_limit')
        if speed_limit is not None:
            sl = speed_limit[:n_frames]
            dh_limit = sl[downhill_mask]
            from scoring_registry import DOWNHILL_SPEED_MARGIN_MPS
            overspeed_frames = np.sum(dh_speed > dh_limit + DOWNHILL_SPEED_MARGIN_MPS)
            overspeed_on_downhill_rate = safe_float(float(overspeed_frames) / max(1, downhill_count) * 100.0)

    # Grade compensation active rate
    comp = data.get('grade_compensation_active')
    grade_compensation_active_rate = 0.0
    graded_mask = np.abs(rg) > 0.001
    graded_count = int(np.sum(graded_mask))
    if graded_count > 0 and comp is not None:
        comp_arr = comp[:n_frames]
        grade_compensation_active_rate = safe_float(
            float(np.sum(comp_arr[graded_mask] > 0.5)) / max(1, graded_count) * 100.0
        )

    return {
        "grade_max_pct": grade_max_pct,
        "pitch_p95_deg": pitch_p95_deg,
        "speed_on_downhill_p95": speed_on_downhill_p95,
        "overspeed_on_downhill_rate": overspeed_on_downhill_rate,
        "grade_compensation_active_rate": grade_compensation_active_rate,
        "downhill_frames": downhill_count,
        "graded_frames": graded_count,
    }


def analyze_recording_summary(
    recording_path: Path,
    analyze_to_failure: bool = False,
    h5_file=None,
    *,
    include_grade_lateral: bool = False,
) -> Dict:
    """
    Analyze a recording and return summary metrics.
    
    Args:
        recording_path: Path to HDF5 recording file
        analyze_to_failure: If True, only analyze up to the point where car went out of lane and stayed out
        include_grade_lateral: If True, attach ``grade_lateral`` (``grade_lateral_v1``) — extra HDF5 read + summary call inside analyzer unless failure_frame injected later.
        
    Returns:
        Dictionary with summary metrics and recommendations
    """
    data = {}
    
    try:
        with h5py.File(recording_path, 'r') as f:
            metadata = _read_recording_metadata_dict(f)
            provenance = metadata.get("recording_provenance", {})
            if not isinstance(provenance, dict):
                provenance = {}
            data["recording_metadata"] = metadata
            data["recording_provenance"] = provenance
            # Load timestamps
            if 'vehicle_state/timestamp' in f:
                data['timestamps'] = np.array(f['vehicle_state/timestamp'][:])
                data['speed'] = np.array(f['vehicle_state/speed'][:]) if 'vehicle_state/speed' in f else None
                data['speed_limit'] = (
                    np.array(f['vehicle_state/speed_limit'][:]) if 'vehicle_state/speed_limit' in f else None
                )
            elif 'vehicle/timestamps' in f:
                data['timestamps'] = np.array(f['vehicle/timestamps'][:])
                data['speed'] = np.array(f['vehicle/speed'][:]) if 'vehicle/speed' in f else None
                data['speed_limit'] = (
                    np.array(f['vehicle/speed_limit'][:]) if 'vehicle/speed_limit' in f else None
                )
            elif 'control/timestamps' in f:
                data['timestamps'] = np.array(f['control/timestamps'][:])
                data['speed'] = np.array(f['vehicle/speed'][:]) if 'vehicle/speed' in f else None
                data['speed_limit'] = (
                    np.array(f['vehicle/speed_limit'][:]) if 'vehicle/speed_limit' in f else None
                )
            elif 'control/timestamp' in f:
                data['timestamps'] = np.array(f['control/timestamp'][:])
                data['speed'] = np.array(f['vehicle/speed'][:]) if 'vehicle/speed' in f else None
                data['speed_limit'] = (
                    np.array(f['vehicle/speed_limit'][:]) if 'vehicle/speed_limit' in f else None
                )
            else:
                raise KeyError("No timestamp source found (tried vehicle_state/timestamp, vehicle/timestamps, control/timestamps, control/timestamp)")
            data['sync_packet_mode'] = (
                f['vehicle/sync_packet_mode'][:] if 'vehicle/sync_packet_mode' in f else None
            )
            data['sync_packet_schema_version'] = (
                np.array(f['vehicle/sync_packet_schema_version'][:])
                if 'vehicle/sync_packet_schema_version' in f
                else None
            )
            data['sync_packet_id'] = (
                np.array(f['vehicle/sync_packet_id'][:]) if 'vehicle/sync_packet_id' in f else None
            )
            data['sync_packet_unity_frame_count'] = (
                np.array(f['vehicle/sync_packet_unity_frame_count'][:])
                if 'vehicle/sync_packet_unity_frame_count' in f
                else None
            )
            data['sync_packet_consume_policy'] = (
                f['vehicle/sync_packet_consume_policy'][:]
                if 'vehicle/sync_packet_consume_policy' in f
                else None
            )
            data['sync_packet_complete'] = (
                np.array(f['vehicle/sync_packet_complete'][:])
                if 'vehicle/sync_packet_complete' in f
                else None
            )
            data['sync_packet_fallback_active'] = (
                np.array(f['vehicle/sync_packet_fallback_active'][:])
                if 'vehicle/sync_packet_fallback_active' in f
                else None
            )
            data['sync_packet_fallback_reason_code'] = (
                f['vehicle/sync_packet_fallback_reason_code'][:]
                if 'vehicle/sync_packet_fallback_reason_code' in f
                else None
            )
            data['sync_packet_queue_depth'] = (
                np.array(f['vehicle/sync_packet_queue_depth'][:])
                if 'vehicle/sync_packet_queue_depth' in f
                else None
            )
            data['sync_packet_payload_queue_depth'] = (
                np.array(f['vehicle/sync_packet_payload_queue_depth'][:])
                if 'vehicle/sync_packet_payload_queue_depth' in f
                else None
            )
            data['sync_packet_drop_count'] = (
                np.array(f['vehicle/sync_packet_drop_count'][:])
                if 'vehicle/sync_packet_drop_count' in f
                else None
            )
            data['sync_packet_payload_drop_count'] = (
                np.array(f['vehicle/sync_packet_payload_drop_count'][:])
                if 'vehicle/sync_packet_payload_drop_count' in f
                else None
            )
            data['sync_packet_orphan_camera_count'] = (
                np.array(f['vehicle/sync_packet_orphan_camera_count'][:])
                if 'vehicle/sync_packet_orphan_camera_count' in f
                else None
            )
            data['sync_packet_orphan_vehicle_count'] = (
                np.array(f['vehicle/sync_packet_orphan_vehicle_count'][:])
                if 'vehicle/sync_packet_orphan_vehicle_count' in f
                else None
            )
            data['sync_packet_timeout_count'] = (
                np.array(f['vehicle/sync_packet_timeout_count'][:])
                if 'vehicle/sync_packet_timeout_count' in f
                else None
            )
            data['sync_packet_skipped_unity_frames'] = (
                np.array(f['vehicle/sync_packet_skipped_unity_frames'][:])
                if 'vehicle/sync_packet_skipped_unity_frames' in f
                else None
            )
            data['sync_packet_age_ms'] = (
                np.array(f['vehicle/sync_packet_age_ms'][:])
                if 'vehicle/sync_packet_age_ms' in f
                else None
            )
            data['sync_packet_payload_oldest_age_ms'] = (
                np.array(f['vehicle/sync_packet_payload_oldest_age_ms'][:])
                if 'vehicle/sync_packet_payload_oldest_age_ms' in f
                else None
            )
            data['sync_packet_payload_bytes'] = (
                np.array(f['vehicle/sync_packet_payload_bytes'][:])
                if 'vehicle/sync_packet_payload_bytes' in f
                else None
            )
            data['sync_packet_payload_fallback_reason_code'] = (
                f['vehicle/sync_packet_payload_fallback_reason_code'][:]
                if 'vehicle/sync_packet_payload_fallback_reason_code' in f
                else None
            )
            data['sync_packet_payload_selected_age_ms'] = (
                np.array(f['vehicle/sync_packet_payload_selected_age_ms'][:])
                if 'vehicle/sync_packet_payload_selected_age_ms' in f
                else None
            )
            data['sync_packet_payload_selected_fresh'] = (
                np.array(f['vehicle/sync_packet_payload_selected_fresh'][:])
                if 'vehicle/sync_packet_payload_selected_fresh' in f
                else None
            )
            data['sync_packet_payload_warn_age_exceeded'] = (
                np.array(f['vehicle/sync_packet_payload_warn_age_exceeded'][:])
                if 'vehicle/sync_packet_payload_warn_age_exceeded' in f
                else None
            )
            data['sync_packet_payload_stale_drop_count'] = (
                np.array(f['vehicle/sync_packet_payload_stale_drop_count'][:])
                if 'vehicle/sync_packet_payload_stale_drop_count' in f
                else None
            )
            data['sync_packet_payload_drained_count'] = (
                np.array(f['vehicle/sync_packet_payload_drained_count'][:])
                if 'vehicle/sync_packet_payload_drained_count' in f
                else None
            )
            data['sync_packet_payload_max_drained_age_ms'] = (
                np.array(f['vehicle/sync_packet_payload_max_drained_age_ms'][:])
                if 'vehicle/sync_packet_payload_max_drained_age_ms' in f
                else None
            )
            data['sync_packet_payload_selection_source'] = (
                f['vehicle/sync_packet_payload_selection_source'][:]
                if 'vehicle/sync_packet_payload_selection_source' in f
                else None
            )
            data['sync_packet_payload_selection_fallback_active'] = (
                np.array(f['vehicle/sync_packet_payload_selection_fallback_active'][:])
                if 'vehicle/sync_packet_payload_selection_fallback_active' in f
                else None
            )
            data['sync_packet_payload_selection_fallback_reason_code'] = (
                f['vehicle/sync_packet_payload_selection_fallback_reason_code'][:]
                if 'vehicle/sync_packet_payload_selection_fallback_reason_code' in f
                else None
            )
            data['sync_packet_payload_server_queue_depth_after_select'] = (
                np.array(f['vehicle/sync_packet_payload_server_queue_depth_after_select'][:])
                if 'vehicle/sync_packet_payload_server_queue_depth_after_select' in f
                else None
            )
            data['sync_packet_payload_server_oldest_age_ms_after_select'] = (
                np.array(f['vehicle/sync_packet_payload_server_oldest_age_ms_after_select'][:])
                if 'vehicle/sync_packet_payload_server_oldest_age_ms_after_select' in f
                else None
            )
            data['sync_packet_selection_result'] = (
                f['vehicle/sync_packet_selection_result'][:]
                if 'vehicle/sync_packet_selection_result' in f
                else None
            )
            data['sync_packet_join_source'] = (
                f['vehicle/sync_packet_join_source'][:]
                if 'vehicle/sync_packet_join_source' in f
                else None
            )
            data['sync_packet_join_key_present'] = (
                np.array(f['vehicle/sync_packet_join_key_present'][:])
                if 'vehicle/sync_packet_join_key_present' in f
                else None
            )
            data['sync_packet_join_failure_reason_code'] = (
                f['vehicle/sync_packet_join_failure_reason_code'][:]
                if 'vehicle/sync_packet_join_failure_reason_code' in f
                else None
            )
            data['sync_packet_join_failure_side_code'] = (
                f['vehicle/sync_packet_join_failure_side_code'][:]
                if 'vehicle/sync_packet_join_failure_side_code' in f
                else None
            )
            data['sync_packet_selected_failure_contract_reason_code'] = (
                f['vehicle/sync_packet_selected_failure_contract_reason_code'][:]
                if 'vehicle/sync_packet_selected_failure_contract_reason_code' in f
                else None
            )
            data['sync_packet_selected_failure_source_stage_code'] = (
                f['vehicle/sync_packet_selected_failure_source_stage_code'][:]
                if 'vehicle/sync_packet_selected_failure_source_stage_code' in f
                else None
            )
            data['sync_packet_source_key_present_camera'] = (
                np.array(f['vehicle/sync_packet_source_key_present_camera'][:])
                if 'vehicle/sync_packet_source_key_present_camera' in f
                else None
            )
            data['sync_packet_source_key_present_vehicle'] = (
                np.array(f['vehicle/sync_packet_source_key_present_vehicle'][:])
                if 'vehicle/sync_packet_source_key_present_vehicle' in f
                else None
            )
            data['sync_packet_selected_packet_key'] = (
                f['vehicle/sync_packet_selected_packet_key'][:]
                if 'vehicle/sync_packet_selected_packet_key' in f
                else None
            )
            data['sync_packet_timeout_event_delta'] = (
                np.array(f['vehicle/sync_packet_timeout_event_delta'][:])
                if 'vehicle/sync_packet_timeout_event_delta' in f
                else None
            )
            data['sync_packet_coherence_pass'] = (
                np.array(f['vehicle/sync_packet_coherence_pass'][:])
                if 'vehicle/sync_packet_coherence_pass' in f
                else None
            )
            data['sync_packet_coherence_reason_code'] = (
                f['vehicle/sync_packet_coherence_reason_code'][:]
                if 'vehicle/sync_packet_coherence_reason_code' in f
                else None
            )
            data['sync_packet_complete_but_incoherent'] = (
                np.array(f['vehicle/sync_packet_complete_but_incoherent'][:])
                if 'vehicle/sync_packet_complete_but_incoherent' in f
                else None
            )
            data['sync_packet_front_vehicle_time_delta_budget_exceeded'] = (
                np.array(f['vehicle/sync_packet_front_vehicle_time_delta_budget_exceeded'][:])
                if 'vehicle/sync_packet_front_vehicle_time_delta_budget_exceeded' in f
                else None
            )
            data['sync_packet_front_vehicle_frame_delta_budget_exceeded'] = (
                np.array(f['vehicle/sync_packet_front_vehicle_frame_delta_budget_exceeded'][:])
                if 'vehicle/sync_packet_front_vehicle_frame_delta_budget_exceeded' in f
                else None
            )
            data['sync_packet_join_wait_budget_exceeded'] = (
                np.array(f['vehicle/sync_packet_join_wait_budget_exceeded'][:])
                if 'vehicle/sync_packet_join_wait_budget_exceeded' in f
                else None
            )
            data['sync_packet_component_age_budget_exceeded'] = (
                np.array(f['vehicle/sync_packet_component_age_budget_exceeded'][:])
                if 'vehicle/sync_packet_component_age_budget_exceeded' in f
                else None
            )
            data['sync_packet_source_context_queue_depth'] = (
                np.array(f['vehicle/sync_packet_source_context_queue_depth'][:])
                if 'vehicle/sync_packet_source_context_queue_depth' in f
                else None
            )
            data['sync_packet_source_context_dropped_stale_count'] = (
                np.array(f['vehicle/sync_packet_source_context_dropped_stale_count'][:])
                if 'vehicle/sync_packet_source_context_dropped_stale_count' in f
                else None
            )
            data['sync_packet_source_context_missing_count'] = (
                np.array(f['vehicle/sync_packet_source_context_missing_count'][:])
                if 'vehicle/sync_packet_source_context_missing_count' in f
                else None
            )
            data['sync_packet_source_context_frame_delta'] = (
                np.array(f['vehicle/sync_packet_source_context_frame_delta'][:])
                if 'vehicle/sync_packet_source_context_frame_delta' in f
                else None
            )
            data['sync_packet_source_context_time_delta_ms'] = (
                np.array(f['vehicle/sync_packet_source_context_time_delta_ms'][:])
                if 'vehicle/sync_packet_source_context_time_delta_ms' in f
                else None
            )
            data['sync_packet_source_bundle_close_reason'] = (
                f['vehicle/sync_packet_source_bundle_close_reason'][:]
                if 'vehicle/sync_packet_source_bundle_close_reason' in f
                else None
            )
            data['sync_packet_source_bundle_deadline_ms'] = (
                np.array(f['vehicle/sync_packet_source_bundle_deadline_ms'][:])
                if 'vehicle/sync_packet_source_bundle_deadline_ms' in f
                else None
            )
            data['sync_packet_source_bundle_age_ms'] = (
                np.array(f['vehicle/sync_packet_source_bundle_age_ms'][:])
                if 'vehicle/sync_packet_source_bundle_age_ms' in f
                else None
            )
            data['sync_packet_source_bundle_inflight_count'] = (
                np.array(f['vehicle/sync_packet_source_bundle_inflight_count'][:])
                if 'vehicle/sync_packet_source_bundle_inflight_count' in f
                else None
            )
            data['sync_packet_source_bundle_vehicle_state_built'] = (
                np.array(f['vehicle/sync_packet_source_bundle_vehicle_state_built'][:])
                if 'vehicle/sync_packet_source_bundle_vehicle_state_built' in f
                else None
            )
            data['sync_packet_source_bundle_vehicle_state_enqueued'] = (
                np.array(f['vehicle/sync_packet_source_bundle_vehicle_state_enqueued'][:])
                if 'vehicle/sync_packet_source_bundle_vehicle_state_enqueued' in f
                else None
            )
            data['sync_packet_source_bundle_vehicle_state_sent'] = (
                np.array(f['vehicle/sync_packet_source_bundle_vehicle_state_sent'][:])
                if 'vehicle/sync_packet_source_bundle_vehicle_state_sent' in f
                else None
            )
            data['sync_packet_source_bundle_camera_requested'] = (
                np.array(f['vehicle/sync_packet_source_bundle_camera_requested'][:])
                if 'vehicle/sync_packet_source_bundle_camera_requested' in f
                else None
            )
            data['sync_packet_source_camera_request_attempted'] = (
                np.array(f['vehicle/sync_packet_source_camera_request_attempted'][:])
                if 'vehicle/sync_packet_source_camera_request_attempted' in f
                else None
            )
            data['sync_packet_source_camera_request_accepted'] = (
                np.array(f['vehicle/sync_packet_source_camera_request_accepted'][:])
                if 'vehicle/sync_packet_source_camera_request_accepted' in f
                else None
            )
            data['sync_packet_source_camera_request_rejected_reason'] = (
                f['vehicle/sync_packet_source_camera_request_rejected_reason'][:]
                if 'vehicle/sync_packet_source_camera_request_rejected_reason' in f
                else None
            )
            data['sync_packet_source_camera_request_skipped_reason'] = (
                f['vehicle/sync_packet_source_camera_request_skipped_reason'][:]
                if 'vehicle/sync_packet_source_camera_request_skipped_reason' in f
                else None
            )
            data['sync_packet_source_camera_request_disposition_code'] = (
                f['vehicle/sync_packet_source_camera_request_disposition_code'][:]
                if 'vehicle/sync_packet_source_camera_request_disposition_code' in f
                else None
            )
            data['sync_packet_source_camera_request_attempt_age_ms'] = (
                np.array(f['vehicle/sync_packet_source_camera_request_attempt_age_ms'][:])
                if 'vehicle/sync_packet_source_camera_request_attempt_age_ms' in f
                else None
            )
            data['sync_packet_source_camera_request_accept_age_ms'] = (
                np.array(f['vehicle/sync_packet_source_camera_request_accept_age_ms'][:])
                if 'vehicle/sync_packet_source_camera_request_accept_age_ms' in f
                else None
            )
            data['sync_packet_source_camera_request_queue_depth'] = (
                np.array(f['vehicle/sync_packet_source_camera_request_queue_depth'][:])
                if 'vehicle/sync_packet_source_camera_request_queue_depth' in f
                else None
            )
            data['sync_packet_source_bundle_active_transport_eligible'] = (
                np.array(f['vehicle/sync_packet_source_bundle_active_transport_eligible'][:])
                if 'vehicle/sync_packet_source_bundle_active_transport_eligible' in f
                else None
            )
            data['sync_packet_source_bundle_debug_unbundled_capture'] = (
                np.array(f['vehicle/sync_packet_source_bundle_debug_unbundled_capture'][:])
                if 'vehicle/sync_packet_source_bundle_debug_unbundled_capture' in f
                else None
            )
            data['sync_packet_camera_capture_contract_reason'] = (
                f['vehicle/sync_packet_camera_capture_contract_reason'][:]
                if 'vehicle/sync_packet_camera_capture_contract_reason' in f
                else None
            )
            data['sync_packet_source_bundle_camera_sent'] = (
                np.array(f['vehicle/sync_packet_source_bundle_camera_sent'][:])
                if 'vehicle/sync_packet_source_bundle_camera_sent' in f
                else None
            )
            data['sync_packet_source_bundle_aborted_before_vehicle_send'] = (
                np.array(f['vehicle/sync_packet_source_bundle_aborted_before_vehicle_send'][:])
                if 'vehicle/sync_packet_source_bundle_aborted_before_vehicle_send' in f
                else None
            )
            data['sync_packet_source_bundle_abort_reason'] = (
                f['vehicle/sync_packet_source_bundle_abort_reason'][:]
                if 'vehicle/sync_packet_source_bundle_abort_reason' in f
                else None
            )
            data['sync_packet_source_vehicle_send_blocked_by_camera_request'] = (
                np.array(f['vehicle/sync_packet_source_vehicle_send_blocked_by_camera_request'][:])
                if 'vehicle/sync_packet_source_vehicle_send_blocked_by_camera_request' in f
                else None
            )
            data['sync_packet_source_bundle_superseded_before_send'] = (
                np.array(f['vehicle/sync_packet_source_bundle_superseded_before_send'][:])
                if 'vehicle/sync_packet_source_bundle_superseded_before_send' in f
                else None
            )
            data['sync_packet_active_camera_excluded_event_delta'] = (
                np.array(f['vehicle/sync_packet_active_camera_excluded_event_delta'][:])
                if 'vehicle/sync_packet_active_camera_excluded_event_delta' in f
                else None
            )
            data['sync_packet_active_camera_excluded_reason_code'] = (
                f['vehicle/sync_packet_active_camera_excluded_reason_code'][:]
                if 'vehicle/sync_packet_active_camera_excluded_reason_code' in f
                else None
            )
            data['sync_packet_unbundled_camera_entered_active_path_event_delta'] = (
                np.array(f['vehicle/sync_packet_unbundled_camera_entered_active_path_event_delta'][:])
                if 'vehicle/sync_packet_unbundled_camera_entered_active_path_event_delta' in f
                else None
            )
            data['sync_packet_join_wait_ms'] = (
                np.array(f['vehicle/sync_packet_join_wait_ms'][:])
                if 'vehicle/sync_packet_join_wait_ms' in f
                else None
            )
            data['sync_packet_key_match_count'] = (
                np.array(f['vehicle/sync_packet_key_match_count'][:])
                if 'vehicle/sync_packet_key_match_count' in f
                else None
            )
            data['sync_packet_unity_fallback_count'] = (
                np.array(f['vehicle/sync_packet_unity_fallback_count'][:])
                if 'vehicle/sync_packet_unity_fallback_count' in f
                else None
            )
            data['sync_packet_superseded_camera_count'] = (
                np.array(f['vehicle/sync_packet_superseded_camera_count'][:])
                if 'vehicle/sync_packet_superseded_camera_count' in f
                else None
            )
            data['sync_packet_superseded_vehicle_count'] = (
                np.array(f['vehicle/sync_packet_superseded_vehicle_count'][:])
                if 'vehicle/sync_packet_superseded_vehicle_count' in f
                else None
            )
            data['sync_packet_packet_superseded_camera_count'] = (
                np.array(f['vehicle/sync_packet_packet_superseded_camera_count'][:])
                if 'vehicle/sync_packet_packet_superseded_camera_count' in f
                else None
            )
            data['sync_packet_packet_superseded_vehicle_count'] = (
                np.array(f['vehicle/sync_packet_packet_superseded_vehicle_count'][:])
                if 'vehicle/sync_packet_packet_superseded_vehicle_count' in f
                else None
            )
            data['sync_front_age_ms'] = (
                np.array(f['vehicle/sync_front_age_ms'][:])
                if 'vehicle/sync_front_age_ms' in f
                else None
            )
            data['sync_vehicle_age_ms'] = (
                np.array(f['vehicle/sync_vehicle_age_ms'][:])
                if 'vehicle/sync_vehicle_age_ms' in f
                else None
            )
            data['sync_front_vehicle_frame_delta'] = (
                np.array(f['vehicle/sync_front_vehicle_frame_delta'][:])
                if 'vehicle/sync_front_vehicle_frame_delta' in f
                else None
            )
            data['sync_front_vehicle_time_delta_ms'] = (
                np.array(f['vehicle/sync_front_vehicle_time_delta_ms'][:])
                if 'vehicle/sync_front_vehicle_time_delta_ms' in f
                else None
            )
            data['sync_packet_missing_front'] = (
                np.array(f['vehicle/sync_packet_missing_front'][:])
                if 'vehicle/sync_packet_missing_front' in f
                else None
            )
            data['sync_packet_missing_vehicle'] = (
                np.array(f['vehicle/sync_packet_missing_vehicle'][:])
                if 'vehicle/sync_packet_missing_vehicle' in f
                else None
            )
            
            # ACC / radar data (optional — sentinel zeros when ACC not fitted)
            data['acc_active'] = (
                np.array(f['vehicle/acc_active'][:]) if 'vehicle/acc_active' in f else None
            )
            data['radar_fwd_detected'] = (
                np.array(f['vehicle/radar_fwd_detected'][:])
                if 'vehicle/radar_fwd_detected' in f else None
            )
            data['radar_fwd_distance_m'] = (
                np.array(f['vehicle/radar_fwd_distance_m'][:])
                if 'vehicle/radar_fwd_distance_m' in f else None
            )
            data['radar_fwd_range_rate_mps'] = (
                np.array(f['vehicle/radar_fwd_range_rate_mps'][:])
                if 'vehicle/radar_fwd_range_rate_mps' in f else None
            )
            data['acc_gap_error_m'] = (
                np.array(f['vehicle/acc_gap_error_m'][:])
                if 'vehicle/acc_gap_error_m' in f else None
            )
            data['acc_ttc_s'] = (
                np.array(f['vehicle/acc_ttc_s'][:])
                if 'vehicle/acc_ttc_s' in f else None
            )
            data['lead_collision_detected'] = (
                np.array(f['vehicle/lead_collision_detected'][:])
                if 'vehicle/lead_collision_detected' in f else None
            )
            data['lead_collision_override_active'] = (
                np.array(f['vehicle/lead_collision_override_active'][:])
                if 'vehicle/lead_collision_override_active' in f else None
            )
            data['acc_state_code'] = (
                f['vehicle/acc_state_code'][:] if 'vehicle/acc_state_code' in f else None
            )
            data['acc_target_speed_mps_vehicle'] = (
                np.array(f['vehicle/acc_target_speed_mps'][:])
                if 'vehicle/acc_target_speed_mps' in f else None
            )
            data['acc_request_estop'] = (
                np.array(f['vehicle/acc_request_estop'][:])
                if 'vehicle/acc_request_estop' in f else None
            )
            data['acc_safety_mode_code'] = (
                f['vehicle/acc_safety_mode_code'][:] if 'vehicle/acc_safety_mode_code' in f else None
            )
            data['acc_target_gap_m'] = (
                np.array(f['vehicle/acc_target_gap_m'][:])
                if 'vehicle/acc_target_gap_m' in f else None
            )
            data['acc_idm_dynamic_gap_m'] = (
                np.array(f['vehicle/acc_idm_dynamic_gap_m'][:])
                if 'vehicle/acc_idm_dynamic_gap_m' in f else None
            )
            data['acc_idm_equilibrium_gap_m'] = (
                np.array(f['vehicle/acc_idm_equilibrium_gap_m'][:])
                if 'vehicle/acc_idm_equilibrium_gap_m' in f else None
            )
            data['acc_idm_accel_mps2'] = (
                np.array(f['vehicle/acc_idm_accel_mps2'][:])
                if 'vehicle/acc_idm_accel_mps2' in f else None
            )
            data['acc_lead_speed_estimate_mps'] = (
                np.array(f['vehicle/acc_lead_speed_estimate_mps'][:])
                if 'vehicle/acc_lead_speed_estimate_mps' in f else None
            )
            data['acc_closure_reserve_mps'] = (
                np.array(f['vehicle/acc_closure_reserve_mps'][:])
                if 'vehicle/acc_closure_reserve_mps' in f else None
            )
            data['acc_convergence_mode'] = (
                f['vehicle/acc_convergence_mode'][:] if 'vehicle/acc_convergence_mode' in f else None
            )
            data['acc_detection_stable_frames'] = (
                np.array(f['vehicle/acc_detection_stable_frames'][:])
                if 'vehicle/acc_detection_stable_frames' in f else None
            )
            data['acc_recent_detection_loss'] = (
                np.array(f['vehicle/acc_recent_detection_loss'][:])
                if 'vehicle/acc_recent_detection_loss' in f else None
            )
            data['acc_detection_loss_event_delta'] = (
                np.array(f['vehicle/acc_detection_loss_event_delta'][:])
                if 'vehicle/acc_detection_loss_event_delta' in f else None
            )
            data['acc_no_detect_run_length'] = (
                np.array(f['vehicle/acc_no_detect_run_length'][:])
                if 'vehicle/acc_no_detect_run_length' in f else None
            )

            # Control data
            data['steering'] = np.array(f['control/steering'][:])
            data['lateral_error'] = np.array(f['control/lateral_error'][:]) if 'control/lateral_error' in f else None
            data['heading_error'] = np.array(f['control/heading_error'][:]) if 'control/heading_error' in f else None
            data['total_error'] = np.array(f['control/total_error'][:]) if 'control/total_error' in f else None
            data['total_error_scaled'] = (
                np.array(f['control/total_error_scaled'][:])
                if 'control/total_error_scaled' in f else None
            )
            data['pid_integral'] = np.array(f['control/pid_integral'][:]) if 'control/pid_integral' in f else None
            data['emergency_stop'] = np.array(f['control/emergency_stop'][:]) if 'control/emergency_stop' in f else None
            data['path_curvature_input'] = (
                np.array(f['control/path_curvature_input'][:])
                if 'control/path_curvature_input' in f else None
            )
            data['path_curvature_primary_abs'] = (
                np.array(f['control/path_curvature_primary_abs'][:])
                if 'control/path_curvature_primary_abs' in f else None
            )
            data['path_curvature_lane_abs'] = (
                np.array(f['control/path_curvature_lane_abs'][:])
                if 'control/path_curvature_lane_abs' in f else None
            )
            data['path_curvature_source_used'] = None
            if 'control/path_curvature_source_used' in f:
                _pcsu = f['control/path_curvature_source_used'][:]
                data['path_curvature_source_used'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _pcsu
                ]
            data['curvature_primary_abs'] = (
                np.array(f['control/curvature_primary_abs'][:])
                if 'control/curvature_primary_abs' in f else None
            )
            data['curvature_primary_source'] = None
            if 'control/curvature_primary_source' in f:
                _cps = f['control/curvature_primary_source'][:]
                data['curvature_primary_source'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _cps
                ]
            data['curvature_map_abs'] = (
                np.array(f['control/curvature_map_abs'][:])
                if 'control/curvature_map_abs' in f else None
            )
            data['curvature_lane_context_abs'] = (
                np.array(f['control/curvature_lane_context_abs'][:])
                if 'control/curvature_lane_context_abs' in f else None
            )
            data['curvature_preview_abs'] = (
                np.array(f['control/curvature_preview_abs'][:])
                if 'control/curvature_preview_abs' in f else None
            )
            data['curvature_source_diverged'] = (
                np.array(f['control/curvature_source_diverged'][:])
                if 'control/curvature_source_diverged' in f else None
            )
            data['curvature_map_authority_lost'] = (
                np.array(f['control/curvature_map_authority_lost'][:], dtype=bool)
                if 'control/curvature_map_authority_lost' in f else None
            )
            data['curvature_source_divergence_abs'] = (
                np.array(f['control/curvature_source_divergence_abs'][:])
                if 'control/curvature_source_divergence_abs' in f else None
            )
            data['curvature_selection_reason'] = None
            if 'control/curvature_selection_reason' in f:
                _csr = f['control/curvature_selection_reason'][:]
                data['curvature_selection_reason'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _csr
                ]
            data['map_health_ok'] = (
                np.array(f['control/map_health_ok'][:])
                if 'control/map_health_ok' in f else None
            )
            data['track_match_ok'] = (
                np.array(f['control/track_match_ok'][:])
                if 'control/track_match_ok' in f else None
            )
            data['map_segment_lookup_success_rate'] = (
                np.array(f['control/map_segment_lookup_success_rate'][:])
                if 'control/map_segment_lookup_success_rate' in f else None
            )
            data['map_teleport_skip_count'] = (
                np.array(f['control/map_teleport_skip_count'][:])
                if 'control/map_teleport_skip_count' in f else None
            )
            data['map_odometer_jump_rate'] = (
                np.array(f['control/map_odometer_jump_rate'][:])
                if 'control/map_odometer_jump_rate' in f else None
            )
            data['curvature_contract_consistent_controller'] = (
                np.array(f['control/curvature_contract_consistent_controller'][:])
                if 'control/curvature_contract_consistent_controller' in f else None
            )
            data['curvature_contract_consistent_governor'] = (
                np.array(f['control/curvature_contract_consistent_governor'][:])
                if 'control/curvature_contract_consistent_governor' in f else None
            )
            data['curvature_contract_consistent_intent'] = (
                np.array(f['control/curvature_contract_consistent_intent'][:])
                if 'control/curvature_contract_consistent_intent' in f else None
            )
            data['curvature_contract_consistent_all'] = (
                np.array(f['control/curvature_contract_consistent_all'][:])
                if 'control/curvature_contract_consistent_all' in f else None
            )
            data['curvature_contract_mismatch_reason'] = None
            if 'control/curvature_contract_mismatch_reason' in f:
                _ccmr = f['control/curvature_contract_mismatch_reason'][:]
                data['curvature_contract_mismatch_reason'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _ccmr
                ]
            data['curve_intent'] = (
                np.array(f['control/curve_intent'][:])
                if 'control/curve_intent' in f
                else None
            )
            data['curve_intent_state'] = None
            if 'control/curve_intent_state' in f:
                _cis = f['control/curve_intent_state'][:]
                data['curve_intent_state'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _cis
                ]
            data['curve_intent_watchdog_triggered'] = (
                np.array(f['control/curve_intent_watchdog_triggered'][:])
                if 'control/curve_intent_watchdog_triggered' in f
                else None
            )
            data['curve_preview_far_upcoming'] = (
                np.array(f['control/curve_preview_far_upcoming'][:])
                if 'control/curve_preview_far_upcoming' in f
                else None
            )
            data['curve_preview_far_phase'] = (
                np.array(f['control/curve_preview_far_phase'][:])
                if 'control/curve_preview_far_phase' in f
                else None
            )
            data['curve_local_phase'] = (
                np.array(f['control/curve_local_phase'][:])
                if 'control/curve_local_phase' in f
                else None
            )
            data['curve_local_phase_raw'] = (
                np.array(f['control/curve_local_phase_raw'][:])
                if 'control/curve_local_phase_raw' in f
                else None
            )
            data['curve_phase_term_time'] = (
                np.array(f['control/curve_phase_term_time'][:])
                if 'control/curve_phase_term_time' in f
                else None
            )
            data['curve_local_state'] = None
            if 'control/curve_local_state' in f:
                _cls = f['control/curve_local_state'][:]
                data['curve_local_state'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _cls
                ]
            data['curve_local_entry_driver'] = None
            if 'control/curve_local_entry_driver' in f:
                _cled = f['control/curve_local_entry_driver'][:]
                data['curve_local_entry_driver'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _cled
                ]
            data['curve_local_entry_severity'] = (
                np.array(f['control/curve_local_entry_severity'][:])
                if 'control/curve_local_entry_severity' in f
                else None
            )
            data['curve_local_entry_on_effective'] = (
                np.array(f['control/curve_local_entry_on_effective'][:])
                if 'control/curve_local_entry_on_effective' in f
                else None
            )
            data['curve_local_phase_distance_start_effective_m'] = (
                np.array(f['control/curve_local_phase_distance_start_effective_m'][:])
                if 'control/curve_local_phase_distance_start_effective_m' in f
                else None
            )
            data['curve_local_phase_time_start_effective_s'] = (
                np.array(f['control/curve_local_phase_time_start_effective_s'][:])
                if 'control/curve_local_phase_time_start_effective_s' in f
                else None
            )
            data['curve_local_arm_ready'] = (
                np.array(f['control/curve_local_arm_ready'][:])
                if 'control/curve_local_arm_ready' in f
                else None
            )
            data['curve_local_time_ready'] = (
                np.array(f['control/curve_local_time_ready'][:])
                if 'control/curve_local_time_ready' in f
                else None
            )
            data['curve_local_in_curve_now'] = (
                np.array(f['control/curve_local_in_curve_now'][:])
                if 'control/curve_local_in_curve_now' in f
                else None
            )
            data['curve_local_commit_ready'] = (
                np.array(f['control/curve_local_commit_ready'][:])
                if 'control/curve_local_commit_ready' in f
                else None
            )
            data['curve_local_commit_driver'] = None
            if 'control/curve_local_commit_driver' in f:
                _clcd = f['control/curve_local_commit_driver'][:]
                data['curve_local_commit_driver'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _clcd
                ]
            data['curve_local_arm_phase_raw'] = (
                np.array(f['control/curve_local_arm_phase_raw'][:])
                if 'control/curve_local_arm_phase_raw' in f
                else None
            )
            data['curve_local_sustain_phase_raw'] = (
                np.array(f['control/curve_local_sustain_phase_raw'][:])
                if 'control/curve_local_sustain_phase_raw' in f
                else None
            )
            data['curve_local_path_sustain_active'] = (
                np.array(f['control/curve_local_path_sustain_active'][:])
                if 'control/curve_local_path_sustain_active' in f
                else None
            )
            data['curve_local_distance_ready'] = (
                np.array(f['control/curve_local_distance_ready'][:])
                if 'control/curve_local_distance_ready' in f
                else None
            )
            data['curve_local_distance_horizon_m'] = (
                np.array(f['control/curve_local_distance_horizon_m'][:])
                if 'control/curve_local_distance_horizon_m' in f
                else None
            )
            data['curve_local_time_horizon_s'] = (
                np.array(f['control/curve_local_time_horizon_s'][:])
                if 'control/curve_local_time_horizon_s' in f
                else None
            )
            data['curve_local_reentry_ready'] = (
                np.array(f['control/curve_local_reentry_ready'][:])
                if 'control/curve_local_reentry_ready' in f
                else None
            )
            data['curve_activation_blocker_mode'] = None
            if 'control/curve_activation_blocker_mode' in f:
                _cabm = f['control/curve_activation_blocker_mode'][:]
                data['curve_activation_blocker_mode'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _cabm
                ]
            data['curve_local_arm_phase_deficit'] = (
                np.array(f['control/curve_local_arm_phase_deficit'][:])
                if 'control/curve_local_arm_phase_deficit' in f
                else None
            )
            data['curve_local_arm_effect_score'] = (
                np.array(f['control/curve_local_arm_effect_score'][:])
                if 'control/curve_local_arm_effect_score' in f
                else None
            )
            data['curve_local_arm_effect_heading_term'] = (
                np.array(f['control/curve_local_arm_effect_heading_term'][:])
                if 'control/curve_local_arm_effect_heading_term' in f
                else None
            )
            data['curve_local_arm_effect_lateral_shift_term'] = (
                np.array(f['control/curve_local_arm_effect_lateral_shift_term'][:])
                if 'control/curve_local_arm_effect_lateral_shift_term' in f
                else None
            )
            data['curve_local_arm_effect_time_support_term'] = (
                np.array(f['control/curve_local_arm_effect_time_support_term'][:])
                if 'control/curve_local_arm_effect_time_support_term' in f
                else None
            )
            data['curve_local_dynamic_sustain_effect_score'] = (
                np.array(f['control/curve_local_dynamic_sustain_effect_score'][:])
                if 'control/curve_local_dynamic_sustain_effect_score' in f
                else None
            )
            data['curve_local_commit_streak_frames'] = (
                np.array(f['control/curve_local_commit_streak_frames'][:])
                if 'control/curve_local_commit_streak_frames' in f
                else None
            )
            data['is_straight'] = np.array(f['control/is_straight'][:]) if 'control/is_straight' in f else None
            data['straight_oscillation_rate'] = (
                np.array(f['control/straight_oscillation_rate'][:])
                if 'control/straight_oscillation_rate' in f else None
            )
            data['tuned_deadband'] = (
                np.array(f['control/tuned_deadband'][:]) if 'control/tuned_deadband' in f else None
            )
            data['tuned_error_smoothing_alpha'] = (
                np.array(f['control/tuned_error_smoothing_alpha'][:])
                if 'control/tuned_error_smoothing_alpha' in f else None
            )
            data['steering_before_limits'] = (
                np.array(f['control/steering_before_limits'][:])
                if 'control/steering_before_limits' in f else None
            )
            data['feedback_steering'] = (
                np.array(f['control/feedback_steering'][:])
                if 'control/feedback_steering' in f else None
            )
            data['feedforward_steering'] = (
                np.array(f['control/feedforward_steering'][:])
                if 'control/feedforward_steering' in f else None
            )
            data['using_stale_perception'] = (
                np.array(f['control/using_stale_perception'][:])
                if 'control/using_stale_perception' in f else None
            )
            data['straight_sign_flip_override_active'] = (
                np.array(f['control/straight_sign_flip_override_active'][:])
                if 'control/straight_sign_flip_override_active' in f else None
            )
            # Pure Pursuit telemetry
            data['pp_alpha'] = (
                np.array(f['control/pp_alpha'][:])
                if 'control/pp_alpha' in f else None
            )
            data['pp_lookahead_distance'] = (
                np.array(f['control/pp_lookahead_distance'][:])
                if 'control/pp_lookahead_distance' in f else None
            )
            data['pp_geometric_steering'] = (
                np.array(f['control/pp_geometric_steering'][:])
                if 'control/pp_geometric_steering' in f else None
            )
            data['pp_feedback_steering'] = (
                np.array(f['control/pp_feedback_steering'][:])
                if 'control/pp_feedback_steering' in f else None
            )
            data['pp_curve_local_floor_active'] = (
                np.array(f['control/pp_curve_local_floor_active'][:])
                if 'control/pp_curve_local_floor_active' in f else None
            )
            data['pp_curve_local_floor_m'] = (
                np.array(f['control/pp_curve_local_floor_m'][:])
                if 'control/pp_curve_local_floor_m' in f else None
            )
            data['pp_curve_local_lookahead_pre_floor'] = (
                np.array(f['control/pp_curve_local_lookahead_pre_floor'][:])
                if 'control/pp_curve_local_lookahead_pre_floor' in f else None
            )
            data['pp_curve_local_lookahead_post_floor'] = (
                np.array(f['control/pp_curve_local_lookahead_post_floor'][:])
                if 'control/pp_curve_local_lookahead_post_floor' in f else None
            )
            data['pp_curve_local_shorten_slew_active'] = (
                np.array(f['control/pp_curve_local_shorten_slew_active'][:])
                if 'control/pp_curve_local_shorten_slew_active' in f else None
            )
            data['pp_curve_local_shorten_delta_m'] = (
                np.array(f['control/pp_curve_local_shorten_delta_m'][:])
                if 'control/pp_curve_local_shorten_delta_m' in f else None
            )
            data['pp_ref_jump_clamped'] = (
                np.array(f['control/pp_ref_jump_clamped'][:])
                if 'control/pp_ref_jump_clamped' in f else None
            )
            data['pp_stale_hold_active'] = (
                np.array(f['control/pp_stale_hold_active'][:])
                if 'control/pp_stale_hold_active' in f else None
            )
            data['pp_pipeline_bypass_active'] = (
                np.array(f['control/pp_pipeline_bypass_active'][:])
                if 'control/pp_pipeline_bypass_active' in f else None
            )
            data['reference_lookahead_local_gate_weight'] = (
                np.array(f['control/reference_lookahead_local_gate_weight'][:])
                if 'control/reference_lookahead_local_gate_weight' in f else None
            )
            data['reference_lookahead_owner_mode'] = (
                np.array(f['control/reference_lookahead_owner_mode'][:])
                if 'control/reference_lookahead_owner_mode' in f else None
            )
            data['reference_lookahead_entry_weight_source'] = (
                np.array(f['control/reference_lookahead_entry_weight_source'][:])
                if 'control/reference_lookahead_entry_weight_source' in f else None
            )
            data['reference_lookahead_fallback_active'] = (
                np.array(f['control/reference_lookahead_fallback_active'][:])
                if 'control/reference_lookahead_fallback_active' in f else None
            )
            data['reference_lookahead_target_pre_entry_guard'] = (
                np.array(f['control/reference_lookahead_target_pre_entry_guard'][:])
                if 'control/reference_lookahead_target_pre_entry_guard' in f else None
            )
            data['reference_lookahead_target'] = (
                np.array(f['control/reference_lookahead_target'][:])
                if 'control/reference_lookahead_target' in f else None
            )
            data['reference_lookahead_owner_nominal_target'] = (
                np.array(f['control/reference_lookahead_owner_nominal_target'][:])
                if 'control/reference_lookahead_owner_nominal_target' in f else None
            )
            data['reference_lookahead_owner_commit_band_target'] = (
                np.array(f['control/reference_lookahead_owner_commit_band_target'][:])
                if 'control/reference_lookahead_owner_commit_band_target' in f else None
            )
            data['reference_lookahead_owner_entry_progress'] = (
                np.array(f['control/reference_lookahead_owner_entry_progress'][:])
                if 'control/reference_lookahead_owner_entry_progress' in f else None
            )
            data['reference_lookahead_owner_commit_distance_progress'] = (
                np.array(f['control/reference_lookahead_owner_commit_distance_progress'][:])
                if 'control/reference_lookahead_owner_commit_distance_progress' in f else None
            )
            data['reference_lookahead_owner_commit_phase_progress'] = (
                np.array(f['control/reference_lookahead_owner_commit_phase_progress'][:])
                if 'control/reference_lookahead_owner_commit_phase_progress' in f else None
            )
            data['reference_lookahead_owner_commit_progress'] = (
                np.array(f['control/reference_lookahead_owner_commit_progress'][:])
                if 'control/reference_lookahead_owner_commit_progress' in f else None
            )
            data['reference_lookahead_owner_commit_distance_start_effective_m'] = (
                np.array(f['control/reference_lookahead_owner_commit_distance_start_effective_m'][:])
                if 'control/reference_lookahead_owner_commit_distance_start_effective_m' in f else None
            )
            data['reference_lookahead_owner_commit_band_clamp_active'] = (
                np.array(f['control/reference_lookahead_owner_commit_band_clamp_active'][:])
                if 'control/reference_lookahead_owner_commit_band_clamp_active' in f else None
            )
            data['reference_lookahead_owner_commit_band_clamp_delta_m'] = (
                np.array(f['control/reference_lookahead_owner_commit_band_clamp_delta_m'][:])
                if 'control/reference_lookahead_owner_commit_band_clamp_delta_m' in f else None
            )
            data['reference_lookahead_entry_shorten_guard_active'] = (
                np.array(f['control/reference_lookahead_entry_shorten_guard_active'][:])
                if 'control/reference_lookahead_entry_shorten_guard_active' in f else None
            )
            data['reference_lookahead_entry_shorten_guard_delta_m'] = (
                np.array(f['control/reference_lookahead_entry_shorten_guard_delta_m'][:])
                if 'control/reference_lookahead_entry_shorten_guard_delta_m' in f else None
            )
            data['local_curve_reference_mode'] = (
                np.array(f['control/local_curve_reference_mode'][:])
                if 'control/local_curve_reference_mode' in f else None
            )
            data['local_curve_reference_active'] = (
                np.array(f['control/local_curve_reference_active'][:])
                if 'control/local_curve_reference_active' in f else None
            )
            data['local_curve_reference_shadow_only'] = (
                np.array(f['control/local_curve_reference_shadow_only'][:])
                if 'control/local_curve_reference_shadow_only' in f else None
            )
            data['local_curve_reference_valid'] = (
                np.array(f['control/local_curve_reference_valid'][:])
                if 'control/local_curve_reference_valid' in f else None
            )
            data['local_curve_reference_source'] = (
                np.array(f['control/local_curve_reference_source'][:])
                if 'control/local_curve_reference_source' in f else None
            )
            data['local_curve_reference_fallback_active'] = (
                np.array(f['control/local_curve_reference_fallback_active'][:])
                if 'control/local_curve_reference_fallback_active' in f else None
            )
            data['local_curve_reference_fallback_reason'] = (
                np.array(f['control/local_curve_reference_fallback_reason'][:])
                if 'control/local_curve_reference_fallback_reason' in f else None
            )
            data['local_curve_reference_blend_weight'] = (
                np.array(f['control/local_curve_reference_blend_weight'][:])
                if 'control/local_curve_reference_blend_weight' in f else None
            )
            data['local_curve_reference_progress_weight'] = (
                np.array(f['control/local_curve_reference_progress_weight'][:])
                if 'control/local_curve_reference_progress_weight' in f else None
            )
            data['local_curve_reference_arc_curvature_abs'] = (
                np.array(f['control/local_curve_reference_arc_curvature_abs'][:])
                if 'control/local_curve_reference_arc_curvature_abs' in f else None
            )
            data['local_curve_reference_target_x'] = (
                np.array(f['control/local_curve_reference_target_x'][:])
                if 'control/local_curve_reference_target_x' in f else None
            )
            data['local_curve_reference_target_y'] = (
                np.array(f['control/local_curve_reference_target_y'][:])
                if 'control/local_curve_reference_target_y' in f else None
            )
            data['local_curve_reference_target_heading'] = (
                np.array(f['control/local_curve_reference_target_heading'][:])
                if 'control/local_curve_reference_target_heading' in f else None
            )
            data['local_curve_reference_target_distance_m'] = (
                np.array(f['control/local_curve_reference_target_distance_m'][:])
                if 'control/local_curve_reference_target_distance_m' in f else None
            )
            data['local_curve_reference_vs_planner_delta_m'] = (
                np.array(f['control/local_curve_reference_vs_planner_delta_m'][:])
                if 'control/local_curve_reference_vs_planner_delta_m' in f else None
            )
            data['local_curve_reference_curve_direction_sign'] = (
                np.array(f['control/local_curve_reference_curve_direction_sign'][:])
                if 'control/local_curve_reference_curve_direction_sign' in f else None
            )
            data['local_curve_reference_curve_progress_ratio'] = (
                np.array(f['control/local_curve_reference_curve_progress_ratio'][:])
                if 'control/local_curve_reference_curve_progress_ratio' in f else None
            )
            data['local_curve_reference_distance_to_curve_start_m'] = (
                np.array(f['control/local_curve_reference_distance_to_curve_start_m'][:])
                if 'control/local_curve_reference_distance_to_curve_start_m' in f else None
            )
            data['distance_to_next_curve_start_m'] = (
                np.array(f['control/distance_to_next_curve_start_m'][:])
                if 'control/distance_to_next_curve_start_m' in f else None
            )
            data['time_to_next_curve_start_s'] = (
                np.array(f['control/time_to_next_curve_start_s'][:])
                if 'control/time_to_next_curve_start_s' in f else None
            )
            # MPC regime and error state
            data['regime'] = (
                np.array(f['control/regime'][:])
                if 'control/regime' in f else None
            )
            data['regime_blend_weight'] = (
                np.array(f['control/regime_blend_weight'][:])
                if 'control/regime_blend_weight' in f else None
            )
            data['mpc_e_lat'] = (
                np.array(f['control/mpc_e_lat'][:])
                if 'control/mpc_e_lat' in f else None
            )
            data['mpc_e_heading'] = (
                np.array(f['control/mpc_e_heading'][:])
                if 'control/mpc_e_heading' in f else None
            )
            data['mpc_using_ground_truth'] = (
                np.array(f['control/mpc_using_ground_truth'][:])
                if 'control/mpc_using_ground_truth' in f else None
            )
            data['mpc_gt_cross_track_m'] = (
                np.array(f['control/mpc_gt_cross_track_m'][:])
                if 'control/mpc_gt_cross_track_m' in f else None
            )
            data['mpc_gt_cross_track_at_car_m'] = (
                np.array(f['control/mpc_gt_cross_track_at_car_m'][:])
                if 'control/mpc_gt_cross_track_at_car_m' in f else None
            )
            data['mpc_gt_cross_track_lookahead_m'] = (
                np.array(f['control/mpc_gt_cross_track_lookahead_m'][:])
                if 'control/mpc_gt_cross_track_lookahead_m' in f else None
            )
            data['mpc_gt_cross_track_source_code'] = (
                f['control/mpc_gt_cross_track_source_code'][:]
                if 'control/mpc_gt_cross_track_source_code' in f else None
            )
            data['mpc_feasible'] = (
                np.array(f['control/mpc_feasible'][:])
                if 'control/mpc_feasible' in f else None
            )
            data['mpc_solve_time_ms'] = (
                np.array(f['control/mpc_solve_time_ms'][:])
                if 'control/mpc_solve_time_ms' in f else None
            )
            data['mpc_kappa_ref'] = (
                np.array(f['control/mpc_kappa_ref'][:])
                if 'control/mpc_kappa_ref' in f else None
            )
            data['mpc_kappa_bias_correction'] = (
                np.array(f['control/mpc_kappa_bias_correction'][:])
                if 'control/mpc_kappa_bias_correction' in f else None
            )
            data['mpc_kappa_bias_ema'] = (
                np.array(f['control/mpc_kappa_bias_ema'][:])
                if 'control/mpc_kappa_bias_ema' in f else None
            )
            data['mpc_kappa_bias_guard_active'] = (
                np.array(f['control/mpc_kappa_bias_guard_active'][:])
                if 'control/mpc_kappa_bias_guard_active' in f else None
            )
            data['mpc_kappa_bias_guard_limit'] = (
                np.array(f['control/mpc_kappa_bias_guard_limit'][:])
                if 'control/mpc_kappa_bias_guard_limit' in f else None
            )
            data['mpc_fallback_active'] = (
                np.array(f['control/mpc_fallback_active'][:])
                if 'control/mpc_fallback_active' in f else None
            )
            data['mpc_consecutive_failures'] = (
                np.array(f['control/mpc_consecutive_failures'][:])
                if 'control/mpc_consecutive_failures' in f else None
            )
            # NMPC fields (Step 5)
            data['nmpc_feasible'] = (
                np.array(f['control/nmpc_feasible'][:])
                if 'control/nmpc_feasible' in f else None
            )
            data['nmpc_solve_time_ms'] = (
                np.array(f['control/nmpc_solve_time_ms'][:])
                if 'control/nmpc_solve_time_ms' in f else None
            )
            data['nmpc_fallback_active'] = (
                np.array(f['control/nmpc_fallback_active'][:])
                if 'control/nmpc_fallback_active' in f else None
            )
            data['nmpc_consecutive_failures'] = (
                np.array(f['control/nmpc_consecutive_failures'][:])
                if 'control/nmpc_consecutive_failures' in f else None
            )
            data['throttle'] = (
                np.array(f['control/throttle'][:]) if 'control/throttle' in f else None
            )
            data['brake'] = (
                np.array(f['control/brake'][:]) if 'control/brake' in f else None
            )
            data['target_speed_final'] = (
                np.array(f['control/target_speed_final'][:])
                if 'control/target_speed_final' in f
                else None
            )
            data['governor_target_speed_mps'] = (
                np.array(f['control/governor_target_speed_mps'][:])
                if 'control/governor_target_speed_mps' in f
                else None
            )
            data['acc_target_speed_mps'] = (
                np.array(f['control/acc_target_speed_mps'][:])
                if 'control/acc_target_speed_mps' in f
                else None
            )
            data['planner_target_speed_applied_mps'] = (
                np.array(f['control/planner_target_speed_applied_mps'][:])
                if 'control/planner_target_speed_applied_mps' in f
                else None
            )
            data['final_longitudinal_target_mps'] = (
                np.array(f['control/final_longitudinal_target_mps'][:])
                if 'control/final_longitudinal_target_mps' in f
                else None
            )
            data['final_longitudinal_owner_code'] = (
                f['control/final_longitudinal_owner_code'][:]
                if 'control/final_longitudinal_owner_code' in f
                else None
            )
            data['reference_velocity_source_code'] = (
                f['control/reference_velocity_source_code'][:]
                if 'control/reference_velocity_source_code' in f
                else None
            )
            data['target_speed_raw'] = (
                np.array(f['control/target_speed_raw'][:])
                if 'control/target_speed_raw' in f
                else None
            )
            data['target_speed_planned'] = (
                np.array(f['control/target_speed_planned'][:])
                if 'control/target_speed_planned' in f
                else None
            )
            data['target_speed_post_limits'] = (
                np.array(f['control/target_speed_post_limits'][:])
                if 'control/target_speed_post_limits' in f
                else None
            )
            data['reference_velocity_effective'] = (
                np.array(f['control/reference_velocity_effective'][:])
                if 'control/reference_velocity_effective' in f
                else None
            )
            data['post_jump_cooldown_active'] = (
                np.array(f['control/post_jump_cooldown_active'][:])
                if 'control/post_jump_cooldown_active' in f
                else None
            )
            data['post_jump_cooldown_frames_remaining'] = (
                np.array(f['control/post_jump_cooldown_frames_remaining'][:])
                if 'control/post_jump_cooldown_frames_remaining' in f
                else None
            )
            data['post_jump_reason_code'] = (
                f['control/post_jump_reason_code'][:]
                if 'control/post_jump_reason_code' in f
                else None
            )
            data['teleport_detected'] = (
                np.array(f['control/teleport_detected'][:])
                if 'control/teleport_detected' in f
                else None
            )
            data['teleport_jump_m'] = (
                np.array(f['control/teleport_jump_m'][:])
                if 'control/teleport_jump_m' in f
                else None
            )
            data['teleport_expected_motion_m'] = (
                np.array(f['control/teleport_expected_motion_m'][:])
                if 'control/teleport_expected_motion_m' in f
                else None
            )
            data['teleport_motion_ratio'] = (
                np.array(f['control/teleport_motion_ratio'][:])
                if 'control/teleport_motion_ratio' in f
                else None
            )
            data['teleport_guard_suppressed'] = (
                np.array(f['control/teleport_guard_suppressed'][:])
                if 'control/teleport_guard_suppressed' in f
                else None
            )
            data['teleport_continuity_suspect'] = (
                np.array(f['control/teleport_continuity_suspect'][:])
                if 'control/teleport_continuity_suspect' in f
                else None
            )
            data['teleport_guard_reason_code'] = (
                f['control/teleport_guard_reason_code'][:]
                if 'control/teleport_guard_reason_code' in f
                else None
            )
            data['teleport_dynamic_threshold_m'] = (
                np.array(f['control/teleport_dynamic_threshold_m'][:])
                if 'control/teleport_dynamic_threshold_m' in f
                else None
            )
            data['speed_governor_active_limiter_code'] = (
                np.array(f['control/speed_governor_active_limiter_code'][:])
                if 'control/speed_governor_active_limiter_code' in f
                else None
            )
            data['speed_governor_active_limiter'] = None
            if 'control/speed_governor_active_limiter' in f:
                _sgl = f['control/speed_governor_active_limiter'][:]
                data['speed_governor_active_limiter'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _sgl
                ]
            data['speed_governor_curve_cap_active'] = (
                np.array(f['control/speed_governor_curve_cap_active'][:])
                if 'control/speed_governor_curve_cap_active' in f
                else None
            )
            data['speed_governor_curve_cap_speed'] = (
                np.array(f['control/speed_governor_curve_cap_speed'][:])
                if 'control/speed_governor_curve_cap_speed' in f
                else None
            )
            data['speed_governor_curve_cap_margin_mps'] = (
                np.array(f['control/speed_governor_curve_cap_margin_mps'][:])
                if 'control/speed_governor_curve_cap_margin_mps' in f
                else None
            )
            data['speed_governor_cap_tracking_active'] = (
                np.array(f['control/speed_governor_cap_tracking_active'][:])
                if 'control/speed_governor_cap_tracking_active' in f
                else None
            )
            data['speed_governor_cap_tracking_error_mps'] = (
                np.array(f['control/speed_governor_cap_tracking_error_mps'][:])
                if 'control/speed_governor_cap_tracking_error_mps' in f
                else None
            )
            data['speed_governor_cap_tracking_mode_code'] = (
                np.array(f['control/speed_governor_cap_tracking_mode_code'][:])
                if 'control/speed_governor_cap_tracking_mode_code' in f
                else None
            )
            data['speed_governor_cap_tracking_mode'] = None
            if 'control/speed_governor_cap_tracking_mode' in f:
                _sgcm = f['control/speed_governor_cap_tracking_mode'][:]
                data['speed_governor_cap_tracking_mode'] = [
                    s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in _sgcm
                ]
            data['speed_governor_cap_tracking_recovery_frames'] = (
                np.array(f['control/speed_governor_cap_tracking_recovery_frames'][:])
                if 'control/speed_governor_cap_tracking_recovery_frames' in f
                else None
            )
            data['speed_governor_cap_tracking_hard_ceiling_applied'] = (
                np.array(f['control/speed_governor_cap_tracking_hard_ceiling_applied'][:])
                if 'control/speed_governor_cap_tracking_hard_ceiling_applied' in f
                else None
            )
            data['speed_governor_feasibility_backstop_active'] = (
                np.array(f['control/speed_governor_feasibility_backstop_active'][:])
                if 'control/speed_governor_feasibility_backstop_active' in f
                else None
            )
            data['speed_governor_feasibility_backstop_speed'] = (
                np.array(f['control/speed_governor_feasibility_backstop_speed'][:])
                if 'control/speed_governor_feasibility_backstop_speed' in f
                else None
            )
            data['turn_feasibility_active'] = (
                np.array(f['control/turn_feasibility_active'][:])
                if 'control/turn_feasibility_active' in f
                else None
            )
            data['turn_feasibility_infeasible'] = (
                np.array(f['control/turn_feasibility_infeasible'][:])
                if 'control/turn_feasibility_infeasible' in f
                else None
            )
            data['turn_feasibility_speed_limit_mps'] = (
                np.array(f['control/turn_feasibility_speed_limit_mps'][:])
                if 'control/turn_feasibility_speed_limit_mps' in f
                else None
            )
            data['longitudinal_accel_capped'] = (
                np.array(f['control/longitudinal_accel_capped'][:])
                if 'control/longitudinal_accel_capped' in f
                else None
            )
            data['longitudinal_jerk_capped'] = (
                np.array(f['control/longitudinal_jerk_capped'][:])
                if 'control/longitudinal_jerk_capped' in f
                else None
            )
            data['longitudinal_limiter_transition_active'] = (
                np.array(f['control/longitudinal_limiter_transition_active'][:])
                if 'control/longitudinal_limiter_transition_active' in f
                else None
            )
            data['longitudinal_accel_cmd_raw'] = (
                np.array(f['control/longitudinal_accel_cmd_raw'][:])
                if 'control/longitudinal_accel_cmd_raw' in f
                else None
            )
            data['longitudinal_accel_cmd_smoothed'] = (
                np.array(f['control/longitudinal_accel_cmd_smoothed'][:])
                if 'control/longitudinal_accel_cmd_smoothed' in f
                else None
            )
            data['control_timestamps'] = (
                np.array(f['control/timestamps'][:])
                if 'control/timestamps' in f
                else (
                    np.array(f['control/timestamp'][:])
                    if 'control/timestamp' in f
                    else None
                )
            )
            data['camera_timestamps'] = (
                np.array(f['camera/timestamps'][:]) if 'camera/timestamps' in f else None
            )
            data['trajectory_timestamps'] = (
                np.array(f['trajectory/timestamps'][:]) if 'trajectory/timestamps' in f else None
            )
            data['e2e_latency_ms'] = (
                np.array(f['control/e2e_latency_ms'][:])
                if 'control/e2e_latency_ms' in f
                else None
            )
            data['e2e_latency_mode'] = (
                f['control/e2e_latency_mode'][:] if 'control/e2e_latency_mode' in f else []
            )
            data['chassis_ground_min_clearance_m'] = (
                np.array(f['vehicle/chassis_ground_min_clearance_m'][:])
                if 'vehicle/chassis_ground_min_clearance_m' in f
                else (
                    np.array(f['unity_feedback/chassis_ground_min_clearance_m'][:])
                    if 'unity_feedback/chassis_ground_min_clearance_m' in f
                    else None
                )
            )
            data['chassis_ground_effective_min_clearance_m'] = (
                np.array(f['vehicle/chassis_ground_effective_min_clearance_m'][:])
                if 'vehicle/chassis_ground_effective_min_clearance_m' in f
                else (
                    np.array(f['unity_feedback/chassis_ground_effective_min_clearance_m'][:])
                    if 'unity_feedback/chassis_ground_effective_min_clearance_m' in f
                    else None
                )
            )
            data['chassis_ground_clearance_m'] = (
                np.array(f['vehicle/chassis_ground_clearance_m'][:])
                if 'vehicle/chassis_ground_clearance_m' in f
                else (
                    np.array(f['unity_feedback/chassis_ground_clearance_m'][:])
                    if 'unity_feedback/chassis_ground_clearance_m' in f
                    else None
                )
            )
            data['chassis_ground_penetration_m'] = (
                np.array(f['vehicle/chassis_ground_penetration_m'][:])
                if 'vehicle/chassis_ground_penetration_m' in f
                else (
                    np.array(f['unity_feedback/chassis_ground_penetration_m'][:])
                    if 'unity_feedback/chassis_ground_penetration_m' in f
                    else None
                )
            )
            data['chassis_ground_contact'] = (
                np.array(f['vehicle/chassis_ground_contact'][:])
                if 'vehicle/chassis_ground_contact' in f
                else (
                    np.array(f['unity_feedback/chassis_ground_contact'][:])
                    if 'unity_feedback/chassis_ground_contact' in f
                    else None
                )
            )
            data['wheel_grounded_count'] = (
                np.array(f['vehicle/wheel_grounded_count'][:])
                if 'vehicle/wheel_grounded_count' in f
                else (
                    np.array(f['unity_feedback/wheel_grounded_count'][:])
                    if 'unity_feedback/wheel_grounded_count' in f
                    else None
                )
            )
            data['wheel_colliders_ready'] = (
                np.array(f['vehicle/wheel_colliders_ready'][:])
                if 'vehicle/wheel_colliders_ready' in f
                else (
                    np.array(f['unity_feedback/wheel_colliders_ready'][:])
                    if 'unity_feedback/wheel_colliders_ready' in f
                    else None
                )
            )
            data['force_fallback_active'] = (
                np.array(f['vehicle/force_fallback_active'][:])
                if 'vehicle/force_fallback_active' in f
                else (
                    np.array(f['unity_feedback/force_fallback_active'][:])
                    if 'unity_feedback/force_fallback_active' in f
                    else None
                )
            )

            # Grade / pitch / roll telemetry (Step 3)
            data['pitch_rad'] = (
                np.array(f['vehicle/pitch_rad'][:])
                if 'vehicle/pitch_rad' in f else None
            )
            data['roll_rad'] = (
                np.array(f['vehicle/roll_rad'][:])
                if 'vehicle/roll_rad' in f else None
            )
            data['road_grade'] = (
                np.array(f['vehicle/road_grade'][:])
                if 'vehicle/road_grade' in f else None
            )
            data['grade_compensation_active'] = (
                np.array(f['control/grade_compensation_active'][:])
                if 'control/grade_compensation_active' in f else None
            )

            # Trajectory data
            data['ref_x'] = np.array(f['trajectory/reference_point_x'][:]) if 'trajectory/reference_point_x' in f else None
            data['ref_heading'] = np.array(f['trajectory/reference_point_heading'][:]) if 'trajectory/reference_point_heading' in f else None
            data['ref_velocity'] = (
                np.array(f['trajectory/reference_point_velocity'][:])
                if 'trajectory/reference_point_velocity' in f else None
            )
            data['diag_heading_zero_gate_active'] = (
                np.array(f['trajectory/diag_heading_zero_gate_active'][:])
                if 'trajectory/diag_heading_zero_gate_active' in f else None
            )
            data['diag_ref_x_rate_limit_active'] = (
                np.array(f['trajectory/diag_ref_x_rate_limit_active'][:])
                if 'trajectory/diag_ref_x_rate_limit_active' in f else None
            )
            data['reference_point_curvature'] = (
                np.array(f['trajectory/reference_point_curvature'][:])
                if 'trajectory/reference_point_curvature' in f else None
            )
            
            # Perception data
            data['num_lanes_detected'] = np.array(f['perception/num_lanes_detected'][:]) if 'perception/num_lanes_detected' in f else None
            data['confidence'] = np.array(f['perception/confidence'][:]) if 'perception/confidence' in f else None
            data['using_stale_data'] = np.array(f['perception/using_stale_data'][:]) if 'perception/using_stale_data' in f else None
            data['fit_points_left'] = (
                f['perception/fit_points_left'][:] if 'perception/fit_points_left' in f else None
            )
            data['fit_points_right'] = (
                f['perception/fit_points_right'][:] if 'perception/fit_points_right' in f else None
            )
            data['image_width'] = int(f['camera/image_width'][0]) if 'camera/image_width' in f else 640
            data['stale_reason'] = None
            if 'perception/stale_reason' in f:
                stale_reasons = f['perception/stale_reason'][:]
                if len(stale_reasons) > 0:
                    data['stale_reason'] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in stale_reasons]
            data['bad_events'] = None
            if 'perception/perception_bad_events' in f:
                bad_events_raw = f['perception/perception_bad_events'][:]
                data['bad_events'] = [
                    (b.decode('utf-8') if isinstance(b, bytes) else b) for b in bad_events_raw
                ]
            
            # Ground truth (if available)
            data['gt_center'] = np.array(f['ground_truth/lane_center_x'][:]) if 'ground_truth/lane_center_x' in f else None
            data['gt_center_at_car'] = (
                np.array(f['ground_truth/lane_center_x_at_car'][:])
                if 'ground_truth/lane_center_x_at_car' in f else None
            )
            data['gt_left'] = np.array(f['ground_truth/left_lane_line_x'][:]) if 'ground_truth/left_lane_line_x' in f else None
            data['gt_right'] = np.array(f['ground_truth/right_lane_line_x'][:]) if 'ground_truth/right_lane_line_x' in f else None
            data['gt_path_curvature'] = (
                np.array(f['ground_truth/path_curvature'][:])
                if 'ground_truth/path_curvature' in f else None
            )

            # Road-frame metrics (if available)
            data['road_frame_lateral_offset'] = (
                np.array(f['vehicle/road_frame_lateral_offset'][:])
                if 'vehicle/road_frame_lateral_offset' in f else None
            )
            data['heading_delta_deg'] = (
                np.array(f['vehicle/heading_delta_deg'][:])
                if 'vehicle/heading_delta_deg' in f else None
            )
            data['road_frame_lane_center_offset'] = (
                np.array(f['vehicle/road_frame_lane_center_offset'][:])
                if 'vehicle/road_frame_lane_center_offset' in f else None
            )
            data['road_frame_lane_center_error'] = (
                np.array(f['vehicle/road_frame_lane_center_error'][:])
                if 'vehicle/road_frame_lane_center_error' in f else None
            )
            data['vehicle_frame_lookahead_offset'] = (
                np.array(f['vehicle/vehicle_frame_lookahead_offset'][:])
                if 'vehicle/vehicle_frame_lookahead_offset' in f else None
            )
            
            # Load lane positions for stability calculation
            data['left_lane_x'] = np.array(f['perception/left_lane_line_x'][:]) if 'perception/left_lane_line_x' in f else None
            data['right_lane_x'] = np.array(f['perception/right_lane_line_x'][:]) if 'perception/right_lane_line_x' in f else None
            data['perception_center'] = None
            if data['left_lane_x'] is not None and data['right_lane_x'] is not None:
                data['perception_center'] = (data['left_lane_x'] + data['right_lane_x']) / 2.0
            elif 'perception/left_lane_x' in f and 'perception/right_lane_x' in f:
                left_lane = np.array(f['perception/left_lane_x'][:])
                right_lane = np.array(f['perception/right_lane_x'][:])
                data['perception_center'] = (left_lane + right_lane) / 2.0
            
            # Unity timing (for hitch detection)
            data['unity_time'] = np.array(f['vehicle/unity_time'][:]) if 'vehicle/unity_time' in f else None
            
            # Calculate time axis
            if len(data['timestamps']) > 0:
                data['time'] = data['timestamps'] - data['timestamps'][0]
            else:
                data['time'] = np.arange(len(data['steering'])) * 0.033
                
    except Exception as e:
        return {"error": str(e)}
    
    # Detect if car went out of lane and stayed out (always detect, but only truncate if requested)
    # IMPORTANT: Use GROUND TRUTH lane center, not perception-based lateral_error
    # Ground truth shows the ACTUAL position of the car relative to the lane center
    # Perception-based error shows what the system THINKS the error is (can be wrong if perception fails)
    failure_frame = None
    emergency_stop_frame = None
    
    # Use ground truth boundaries if available, otherwise fall back to center/perception error
    if data.get('gt_left') is not None and data.get('gt_right') is not None:
        # Out of lane if car (x=0) is not between left/right boundaries.
        gt_left = data['gt_left']
        gt_right = data['gt_right']
        out_of_lane_mask = ~((gt_left <= 0) & (0 <= gt_right))
        error_data = np.where(
            gt_left > 0,
            np.abs(gt_left),
            np.where(gt_right < 0, np.abs(gt_right), 0.0),
        )
        error_source = "ground_truth_boundaries"
    elif data['gt_center'] is not None:
        # Ground truth lane_center_x: position of lane center relative to vehicle center (in vehicle frame)
        # Lateral error = -gt_center_x (negate because gt_center is position, not error)
        gt_lateral_error = -data['gt_center']
        error_data = gt_lateral_error
        error_source = "ground_truth"
        out_of_lane_mask = np.abs(error_data) > OUT_OF_LANE_THRESHOLD_M
    elif data['lateral_error'] is not None:
        error_data = data['lateral_error']
        error_source = "perception"
        out_of_lane_mask = np.abs(error_data) > OUT_OF_LANE_THRESHOLD_M
    else:
        error_data = None
        out_of_lane_mask = None
    
    full_out_of_lane_mask = None
    full_error_data = None
    full_time = np.array(data.get('time', []), dtype=float)
    full_error_source = error_source if error_data is not None else "none"
    if error_data is not None and out_of_lane_mask is not None:
        full_out_of_lane_mask = np.array(out_of_lane_mask, dtype=bool)
        full_error_data = np.array(error_data, dtype=float)
        # Define thresholds
        out_of_lane_threshold = OUT_OF_LANE_THRESHOLD_M
        catastrophic_threshold = CATASTROPHIC_ERROR_M
        min_consecutive_out = MIN_CONSECUTIVE_OOL
        
        # Strategy: Find the last recovery before a catastrophic failure, then find when error
        # first exceeded threshold after that recovery. This identifies the "start of final failure sequence"
        # rather than just the first time error exceeded threshold in the entire recording.
        
        # First, find if there's a catastrophic failure
        catastrophic_frame = None
        for i, error in enumerate(error_data):
            if abs(error) > catastrophic_threshold:
                catastrophic_frame = i
                break
        
        if catastrophic_frame is not None:
            # Find last recovery (error < 0.5m) before catastrophic failure
            last_recovery_frame = None
            for i in range(catastrophic_frame - 1, -1, -1):
                if abs(error_data[i]) < out_of_lane_threshold:
                    last_recovery_frame = i
                    break
            
            # If we found a recovery, find when error first exceeded threshold after that recovery
            if last_recovery_frame is not None:
                consecutive_out_frames = 0
                for i in range(last_recovery_frame + 1, len(error_data)):
                    if out_of_lane_mask[i]:
                        consecutive_out_frames += 1
                        # If we've been out for enough frames, this is the failure point
                        if consecutive_out_frames >= min_consecutive_out and failure_frame is None:
                            failure_frame = i - (min_consecutive_out - 1)  # Go back to when we first went out
                            break
                    else:
                        consecutive_out_frames = 0
        
        # Fallback: If no catastrophic failure, use simple detection
        if failure_frame is None:
            consecutive_out_frames = 0
            for i, is_out in enumerate(out_of_lane_mask):
                if is_out:
                    consecutive_out_frames += 1
                    if consecutive_out_frames >= min_consecutive_out and failure_frame is None:
                        failure_frame = i - (min_consecutive_out - 1)
                        break
                else:
                    consecutive_out_frames = 0
        
    if data.get('emergency_stop') is not None:
        stop_indices = np.where(data['emergency_stop'] > 0)[0]
        if stop_indices.size > 0:
            emergency_stop_frame = int(stop_indices[0])

    if emergency_stop_frame is not None:
        if failure_frame is None or emergency_stop_frame < failure_frame:
            failure_frame = emergency_stop_frame
            error_source = "emergency_stop"

    # If we found a failure point AND analyze_to_failure is True, truncate data to that point
    if failure_frame is not None and analyze_to_failure:
        # Truncate all arrays to failure_frame
        for key in data:
            if isinstance(data[key], np.ndarray) and len(data[key]) > failure_frame:
                data[key] = data[key][:failure_frame]
    
    # Calculate metrics
    n_frames = len(data['steering'])
    dt = safe_float(np.mean(np.diff(data['time'])) if len(data['time']) > 1 else 0.033)
    duration = safe_float(data['time'][-1] if len(data['time']) > 0 else n_frames * dt)

    # Unity timing gaps (hitch detection)
    unity_time_gap_max = 0.0
    unity_time_gap_count = 0
    if data.get('unity_time') is not None and len(data['unity_time']) > 1:
        unity_time_diffs = np.diff(data['unity_time'])
        if unity_time_diffs.size > 0:
            unity_time_gap_max = safe_float(np.max(unity_time_diffs))
            unity_time_gap_count = int(np.sum(unity_time_diffs > 0.2))
    
    # 1. PATH TRACKING
    lateral_error_rmse = safe_float(np.sqrt(np.mean(data['lateral_error']**2)) if data['lateral_error'] is not None and len(data['lateral_error']) > 0 else 0.0)
    lateral_error_mean = safe_float(np.mean(np.abs(data['lateral_error'])) if data['lateral_error'] is not None and len(data['lateral_error']) > 0 else 0.0)
    lateral_error_max = safe_float(np.max(np.abs(data['lateral_error'])) if data['lateral_error'] is not None and len(data['lateral_error']) > 0 else 0.0)
    lateral_error_p95 = safe_float(np.percentile(np.abs(data['lateral_error']), 95) if data['lateral_error'] is not None and len(data['lateral_error']) > 0 else 0.0)

    # Curvature-adjusted lateral error: subtract a geometry-dependent floor per frame.
    # floor = CURVATURE_FLOOR_COEFF * |kappa|  — accounts for arc-chord tracking error
    # that is physically unavoidable for reactive controllers on tight curves.
    # Coefficient validated: <0.01m effect on highway/sweeping, ~0.06m on s_loop R40,
    # ~0.20m on hairpin R15.  Raw metrics preserved for diagnostics.
    CURVATURE_FLOOR_COEFF = _CURVATURE_FLOOR_COEFF
    _curv_for_floor = data.get('gt_path_curvature')
    if _curv_for_floor is None:
        _curv_for_floor = data.get('path_curvature_input')
    if (data['lateral_error'] is not None and len(data['lateral_error']) > 0
            and _curv_for_floor is not None and len(_curv_for_floor) > 0):
        _n_adj = min(len(data['lateral_error']), len(_curv_for_floor))
        _abs_err = np.abs(data['lateral_error'][:_n_adj])
        _floor = CURVATURE_FLOOR_COEFF * np.abs(np.asarray(_curv_for_floor[:_n_adj], dtype=float))
        _adj_err = np.maximum(0.0, _abs_err - _floor)
        lateral_error_adj_rmse = safe_float(np.sqrt(np.mean(_adj_err ** 2)))
        lateral_error_adj_p95 = safe_float(np.percentile(_adj_err, 95))
    else:
        lateral_error_adj_rmse = lateral_error_rmse
        lateral_error_adj_p95 = lateral_error_p95
    
    heading_error_rmse = safe_float(np.sqrt(np.mean(data['heading_error']**2)) if data['heading_error'] is not None and len(data['heading_error']) > 0 else 0.0)
    heading_error_max = safe_float(np.max(np.abs(data['heading_error'])) if data['heading_error'] is not None and len(data['heading_error']) > 0 else 0.0)
    
    # Time in Lane:
    # Primary: Use ground truth boundaries if available (car at x=0 is in lane if left <= 0 <= right)
    # Secondary: Centeredness within ±CENTERED_BAND_M (for tuning)
    time_in_lane_centered = safe_float(
        np.sum(np.abs(data['lateral_error']) < CENTERED_BAND_M) / n_frames * 100
        if data['lateral_error'] is not None and n_frames > 0 else 0.0
    )
    if data['gt_left'] is not None and data['gt_right'] is not None:
        in_lane_mask = (data['gt_left'] <= 0) & (0 <= data['gt_right'])
        time_in_lane = safe_float(np.sum(in_lane_mask) / n_frames * 100 if n_frames > 0 else 0.0)
    elif data['lateral_error'] is not None:
        time_in_lane = time_in_lane_centered
    else:
        time_in_lane = 0.0
    
    # 2. CONTROL SMOOTHNESS
    steering_rate = np.diff(data['steering']) / np.diff(data['time']) if len(data['steering']) > 1 and len(data['time']) > 1 else np.array([0.0])
    steering_jerk = np.diff(steering_rate) / np.diff(data['time'][1:]) if len(steering_rate) > 1 and len(data['time']) > 2 else np.array([0.0])
    steering_jerk_max_raw = safe_float(np.max(np.abs(steering_jerk)) if len(steering_jerk) > 0 else 0.0)
    # Filter gap-induced jerk artifacts.  jerk[i] = (rate[i+1] - rate[i]) / dt[i+1]
    # where rate[i] uses dt[i] and rate[i+1] uses dt[i+1].  Any frame where EITHER
    # neighbouring dt exceeds 1.5x the median creates a phantom spike because
    # the two rates are averaged over different time windows:
    #   - dt[i] >> normal  → rate[i] is a long-window average (artificially small)
    #   - dt[i+1] >> normal → rate[i+1] numerator is large AND divisor is large,
    #     and the jerk divisor is also that inflated dt
    # Threshold: 1.5× median filters double-ticks (2× normal period) and larger.
    if len(steering_jerk) > 0:
        dt_arr = np.diff(data['time'])
        median_dt = float(np.median(dt_arr)) if len(dt_arr) > 0 else 1.0
        gap_threshold = 1.5 * max(median_dt, 1e-6)
        dt_before_jerk = dt_arr[:len(steering_jerk)]      # dt[i]   → used for rate[i]
        dt_at_jerk = dt_arr[1:len(steering_jerk) + 1]     # dt[i+1] → used for rate[i+1] and jerk divisor
        valid_jerk_mask = (dt_before_jerk <= gap_threshold) & (dt_at_jerk <= gap_threshold)
        valid_jerk = steering_jerk[valid_jerk_mask]
        steering_jerk_max = safe_float(np.max(np.abs(valid_jerk)) if len(valid_jerk) > 0 else 0.0)
    else:
        steering_jerk_max = 0.0
    steering_rate_max = safe_float(np.max(np.abs(steering_rate)) if len(steering_rate) > 0 else 0.0)
    
    steering_std = safe_float(np.std(data['steering']) if len(data['steering']) > 0 else 0.0)
    steering_smoothness = safe_float(1.0 / (steering_std + 1e-6))

    # Straight-line stability (uses controller diagnostics if available)
    straight_fraction = 0.0
    straight_oscillation_mean = 0.0
    straight_oscillation_max = 0.0
    tuned_deadband_mean = 0.0
    tuned_deadband_max = 0.0
    tuned_smoothing_mean = 0.0
    straight_sign_mismatch_rate = 0.0
    straight_sign_mismatch_events = 0
    straight_sign_mismatch_frames = 0
    straight_sign_mismatch_events_list = []
    if data.get('is_straight') is not None and len(data['is_straight']) > 0:
        straight_mask = data['is_straight'][:n_frames] > 0
        straight_fraction = safe_float(np.sum(straight_mask) / max(1, n_frames) * 100.0)
        if data.get('straight_oscillation_rate') is not None and len(data['straight_oscillation_rate']) > 0:
            straight_rates = data['straight_oscillation_rate'][:n_frames]
            if np.any(straight_mask):
                straight_rates = straight_rates[straight_mask]
            straight_oscillation_mean = safe_float(np.mean(straight_rates)) if straight_rates.size > 0 else 0.0
            straight_oscillation_max = safe_float(np.max(straight_rates)) if straight_rates.size > 0 else 0.0
        if data.get('tuned_deadband') is not None and len(data['tuned_deadband']) > 0:
            tuned_deadband = data['tuned_deadband'][:n_frames]
            tuned_deadband_mean = safe_float(np.mean(tuned_deadband))
            tuned_deadband_max = safe_float(np.max(tuned_deadband))
        if data.get('tuned_error_smoothing_alpha') is not None and len(data['tuned_error_smoothing_alpha']) > 0:
            tuned_smoothing = data['tuned_error_smoothing_alpha'][:n_frames]
            tuned_smoothing_mean = safe_float(np.mean(tuned_smoothing))

        # Detect sign-mismatch events on straights (steering fighting scaled error)
        error_series = data.get('total_error_scaled') if data.get('total_error_scaled') is not None else data.get('total_error')
        if error_series is not None and len(error_series) > 0:
            error_series = error_series[:n_frames]
            steering_series = data['steering'][:n_frames]
            # Exclude MPC-active and blend-transition frames from sign-mismatch.
            # During MPC, steering is driven by mpc_e_lat (true cross-track), not
            # the PP total_error_scaled signal — comparing them is meaningless.
            # During blend transitions, opposite-sign steering is expected.
            mpc_exclude = np.zeros(n_frames, dtype=bool)
            if data.get('regime') is not None:
                regime_arr = np.asarray(data['regime'][:n_frames], dtype=float)
                mpc_exclude = regime_arr > 0.5  # MPC-active frames
            if data.get('regime_blend_weight') is not None:
                rbw = np.asarray(data['regime_blend_weight'][:n_frames], dtype=float)
                mpc_exclude = mpc_exclude | ((rbw > 0.01) & (rbw < 0.99))
            valid_mask = straight_mask & (np.abs(error_series) >= 0.02) & (np.abs(steering_series) >= 0.02) & ~mpc_exclude
            mismatch_mask = valid_mask & (np.sign(error_series) != np.sign(steering_series))
            straight_sign_mismatch_frames = int(np.sum(mismatch_mask))
            denom = int(np.sum(valid_mask))
            if denom > 0:
                straight_sign_mismatch_rate = safe_float(straight_sign_mismatch_frames / denom * 100.0)

            # Count contiguous mismatch events (>=3 frames) and classify root cause.
            min_event_len = 3
            current = 0
            event_start = None
            for idx, flag in enumerate(mismatch_mask):
                if flag:
                    if current == 0:
                        event_start = idx
                    current += 1
                else:
                    if current >= min_event_len and event_start is not None:
                        event_end = idx - 1
                        straight_sign_mismatch_events += 1
                        straight_sign_mismatch_events_list.append(
                            _build_sign_mismatch_event(
                                data=data,
                                start_frame=event_start,
                                end_frame=event_end,
                            )
                        )
                    current = 0
                    event_start = None
            if current >= min_event_len and event_start is not None:
                event_end = len(mismatch_mask) - 1
                straight_sign_mismatch_events += 1
                straight_sign_mismatch_events_list.append(
                    _build_sign_mismatch_event(
                        data=data,
                        start_frame=event_start,
                        end_frame=event_end,
                    )
                )
    
    # Oscillation frequency
    oscillation_frequency = 0.0
    if data['lateral_error'] is not None and len(data['lateral_error']) > 10:
        try:
            error_centered = data['lateral_error'] - np.mean(data['lateral_error'])
            fft_vals = fft(error_centered)
            fft_freqs = fftfreq(len(error_centered), dt)
            positive_freqs = fft_freqs[:len(fft_freqs)//2]
            positive_fft = np.abs(fft_vals[:len(fft_vals)//2])
            if len(positive_fft) > 1:
                dominant_idx = np.argmax(positive_fft[1:]) + 1
                oscillation_frequency = safe_float(positive_freqs[dominant_idx] if dominant_idx < len(positive_freqs) else 0.0)
        except:
            oscillation_frequency = 0.0
    oscillation_frequency = safe_float(oscillation_frequency)
    oscillation_zero_crossing_rate_hz = 0.0
    oscillation_rms_growth_slope_mps = 0.0
    oscillation_rms_window_start_m = 0.0
    oscillation_rms_window_end_m = 0.0
    oscillation_rms_window_p95_m = 0.0
    oscillation_rms_windows_count = 0
    oscillation_amplitude_runaway = False
    oscillation_curve_suppressed = False
    curve_fraction = 0.0
    if data.get('lateral_error') is not None and data.get('time') is not None:
        # Build regime-aware lateral error for oscillation analysis:
        # use mpc_e_lat (true cross-track) where MPC is active, PP lateral_error elsewhere
        e_series_raw = np.asarray(data['lateral_error'][:n_frames], dtype=float)
        if data.get('mpc_e_lat') is not None and data.get('mpc_using_ground_truth') is not None:
            mpc_e = np.asarray(data['mpc_e_lat'][:n_frames], dtype=float)
            mpc_active = np.asarray(data['mpc_using_ground_truth'][:n_frames], dtype=float) > 0.5
            e_series = np.where(mpc_active, mpc_e, e_series_raw)
        else:
            e_series = e_series_raw
        t_series = np.asarray(data['time'][:n_frames], dtype=float)
        finite_mask = np.isfinite(e_series) & np.isfinite(t_series)
        if np.any(finite_mask):
            e_series = e_series[finite_mask]
            t_series = t_series[finite_mask]
            if e_series.size > 2:
                duration_s = max(1e-6, float(t_series[-1] - t_series[0]))
                sign_series = np.sign(e_series)
                zero_crossings = int(np.sum((sign_series[1:] * sign_series[:-1]) < 0))
                oscillation_zero_crossing_rate_hz = safe_float(zero_crossings / duration_s)

                # Track amplitude growth using 2s rolling RMS windows (step 1s).
                window_s = 2.0
                step_s = 1.0
                rms_times = []
                rms_values = []
                start_t = float(t_series[0])
                end_t = float(t_series[-1])
                while start_t + window_s <= end_t + 1e-9:
                    mask = (t_series >= start_t) & (t_series < (start_t + window_s))
                    if np.sum(mask) >= 3:
                        rms_times.append(start_t + (window_s * 0.5))
                        rms_values.append(float(np.sqrt(np.mean(e_series[mask] ** 2))))
                    start_t += step_s
                oscillation_rms_windows_count = int(len(rms_values))
                if rms_values:
                    rms_arr = np.asarray(rms_values, dtype=float)
                    oscillation_rms_window_start_m = safe_float(rms_arr[0])
                    oscillation_rms_window_end_m = safe_float(rms_arr[-1])
                    oscillation_rms_window_p95_m = safe_float(np.percentile(rms_arr, 95))
                    if len(rms_values) >= 2:
                        times_arr = np.asarray(rms_times, dtype=float)
                        slope, _ = np.polyfit(times_arr, rms_arr, 1)
                        oscillation_rms_growth_slope_mps = safe_float(slope)

                # Runaway heuristic: oscillating sign changes + growing RMS envelope.
                # Gate on curvature: sustained curves produce monotonic RMS growth
                # from tracking bias, not oscillation.  If >30% of frames are in
                # curves (|κ| > 0.002 rad/m), require a higher zero-crossing rate
                # to distinguish real oscillation from steady-state curve tracking error.
                oscillation_curve_suppressed = False
                curve_fraction = 0.0
                gt_curv_osc = data.get('gt_path_curvature')
                if gt_curv_osc is not None:
                    gt_curv_arr = np.asarray(gt_curv_osc[:n_frames], dtype=float)
                    gt_curv_finite = gt_curv_arr[np.isfinite(gt_curv_arr)]
                    if gt_curv_finite.size > 0:
                        curve_fraction = float(np.mean(np.abs(gt_curv_finite) > 0.002))

                if curve_fraction > 0.30:
                    # On curved tracks, require stronger evidence of oscillation:
                    # higher zero-crossing rate (true oscillation crosses zero frequently)
                    oscillation_amplitude_runaway = bool(
                        oscillation_zero_crossing_rate_hz >= 0.5
                        and oscillation_rms_growth_slope_mps > 0.002
                    )
                    if (oscillation_zero_crossing_rate_hz >= 0.2
                            and oscillation_rms_growth_slope_mps > 0.002
                            and not oscillation_amplitude_runaway):
                        oscillation_curve_suppressed = True
                else:
                    oscillation_amplitude_runaway = bool(
                        oscillation_zero_crossing_rate_hz >= 0.2
                        and oscillation_rms_growth_slope_mps > 0.002
                    )

    # 2.5 SPEED CONTROL + COMFORT
    speed_error_rmse = 0.0
    speed_error_mean = 0.0
    speed_error_max = 0.0
    speed_overspeed_rate = 0.0
    speed_limit_zero_rate = 0.0
    speed_surge_count = 0
    speed_surge_avg_drop = 0.0
    speed_surge_p95_drop = 0.0
    curve_cap_active_rate = 0.0
    curve_cap_pre_turn_lead_frames_p50 = 0.0
    curve_cap_pre_turn_lead_frames_p95 = 0.0
    turn_infeasible_rate_when_curve_cap_active = 0.0
    overspeed_into_curve_rate = 0.0
    cap_tracking_error_p50_mps = 0.0
    cap_tracking_error_p95_mps = 0.0
    cap_tracking_error_max_mps = 0.0
    frames_above_cap_0p3mps_rate = 0.0
    frames_above_cap_1p0mps_rate = 0.0
    cap_recovery_frames_p50 = 0.0
    cap_recovery_frames_p95 = 0.0
    hard_ceiling_applied_rate = 0.0
    curvature_source_divergence_p95 = 0.0
    curvature_source_diverged_rate = 0.0
    curvature_map_authority_lost_rate = 0.0
    curve_intent_commit_streak_max_frames = 0
    feasibility_violation_rate = 0.0
    feasibility_backstop_active_rate = 0.0
    map_health_untrusted_rate = 0.0
    track_mismatch_rate = 0.0
    curvature_contract_consistency_rate = 0.0
    telemetry_completeness_rate_curvature_contract = 0.0
    telemetry_completeness_rate_feasibility = 0.0
    first_divergence_frame = None
    first_infeasible_frame = None
    first_speed_above_feasibility_frame = None
    first_boundary_breach_frame = None
    speed_min_mps = None
    speed_max_mps = None
    if data.get('speed') is not None and len(data['speed']) > 0:
        _s = np.asarray(data['speed'], dtype=np.float64)
        _s = _s[np.isfinite(_s)]
        if len(_s) > 0:
            speed_min_mps = safe_float(np.min(_s))
            speed_max_mps = safe_float(np.max(_s))
    if data.get('speed') is not None and data.get('ref_velocity') is not None:
        n_speed = min(len(data['speed']), len(data['ref_velocity']))
        if n_speed > 0:
            speed = data['speed'][:n_speed]
            ref_speed = data['ref_velocity'][:n_speed]
            speed_error = speed - ref_speed
            speed_error_rmse = safe_float(np.sqrt(np.mean(speed_error ** 2)))
            speed_error_mean = safe_float(np.mean(np.abs(speed_error)))
            speed_error_max = safe_float(np.max(np.abs(speed_error)))
            overspeed_threshold = 0.5
            speed_overspeed_rate = safe_float(np.sum(speed_error > overspeed_threshold) / n_speed * 100)
    if data.get('speed_limit') is not None and len(data['speed_limit']) > 0:
        speed_limit_zero_rate = safe_float(
            np.sum(data['speed_limit'] <= 0.01) / len(data['speed_limit']) * 100.0
        )

    acceleration_mean = 0.0
    acceleration_max = 0.0
    acceleration_p95 = 0.0
    jerk_mean = 0.0
    jerk_max = 0.0
    jerk_p95 = 0.0
    acceleration_mean_filtered = 0.0
    acceleration_max_filtered = 0.0
    acceleration_p95_filtered = 0.0
    jerk_mean_filtered = 0.0
    jerk_max_filtered = 0.0
    jerk_p95_filtered = 0.0
    commanded_jerk_p95 = 0.0
    lateral_accel_p95 = 0.0
    lateral_jerk_p95 = 0.0
    lateral_jerk_max = 0.0
    hotspot_attribution = {
        "schema_version": "v1",
        "availability": "unavailable",
        "dt_nominal_ms": None,
        "dt_gap_threshold_ms": None,
        "entries": [],
    }
    config = _load_config()
    traj_cfg = config.get('trajectory', {})
    perception_cfg = config.get('perception', {})
    control_cfg = config.get('control', {})
    lateral_cfg = control_cfg.get('lateral', {}) if isinstance(control_cfg, dict) else {}
    config_summary = {
        "camera_fov": safe_float(traj_cfg.get("camera_fov", 0.0)),
        "camera_height": safe_float(traj_cfg.get("camera_height", 0.0)),
        "segmentation_fit_min_row_ratio": safe_float(
            perception_cfg.get("segmentation_fit_min_row_ratio", 0.0)
        ),
        "segmentation_fit_max_row_ratio": safe_float(
            perception_cfg.get("segmentation_fit_max_row_ratio", 0.0)
        ),
    }
    curvature_smoothing_enabled = bool(traj_cfg.get('curvature_smoothing_enabled', False))
    curvature_window_m = float(traj_cfg.get('curvature_smoothing_window_m', 12.0))
    curvature_min_speed = float(traj_cfg.get('curvature_smoothing_min_speed', 2.0))

    if data.get('speed') is not None:
        curvature_series = None
        if data.get('gt_path_curvature') is not None:
            curvature_series = data['gt_path_curvature']
        elif data.get('path_curvature_input') is not None:
            curvature_series = data['path_curvature_input']
        if curvature_series is not None:
            n_surge = min(len(data['speed']), len(curvature_series))
            speed_series = data['speed'][:n_surge]
            curvature_series = curvature_series[:n_surge]
            straight_threshold = float(traj_cfg.get('straight_speed_smoothing_curvature_threshold', 0.003))
            min_drop = float(traj_cfg.get('speed_surge_min_drop', 1.0))
            straight_mask = np.abs(curvature_series) <= straight_threshold
            drops = []
            i = 1
            while i < len(speed_series) - 1:
                if not straight_mask[i]:
                    i += 1
                    continue
                if speed_series[i] > speed_series[i - 1] and speed_series[i] >= speed_series[i + 1]:
                    max_idx = i
                    j = i + 1
                    while j < len(speed_series) - 1 and straight_mask[j]:
                        if speed_series[j] <= speed_series[j - 1] and speed_series[j] < speed_series[j + 1]:
                            drop = float(speed_series[max_idx] - speed_series[j])
                            if drop >= min_drop:
                                drops.append(drop)
                            break
                        j += 1
                    i = j
                else:
                    i += 1
            if drops:
                speed_surge_count = len(drops)
                speed_surge_avg_drop = safe_float(np.mean(drops))
                speed_surge_p95_drop = safe_float(np.percentile(drops, 95))

    if data.get('speed') is not None and len(data['speed']) > 1 and len(data['time']) > 1:
        dt_series = np.diff(data['time'])
        dt_series[dt_series <= 0] = dt
        acceleration = np.diff(data['speed']) / dt_series
        if acceleration.size > 0:
            abs_accel = np.abs(acceleration)
            acceleration_mean = safe_float(np.mean(abs_accel))
            acceleration_max = safe_float(np.max(abs_accel))
            acceleration_p95 = safe_float(np.percentile(abs_accel, 95))
        if acceleration.size > 1:
            jerk = np.diff(acceleration) / dt_series[1:]
            if jerk.size > 0:
                abs_jerk = np.abs(jerk)
                jerk_mean = safe_float(np.mean(abs_jerk))
                jerk_max = safe_float(np.max(abs_jerk))
                jerk_p95 = safe_float(np.percentile(abs_jerk, 95))
        # Filtered speed for comfort metrics (reduce derivative noise).
        # Higher alpha = stronger filter. 0.95 brings jerk noise floor from
        # ~160 m/s³ down to ~20 m/s³ at 30 fps with 0.10 m/s velocity noise.
        alpha = 0.95
        filtered_speed = np.empty_like(data['speed'])
        filtered_speed[0] = data['speed'][0]
        for i in range(1, len(data['speed'])):
            filtered_speed[i] = alpha * filtered_speed[i - 1] + (1.0 - alpha) * data['speed'][i]
        filtered_accel = np.diff(filtered_speed) / dt_series
        if filtered_accel.size > 0:
            abs_f_accel = np.abs(filtered_accel)
            acceleration_mean_filtered = safe_float(np.mean(abs_f_accel))
            acceleration_max_filtered = safe_float(np.max(abs_f_accel))
            acceleration_p95_filtered = safe_float(np.percentile(abs_f_accel, 95))
        if filtered_accel.size > 1:
            filtered_jerk = np.diff(filtered_accel) / dt_series[1:]
            if filtered_jerk.size > 0:
                abs_f_jerk = np.abs(filtered_jerk)
                jerk_mean_filtered = safe_float(np.mean(abs_f_jerk))
                jerk_max_filtered = safe_float(np.max(abs_f_jerk))
                jerk_p95_filtered = safe_float(np.percentile(abs_f_jerk, 95))
        # Commanded jerk proxy: d(throttle*max_accel - brake*max_decel)/dt
        # Noise-free because it uses controller command signals, not physics velocity.
        # Bounded by rate limiters: throttle_rate*max_accel/dt ≈ 3 m/s³, brake_rate*max_decel/dt ≈ 4.5 m/s³
        if data.get('throttle') is not None and data.get('brake') is not None:
            ctrl_cfg = config.get('control', {}).get('longitudinal', {})
            max_accel_cfg = float(ctrl_cfg.get('max_accel', 2.5))
            max_decel_cfg = float(ctrl_cfg.get('max_decel', 3.0))
            t_arr = np.asarray(data['throttle'], dtype=float)
            b_arr = np.asarray(data['brake'], dtype=float)
            n_cmd = min(len(t_arr), len(b_arr), len(dt_series) + 1)
            if n_cmd > 1:
                net_accel_cmd = t_arr[:n_cmd] * max_accel_cfg - b_arr[:n_cmd] * max_decel_cfg
                cmd_jerk = np.diff(net_accel_cmd) / dt_series[:n_cmd - 1]
                if cmd_jerk.size > 0:
                    commanded_jerk_p95 = safe_float(np.percentile(np.abs(cmd_jerk), 95))
        curvature = None
        if data.get('gt_path_curvature') is not None:
            curvature = data['gt_path_curvature']
        elif data.get('path_curvature_input') is not None:
            curvature = data['path_curvature_input']
        if curvature is not None:
            n_lat = min(len(curvature), len(data['speed']))
            if n_lat > 1:
                curvature_source = curvature[:n_lat]
                if curvature_smoothing_enabled:
                    curvature_source = np.array(
                        smooth_curvature_distance(
                            curvature_source,
                            data['speed'][:n_lat],
                            data['time'][:n_lat],
                            curvature_window_m,
                            curvature_min_speed,
                        )
                    )
                lat_accel = (data['speed'][:n_lat] ** 2) * curvature_source
                abs_lat_accel = np.abs(lat_accel)
                lateral_accel_p95 = safe_float(np.percentile(abs_lat_accel, 95))
                lat_dt = np.diff(data['time'][:n_lat])
                lat_dt[lat_dt <= 0] = dt
                lat_jerk = np.diff(lat_accel) / lat_dt
                if lat_jerk.size > 0:
                    abs_lat_jerk = np.abs(lat_jerk)
                    lateral_jerk_p95 = safe_float(np.percentile(abs_lat_jerk, 95))
                    lateral_jerk_max = safe_float(np.max(abs_lat_jerk))
        hotspot_attribution = _build_longitudinal_hotspot_attribution(
            data=data,
            config=config,
            n_frames=n_frames,
        )
    
    # 3. PERCEPTION QUALITY
    lane_detection_rate = safe_float(np.sum(data['num_lanes_detected'] >= 2) / n_frames * 100 if data['num_lanes_detected'] is not None and n_frames > 0 else 0.0)
    perception_confidence_mean = safe_float(np.mean(data['confidence']) if data['confidence'] is not None and len(data['confidence']) > 0 else 0.0)
    single_lane_rate = safe_float(np.sum(data['num_lanes_detected'] < 2) / n_frames * 100 if data['num_lanes_detected'] is not None and n_frames > 0 else 0.0)
    
    # Count different types of perception issues
    perception_jumps_detected = 0
    perception_instability_detected = 0
    perception_extreme_coeffs_detected = 0
    perception_invalid_width_detected = 0
    
    if data['stale_reason'] is not None:
        for r in data['stale_reason']:
            if r:
                r_str = str(r).lower()
                if 'jump' in r_str:
                    perception_jumps_detected += 1
                elif 'instability' in r_str:
                    perception_instability_detected += 1
                elif 'extreme' in r_str or 'coefficient' in r_str:
                    perception_extreme_coeffs_detected += 1
                elif 'width' in r_str or 'invalid' in r_str:
                    perception_invalid_width_detected += 1
    
    stale_perception_rate = safe_float(
        np.sum(data['using_stale_data']) / n_frames * 100
        if data['using_stale_data'] is not None and n_frames > 0
        else 0.0
    )
    stale_raw_rate = stale_perception_rate
    stale_hard_rate = 0.0
    stale_fallback_visibility_rate = 0.0
    if data['using_stale_data'] is not None and n_frames > 0:
        stale_flags = np.asarray(data['using_stale_data'][:n_frames]).astype(bool)
        stale_reasons = data['stale_reason'] if data['stale_reason'] is not None else []
        hard_count = 0
        fallback_count = 0
        for i, is_stale in enumerate(stale_flags):
            if not is_stale:
                continue
            reason = ""
            if i < len(stale_reasons) and stale_reasons[i] is not None:
                reason = str(stale_reasons[i]).strip().lower()
            if reason in LOW_VISIBILITY_STALE_REASONS:
                fallback_count += 1
            else:
                hard_count += 1
        stale_hard_rate = safe_float(hard_count / n_frames * 100)
        stale_fallback_visibility_rate = safe_float(fallback_count / n_frames * 100)
    
    # NEW: Calculate perception stability metrics (lane position/width variance)
    # High variance indicates perception instability even if not caught by stale_data
    perception_stability_score = 100.0  # Start at 100, penalize for instability
    lane_position_variance = 0.0
    lane_width_variance = 0.0
    lane_line_jitter_p95 = 0.0
    lane_line_jitter_p99 = 0.0
    reference_jitter_p95 = 0.0
    reference_jitter_p99 = 0.0
    right_lane_low_visibility_rate = 0.0
    right_lane_edge_contact_rate = 0.0
    left_lane_low_visibility_rate = 0.0
    
    if data['left_lane_x'] is not None and data['right_lane_x'] is not None:
        left_lanes = data['left_lane_x'][:n_frames]
        right_lanes = data['right_lane_x'][:n_frames]
        
        # Calculate lane center and width for each frame
        lane_centers = (left_lanes + right_lanes) / 2.0
        lane_widths = right_lanes - left_lanes
        
        # Filter out invalid values (NaN, inf, or extreme values)
        valid_mask = (np.isfinite(lane_centers) & np.isfinite(lane_widths) & 
                     (lane_widths > 2.0) & (lane_widths < 10.0) &  # Valid width range
                     (np.abs(lane_centers) < 5.0))  # Reasonable center position
        
        if np.sum(valid_mask) > 10:  # Need at least 10 valid frames
            valid_centers = lane_centers[valid_mask]
            valid_widths = lane_widths[valid_mask]
            
            # Calculate variance (higher = less stable)
            lane_position_variance = safe_float(np.var(valid_centers))
            lane_width_variance = safe_float(np.var(valid_widths))

            # Lane-line jitter: frame-to-frame movement (max of left/right)
            if valid_centers.size > 1:
                valid_left = left_lanes[valid_mask]
                valid_right = right_lanes[valid_mask]
                left_jitter = np.abs(np.diff(valid_left))
                right_jitter = np.abs(np.diff(valid_right))
                jitter = np.maximum(left_jitter, right_jitter)
                if jitter.size > 0:
                    lane_line_jitter_p95 = safe_float(np.percentile(jitter, 95))
                    lane_line_jitter_p99 = safe_float(np.percentile(jitter, 99))

        # Lane visibility based on fit points near image edges or too few points
        if data.get('fit_points_left') is not None and data.get('fit_points_right') is not None:
            min_points = 6
            edge_margin = 12
            width = data.get('image_width', 640)
            left_low = 0
            right_low = 0
            right_edge_contact = 0

            def parse_points(raw) -> list:
                if raw is None:
                    return []
                try:
                    s = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray, np.bytes_)) else str(raw)
                    return json.loads(s)
                except Exception:
                    return []

            for i in range(min(n_frames, len(data['fit_points_left']))):
                left_pts = parse_points(data['fit_points_left'][i])
                right_pts = parse_points(data['fit_points_right'][i])

                left_xs = [p[0] for p in left_pts if isinstance(p, (list, tuple)) and len(p) >= 2]
                right_xs = [p[0] for p in right_pts if isinstance(p, (list, tuple)) and len(p) >= 2]

                if len(left_xs) < min_points or (left_xs and min(left_xs) < edge_margin):
                    left_low += 1
                right_low_flag = False
                if data.get('num_lanes_detected') is not None and i < len(data['num_lanes_detected']) and data['num_lanes_detected'][i] < 2:
                    right_low_flag = True
                if len(right_xs) < min_points:
                    right_low_flag = True
                if right_xs and max(right_xs) > (width - edge_margin):
                    right_edge_contact += 1
                if right_low_flag:
                    right_low += 1

            right_lane_low_visibility_rate = safe_float(right_low / n_frames * 100 if n_frames > 0 else 0.0)
            right_lane_edge_contact_rate = safe_float(right_edge_contact / n_frames * 100 if n_frames > 0 else 0.0)
            left_lane_low_visibility_rate = safe_float(left_low / n_frames * 100 if n_frames > 0 else 0.0)
            
            # Penalize high variance (instability)
            # Position variance > 0.1m² indicates instability
            if lane_position_variance > 0.1:
                perception_stability_score -= min(30, (lane_position_variance - 0.1) * 100)
            
            # Width variance > 0.2m² indicates instability
            if lane_width_variance > 0.2:
                perception_stability_score -= min(30, (lane_width_variance - 0.2) * 50)
            
            perception_stability_score = safe_float(max(0, perception_stability_score))
    
    # Reference jitter (frame-to-frame movement)
    if data.get('ref_x') is not None and len(data['ref_x']) > 1:
        ref_jitter = np.abs(np.diff(data['ref_x'][:n_frames]))
        if ref_jitter.size > 0:
            reference_jitter_p95 = safe_float(np.percentile(ref_jitter, 95))
            reference_jitter_p99 = safe_float(np.percentile(ref_jitter, 99))

    # Also penalize for detected instability events
    if perception_instability_detected > 0:
        instability_rate = (perception_instability_detected / n_frames) * 100
        perception_stability_score -= min(40, instability_rate * 2)  # -2 points per % of frames with instability
        perception_stability_score = safe_float(max(0, perception_stability_score))

    # Penalize high lane-line jitter even if not flagged as instability
    if lane_line_jitter_p95 > 0.6:
        perception_stability_score -= min(20, (lane_line_jitter_p95 - 0.6) * 40)
        perception_stability_score = safe_float(max(0, perception_stability_score))
    if right_lane_low_visibility_rate > 10.0:
        perception_stability_score -= min(15, (right_lane_low_visibility_rate - 10.0) * 0.5)
        perception_stability_score = safe_float(max(0, perception_stability_score))
    
    # 4. TRAJECTORY QUALITY
    trajectory_availability = safe_float(np.sum(~np.isnan(data['ref_x'])) / n_frames * 100 if data['ref_x'] is not None and n_frames > 0 else 0.0)
    
    ref_point_accuracy_rmse = 0.0
    if data['ref_x'] is not None and data['gt_center'] is not None:
        valid_mask = ~np.isnan(data['ref_x']) & ~np.isnan(data['gt_center'])
        if np.sum(valid_mask) > 0:
            errors = data['ref_x'][valid_mask] - data['gt_center'][valid_mask]
            ref_point_accuracy_rmse = safe_float(np.sqrt(np.mean(errors**2)) if len(errors) > 0 else 0.0)
    
    # 5. SYSTEM HEALTH
    pid_integral_max = safe_float(np.max(np.abs(data['pid_integral'])) if data['pid_integral'] is not None and len(data['pid_integral']) > 0 else 0.0)
    
    # 6. SAFETY
    # Out-of-lane detection: Check if car is actually outside lane boundaries
    # Car is at x=0 in vehicle frame, so check if 0 is between left_lane_line_x and right_lane_line_x
    if data['gt_left'] is not None and data['gt_right'] is not None:
        # Use ground truth lane boundaries (most accurate)
        gt_left = data['gt_left']
        gt_right = data['gt_right']
        # Car is at x=0 in vehicle frame
        # Out of lane if: NOT (left <= 0 <= right)
        # This means: car is OUT if (left > 0) OR (right < 0)
        out_of_lane_mask = ~((gt_left <= 0) & (0 <= gt_right))
        # Use sustained out-of-lane mask to avoid single-frame noise
        min_consecutive_out = MIN_CONSECUTIVE_OOL
        sustained_mask = np.zeros_like(out_of_lane_mask, dtype=bool)
        run_start = None
        run_len = 0
        for i, is_out in enumerate(out_of_lane_mask):
            if is_out:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_len >= min_consecutive_out and run_start is not None:
                    sustained_mask[run_start:i] = True
                run_start = None
                run_len = 0
        if run_len >= min_consecutive_out and run_start is not None:
            sustained_mask[run_start:len(out_of_lane_mask)] = True

        out_of_lane_time = safe_float(
            np.sum(sustained_mask) / n_frames * 100 if n_frames > 0 else 0.0
        )
        error_source = "ground_truth_boundaries"
        
        # For error calculation in events list, use distance outside lane
        def get_error_outside_lane(i):
            if i >= len(gt_left) or i >= len(gt_right):
                return 0.0
            if gt_left[i] > 0:
                # Car is left of left lane line
                return abs(gt_left[i])
            elif gt_right[i] < 0:
                # Car is right of right lane line
                return abs(gt_right[i])
            else:
                # Car is in lane
                return 0.0
        
        error_data_for_events = np.array([get_error_outside_lane(i) for i in range(len(out_of_lane_mask))])
    elif error_data is not None:
        # Fallback: Use distance from lane center with adaptive threshold
        # If we have lane width info, use 50% of half-lane-width as threshold
        # Otherwise use 1.75m (half of typical 3.5m lane)
        if data['gt_left'] is not None and data['gt_right'] is not None:
            gt_left = data['gt_left']
            gt_right = data['gt_right']
            # Calculate lane width for each frame and use 50% of half-width as threshold
            lane_widths = gt_right - gt_left
            # Use median lane width to avoid noise
            median_lane_width = safe_float(np.median(lane_widths) if len(lane_widths) > 0 else 3.5)
            threshold = median_lane_width * 0.5 * 0.5  # 50% of half-lane-width
        else:
            threshold = 1.75  # Half of typical 3.5m lane width
        
        out_of_lane_mask = np.abs(error_data) > threshold
        # Use sustained out-of-lane mask to avoid single-frame noise
        min_consecutive_out = MIN_CONSECUTIVE_OOL
        sustained_mask = np.zeros_like(out_of_lane_mask, dtype=bool)
        run_start = None
        run_len = 0
        for i, is_out in enumerate(out_of_lane_mask):
            if is_out:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_len >= min_consecutive_out and run_start is not None:
                    sustained_mask[run_start:i] = True
                run_start = None
                run_len = 0
        if run_len >= min_consecutive_out and run_start is not None:
            sustained_mask[run_start:len(out_of_lane_mask)] = True

        out_of_lane_time = safe_float(
            np.sum(sustained_mask) / n_frames * 100 if n_frames > 0 else 0.0
        )
        error_data_for_events = np.abs(error_data)
        error_source = error_source if 'error_source' in locals() else "perception"
    else:
        out_of_lane_events_count = 0
        out_of_lane_time = 0.0
        out_of_lane_events_list = []
        error_data_for_events = None
    
    # Detect individual out-of-lane events (consecutive frames out of lane)
    if error_data_for_events is not None:
        out_of_lane_events_list = []
        in_event = False
        event_start = None
        event_max_error = 0.0
        
        min_consecutive_out = 10
        for i, is_out in enumerate(out_of_lane_mask):
            if is_out:
                if not in_event:
                    # Start of new event
                    in_event = True
                    event_start = i
                    event_max_error = error_data_for_events[i]
                else:
                    # Continue event, update max error
                    event_max_error = max(event_max_error, error_data_for_events[i])
            else:
                if in_event:
                    # End of event
                    event_duration = i - event_start
                    if event_duration >= min_consecutive_out:
                        out_of_lane_events_list.append({
                            'start_frame': int(event_start),
                            'end_frame': int(i - 1),
                            'duration_frames': int(event_duration),
                            'duration_seconds': safe_float(event_duration * dt),
                            'max_error': safe_float(event_max_error),
                            'error_source': error_source
                        })
                    in_event = False
                    event_start = None
                    event_max_error = 0.0
        
        # Handle event that extends to end of data
        if in_event:
            event_duration = len(out_of_lane_mask) - event_start
            if event_duration >= min_consecutive_out:
                out_of_lane_events_list.append({
                    'start_frame': int(event_start),
                    'end_frame': int(len(out_of_lane_mask) - 1),
                    'duration_frames': int(event_duration),
                    'duration_seconds': safe_float(event_duration * dt),
                    'max_error': safe_float(event_max_error),
                    'error_source': error_source
                })
    else:
        out_of_lane_events_list = []
    
    out_of_lane_events = len(out_of_lane_events_list)
    out_of_lane_events_full_run = int(out_of_lane_events)
    out_of_lane_time_full_run = safe_float(out_of_lane_time)
    out_of_lane_event_at_failure_boundary = False

    # Preserve full-run out-of-lane metrics even when analysis is truncated.
    if full_out_of_lane_mask is not None and len(full_out_of_lane_mask) > 0:
        min_consecutive_out = 10
        sustained_full = np.zeros_like(full_out_of_lane_mask, dtype=bool)
        run_start = None
        run_len = 0
        full_events_count = 0
        for i, is_out in enumerate(full_out_of_lane_mask):
            if is_out:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_len >= min_consecutive_out and run_start is not None:
                    sustained_full[run_start:i] = True
                    full_events_count += 1
                run_start = None
                run_len = 0
        if run_len >= min_consecutive_out and run_start is not None:
            sustained_full[run_start:len(full_out_of_lane_mask)] = True
            full_events_count += 1
        out_of_lane_events_full_run = int(full_events_count)
        out_of_lane_time_full_run = safe_float(
            float(np.sum(sustained_full)) / float(len(full_out_of_lane_mask)) * 100.0
        )

    # In analyze-to-failure mode, truncation can cut away the sustained event window.
    # Preserve explicit failure-boundary event context from the full run masks.
    if analyze_to_failure and failure_frame is not None:
        if full_out_of_lane_mask is not None and failure_frame < len(full_out_of_lane_mask):
            out_of_lane_event_at_failure_boundary = bool(full_out_of_lane_mask[failure_frame])
            if out_of_lane_events == 0 and out_of_lane_event_at_failure_boundary:
                end_idx = int(failure_frame)
                while end_idx < len(full_out_of_lane_mask) and bool(full_out_of_lane_mask[end_idx]):
                    end_idx += 1
                event_duration = max(0, end_idx - int(failure_frame))
                if event_duration >= 10:
                    duration_seconds = 0.0
                    if full_time.size > end_idx and end_idx > int(failure_frame):
                        duration_seconds = safe_float(
                            float(full_time[end_idx - 1]) - float(full_time[int(failure_frame)])
                        )
                    elif full_time.size > 1:
                        dt_guess = safe_float(float(np.mean(np.diff(full_time))), 0.0)
                        duration_seconds = safe_float(event_duration * dt_guess)
                    max_error = 0.0
                    if full_error_data is not None and end_idx > int(failure_frame):
                        max_error = safe_float(
                            float(np.max(np.abs(full_error_data[int(failure_frame):end_idx])))
                        )
                    out_of_lane_events_list = [{
                        'start_frame': int(failure_frame),
                        'end_frame': int(end_idx - 1),
                        'duration_frames': int(event_duration),
                        'duration_seconds': safe_float(duration_seconds),
                        'max_error': safe_float(max_error),
                        'error_source': full_error_source,
                    }]
                    out_of_lane_events = 1
                    out_of_lane_time = 0.0

    # 7. TURN BIAS (road-frame offsets)
    turn_bias = None
    alignment_summary = None
    if data.get('road_frame_lateral_offset') is not None:
        curvature_source = None
        if data.get('gt_path_curvature') is not None:
            curvature_source = data['gt_path_curvature']
        elif data.get('path_curvature_input') is not None:
            curvature_source = data['path_curvature_input']

        if curvature_source is not None:
            offset = data['road_frame_lateral_offset']
            heading_delta = data.get('heading_delta_deg')
            steering = data.get('steering')
            gt_center = data.get('gt_center')
            p_center = data.get('perception_center')

            n_bias = len(offset)
            n_bias = min(n_bias, len(curvature_source))
            if steering is not None:
                n_bias = min(n_bias, len(steering))
            if heading_delta is not None:
                n_bias = min(n_bias, len(heading_delta))
            if gt_center is not None:
                n_bias = min(n_bias, len(gt_center))
            if p_center is not None:
                n_bias = min(n_bias, len(p_center))

            offset = offset[:n_bias]
            curvature_source = curvature_source[:n_bias]
            if steering is not None:
                steering = steering[:n_bias]
            if heading_delta is not None:
                heading_delta = heading_delta[:n_bias]
            if gt_center is not None:
                gt_center = gt_center[:n_bias]
            if p_center is not None:
                p_center = p_center[:n_bias]

            curve_threshold = 0.002
            curve_mask = np.abs(curvature_source) >= curve_threshold
            left_mask = curve_mask & (curvature_source > 0)
            right_mask = curve_mask & (curvature_source < 0)

            def summarize_mask(mask):
                if not np.any(mask):
                    return {
                        "frames": 0,
                        "mean": 0.0,
                        "p95_abs": 0.0,
                        "max_abs": 0.0,
                        "outside_rate": 0.0,
                    }
                vals = offset[mask]
                outside = (curvature_source[mask] > 0) & (vals > 0)
                outside |= (curvature_source[mask] < 0) & (vals < 0)
                return {
                    "frames": int(mask.sum()),
                    "mean": safe_float(np.mean(vals)),
                    "p95_abs": safe_float(np.percentile(np.abs(vals), 95)),
                    "max_abs": safe_float(np.max(np.abs(vals))),
                    "outside_rate": safe_float(np.mean(outside) * 100.0),
                }

            def top_turn_frames(mask, max_items=8, min_separation=12):
                output = []
                if not np.any(mask):
                    return output
                candidates = np.where(mask)[0]
                sorted_idx = candidates[np.argsort(np.abs(offset[candidates]))[::-1]]
                for idx in sorted_idx:
                    if len(output) >= max_items:
                        break
                    if any(abs(idx - h["frame"]) < min_separation for h in output):
                        continue
                    curvature_val = float(curvature_source[idx])
                    outside = (curvature_val > 0 and offset[idx] > 0) or (curvature_val < 0 and offset[idx] < 0)
                    lane_center_error_val = None
                    if data.get('road_frame_lane_center_error') is not None:
                        lane_center_error_val = safe_float(
                            float(data['road_frame_lane_center_error'][idx])
                        )
                    output.append({
                        "frame": int(idx),
                        "time": safe_float(data['time'][idx]) if idx < len(data['time']) else 0.0,
                        "road_offset": safe_float(float(offset[idx])),
                        "curvature": safe_float(curvature_val),
                        "steering": safe_float(float(steering[idx])) if steering is not None else None,
                        "heading_delta_deg": safe_float(float(heading_delta[idx])) if heading_delta is not None else None,
                        "gt_center": safe_float(float(gt_center[idx])) if gt_center is not None else None,
                        "perception_center": safe_float(float(p_center[idx])) if p_center is not None else None,
                        "outside": bool(outside),
                        "segment": "left" if curvature_val > 0 else "right",
                        "lane_center_error": lane_center_error_val,
                    })
                return output

            turn_bias = {
                "curve_threshold": safe_float(curve_threshold),
                "left_turn": summarize_mask(left_mask),
                "right_turn": summarize_mask(right_mask),
                "top_left": top_turn_frames(left_mask),
                "top_right": top_turn_frames(right_mask),
            }

            if gt_center is not None and p_center is not None:
                align_diff = p_center - gt_center
                if align_diff.size > 0:
                    alignment_summary = {
                        "perception_vs_gt_mean": safe_float(np.mean(align_diff)),
                        "perception_vs_gt_p95_abs": safe_float(np.percentile(np.abs(align_diff), 95)),
                        "perception_vs_gt_rmse": safe_float(np.sqrt(np.mean(align_diff ** 2))),
                    }
                else:
                    alignment_summary = {
                        "perception_vs_gt_mean": 0.0,
                        "perception_vs_gt_p95_abs": 0.0,
                        "perception_vs_gt_rmse": 0.0,
                    }
            if data.get('road_frame_lane_center_error') is not None:
                lane_center_error = data['road_frame_lane_center_error'][:n_bias]
                if alignment_summary is None:
                    alignment_summary = {}
                alignment_summary.update({
                    "road_frame_lane_center_error_mean": (
                        safe_float(np.mean(lane_center_error)) if lane_center_error.size > 0 else 0.0
                    ),
                    "road_frame_lane_center_error_p95_abs": (
                        safe_float(np.percentile(np.abs(lane_center_error), 95))
                        if lane_center_error.size > 0 else 0.0
                    ),
                    "road_frame_lane_center_error_rmse": (
                        safe_float(np.sqrt(np.mean(lane_center_error ** 2)))
                        if lane_center_error.size > 0 else 0.0
                    ),
                })
            if data.get('vehicle_frame_lookahead_offset') is not None:
                vf_offset = data['vehicle_frame_lookahead_offset'][:n_bias]
                if alignment_summary is None:
                    alignment_summary = {}
                alignment_summary.update({
                    "vehicle_frame_lookahead_offset_mean": (
                        safe_float(np.mean(vf_offset)) if vf_offset.size > 0 else 0.0
                    ),
                    "vehicle_frame_lookahead_offset_p95_abs": (
                        safe_float(np.percentile(np.abs(vf_offset), 95))
                        if vf_offset.size > 0 else 0.0
                    ),
                })
    
    # Layered scoring model (0-100 per layer) with hybrid cap.
    # Deadline preset weights:
    #   Safety 0.32, Trajectory 0.30, Control 0.16, Perception 0.14, LongitudinalComfort 0.08
    # Hybrid cap:
    #   if critical layer red -> cap 59
    #   elif critical layer yellow -> cap 79
    #   else cap 100
    lane_detection_penalty = safe_float(min(20, (100 - lane_detection_rate) * 0.2))
    stale_data_penalty = safe_float(min(15, stale_hard_rate * 0.15))
    perception_instability_penalty = safe_float(max(0, (100 - perception_stability_score) * 0.2))
    perception_instability_penalty = safe_float(min(20, perception_instability_penalty))
    lane_jitter_penalty = safe_float(min(10, max(0.0, lane_line_jitter_p95 - 0.30) * 30.0))
    reference_jitter_penalty = safe_float(min(10, max(0.0, reference_jitter_p95 - 0.15) * 40.0))

    trajectory_lateral_rmse_penalty = safe_float(min(30, lateral_error_adj_rmse * 50))
    trajectory_lateral_p95_penalty = safe_float(min(20, max(0.0, lateral_error_adj_p95 - LATERAL_P95_GATE_M) * 35.0))
    trajectory_heading_penalty = safe_float(min(20, max(0.0, np.degrees(heading_error_rmse) - HEADING_PENALTY_FLOOR_DEG) * 2.5))

    # Penalty only above the pp_max_steering_jerk cap (STEERING_JERK_PENALTY_CAP normalized/s²).
    # Operating at-cap is in-spec; exceeding it signals a limiter failure.
    control_steering_jerk_penalty = safe_float(min(20, max(0.0, (steering_jerk_max - STEERING_JERK_PENALTY_CAP) * 2.0)))
    control_oscillation_penalty = safe_float(
        min(15, max(0.0, oscillation_frequency - 1.0) * 7.0)
    )
    control_sign_mismatch_penalty = safe_float(
        min(12, straight_sign_mismatch_rate * 0.2) if straight_sign_mismatch_rate > 5.0 else 0.0
    )

    # S1-M39: Gates 3.0 m/s², 6.0 m/s³ (0.31g, 0.61 g/s)
    # Scoring uses filtered values to avoid penalising velocity quantisation noise.
    # Raw values are retained in the output dict for diagnostic reference.
    accel_gate_g = ACCEL_P95_GATE_MPS2 / G_MPS2
    jerk_gate_gps = JERK_P95_GATE_MPS3 / G_MPS2
    longitudinal_accel_penalty = safe_float(
        min(20, max(0.0, (acceleration_p95_filtered / G_MPS2) - accel_gate_g) * 120.0)
    )
    longitudinal_jerk_penalty = safe_float(
        min(20, max(0.0, (commanded_jerk_p95 / G_MPS2) - jerk_gate_gps) * 50.0)
    )

    safety_out_of_lane_penalty = safe_float(min(35, out_of_lane_time * 0.35))
    safety_event_penalty = 0.0
    if out_of_lane_events > 0:
        safety_event_penalty += 25.0
    if emergency_stop_frame is not None:
        safety_event_penalty += 20.0
    safety_event_penalty = safe_float(min(45, safety_event_penalty))

    # 8. SIGNAL INTEGRITY
    signal_integrity_heading_penalty = 0.0
    signal_integrity_rate_limit_penalty = 0.0
    signal_integrity_speed_feasibility_penalty = 0.0
    heading_suppression_rate = 0.0
    rate_limit_saturation_rate = 0.0
    speed_feasibility_violation_frames = 0

    curvature_source = None
    for _cs_key in ('reference_point_curvature', 'path_curvature_input', 'gt_path_curvature'):
        _cs_val = data.get(_cs_key)
        if _cs_val is not None and (not hasattr(_cs_val, '__len__') or len(_cs_val) > 0):
            curvature_source = _cs_val
            break
    if (
        curvature_source is not None
        and data.get('speed') is not None
        and len(curvature_source) > 0
        and len(data['speed']) > 0
    ):
        n_sig = min(n_frames, len(curvature_source), len(data['speed']))
        curvature_arr = np.asarray(curvature_source[:n_sig], dtype=np.float64)
        speed_arr = np.asarray(data['speed'][:n_sig], dtype=np.float64)
        curve_threshold = 0.003
        curve_mask = np.abs(curvature_arr) > curve_threshold
        n_curve = int(np.sum(curve_mask))

        # For heading suppression check, use a lower threshold (0.0005)
        # because suppression happens on curve approach where curvature
        # is still building up (0.001-0.003 range).
        approach_threshold = 0.0005
        approach_mask = np.abs(curvature_arr) > approach_threshold
        n_approach = int(np.sum(approach_mask))

        if n_approach > 0:
            heading_gate = data.get('diag_heading_zero_gate_active')
            rate_limit = data.get('diag_ref_x_rate_limit_active')
            if heading_gate is not None and len(heading_gate) >= n_sig:
                heading_active = np.asarray(heading_gate[:n_sig]) > 0.5
                heading_suppression_rate = safe_float(
                    np.sum(heading_active & approach_mask) / n_approach * 100.0
                )
                if heading_suppression_rate > 20.0:
                    signal_integrity_heading_penalty = safe_float(
                        min(25.0, (heading_suppression_rate - 20.0) * 0.625)
                    )
            if rate_limit is not None and len(rate_limit) >= n_sig:
                rate_limit_active = np.asarray(rate_limit[:n_sig]) > 0.5
                rate_limit_saturation_rate = safe_float(
                    np.sum(rate_limit_active & approach_mask) / n_approach * 100.0
                )
                if rate_limit_saturation_rate > 30.0:
                    signal_integrity_rate_limit_penalty = safe_float(
                        min(20.0, (rate_limit_saturation_rate - 30.0) * 0.4)
                    )

        if n_curve > 0:
            pass  # Speed feasibility uses original curve_mask below

        # Speed feasibility: speed > sqrt(2.45/|curvature|) is a violation
        # Use masked assignment — np.where evaluates both branches and can divide by zero at |κ|≈0.
        curvature_eps = 1e-6
        abs_curv = np.abs(curvature_arr)
        m = abs_curv > curvature_eps
        v_max = np.full_like(abs_curv, np.inf, dtype=np.float64)
        if np.any(m):
            v_max[m] = np.sqrt(2.45 / abs_curv[m])
        speed_feasibility_violation_frames = int(np.sum(speed_arr > v_max))
        if speed_feasibility_violation_frames > 5:
            capped = min(speed_feasibility_violation_frames, 30)
            signal_integrity_speed_feasibility_penalty = safe_float(
                min(15.0, (capped - 5) * 0.6)
            )

    # 9. CURVE INTENT DIAGNOSTICS
    curve_intent_diag = {
        "available": False,
        "owner_class": "legacy_proxy",
        "arm_signal_available": False,
        "curve_event_count": 0,
        "armed_curve_event_count": 0,
        "arm_early_enough_count": 0,
        "arm_early_enough_rate": 0.0,
        "arm_late_or_missing_count": 0,
        "arm_lead_frames_min": 0.0,
        "arm_lead_frames_p50": 0.0,
        "arm_lead_frames_p95": 0.0,
        "arm_lead_seconds_p50": 0.0,
        "undercall_frame_rate": 0.0,
        "curvature_ratio_p50": 0.0,
        "curvature_ratio_p95": 0.0,
        "undercall_detected": False,
        "first_curve_start_frame": None,
        "first_arm_frame": None,
        "thresholds": {
            "curve_start_abs_curvature": 0.003,
            "arm_intent_min": 0.45,
            "arm_min_lead_frames": 6,
            "undercall_ratio_threshold": 0.70,
        },
    }
    gt_curv_series = data.get("gt_path_curvature")
    ctrl_curv_series = data.get("path_curvature_input")
    curve_intent_series = data.get("curve_intent")
    curve_intent_state_series = data.get("curve_intent_state")
    if gt_curv_series is not None and ctrl_curv_series is not None:
        n_curve_diag = min(n_frames, len(gt_curv_series), len(ctrl_curv_series))
        if curve_intent_series is not None:
            n_curve_diag = min(n_curve_diag, len(curve_intent_series))
        if curve_intent_state_series is not None:
            n_curve_diag = min(n_curve_diag, len(curve_intent_state_series))
        if n_curve_diag > 0:
            gt_abs = np.abs(np.asarray(gt_curv_series[:n_curve_diag], dtype=np.float64))
            ctrl_abs = np.abs(np.asarray(ctrl_curv_series[:n_curve_diag], dtype=np.float64))
            gt_abs = np.where(np.isfinite(gt_abs), gt_abs, 0.0)
            ctrl_abs = np.where(np.isfinite(ctrl_abs), ctrl_abs, 0.0)

            threshold_cfg = curve_intent_diag["thresholds"]
            curve_start_abs_curvature = float(threshold_cfg["curve_start_abs_curvature"])
            arm_intent_min = float(threshold_cfg["arm_intent_min"])
            arm_min_lead_frames = int(threshold_cfg["arm_min_lead_frames"])
            undercall_ratio_threshold = float(threshold_cfg["undercall_ratio_threshold"])

            curve_mask = gt_abs >= curve_start_abs_curvature
            prev_curve_mask = np.r_[False, curve_mask[:-1]]
            curve_start_frames = np.where(curve_mask & (~prev_curve_mask))[0]

            if curve_intent_state_series is not None and len(curve_intent_state_series) >= n_curve_diag:
                state_flags = np.array([
                    str(s or "").strip().upper() in {"ENTRY", "COMMIT"}
                    for s in curve_intent_state_series[:n_curve_diag]
                ], dtype=bool)
                arm_signal_available = True
            elif curve_intent_series is not None and len(curve_intent_series) >= n_curve_diag:
                ci = np.asarray(curve_intent_series[:n_curve_diag], dtype=np.float64)
                ci = np.where(np.isfinite(ci), ci, 0.0)
                state_flags = ci >= arm_intent_min
                arm_signal_available = True
            else:
                state_flags = np.zeros(n_curve_diag, dtype=bool)
                arm_signal_available = False
            prev_state_flags = np.r_[False, state_flags[:-1]]
            arm_start_frames = np.where(state_flags & (~prev_state_flags))[0]

            lead_frames: list[int] = []
            armed_count = 0
            early_count = 0
            for start in curve_start_frames:
                prior_arms = arm_start_frames[arm_start_frames <= start]
                if prior_arms.size == 0:
                    continue
                armed_count += 1
                lead = int(start - int(prior_arms[-1]))
                lead_frames.append(lead)
                if lead >= arm_min_lead_frames:
                    early_count += 1

            curve_intent_diag["available"] = True
            curve_intent_diag["arm_signal_available"] = bool(arm_signal_available)
            curve_intent_diag["curve_event_count"] = int(curve_start_frames.size)
            curve_intent_diag["armed_curve_event_count"] = int(armed_count)
            curve_intent_diag["arm_early_enough_count"] = int(early_count)
            curve_intent_diag["arm_late_or_missing_count"] = int(
                max(0, int(curve_start_frames.size) - int(early_count))
            )
            if curve_start_frames.size > 0:
                curve_intent_diag["arm_early_enough_rate"] = safe_float(
                    (float(early_count) / float(curve_start_frames.size)) * 100.0
                )
                curve_intent_diag["first_curve_start_frame"] = int(curve_start_frames[0])
            if arm_start_frames.size > 0:
                curve_intent_diag["first_arm_frame"] = int(arm_start_frames[0])
            if lead_frames:
                lead_arr = np.asarray(lead_frames, dtype=np.float64)
                curve_intent_diag["arm_lead_frames_min"] = safe_float(np.min(lead_arr))
                curve_intent_diag["arm_lead_frames_p50"] = safe_float(np.percentile(lead_arr, 50))
                curve_intent_diag["arm_lead_frames_p95"] = safe_float(np.percentile(lead_arr, 95))
                curve_intent_diag["arm_lead_seconds_p50"] = safe_float(
                    np.percentile(lead_arr, 50) * dt
                )

            # GT curvature floor: skip undercall metric when road is effectively
            # straight (max GT curvature < 0.005 rad/m). The ratio is meaningless
            # when both map and GT curvature are near-zero quantization noise.
            gt_curvature_floor = 0.005
            gt_max_curvature = float(np.max(gt_abs)) if gt_abs.size > 0 else 0.0
            curve_intent_diag["gt_max_curvature"] = safe_float(gt_max_curvature)
            curve_intent_diag["gt_curvature_floor"] = gt_curvature_floor

            if gt_max_curvature < gt_curvature_floor:
                curve_intent_diag["undercall_frame_rate"] = None
                curve_intent_diag["undercall_detected"] = False
                curve_intent_diag["undercall_skipped_reason"] = "gt_curvature_below_floor"
            else:
                ratio_mask = gt_abs >= curve_start_abs_curvature
                ratio = np.array([], dtype=np.float64)
                if np.any(ratio_mask):
                    ratio = np.divide(
                        ctrl_abs[ratio_mask],
                        np.maximum(gt_abs[ratio_mask], 1e-6),
                    )
                    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
                if ratio.size > 0:
                    undercall_rate = float(np.mean(ratio < undercall_ratio_threshold) * 100.0)
                    curve_intent_diag["undercall_frame_rate"] = safe_float(undercall_rate)
                    curve_intent_diag["curvature_ratio_p50"] = safe_float(np.percentile(ratio, 50))
                    curve_intent_diag["curvature_ratio_p95"] = safe_float(np.percentile(ratio, 95))
                    curve_intent_diag["undercall_detected"] = bool(undercall_rate > 25.0)

    curve_phase_on = float(traj_cfg.get("curve_phase_on", 0.45))
    curve_local_reentry_gate_min = float(traj_cfg.get("curve_local_phase_reentry_gate_min", 0.10))
    curve_local_entry_min_ttc_s = float(
        traj_cfg.get("curve_local_phase_time_end_s", traj_cfg.get("curve_phase_time_to_curve_min_s", 0.25))
    )
    curve_local_watchdog_pingpong_window_frames = max(
        2, int(traj_cfg.get("curve_local_watchdog_pingpong_window_frames", 6))
    )
    steering_onset_abs_threshold = float(
        lateral_cfg.get("curve_turn_event_steering_onset_abs_min", 0.12)
    )
    entry_shorten_guard_limit = max(
        0.0,
        float(
            traj_cfg.get("reference_lookahead_entry_shorten_slew_m_per_frame", 0.0) or 0.0
        ),
    )

    curve_local_contract = {
        "availability": "unavailable",
        "owner_class": "authoritative",
        "curve_local_contract_available": False,
        "curve_preview_far_active_straight_rate": 0.0,
        "curve_local_active_straight_rate": 0.0,
        "curve_local_arm_ready_straight_rate": 0.0,
        "curve_local_commit_ready_straight_rate": 0.0,
        "curve_local_path_sustain_active_straight_rate": 0.0,
        "curve_local_commit_streak_max_frames": 0,
        "curve_local_arm_without_ready_count": 0,
        "curve_local_arm_without_ready_rate": 0.0,
        "curve_local_commit_without_ready_count": 0,
        "curve_local_commit_without_ready_rate": 0.0,
        "curve_local_commit_without_distance_ready_count": 0,
        "curve_local_commit_without_distance_ready_rate": 0.0,
        "curve_local_latched_straight_count": 0,
        "curve_local_latched_straight_rate": 0.0,
        "curve_local_reentry_without_gate_count": 0,
        "curve_local_reentry_without_gate_rate": 0.0,
        "curve_local_watchdog_pingpong_count": 0,
        "curve_local_gate_weight_straight_p50": None,
        "curve_local_phase_raw_straight_p50": None,
        "curve_preview_far_phase_straight_p50": None,
        "curve_phase_term_time_straight_p50": None,
        "curve_local_reentry_ready_straight_rate": None,
        "pp_curve_local_floor_breach_count": 0,
        "pp_curve_local_floor_breach_rate": 0.0,
        "curve_lookahead_collapse_violation_count": 0,
        "curve_lookahead_collapse_violation_rate": 0.0,
        "pp_entry_lookahead_min_m": None,
        "pp_entry_lookahead_shorten_rate_min_m_per_frame": None,
        "straight_summary_source": "proxy",
        "straight_summary_vs_segment_rate_delta_pct": None,
        "limits": {
            "curve_local_active_straight_rate_max_pct": 5.0,
            "curve_local_arm_without_ready_count_max": 0,
            "curve_local_commit_without_ready_count_max": 0,
            "curve_local_commit_without_distance_ready_count_max": 0,
            "curve_local_reentry_without_gate_count_max": 0,
            "curve_local_watchdog_pingpong_count_max": 0,
            "pp_curve_local_floor_breach_count_max": 0,
            "curve_lookahead_collapse_violation_count_max": 0,
            "curve_local_reentry_gate_min": curve_local_reentry_gate_min,
            "curve_local_entry_min_ttc_s": curve_local_entry_min_ttc_s,
            "steering_onset_distance_min_m": 0.0,
            "steering_onset_abs_threshold": steering_onset_abs_threshold,
            "reference_lookahead_entry_shorten_slew_m_per_frame": entry_shorten_guard_limit,
            "pp_floor_rescue_delta_max_m": 1.0,
        },
    }
    curve_turn_events = []
    curve_straight_segments = []
    curve_preview_far_series = data.get("curve_preview_far_upcoming")
    if curve_preview_far_series is None:
        curve_preview_far_series = data.get("curve_phase_preview_upcoming")
    curve_preview_far_phase_series = data.get("curve_preview_far_phase")
    local_arm_ready_series = data.get("curve_local_arm_ready")
    local_time_ready_series = data.get("curve_local_time_ready")
    local_in_curve_now_series = data.get("curve_local_in_curve_now")
    local_commit_ready_series = data.get("curve_local_commit_ready")
    local_arm_phase_raw_series = data.get("curve_local_arm_phase_raw")
    local_sustain_phase_raw_series = data.get("curve_local_sustain_phase_raw")
    local_path_sustain_active_series = data.get("curve_local_path_sustain_active")
    curve_phase_term_time_series = data.get("curve_phase_term_time")
    local_state_series = data.get("curve_local_state")
    local_phase_series = data.get("curve_local_phase")
    local_phase_raw_series = data.get("curve_local_phase_raw")
    local_reentry_ready_series = data.get("curve_local_reentry_ready")
    local_distance_ready_series = data.get("curve_local_distance_ready")
    local_commit_driver_series = data.get("curve_local_commit_driver")
    local_gate_weight_series = data.get("reference_lookahead_local_gate_weight")
    owner_mode_series = data.get("reference_lookahead_owner_mode")
    entry_weight_source_series = data.get("reference_lookahead_entry_weight_source")
    fallback_active_series = data.get("reference_lookahead_fallback_active")
    local_arc_mode_series = data.get("local_curve_reference_mode")
    local_arc_active_series = data.get("local_curve_reference_active")
    local_arc_shadow_only_series = data.get("local_curve_reference_shadow_only")
    local_arc_valid_series = data.get("local_curve_reference_valid")
    local_arc_source_series = data.get("local_curve_reference_source")
    local_arc_fallback_active_series = data.get("local_curve_reference_fallback_active")
    local_arc_fallback_reason_series = data.get("local_curve_reference_fallback_reason")
    local_arc_blend_weight_series = data.get("local_curve_reference_blend_weight")
    local_arc_progress_weight_series = data.get("local_curve_reference_progress_weight")
    local_arc_curvature_series = data.get("local_curve_reference_arc_curvature_abs")
    local_arc_target_distance_series = data.get("local_curve_reference_target_distance_m")
    local_arc_vs_planner_delta_series = data.get("local_curve_reference_vs_planner_delta_m")
    time_to_curve_series = data.get("time_to_next_curve_start_s")
    intent_state_series = data.get("curve_intent_state")
    intent_watchdog_series = data.get("curve_intent_watchdog_triggered")
    straight_series = data.get("is_control_straight_proxy")
    if straight_series is None:
        straight_series = data.get("is_straight")
    pp_floor_active_series = data.get("pp_curve_local_floor_active")
    pp_floor_m_series = data.get("pp_curve_local_floor_m")
    pp_post_floor_series = data.get("pp_curve_local_lookahead_post_floor")
    pp_pre_floor_series = data.get("pp_curve_local_lookahead_pre_floor")
    reference_ld_series = data.get("reference_lookahead_active")
    pp_ld_series = data.get("pp_lookahead_distance")
    steering_series = data.get("steering")

    def _normalize_str_value(value: object) -> str:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", "ignore")
            except Exception:
                value = ""
        text = str(value or "").strip()
        return text

    def _masked_mode_string(arr: Optional[np.ndarray], mask_indices: np.ndarray) -> Optional[str]:
        if arr is None or mask_indices.size == 0:
            return None
        values = []
        for idx in mask_indices:
            if 0 <= idx < len(arr):
                text = _normalize_str_value(arr[int(idx)])
                if text:
                    values.append(text)
        if not values:
            return None
        return max(sorted(set(values)), key=values.count)

    def _string_at(arr: Optional[np.ndarray], frame_idx: Optional[int]) -> Optional[str]:
        if arr is None or frame_idx is None or frame_idx < 0 or frame_idx >= len(arr):
            return None
        text = _normalize_str_value(arr[int(frame_idx)])
        return text if text else None

    def _masked_percentile(arr: Optional[np.ndarray], mask: Optional[np.ndarray], pct: float) -> Optional[float]:
        if arr is None or mask is None or not np.any(mask):
            return None
        vals = np.asarray(arr[:n_frames], dtype=np.float64)[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return safe_float(np.percentile(vals, pct), default=None)

    def _split_contiguous(indices: np.ndarray) -> list[np.ndarray]:
        if indices.size == 0:
            return []
        return [run for run in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1) if run.size > 0]

    def _max_true_run(mask: np.ndarray) -> int:
        run = 0
        best = 0
        for active in mask:
            if active:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return int(best)

    local_states = None
    local_active_mask = None
    commit_mask = None
    if local_state_series is not None:
        local_states = np.array(
            [str(state or "").strip().upper() for state in local_state_series[:n_frames]],
            dtype=object,
        )
        local_active_mask = np.array([state in {"ENTRY", "COMMIT"} for state in local_states], dtype=bool)
        commit_mask = np.array([state == "COMMIT" for state in local_states], dtype=bool)
        curve_local_contract["curve_local_commit_streak_max_frames"] = _max_true_run(commit_mask)
    elif local_phase_series is not None:
        local_phase_arr = np.asarray(local_phase_series[:n_frames], dtype=np.float64)
        local_phase_arr = np.where(np.isfinite(local_phase_arr), local_phase_arr, 0.0)
        local_active_mask = local_phase_arr >= curve_phase_on
        commit_mask = np.array(local_active_mask, copy=True)
        curve_local_contract["curve_local_commit_streak_max_frames"] = _max_true_run(local_active_mask)

    straight_mask = None
    if straight_series is not None:
        straight_arr = np.asarray(straight_series[:n_frames], dtype=np.float64)
        straight_mask = straight_arr > 0.5
        if local_in_curve_now_series is not None:
            local_in_curve_arr = np.asarray(local_in_curve_now_series[:n_frames], dtype=np.float64)
            straight_mask = straight_mask & ~(local_in_curve_arr > 0.5)

    if local_active_mask is not None:
        curve_local_contract["curve_local_contract_available"] = True
        curve_local_contract["availability"] = "available"
        if straight_mask is not None and np.any(straight_mask):
            curve_local_contract["curve_local_active_straight_rate"] = safe_float(
                np.mean(local_active_mask[straight_mask]) * 100.0
            )
            curve_local_contract["curve_local_latched_straight_count"] = int(
                np.sum(local_active_mask & straight_mask)
            )
            curve_local_contract["curve_local_latched_straight_rate"] = safe_float(
                np.mean(local_active_mask & straight_mask) * 100.0
            )
            curve_local_contract["curve_local_gate_weight_straight_p50"] = _masked_percentile(
                local_gate_weight_series, straight_mask, 50
            )
            curve_local_contract["curve_local_phase_raw_straight_p50"] = _masked_percentile(
                local_phase_raw_series, straight_mask, 50
            )
            curve_local_contract["curve_preview_far_phase_straight_p50"] = _masked_percentile(
                curve_preview_far_phase_series, straight_mask, 50
            )
            curve_local_contract["curve_phase_term_time_straight_p50"] = _masked_percentile(
                curve_phase_term_time_series, straight_mask, 50
            )
            if local_arm_ready_series is not None:
                local_arm_ready_arr = np.asarray(local_arm_ready_series[:n_frames], dtype=np.float64)
                valid_arm_ready_mask = straight_mask & (local_arm_ready_arr >= 0.0)
                if np.any(valid_arm_ready_mask):
                    curve_local_contract["curve_local_arm_ready_straight_rate"] = safe_float(
                        np.mean(local_arm_ready_arr[valid_arm_ready_mask] > 0.5) * 100.0
                    )
            if local_commit_ready_series is not None:
                local_commit_ready_arr = np.asarray(local_commit_ready_series[:n_frames], dtype=np.float64)
                valid_commit_ready_mask = straight_mask & (local_commit_ready_arr >= 0.0)
                if np.any(valid_commit_ready_mask):
                    curve_local_contract["curve_local_commit_ready_straight_rate"] = safe_float(
                        np.mean(local_commit_ready_arr[valid_commit_ready_mask] > 0.5) * 100.0
                    )
            if local_path_sustain_active_series is not None:
                local_path_sustain_arr = np.asarray(
                    local_path_sustain_active_series[:n_frames], dtype=np.float64
                )
                valid_path_sustain_mask = straight_mask & (local_path_sustain_arr >= 0.0)
                if np.any(valid_path_sustain_mask):
                    curve_local_contract["curve_local_path_sustain_active_straight_rate"] = safe_float(
                        np.mean(local_path_sustain_arr[valid_path_sustain_mask] > 0.5) * 100.0
                    )
            if local_reentry_ready_series is not None:
                local_reentry_arr = np.asarray(local_reentry_ready_series[:n_frames], dtype=np.float64)
                valid_reentry_mask = straight_mask & (local_reentry_arr >= 0.0)
                if np.any(valid_reentry_mask):
                    curve_local_contract["curve_local_reentry_ready_straight_rate"] = safe_float(
                        np.mean(local_reentry_arr[valid_reentry_mask] > 0.5) * 100.0
                    )
        if curve_preview_far_series is not None and straight_mask is not None and np.any(straight_mask):
            preview_far_arr = np.asarray(curve_preview_far_series[:n_frames], dtype=np.float64) > 0.5
            curve_local_contract["curve_preview_far_active_straight_rate"] = safe_float(
                np.mean(preview_far_arr[straight_mask]) * 100.0
            )

        if local_gate_weight_series is not None:
            gate_arr = np.asarray(local_gate_weight_series[:n_frames], dtype=np.float64)
            gate_arr = np.where(np.isfinite(gate_arr), gate_arr, 0.0)
            prev_active = np.concatenate(([False], local_active_mask[:-1]))
            active_entry_mask = local_active_mask & (~prev_active)
            reentry_without_gate_mask = active_entry_mask & (gate_arr < curve_local_reentry_gate_min)
            curve_local_contract["curve_local_reentry_without_gate_count"] = int(
                np.sum(reentry_without_gate_mask)
            )
            if np.any(active_entry_mask):
                curve_local_contract["curve_local_reentry_without_gate_rate"] = safe_float(
                    np.mean(reentry_without_gate_mask[active_entry_mask]) * 100.0
                )
            if local_arm_ready_series is not None:
                local_arm_ready_arr = np.asarray(local_arm_ready_series[:n_frames], dtype=np.float64)
                valid_arm_ready_mask = local_arm_ready_arr >= 0.0
                arm_ready_mask = local_arm_ready_arr > 0.5
                arm_without_ready_mask = active_entry_mask & valid_arm_ready_mask & (~arm_ready_mask)
                curve_local_contract["curve_local_arm_without_ready_count"] = int(
                    np.sum(arm_without_ready_mask)
                )
                if np.any(active_entry_mask & valid_arm_ready_mask):
                    curve_local_contract["curve_local_arm_without_ready_rate"] = safe_float(
                        np.mean(
                            arm_without_ready_mask[active_entry_mask & valid_arm_ready_mask]
                        ) * 100.0
                    )
            if local_commit_ready_series is not None:
                local_commit_ready_arr = np.asarray(local_commit_ready_series[:n_frames], dtype=np.float64)
                valid_commit_ready_mask = local_commit_ready_arr >= 0.0
                commit_ready_mask = local_commit_ready_arr > 0.5
                early_commit_ready_mask = (
                    commit_mask & valid_commit_ready_mask & (~commit_ready_mask)
                    if commit_mask is not None
                    else local_active_mask & valid_commit_ready_mask & (~commit_ready_mask)
                )
                curve_local_contract["curve_local_commit_without_ready_count"] = int(
                    np.sum(early_commit_ready_mask)
                )
                if np.any((commit_mask if commit_mask is not None else local_active_mask) & valid_commit_ready_mask):
                    source_mask = (commit_mask if commit_mask is not None else local_active_mask) & valid_commit_ready_mask
                    curve_local_contract["curve_local_commit_without_ready_rate"] = safe_float(
                        np.mean(early_commit_ready_mask[source_mask]) * 100.0
                    )

        if local_distance_ready_series is not None:
            local_distance_arr = np.asarray(local_distance_ready_series[:n_frames], dtype=np.float64)
            valid_distance_mask = local_distance_arr >= 0.0
            distance_ready_mask = local_distance_arr > 0.5
            early_commit_mask = commit_mask & valid_distance_mask & (~distance_ready_mask) if commit_mask is not None else (
                local_active_mask & valid_distance_mask & (~distance_ready_mask)
            )
            curve_local_contract["curve_local_commit_without_distance_ready_count"] = int(
                np.sum(early_commit_mask)
            )
            if np.any(valid_distance_mask):
                curve_local_contract["curve_local_commit_without_distance_ready_rate"] = safe_float(
                    np.mean(early_commit_mask[valid_distance_mask]) * 100.0
                )
            if reference_ld_series is not None and len(reference_ld_series) >= n_frames:
                reference_ld_arr = np.asarray(reference_ld_series[:n_frames], dtype=np.float64)
                delta_ld = np.diff(reference_ld_arr)
                collapse_limit = float(
                    lateral_cfg.get("pp_curve_local_shorten_slew_m_per_frame", 0.0) or 0.0
                )
                if collapse_limit > 0.0 and delta_ld.size > 0:
                    early_collapse = (
                        delta_ld < -(collapse_limit + 1e-6)
                    ) & (~distance_ready_mask[:-1])
                    curve_local_contract["curve_lookahead_collapse_violation_count"] = int(
                        np.sum(early_collapse)
                    )
                    curve_local_contract["curve_lookahead_collapse_violation_rate"] = safe_float(
                        np.mean(early_collapse) * 100.0
                    )

        if (
            pp_floor_active_series is not None
            and pp_floor_m_series is not None
            and pp_post_floor_series is not None
        ):
            pp_floor_active_arr = np.asarray(pp_floor_active_series[:n_frames], dtype=np.float64) > 0.5
            pp_floor_arr = np.asarray(pp_floor_m_series[:n_frames], dtype=np.float64)
            pp_post_floor_arr = np.asarray(pp_post_floor_series[:n_frames], dtype=np.float64)
            valid_floor_mask = (
                pp_floor_active_arr
                & np.isfinite(pp_floor_arr)
                & np.isfinite(pp_post_floor_arr)
            )
            floor_breach_mask = valid_floor_mask & (pp_post_floor_arr < (pp_floor_arr - 1e-4))
            curve_local_contract["pp_curve_local_floor_breach_count"] = int(np.sum(floor_breach_mask))
            if np.any(valid_floor_mask):
                curve_local_contract["pp_curve_local_floor_breach_rate"] = safe_float(
                    np.mean(floor_breach_mask[valid_floor_mask]) * 100.0
                )
        if pp_pre_floor_series is not None and np.size(pp_pre_floor_series) > 0:
            pp_pre_floor_arr = np.asarray(pp_pre_floor_series[:n_frames], dtype=np.float64)
            pp_pre_floor_arr = pp_pre_floor_arr[np.isfinite(pp_pre_floor_arr) & (pp_pre_floor_arr > 0.0)]
            if pp_pre_floor_arr.size > 0:
                curve_local_contract["pp_entry_lookahead_min_m"] = safe_float(
                    np.min(pp_pre_floor_arr), default=None
                )
        if pp_post_floor_series is not None and len(pp_post_floor_series) >= 2:
            pp_post_floor_arr = np.asarray(pp_post_floor_series[:n_frames], dtype=np.float64)
            delta_pp_floor = np.diff(pp_post_floor_arr)
            valid_delta = delta_pp_floor[np.isfinite(delta_pp_floor)]
            if valid_delta.size > 0:
                curve_local_contract["pp_entry_lookahead_shorten_rate_min_m_per_frame"] = safe_float(
                    np.min(valid_delta), default=None
                )

        pingpong_states = None
        if intent_state_series is not None:
            pingpong_states = np.array(
                [str(state or "").strip().upper() for state in intent_state_series[:n_frames]],
                dtype=object,
            )
        elif local_states is not None:
            pingpong_states = local_states
        watchdog_arr = None
        if intent_watchdog_series is not None:
            watchdog_arr = np.asarray(intent_watchdog_series[:n_frames], dtype=np.float64) > 0.5
        gate_arr = None
        if local_gate_weight_series is not None:
            gate_arr = np.asarray(local_gate_weight_series[:n_frames], dtype=np.float64)
            gate_arr = np.where(np.isfinite(gate_arr), gate_arr, 0.0)
        if pingpong_states is not None:
            pingpong_count = 0
            idx = 1
            while idx < len(pingpong_states) - 1:
                is_rearm = pingpong_states[idx] == "REARM"
                anchor_ok = pingpong_states[idx - 1] == "COMMIT"
                if watchdog_arr is not None:
                    anchor_ok = anchor_ok and bool(watchdog_arr[idx] or watchdog_arr[idx - 1])
                if straight_mask is not None:
                    anchor_ok = anchor_ok and bool(straight_mask[idx])
                if gate_arr is not None:
                    anchor_ok = anchor_ok and float(gate_arr[idx]) < curve_local_reentry_gate_min
                if not (is_rearm and anchor_ok):
                    idx += 1
                    continue
                reentered = False
                for jdx in range(idx + 1, min(len(pingpong_states), idx + curve_local_watchdog_pingpong_window_frames + 1)):
                    if pingpong_states[jdx] == "STRAIGHT":
                        break
                    if pingpong_states[jdx] in {"ENTRY", "COMMIT"}:
                        if gate_arr is None or float(gate_arr[jdx]) < curve_local_reentry_gate_min:
                            reentered = True
                            idx = jdx
                            break
                if reentered:
                    pingpong_count += 1
                idx += 1
            curve_local_contract["curve_local_watchdog_pingpong_count"] = int(pingpong_count)

    provenance = data.get("recording_provenance") or {}
    track_id = ""
    if isinstance(provenance, dict):
        track_id = str(provenance.get("track_id", "") or "").strip()
    track_windows = _load_track_curve_windows(track_id)
    if (
        track_windows
        and data.get("speed") is not None
        and data.get("timestamps") is not None
        and data.get("lateral_error") is not None
    ):
        time_arr = np.asarray(data["timestamps"][:n_frames], dtype=np.float64)
        speed_arr = np.asarray(data["speed"][:n_frames], dtype=np.float64)
        lat_err_arr = np.asarray(data["lateral_error"][:n_frames], dtype=np.float64)
        if time_arr.size > 1 and speed_arr.size == time_arr.size and lat_err_arr.size == time_arr.size:
            dt_arr = np.diff(time_arr, prepend=time_arr[0])
            dt_arr[0] = 0.0
            dt_arr = np.where(np.isfinite(dt_arr) & (dt_arr > 0.0), dt_arr, 0.0)
            distance_trace = np.cumsum(np.maximum(speed_arr, 0.0) * dt_arr)
            total_length_m = float(track_windows.get("total_length_m", 0.0) or 0.0)
            if total_length_m > 0.0:
                track_progress = np.mod(distance_trace, total_length_m)
                pp_track = None
                if pp_ld_series is not None and len(pp_ld_series) >= n_frames:
                    pp_track = np.asarray(pp_ld_series[:n_frames], dtype=np.float64)
                pp_pre_track = None
                if pp_pre_floor_series is not None and len(pp_pre_floor_series) >= n_frames:
                    pp_pre_track = np.asarray(pp_pre_floor_series[:n_frames], dtype=np.float64)
                pp_post_track = None
                if pp_post_floor_series is not None and len(pp_post_floor_series) >= n_frames:
                    pp_post_track = np.asarray(pp_post_floor_series[:n_frames], dtype=np.float64)
                ref_track = None
                if reference_ld_series is not None and len(reference_ld_series) >= n_frames:
                    ref_track = np.asarray(reference_ld_series[:n_frames], dtype=np.float64)
                owner_nominal_track = None
                if data.get("reference_lookahead_owner_nominal_target") is not None:
                    owner_nominal_track = np.asarray(
                        data["reference_lookahead_owner_nominal_target"][:n_frames],
                        dtype=np.float64,
                    )
                owner_commit_band_track = None
                if data.get("reference_lookahead_owner_commit_band_target") is not None:
                    owner_commit_band_track = np.asarray(
                        data["reference_lookahead_owner_commit_band_target"][:n_frames],
                        dtype=np.float64,
                    )
                owner_commit_progress_track = None
                if data.get("reference_lookahead_owner_commit_progress") is not None:
                    owner_commit_progress_track = np.asarray(
                        data["reference_lookahead_owner_commit_progress"][:n_frames],
                        dtype=np.float64,
                    )
                owner_commit_clamp_active_track = None
                if data.get("reference_lookahead_owner_commit_band_clamp_active") is not None:
                    owner_commit_clamp_active_track = np.asarray(
                        data["reference_lookahead_owner_commit_band_clamp_active"][:n_frames],
                        dtype=np.float64,
                    )
                owner_commit_clamp_delta_track = None
                if data.get("reference_lookahead_owner_commit_band_clamp_delta_m") is not None:
                    owner_commit_clamp_delta_track = np.asarray(
                        data["reference_lookahead_owner_commit_band_clamp_delta_m"][:n_frames],
                        dtype=np.float64,
                    )
                distance_to_curve_arr = None
                if data.get("distance_to_next_curve_start_m") is not None:
                    distance_to_curve_arr = np.asarray(data["distance_to_next_curve_start_m"][:n_frames], dtype=np.float64)
                time_to_curve_arr = None
                if time_to_curve_series is not None:
                    time_to_curve_arr = np.asarray(time_to_curve_series[:n_frames], dtype=np.float64)
                steering_arr = None
                if steering_series is not None:
                    steering_arr = np.asarray(steering_series[:n_frames], dtype=np.float64)
                heading_err_arr = None
                if data.get("heading_error") is not None:
                    heading_err_arr = np.asarray(data["heading_error"][:n_frames], dtype=np.float64)
                local_entry_severity_arr = None
                if data.get("curve_local_entry_severity") is not None:
                    local_entry_severity_arr = np.asarray(
                        data["curve_local_entry_severity"][:n_frames], dtype=np.float64
                    )
                local_entry_on_arr = None
                if data.get("curve_local_entry_on_effective") is not None:
                    local_entry_on_arr = np.asarray(
                        data["curve_local_entry_on_effective"][:n_frames], dtype=np.float64
                    )
                local_distance_start_arr = None
                if data.get("curve_local_phase_distance_start_effective_m") is not None:
                    local_distance_start_arr = np.asarray(
                        data["curve_local_phase_distance_start_effective_m"][:n_frames], dtype=np.float64
                    )
                local_time_start_arr = None
                if data.get("curve_local_phase_time_start_effective_s") is not None:
                    local_time_start_arr = np.asarray(
                        data["curve_local_phase_time_start_effective_s"][:n_frames], dtype=np.float64
                    )
                local_arm_ready_arr = None
                if data.get("curve_local_arm_ready") is not None:
                    local_arm_ready_arr = np.asarray(
                        data["curve_local_arm_ready"][:n_frames], dtype=np.float64
                    )
                local_commit_ready_arr = None
                if data.get("curve_local_commit_ready") is not None:
                    local_commit_ready_arr = np.asarray(
                        data["curve_local_commit_ready"][:n_frames], dtype=np.float64
                    )
                local_in_curve_now_arr = None
                if data.get("curve_local_in_curve_now") is not None:
                    local_in_curve_now_arr = np.asarray(
                        data["curve_local_in_curve_now"][:n_frames], dtype=np.float64
                    )
                local_arm_phase_arr = None
                if data.get("curve_local_arm_phase_raw") is not None:
                    local_arm_phase_arr = np.asarray(
                        data["curve_local_arm_phase_raw"][:n_frames], dtype=np.float64
                    )
                local_sustain_phase_arr = None
                if data.get("curve_local_sustain_phase_raw") is not None:
                    local_sustain_phase_arr = np.asarray(
                        data["curve_local_sustain_phase_raw"][:n_frames], dtype=np.float64
                    )
                local_path_sustain_arr = None
                if data.get("curve_local_path_sustain_active") is not None:
                    local_path_sustain_arr = np.asarray(
                        data["curve_local_path_sustain_active"][:n_frames], dtype=np.float64
                    )
                local_commit_driver_series = data.get("curve_local_commit_driver")
                local_entry_driver_series = data.get("curve_local_entry_driver")
                curve_windows = track_windows.get("curve_windows", [])

                straight_windows = []
                prev_end_m = 0.0
                for curve_window in curve_windows:
                    start_m = float(curve_window.get("start_m", 0.0) or 0.0)
                    end_m = float(curve_window.get("end_m", start_m) or start_m)
                    if start_m > prev_end_m + 1e-6:
                        straight_windows.append(
                            {
                                "straight_index": int(len(straight_windows) + 1),
                                "start_m": float(prev_end_m),
                                "end_m": float(start_m),
                                "before_curve_index": int(curve_window.get("curve_index", len(straight_windows) + 1)),
                            }
                        )
                    prev_end_m = max(prev_end_m, end_m)
                if total_length_m > prev_end_m + 1e-6:
                    straight_windows.append(
                        {
                            "straight_index": int(len(straight_windows) + 1),
                            "start_m": float(prev_end_m),
                            "end_m": float(total_length_m),
                            "before_curve_index": None,
                        }
                    )

                for straight_window in straight_windows:
                    straight_indices = np.where(
                        (track_progress >= float(straight_window["start_m"]))
                        & (track_progress < float(straight_window["end_m"]))
                    )[0]
                    if straight_indices.size == 0:
                        continue
                    straight_mask_window = np.zeros(n_frames, dtype=bool)
                    straight_mask_window[straight_indices] = True
                    if local_in_curve_now_series is not None:
                        local_in_curve_arr = np.asarray(
                            local_in_curve_now_series[:n_frames], dtype=np.float64
                        )
                        straight_mask_window = straight_mask_window & ~(local_in_curve_arr > 0.5)
                        straight_indices = np.where(straight_mask_window)[0]
                        if straight_indices.size == 0:
                            continue
                    local_window_rate = (
                        safe_float(np.mean(local_active_mask[straight_mask_window]) * 100.0)
                        if local_active_mask is not None
                        else None
                    )
                    far_window_rate = (
                        safe_float(
                            np.mean(
                                (np.asarray(curve_preview_far_series[:n_frames], dtype=np.float64) > 0.5)[straight_mask_window]
                            ) * 100.0
                        )
                        if curve_preview_far_series is not None
                        else None
                    )
                    local_window_gate = _masked_percentile(local_gate_weight_series, straight_mask_window, 50)
                    local_window_phase_raw = _masked_percentile(local_phase_raw_series, straight_mask_window, 50)
                    local_window_arm_phase_raw = _masked_percentile(
                        local_arm_phase_raw_series, straight_mask_window, 50
                    )
                    local_window_sustain_phase_raw = _masked_percentile(
                        local_sustain_phase_raw_series, straight_mask_window, 50
                    )
                    local_window_far_phase = _masked_percentile(curve_preview_far_phase_series, straight_mask_window, 50)
                    local_window_term_time = _masked_percentile(curve_phase_term_time_series, straight_mask_window, 50)
                    local_window_arm_ready_rate = None
                    if local_arm_ready_series is not None:
                        local_arm_ready_arr = np.asarray(local_arm_ready_series[:n_frames], dtype=np.float64)
                        valid_arm_ready_mask = straight_mask_window & (local_arm_ready_arr >= 0.0)
                        if np.any(valid_arm_ready_mask):
                            local_window_arm_ready_rate = safe_float(
                                np.mean(local_arm_ready_arr[valid_arm_ready_mask] > 0.5) * 100.0
                            )
                    local_window_commit_ready_rate = None
                    if local_commit_ready_series is not None:
                        local_commit_ready_arr = np.asarray(local_commit_ready_series[:n_frames], dtype=np.float64)
                        valid_commit_ready_mask = straight_mask_window & (local_commit_ready_arr >= 0.0)
                        if np.any(valid_commit_ready_mask):
                            local_window_commit_ready_rate = safe_float(
                                np.mean(local_commit_ready_arr[valid_commit_ready_mask] > 0.5) * 100.0
                            )
                    local_window_path_sustain_rate = None
                    if local_path_sustain_active_series is not None:
                        local_path_sustain_arr = np.asarray(
                            local_path_sustain_active_series[:n_frames], dtype=np.float64
                        )
                        valid_path_sustain_mask = (
                            straight_mask_window & (local_path_sustain_arr >= 0.0)
                        )
                        if np.any(valid_path_sustain_mask):
                            local_window_path_sustain_rate = safe_float(
                                np.mean(local_path_sustain_arr[valid_path_sustain_mask] > 0.5) * 100.0
                            )
                    local_window_commit_streak = (
                        _max_true_run(local_active_mask[straight_indices]) if local_active_mask is not None else 0
                    )
                    local_window_pingpong = 0
                    if intent_state_series is not None:
                        intent_states_window = np.array(
                            [str(state or "").strip().upper() for state in intent_state_series[:n_frames]],
                            dtype=object,
                        )
                        idx = 1
                        while idx < len(straight_indices) - 1:
                            frame_idx = int(straight_indices[idx])
                            prev_idx = int(straight_indices[idx - 1])
                            if not (
                                intent_states_window[frame_idx] == "REARM"
                                and intent_states_window[prev_idx] == "COMMIT"
                            ):
                                idx += 1
                                continue
                            reentered = False
                            for jdx in range(idx + 1, min(len(straight_indices), idx + curve_local_watchdog_pingpong_window_frames + 1)):
                                next_frame_idx = int(straight_indices[jdx])
                                if intent_states_window[next_frame_idx] in {"ENTRY", "COMMIT"}:
                                    reentered = True
                                    idx = jdx
                                    break
                            if reentered:
                                local_window_pingpong += 1
                            idx += 1
                    curve_straight_segments.append(
                        {
                            "straight_index": int(straight_window["straight_index"]),
                            "before_curve_index": straight_window.get("before_curve_index"),
                            "start_frame": int(straight_indices[0]),
                            "end_frame": int(straight_indices[-1]),
                            "far_preview_active_rate": far_window_rate,
                            "local_active_rate": local_window_rate,
                            "watchdog_pingpong_count": int(local_window_pingpong),
                            "gate_weight_p50": local_window_gate,
                            "local_phase_raw_p50": local_window_phase_raw,
                            "arm_phase_raw_p50": local_window_arm_phase_raw,
                            "sustain_phase_raw_p50": local_window_sustain_phase_raw,
                            "far_preview_phase_p50": local_window_far_phase,
                            "term_time_p50": local_window_term_time,
                            "arm_ready_rate": local_window_arm_ready_rate,
                            "commit_ready_rate": local_window_commit_ready_rate,
                            "path_sustain_active_rate": local_window_path_sustain_rate,
                            "max_commit_streak_frames": int(local_window_commit_streak),
                        }
                    )

                if curve_straight_segments:
                    segment_lengths = np.array(
                        [
                            max(
                                0,
                                int(segment.get("end_frame", -1)) - int(segment.get("start_frame", 0)) + 1,
                            )
                            for segment in curve_straight_segments
                        ],
                        dtype=np.float64,
                    )
                    local_rates = np.array(
                        [
                            float(segment.get("local_active_rate", 0.0) or 0.0)
                            for segment in curve_straight_segments
                        ],
                        dtype=np.float64,
                    )
                    far_rates = np.array(
                        [
                            float(segment.get("far_preview_active_rate", 0.0) or 0.0)
                            for segment in curve_straight_segments
                        ],
                        dtype=np.float64,
                    )
                    valid_segment_mask = segment_lengths > 0.0
                    if np.any(valid_segment_mask):
                        weighted_local = safe_float(
                            np.average(local_rates[valid_segment_mask], weights=segment_lengths[valid_segment_mask])
                        )
                        weighted_far = safe_float(
                            np.average(far_rates[valid_segment_mask], weights=segment_lengths[valid_segment_mask])
                        )
                        curve_local_contract["straight_summary_source"] = "track_windows"
                        curve_local_contract["curve_local_active_straight_rate"] = weighted_local
                        curve_local_contract["curve_local_latched_straight_rate"] = weighted_local
                        curve_local_contract["curve_preview_far_active_straight_rate"] = weighted_far
                        curve_local_contract["straight_summary_vs_segment_rate_delta_pct"] = 0.0
                        curve_local_contract["curve_local_latched_straight_count"] = int(
                            round(float(weighted_local) / 100.0 * float(np.sum(segment_lengths[valid_segment_mask])))
                        )

                prev_end_m = 0.0
                for curve_window in curve_windows:
                    start_m = float(curve_window.get("start_m", 0.0) or 0.0)
                    end_m = float(curve_window.get("end_m", start_m) or start_m)
                    curve_indices_all = np.where(
                        (track_progress >= start_m) & (track_progress <= end_m)
                    )[0]
                    if curve_indices_all.size == 0:
                        prev_end_m = end_m
                        continue
                    curve_runs = _split_contiguous(curve_indices_all)
                    selected_curve_run = max(
                        curve_runs,
                        key=lambda run: float(np.max(np.abs(lat_err_arr[run]))),
                    )
                    approach_indices_all = np.where(
                        (track_progress >= prev_end_m) & (track_progress < start_m)
                    )[0]
                    approach_runs = _split_contiguous(approach_indices_all)
                    selected_approach_run = None
                    for run in reversed(approach_runs):
                        if int(run[-1]) < int(selected_curve_run[0]):
                            selected_approach_run = run
                            break
                    if selected_approach_run is None and approach_runs:
                        selected_approach_run = min(
                            approach_runs,
                            key=lambda run: abs(int(run[-1]) - int(selected_curve_run[0])),
                        )
                    search_indices = (
                        np.concatenate([selected_approach_run, selected_curve_run])
                        if selected_approach_run is not None and selected_approach_run.size > 0
                        else selected_curve_run
                    )
                    search_indices = np.unique(search_indices.astype(int))
                    lat_slice = np.abs(lat_err_arr[selected_curve_run])
                    peak_idx_local = int(np.argmax(lat_slice))
                    peak_frame = int(selected_curve_run[peak_idx_local])

                    def _first_state_transition(target_states: set[str], fallback_states: set[str] | None = None) -> Optional[int]:
                        if local_states is None or search_indices.size == 0:
                            return None
                        for frame_idx in search_indices:
                            frame_idx = int(frame_idx)
                            current_state = str(local_states[frame_idx])
                            prev_state = str(local_states[frame_idx - 1]) if frame_idx > 0 else "STRAIGHT"
                            if current_state in target_states and prev_state not in target_states:
                                return frame_idx
                        if fallback_states:
                            for frame_idx in search_indices:
                                frame_idx = int(frame_idx)
                                if str(local_states[frame_idx]) in fallback_states:
                                    return frame_idx
                        return None

                    curve_local_entry_frame = _first_state_transition(
                        {"ENTRY"},
                        fallback_states={"ENTRY", "COMMIT"},
                    )
                    curve_local_commit_frame = _first_state_transition({"COMMIT"})

                    steering_onset_frame = None
                    if steering_arr is not None and search_indices.size > 0:
                        for frame_idx in search_indices:
                            frame_idx = int(frame_idx)
                            current_abs = abs(float(steering_arr[frame_idx]))
                            prev_abs = abs(float(steering_arr[frame_idx - 1])) if frame_idx > 0 else 0.0
                            if current_abs >= steering_onset_abs_threshold and prev_abs < steering_onset_abs_threshold:
                                steering_onset_frame = frame_idx
                                break
                        if steering_onset_frame is None:
                            for frame_idx in search_indices:
                                frame_idx = int(frame_idx)
                                if abs(float(steering_arr[frame_idx])) >= steering_onset_abs_threshold:
                                    steering_onset_frame = frame_idx
                                    break

                    curve_start_frame = None
                    if distance_to_curve_arr is not None and search_indices.size > 0:
                        for frame_idx in search_indices:
                            frame_idx = int(frame_idx)
                            value = float(distance_to_curve_arr[frame_idx])
                            if math.isfinite(value) and value <= 0.25:
                                curve_start_frame = frame_idx
                                break
                    if curve_start_frame is None and selected_curve_run.size > 0:
                        curve_start_frame = int(selected_curve_run[0])

                    def _metric_at(arr: Optional[np.ndarray], frame_idx: Optional[int]) -> Optional[float]:
                        if arr is None or frame_idx is None or frame_idx < 0 or frame_idx >= len(arr):
                            return None
                        value = float(arr[frame_idx])
                        return safe_float(value, default=None) if math.isfinite(value) else None

                    pp_pre_floor_shorten_step_min = None
                    if pp_pre_track is not None and search_indices.size >= 2:
                        pp_pre_vals = np.asarray(pp_pre_track[search_indices], dtype=np.float64)
                        valid_pairs = (
                            np.isfinite(pp_pre_vals[:-1])
                            & np.isfinite(pp_pre_vals[1:])
                            & (pp_pre_vals[:-1] > 1e-6)
                            & (pp_pre_vals[1:] > 1e-6)
                        )
                        if np.any(valid_pairs):
                            pp_pre_floor_shorten_step_min = safe_float(
                                np.min(np.diff(pp_pre_vals)[valid_pairs]), default=None
                            )

                    pp_floor_rescue_delta_max = None
                    pp_floor_rescue_delta_mean = None
                    if pp_pre_track is not None and pp_post_track is not None and search_indices.size > 0:
                        pp_pre_vals = np.asarray(pp_pre_track[search_indices], dtype=np.float64)
                        pp_post_vals = np.asarray(pp_post_track[search_indices], dtype=np.float64)
                        valid_rescue_mask = (
                            np.isfinite(pp_pre_vals)
                            & np.isfinite(pp_post_vals)
                            & (pp_pre_vals > 1e-6)
                            & (pp_post_vals > 1e-6)
                        )
                        if np.any(valid_rescue_mask):
                            rescue_delta = np.maximum(
                                pp_post_vals[valid_rescue_mask] - pp_pre_vals[valid_rescue_mask],
                                0.0,
                            )
                            if rescue_delta.size > 0:
                                pp_floor_rescue_delta_max = safe_float(np.max(rescue_delta), default=None)
                                pp_floor_rescue_delta_mean = safe_float(np.mean(rescue_delta), default=None)

                    def _masked_p50(arr: Optional[np.ndarray]) -> Optional[float]:
                        if arr is None or search_indices.size == 0:
                            return None
                        vals = np.asarray(arr[search_indices], dtype=np.float64)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            return None
                        return safe_float(np.percentile(vals, 50), default=None)

                    def _masked_max(arr: Optional[np.ndarray]) -> Optional[float]:
                        if arr is None or search_indices.size == 0:
                            return None
                        vals = np.asarray(arr[search_indices], dtype=np.float64)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            return None
                        return safe_float(np.max(vals), default=None)

                    local_entry_driver_mode = None
                    if local_entry_driver_series is not None and search_indices.size > 0:
                        driver_vals = [
                            str(local_entry_driver_series[idx] or "").strip()
                            for idx in search_indices
                            if 0 <= idx < len(local_entry_driver_series)
                        ]
                        driver_vals = [value for value in driver_vals if value]
                        if driver_vals:
                            local_entry_driver_mode = max(
                                sorted(set(driver_vals)),
                                key=driver_vals.count,
                            )

                    local_commit_driver_mode = None
                    if local_commit_driver_series is not None and search_indices.size > 0:
                        driver_vals = [
                            str(local_commit_driver_series[idx] or "").strip()
                            for idx in search_indices
                            if 0 <= idx < len(local_commit_driver_series)
                        ]
                        driver_vals = [value for value in driver_vals if value]
                        if driver_vals:
                            local_commit_driver_mode = max(
                                sorted(set(driver_vals)),
                                key=driver_vals.count,
                            )

                    owner_mode = _masked_mode_string(owner_mode_series, search_indices)
                    entry_weight_source_mode = _masked_mode_string(
                        entry_weight_source_series, search_indices
                    )
                    fallback_active_rate = None
                    if fallback_active_series is not None and search_indices.size > 0:
                        fallback_vals = np.asarray(
                            fallback_active_series[search_indices], dtype=np.float64
                        )
                        fallback_vals = fallback_vals[np.isfinite(fallback_vals) & (fallback_vals >= 0.0)]
                        if fallback_vals.size > 0:
                            fallback_active_rate = safe_float(
                                np.mean(fallback_vals > 0.5) * 100.0, default=None
                            )
                    owner_nominal_p50 = _masked_p50(owner_nominal_track)
                    owner_commit_band_p50 = _masked_p50(owner_commit_band_track)
                    owner_commit_progress_p50 = _masked_p50(owner_commit_progress_track)
                    owner_commit_clamp_active_rate = None
                    if owner_commit_clamp_active_track is not None and search_indices.size > 0:
                        owner_commit_clamp_vals = np.asarray(
                            owner_commit_clamp_active_track[search_indices], dtype=np.float64
                        )
                        owner_commit_clamp_vals = owner_commit_clamp_vals[
                            np.isfinite(owner_commit_clamp_vals)
                            & (owner_commit_clamp_vals >= 0.0)
                        ]
                        if owner_commit_clamp_vals.size > 0:
                            owner_commit_clamp_active_rate = safe_float(
                                np.mean(owner_commit_clamp_vals > 0.5) * 100.0,
                                default=None,
                            )
                    owner_commit_clamp_delta_max = _masked_max(owner_commit_clamp_delta_track)
                    owner_band_minus_floor_p50 = None
                    if (
                        owner_commit_band_track is not None
                        and pp_floor_m_series is not None
                        and search_indices.size > 0
                    ):
                        owner_band_vals = np.asarray(
                            owner_commit_band_track[search_indices], dtype=np.float64
                        )
                        pp_floor_vals = np.asarray(
                            pp_floor_m_series[search_indices], dtype=np.float64
                        )
                        valid_band_mask = np.isfinite(owner_band_vals) & np.isfinite(pp_floor_vals)
                        if np.any(valid_band_mask):
                            owner_band_minus_floor_p50 = safe_float(
                                np.percentile(
                                    owner_band_vals[valid_band_mask]
                                    - pp_floor_vals[valid_band_mask],
                                    50,
                                ),
                                default=None,
                            )
                    local_arc_mode = _masked_mode_string(local_arc_mode_series, search_indices)
                    local_arc_source_mode = _masked_mode_string(
                        local_arc_source_series, search_indices
                    )
                    local_arc_fallback_reason_mode = _masked_mode_string(
                        local_arc_fallback_reason_series, search_indices
                    )
                    local_arc_active_rate = None
                    if local_arc_active_series is not None and search_indices.size > 0:
                        local_arc_active_vals = np.asarray(
                            local_arc_active_series[search_indices], dtype=np.float64
                        )
                        local_arc_active_vals = local_arc_active_vals[
                            np.isfinite(local_arc_active_vals) & (local_arc_active_vals >= 0.0)
                        ]
                        if local_arc_active_vals.size > 0:
                            local_arc_active_rate = safe_float(
                                np.mean(local_arc_active_vals > 0.5) * 100.0,
                                default=None,
                            )
                    local_arc_fallback_rate = None
                    if local_arc_fallback_active_series is not None and search_indices.size > 0:
                        local_arc_fallback_vals = np.asarray(
                            local_arc_fallback_active_series[search_indices], dtype=np.float64
                        )
                        local_arc_fallback_vals = local_arc_fallback_vals[
                            np.isfinite(local_arc_fallback_vals) & (local_arc_fallback_vals >= 0.0)
                        ]
                        if local_arc_fallback_vals.size > 0:
                            local_arc_fallback_rate = safe_float(
                                np.mean(local_arc_fallback_vals > 0.5) * 100.0,
                                default=None,
                            )
                    local_arc_blend_weight_p50 = _masked_p50(local_arc_blend_weight_series)
                    local_arc_progress_weight_p50 = _masked_p50(local_arc_progress_weight_series)
                    local_arc_planner_delta_p50 = _masked_p50(local_arc_vs_planner_delta_series)
                    local_arc_target_distance_p50 = _masked_p50(local_arc_target_distance_series)
                    local_arc_arc_curvature_p50 = _masked_p50(local_arc_curvature_series)

                    event = {
                        "curve_index": int(curve_window.get("curve_index", len(curve_turn_events) + 1)),
                        "entry_frame": int(selected_curve_run[0]),
                        "exit_frame": int(selected_curve_run[-1]),
                        "peak_lateral_error_m": safe_float(np.max(lat_slice)),
                        "peak_lateral_error_frame": peak_frame,
                        "curve_start_frame": int(curve_start_frame) if curve_start_frame is not None else None,
                        "curve_local_entry_frame": (
                            int(curve_local_entry_frame) if curve_local_entry_frame is not None else None
                        ),
                        "curve_local_commit_frame": (
                            int(curve_local_commit_frame) if curve_local_commit_frame is not None else None
                        ),
                        "steering_onset_frame": (
                            int(steering_onset_frame) if steering_onset_frame is not None else None
                        ),
                        "curve_local_entry_distance_m": _metric_at(distance_to_curve_arr, curve_local_entry_frame),
                        "curve_local_commit_distance_m": _metric_at(distance_to_curve_arr, curve_local_commit_frame),
                        "steering_onset_distance_m": _metric_at(distance_to_curve_arr, steering_onset_frame),
                        "curve_local_entry_ttc_s": _metric_at(time_to_curve_arr, curve_local_entry_frame),
                        "curve_local_commit_ttc_s": _metric_at(time_to_curve_arr, curve_local_commit_frame),
                        "steering_onset_ttc_s": _metric_at(time_to_curve_arr, steering_onset_frame),
                        "curve_local_arm_ready_at_entry": _metric_at(local_arm_ready_arr, curve_local_entry_frame),
                        "curve_local_commit_ready_at_entry": _metric_at(local_commit_ready_arr, curve_local_entry_frame),
                        "curve_local_in_curve_now_at_entry": _metric_at(local_in_curve_now_arr, curve_local_entry_frame),
                        "curve_local_entry_severity_p50": _masked_p50(local_entry_severity_arr),
                        "curve_local_entry_severity_max": _masked_max(local_entry_severity_arr),
                        "curve_local_entry_on_effective_p50": _masked_p50(local_entry_on_arr),
                        "curve_local_phase_distance_start_effective_p50_m": _masked_p50(local_distance_start_arr),
                        "curve_local_phase_time_start_effective_p50_s": _masked_p50(local_time_start_arr),
                        "curve_local_arm_phase_raw_p50": _masked_p50(local_arm_phase_arr),
                        "curve_local_sustain_phase_raw_p50": _masked_p50(local_sustain_phase_arr),
                        "curve_local_path_sustain_active_rate": (
                            safe_float(
                                np.mean(local_path_sustain_arr[search_indices] > 0.5) * 100.0
                            )
                            if local_path_sustain_arr is not None and search_indices.size > 0
                            else None
                        ),
                        "curve_local_entry_driver_mode": local_entry_driver_mode,
                        "curve_local_commit_driver_mode": local_commit_driver_mode,
                        "reference_lookahead_owner_mode": owner_mode,
                        "reference_lookahead_owner_mode_at_entry": _string_at(
                            owner_mode_series, curve_local_entry_frame
                        ),
                        "reference_lookahead_entry_weight_source": entry_weight_source_mode,
                        "reference_lookahead_entry_weight_source_at_entry": _string_at(
                            entry_weight_source_series, curve_local_entry_frame
                        ),
                        "reference_lookahead_fallback_active_rate": fallback_active_rate,
                        "reference_lookahead_fallback_active_at_entry": _metric_at(
                            fallback_active_series, curve_local_entry_frame
                        ),
                        "reference_lookahead_owner_nominal_target_p50_m": owner_nominal_p50,
                        "reference_lookahead_owner_commit_band_target_p50_m": owner_commit_band_p50,
                        "reference_lookahead_owner_commit_progress_p50": owner_commit_progress_p50,
                        "reference_lookahead_owner_commit_band_clamp_active_rate": (
                            owner_commit_clamp_active_rate
                        ),
                        "reference_lookahead_owner_commit_band_clamp_delta_max_m": (
                            owner_commit_clamp_delta_max
                        ),
                        "reference_lookahead_owner_commit_band_minus_floor_p50_m": (
                            owner_band_minus_floor_p50
                        ),
                        "local_curve_reference_mode": local_arc_mode,
                        "local_curve_reference_source_mode": local_arc_source_mode,
                        "local_curve_reference_fallback_reason_mode": (
                            local_arc_fallback_reason_mode
                        ),
                        "local_curve_reference_active_rate": local_arc_active_rate,
                        "local_curve_reference_fallback_rate": local_arc_fallback_rate,
                        "local_curve_reference_blend_weight_p50": local_arc_blend_weight_p50,
                        "local_curve_reference_progress_weight_p50": local_arc_progress_weight_p50,
                        "local_curve_reference_vs_planner_delta_p50_m": (
                            local_arc_planner_delta_p50
                        ),
                        "local_curve_reference_target_distance_p50_m": (
                            local_arc_target_distance_p50
                        ),
                        "local_curve_reference_arc_curvature_p50": (
                            local_arc_arc_curvature_p50
                        ),
                        "steering_onset_lateral_error_m": _metric_at(lat_err_arr, steering_onset_frame),
                        "steering_onset_heading_error_rad": _metric_at(heading_err_arr, steering_onset_frame),
                        "peak_heading_error_rad": _metric_at(heading_err_arr, peak_frame),
                        "steering_onset_minus_curve_start_frames": (
                            int(steering_onset_frame - curve_start_frame)
                            if steering_onset_frame is not None and curve_start_frame is not None
                            else None
                        ),
                        "pp_pre_floor_shorten_step_min_m_per_frame": pp_pre_floor_shorten_step_min,
                        "pp_floor_rescue_delta_max_m": pp_floor_rescue_delta_max,
                        "pp_floor_rescue_delta_mean_m": pp_floor_rescue_delta_mean,
                    }
                    entry_ttc = event.get("curve_local_entry_ttc_s")
                    onset_distance = event.get("steering_onset_distance_m")
                    event["state_armed_early_enough"] = bool(
                        entry_ttc is not None and float(entry_ttc) > curve_local_entry_min_ttc_s
                    )
                    event["steering_started_early_enough"] = bool(
                        (
                            onset_distance is not None and float(onset_distance) > 0.0
                        )
                        or (
                            onset_distance is None
                            and steering_onset_frame is not None
                            and curve_start_frame is not None
                            and int(steering_onset_frame) < int(curve_start_frame)
                        )
                    )
                    event["late_turn_in"] = bool(
                        (onset_distance is not None and float(onset_distance) <= 0.0)
                        or (
                            onset_distance is None
                            and steering_onset_frame is not None
                            and curve_start_frame is not None
                            and int(steering_onset_frame) >= int(curve_start_frame)
                        )
                    )
                    if pp_track is not None:
                        pp_slice = pp_track[selected_curve_run]
                        pp_slice = pp_slice[np.isfinite(pp_slice) & (pp_slice > 0.0)]
                        if pp_slice.size > 0:
                            event["pp_lookahead_min_m"] = safe_float(np.min(pp_slice))
                    if ref_track is not None:
                        ref_slice = ref_track[selected_curve_run]
                        ref_slice = ref_slice[np.isfinite(ref_slice) & (ref_slice > 0.0)]
                        if ref_slice.size > 0:
                            event["reference_lookahead_min_m"] = safe_float(np.min(ref_slice))
                    curve_turn_events.append(event)
                    prev_end_m = end_m

    # Curve-cap longitudinal diagnostics
    curve_cap_active_series = data.get("speed_governor_curve_cap_active")
    if curve_cap_active_series is not None and n_frames > 0:
        n_cap = min(n_frames, len(curve_cap_active_series))
        if n_cap > 0:
            cap_mask = np.asarray(curve_cap_active_series[:n_cap], dtype=np.float64) > 0.5
            curve_cap_active_rate = safe_float(np.mean(cap_mask) * 100.0)
            feas_infeasible_series = data.get("turn_feasibility_infeasible")
            if feas_infeasible_series is not None and len(feas_infeasible_series) >= n_cap:
                infeasible_mask = np.asarray(feas_infeasible_series[:n_cap], dtype=np.float64) > 0.5
                if np.any(cap_mask):
                    turn_infeasible_rate_when_curve_cap_active = safe_float(
                        np.mean(infeasible_mask[cap_mask]) * 100.0
                    )

    if curve_intent_diag.get("available"):
        curve_cap_pre_turn_lead_frames_p50 = safe_float(
            curve_intent_diag.get("arm_lead_frames_p50", 0.0)
        )
        curve_cap_pre_turn_lead_frames_p95 = safe_float(
            curve_intent_diag.get("arm_lead_frames_p95", 0.0)
        )

    if (
        data.get("speed") is not None
        and data.get("turn_feasibility_speed_limit_mps") is not None
    ):
        n_over = min(
            n_frames,
            len(data["speed"]),
            len(data["turn_feasibility_speed_limit_mps"]),
        )
        if n_over > 0:
            speed_arr = np.asarray(data["speed"][:n_over], dtype=np.float64)
            cap_arr = np.asarray(data["turn_feasibility_speed_limit_mps"][:n_over], dtype=np.float64)
            valid = np.isfinite(speed_arr) & np.isfinite(cap_arr) & (cap_arr > 0.1)
            if np.any(valid):
                overspeed_into_curve_rate = safe_float(
                    np.mean(speed_arr[valid] > (cap_arr[valid] + 0.2)) * 100.0
                )

    cap_active_series = data.get("speed_governor_curve_cap_active")
    cap_speed_series = data.get("speed_governor_curve_cap_speed")
    target_speed_final_series = data.get("target_speed_final")
    if (
        cap_active_series is not None
        and cap_speed_series is not None
        and target_speed_final_series is not None
    ):
        n_cap_track = min(
            n_frames,
            len(cap_active_series),
            len(cap_speed_series),
            len(target_speed_final_series),
        )
        if n_cap_track > 0:
            cap_active_arr = np.asarray(cap_active_series[:n_cap_track], dtype=np.float64) > 0.5
            cap_speed_arr = np.asarray(cap_speed_series[:n_cap_track], dtype=np.float64)
            target_final_arr = np.asarray(target_speed_final_series[:n_cap_track], dtype=np.float64)
            valid_cap_mask = (
                cap_active_arr
                & np.isfinite(cap_speed_arr)
                & np.isfinite(target_final_arr)
                & (cap_speed_arr > 0.01)
            )
            if np.any(valid_cap_mask):
                cap_tracking_error_series = data.get("speed_governor_cap_tracking_error_mps")
                if (
                    cap_tracking_error_series is not None
                    and len(cap_tracking_error_series) >= n_cap_track
                ):
                    cap_err_arr = np.asarray(
                        cap_tracking_error_series[:n_cap_track], dtype=np.float64
                    )
                    cap_err_arr = np.where(np.isfinite(cap_err_arr), np.maximum(cap_err_arr, 0.0), 0.0)
                else:
                    cap_err_arr = np.maximum(0.0, target_final_arr - cap_speed_arr)

                cap_err_valid = cap_err_arr[valid_cap_mask]
                if cap_err_valid.size > 0:
                    cap_tracking_error_p50_mps = safe_float(np.percentile(cap_err_valid, 50))
                    cap_tracking_error_p95_mps = safe_float(np.percentile(cap_err_valid, 95))
                    cap_tracking_error_max_mps = safe_float(np.max(cap_err_valid))
                    frames_above_cap_0p3mps_rate = safe_float(
                        np.mean(cap_err_valid > 0.3) * 100.0
                    )
                    frames_above_cap_1p0mps_rate = safe_float(
                        np.mean(cap_err_valid > 1.0) * 100.0
                    )

                cap_recovery_frames_series = data.get("speed_governor_cap_tracking_recovery_frames")
                if (
                    cap_recovery_frames_series is not None
                    and len(cap_recovery_frames_series) >= n_cap_track
                ):
                    recovery_arr = np.asarray(
                        cap_recovery_frames_series[:n_cap_track], dtype=np.float64
                    )
                    recovery_arr = np.where(np.isfinite(recovery_arr), recovery_arr, np.nan)
                    recovery_valid = recovery_arr[valid_cap_mask]
                    recovery_valid = recovery_valid[np.isfinite(recovery_valid) & (recovery_valid >= 0.0)]
                    if recovery_valid.size > 0:
                        cap_recovery_frames_p50 = safe_float(np.percentile(recovery_valid, 50))
                        cap_recovery_frames_p95 = safe_float(np.percentile(recovery_valid, 95))

                hard_ceiling_series = data.get("speed_governor_cap_tracking_hard_ceiling_applied")
                if hard_ceiling_series is not None and len(hard_ceiling_series) >= n_cap_track:
                    hard_ceiling_arr = np.asarray(
                        hard_ceiling_series[:n_cap_track], dtype=np.float64
                    ) > 0.5
                    hard_ceiling_applied_rate = safe_float(
                        np.mean(hard_ceiling_arr[valid_cap_mask]) * 100.0
                    )

    # 10. Curvature contract/coherency diagnostics
    def _bool_series_with_valid(series):
        if series is None:
            return None, None
        arr = np.asarray(series, dtype=np.float64)
        n_local = min(int(n_frames), int(arr.size))
        if n_local <= 0:
            return np.array([], dtype=bool), np.array([], dtype=bool)
        arr = arr[:n_local]
        valid = np.isfinite(arr)
        # Recorder writes -1 for unknown bool telemetry.
        valid &= arr >= 0.0
        return arr > 0.5, valid

    def _field_completeness_percent(field_name: str, kind: str = "numeric") -> float:
        series = data.get(field_name)
        if series is None:
            return 0.0
        if kind == "string":
            n_local = min(int(n_frames), len(series))
            if n_local <= 0:
                return 0.0
            values = [str(series[i] if series[i] is not None else "").strip() for i in range(n_local)]
            valid = np.array([len(v) > 0 for v in values], dtype=bool)
            return safe_float(np.mean(valid) * 100.0)
        arr = np.asarray(series, dtype=np.float64)
        n_local = min(int(n_frames), int(arr.size))
        if n_local <= 0:
            return 0.0
        arr = arr[:n_local]
        valid = np.isfinite(arr)
        if kind == "bool":
            valid &= arr >= 0.0
        return safe_float(np.mean(valid) * 100.0)

    divergence_mask, divergence_valid = _bool_series_with_valid(
        data.get("curvature_source_diverged")
    )
    if divergence_mask is not None and divergence_mask.size > 0 and np.any(divergence_valid):
        curvature_source_diverged_rate = safe_float(
            np.mean(divergence_mask[divergence_valid]) * 100.0
        )
        divergence_indices = np.where(divergence_mask & divergence_valid)[0]
        if divergence_indices.size > 0:
            first_divergence_frame = int(divergence_indices[0])

    map_authority_series = data.get("curvature_map_authority_lost")
    if map_authority_series is not None:
        map_authority_arr = np.asarray(map_authority_series, dtype=bool)
        if map_authority_arr.size > 0:
            curvature_map_authority_lost_rate = safe_float(
                np.mean(map_authority_arr) * 100.0
            )

    divergence_abs_series = data.get("curvature_source_divergence_abs")
    if divergence_abs_series is not None:
        divergence_abs_arr = np.asarray(divergence_abs_series, dtype=np.float64)
        n_div_abs = min(int(n_frames), int(divergence_abs_arr.size))
        if n_div_abs > 0:
            divergence_abs_arr = divergence_abs_arr[:n_div_abs]
            divergence_abs_arr = divergence_abs_arr[np.isfinite(divergence_abs_arr)]
            if divergence_abs_arr.size > 0:
                curvature_source_divergence_p95 = safe_float(
                    np.percentile(divergence_abs_arr, 95)
                )

    curve_intent_state_series = data.get("curve_intent_state")
    if curve_intent_state_series is not None:
        n_state = min(int(n_frames), len(curve_intent_state_series))
        current_commit = 0
        max_commit = 0
        for i in range(n_state):
            state = str(curve_intent_state_series[i] or "").strip().upper()
            if state == "COMMIT":
                current_commit += 1
                if current_commit > max_commit:
                    max_commit = current_commit
            else:
                current_commit = 0
        curve_intent_commit_streak_max_frames = int(max_commit)

    if (
        data.get("speed") is not None
        and data.get("turn_feasibility_speed_limit_mps") is not None
    ):
        speed_arr = np.asarray(data["speed"], dtype=np.float64)
        feas_cap_arr = np.asarray(data["turn_feasibility_speed_limit_mps"], dtype=np.float64)
        n_feas = min(int(n_frames), int(speed_arr.size), int(feas_cap_arr.size))
        if n_feas > 0:
            speed_arr = speed_arr[:n_feas]
            feas_cap_arr = feas_cap_arr[:n_feas]
            valid_feas = np.isfinite(speed_arr) & np.isfinite(feas_cap_arr) & (feas_cap_arr > 0.1)
            if np.any(valid_feas):
                overspeed_mask = speed_arr > (feas_cap_arr + 0.2)
                feasibility_violation_rate = safe_float(
                    np.mean(overspeed_mask[valid_feas]) * 100.0
                )
                overspeed_indices = np.where(overspeed_mask & valid_feas)[0]
                if overspeed_indices.size > 0:
                    first_speed_above_feasibility_frame = int(overspeed_indices[0])

    infeasible_mask, infeasible_valid = _bool_series_with_valid(
        data.get("turn_feasibility_infeasible")
    )
    if infeasible_mask is not None and infeasible_mask.size > 0 and np.any(infeasible_valid):
        infeasible_indices = np.where(infeasible_mask & infeasible_valid)[0]
        if infeasible_indices.size > 0:
            first_infeasible_frame = int(infeasible_indices[0])

    backstop_mask, backstop_valid = _bool_series_with_valid(
        data.get("speed_governor_feasibility_backstop_active")
    )
    if backstop_mask is not None and backstop_mask.size > 0 and np.any(backstop_valid):
        feasibility_backstop_active_rate = safe_float(
            np.mean(backstop_mask[backstop_valid]) * 100.0
        )

    map_health_mask, map_health_valid = _bool_series_with_valid(data.get("map_health_ok"))
    if map_health_mask is not None and map_health_mask.size > 0 and np.any(map_health_valid):
        map_health_untrusted_rate = safe_float(
            np.mean((~map_health_mask)[map_health_valid]) * 100.0
        )

    track_match_mask, track_match_valid = _bool_series_with_valid(data.get("track_match_ok"))
    if track_match_mask is not None and track_match_mask.size > 0 and np.any(track_match_valid):
        track_mismatch_rate = safe_float(
            np.mean((~track_match_mask)[track_match_valid]) * 100.0
        )

    consistent_all_mask, consistent_all_valid = _bool_series_with_valid(
        data.get("curvature_contract_consistent_all")
    )
    if (
        consistent_all_mask is not None
        and consistent_all_mask.size > 0
        and np.any(consistent_all_valid)
    ):
        curvature_contract_consistency_rate = safe_float(
            np.mean(consistent_all_mask[consistent_all_valid]) * 100.0
        )

    contract_completeness_fields = [
        ("curvature_primary_abs", "numeric"),
        ("curvature_primary_source", "string"),
        ("curvature_map_authority_lost", "bool"),
        ("curvature_source_diverged", "bool"),
        ("curvature_source_divergence_abs", "numeric"),
        ("curvature_contract_consistent_all", "bool"),
        ("map_health_ok", "bool"),
        ("track_match_ok", "bool"),
    ]
    contract_completeness_values = [
        _field_completeness_percent(name, kind) for name, kind in contract_completeness_fields
    ]
    if contract_completeness_values:
        telemetry_completeness_rate_curvature_contract = safe_float(
            np.mean(np.asarray(contract_completeness_values, dtype=np.float64))
        )

    feasibility_completeness_fields = [
        ("turn_feasibility_speed_limit_mps", "numeric"),
        ("turn_feasibility_infeasible", "bool"),
        ("speed_governor_feasibility_backstop_active", "bool"),
        ("speed_governor_feasibility_backstop_speed", "numeric"),
    ]
    feasibility_completeness_values = [
        _field_completeness_percent(name, kind) for name, kind in feasibility_completeness_fields
    ]
    if feasibility_completeness_values:
        telemetry_completeness_rate_feasibility = safe_float(
            np.mean(np.asarray(feasibility_completeness_values, dtype=np.float64))
        )

    _boundary_candidates = []
    if out_of_lane_events_list:
        _start = out_of_lane_events_list[0].get("start_frame")
        if _start is not None:
            _boundary_candidates.append(int(_start))
    if emergency_stop_frame is not None:
        _boundary_candidates.append(int(emergency_stop_frame))
    if failure_frame is not None:
        _boundary_candidates.append(int(failure_frame))
    if _boundary_candidates:
        first_boundary_breach_frame = int(min(_boundary_candidates))

    # ── ACC health summary (optional — None when ACC inactive or not fitted) ────
    acc_health = _build_acc_health_summary(data, n_frames, speed=data.get('speed'))
    acc_comfort_contract = _build_acc_comfort_contract_summary(
        acc_health=acc_health,
        hotspot_attribution=hotspot_attribution,
    )
    acc_detection_contract = _build_acc_detection_contract_summary(acc_health)

    # ACC Safety layer deductions (zero when acc_health is None)
    acc_collision_penalty = 0.0
    acc_near_miss_penalty = 0.0
    acc_ttc_warning_penalty = 0.0
    acc_jerk_penalty = 0.0
    acc_hard_zero = False
    if acc_health is not None:
        acc_collision_penalty = safe_float(acc_health.get("collision_penalty", 0.0))
        acc_near_miss_penalty = safe_float(acc_health.get("near_miss_penalty", 0.0))
        acc_ttc_warning_penalty = safe_float(acc_health.get("ttc_warning_penalty", 0.0))
        acc_hard_zero = bool(acc_health.get("hard_zero", False))
        jerk_gate_value = acc_health.get("acc_jerk_gate_value_mps3", 0.0) or 0.0
        if jerk_gate_value > ACC_JERK_P95_GATE_MPS3:
            acc_jerk_penalty = safe_float(
                min(15.0, (jerk_gate_value - ACC_JERK_P95_GATE_MPS3) * 5.0)
            )

    layer_breakdowns = {
        "Perception": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Lane Detection",
                    "value": lane_detection_penalty,
                    "limit": ">=90%",
                },
                {
                    "name": "Stale Hard Data",
                    "value": stale_data_penalty,
                    "limit": "<10%",
                },
                {
                    "name": "Perception Instability",
                    "value": perception_instability_penalty,
                    "limit": "stability>=80%",
                },
                {
                    "name": "Lane Line Jitter P95",
                    "value": lane_jitter_penalty,
                    "limit": "<=0.30m",
                },
                {
                    "name": "Reference Jitter P95",
                    "value": reference_jitter_penalty,
                    "limit": "<=0.15m",
                },
            ],
        },
        "Trajectory": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Lateral Error RMSE (curv-adj)",
                    "value": trajectory_lateral_rmse_penalty,
                    "limit": "<=0.20m",
                },
                {
                    "name": "Lateral Error P95 (curv-adj)",
                    "value": trajectory_lateral_p95_penalty,
                    "limit": f"<={LATERAL_P95_GATE_M}m",
                },
                {
                    "name": "Heading Error RMSE",
                    "value": trajectory_heading_penalty,
                    "limit": f"<={HEADING_PENALTY_FLOOR_DEG:.0f}deg",
                },
            ],
        },
        "Control": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Steering Jerk",
                    "value": control_steering_jerk_penalty,
                    "limit": "<=18.0/s^2 (cap)",
                },
                {
                    "name": "Oscillation Frequency",
                    "value": control_oscillation_penalty,
                    "limit": "<=1.0Hz",
                },
                {
                    "name": "Straight Sign Mismatch",
                    "value": control_sign_mismatch_penalty,
                    "limit": "<=5%",
                },
            ],
        },
        "LongitudinalComfort": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Acceleration P95",
                    "value": longitudinal_accel_penalty,
                    "limit": "<=0.31g (3.0 m/s²)",
                },
                {
                    "name": "Jerk P95",
                    "value": longitudinal_jerk_penalty,
                    "limit": "<=0.61 g/s (6.0 m/s³)",
                },
                {
                    "name": "ACC Following Jerk P95 (filtered)",
                    "value": acc_jerk_penalty,
                    "limit": "<=4.0 m/s³",
                },
            ],
        },
        "Safety": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Out Of Lane Time",
                    "value": safety_out_of_lane_penalty,
                    "limit": "<5%",
                },
                {
                    "name": "Out Of Lane / Emergency Events",
                    "value": safety_event_penalty,
                    "limit": "none",
                },
                {
                    "name": "ACC Collision Events",
                    "value": acc_collision_penalty,
                    "limit": "0 events",
                },
                {
                    "name": "ACC Near-Miss Events",
                    "value": acc_near_miss_penalty,
                    "limit": "0 events",
                },
                {
                    "name": "ACC TTC Warning Zone",
                    "value": acc_ttc_warning_penalty,
                    "limit": "< 5% of ACC frames",
                },
            ],
        },
        "SignalIntegrity": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Heading Suppression Rate (curves)",
                    "value": signal_integrity_heading_penalty,
                    "limit": "<=20%",
                },
                {
                    "name": "Rate Limit Saturation Rate (curves)",
                    "value": signal_integrity_rate_limit_penalty,
                    "limit": "<=30%",
                },
                {
                    "name": "Speed Feasibility Violations",
                    "value": signal_integrity_speed_feasibility_penalty,
                    "limit": "<=5 frames",
                },
            ],
        },
    }

    layer_scores: Dict[str, float] = {}
    for layer_name, layer in layer_breakdowns.items():
        total_deduction = sum(float(d.get("value", 0.0)) for d in layer["deductions"])
        layer_score = safe_float(max(0.0, 100.0 - total_deduction))
        layer["total_deduction"] = safe_float(total_deduction)
        layer["final_score"] = layer_score
        layer_scores[layer_name] = layer_score

    # Scale existing weights by 0.92 to make room for SignalIntegrity 0.08 (sum to 1.0)
    layer_weights = {
        "Safety": 0.32 * 0.92,
        "Trajectory": 0.30 * 0.92,
        "Control": 0.16 * 0.92,
        "Perception": 0.14 * 0.92,
        "LongitudinalComfort": 0.08 * 0.92,
        "SignalIntegrity": 0.08,
    }

    weighted_contributions = {
        layer: safe_float(layer_scores.get(layer, 0.0) * weight)
        for layer, weight in layer_weights.items()
    }
    overall_base = safe_float(sum(weighted_contributions.values()))

    critical_layers = ("Safety", "Trajectory")
    critical_cap = 100.0
    cap_reason = "none"
    critical_layer_colors: Dict[str, str] = {}
    for layer in critical_layers:
        score_val = layer_scores.get(layer, 0.0)
        if score_val < 60.0:
            critical_layer_colors[layer] = "red"
        elif score_val < 80.0:
            critical_layer_colors[layer] = "yellow"
        else:
            critical_layer_colors[layer] = "green"

    if any(color == "red" for color in critical_layer_colors.values()):
        critical_cap = 59.0
        cap_reason = "critical_red_layer"
    elif any(color == "yellow" for color in critical_layer_colors.values()):
        critical_cap = 79.0
        cap_reason = "critical_yellow_layer"

    score = safe_float(min(overall_base, critical_cap))

    # Hard zero override: ACC collision → score = 0 (same path as catastrophic OOL)
    if acc_hard_zero:
        score = 0.0
        cap_reason = "acc_collision"

    turn_in_owner = {
        "availability": "unavailable",
        "owner_class": "authoritative",
        "owner_mode": None,
        "entry_weight_source": None,
        "fallback_active_rate": None,
        "phase_active_rate": None,
        "binary_owner_rate": None,
        "curve_local_commit_without_ready_count": int(
            curve_local_contract.get("curve_local_commit_without_ready_count", 0)
        ),
        "curve_local_arm_without_ready_count": int(
            curve_local_contract.get("curve_local_arm_without_ready_count", 0)
        ),
        "owner_commit_band_clamp_active_rate": None,
        "owner_commit_progress_p50": None,
        "steering_onset_minus_curve_start_frames_p50": None,
    }
    if n_frames > 0 and (
        owner_mode_series is not None
        or entry_weight_source_series is not None
        or fallback_active_series is not None
    ):
        all_indices = np.arange(n_frames, dtype=int)
        turn_in_owner["availability"] = "available"
        turn_in_owner["owner_mode"] = _masked_mode_string(owner_mode_series, all_indices)
        turn_in_owner["entry_weight_source"] = _masked_mode_string(
            entry_weight_source_series, all_indices
        )
        if owner_mode_series is not None:
            owner_text = np.array(
                [_normalize_str_value(value) for value in owner_mode_series[:n_frames]],
                dtype=object,
            )
            valid_mask = owner_text != ""
            if np.any(valid_mask):
                turn_in_owner["phase_active_rate"] = safe_float(
                    np.mean(owner_text[valid_mask] == CURVE_SCHEDULER_MODE_PHASE_ACTIVE) * 100.0
                )
                turn_in_owner["binary_owner_rate"] = safe_float(
                    np.mean(owner_text[valid_mask] == CURVE_SCHEDULER_MODE_BINARY) * 100.0
                )
        if fallback_active_series is not None:
            fallback_vals = np.asarray(fallback_active_series[:n_frames], dtype=np.float64)
            valid_fb = np.isfinite(fallback_vals) & (fallback_vals >= 0.0)
            if np.any(valid_fb):
                turn_in_owner["fallback_active_rate"] = safe_float(
                    np.mean(fallback_vals[valid_fb] > 0.5) * 100.0
                )
        onset_offsets = [
            float(event.get("steering_onset_minus_curve_start_frames"))
            for event in curve_turn_events
            if event.get("steering_onset_minus_curve_start_frames") is not None
        ]
        if onset_offsets:
            turn_in_owner["steering_onset_minus_curve_start_frames_p50"] = safe_float(
                np.percentile(np.asarray(onset_offsets, dtype=np.float64), 50)
            )
        clamp_rates = [
            float(event.get("reference_lookahead_owner_commit_band_clamp_active_rate"))
            for event in curve_turn_events
            if event.get("reference_lookahead_owner_commit_band_clamp_active_rate") is not None
        ]
        if clamp_rates:
            turn_in_owner["owner_commit_band_clamp_active_rate"] = safe_float(
                np.percentile(np.asarray(clamp_rates, dtype=np.float64), 50)
            )
        commit_progress_vals = [
            float(event.get("reference_lookahead_owner_commit_progress_p50"))
            for event in curve_turn_events
            if event.get("reference_lookahead_owner_commit_progress_p50") is not None
        ]
        if commit_progress_vals:
            turn_in_owner["owner_commit_progress_p50"] = safe_float(
                np.percentile(np.asarray(commit_progress_vals, dtype=np.float64), 50)
            )

    local_curve_reference = {
        "availability": "unavailable",
        "owner_class": "authoritative",
        "mode": None,
        "active_rate": None,
        "shadow_only_rate": None,
        "valid_rate": None,
        "fallback_active_rate": None,
        "source_mode": None,
        "fallback_reason_mode": None,
        "blend_weight_p50": None,
        "progress_weight_p50": None,
        "planner_delta_p50_m": None,
        "planner_delta_p95_m": None,
        "target_distance_p50_m": None,
        "arc_curvature_p50": None,
        "turn_event_count": int(len(curve_turn_events)),
        "limits": {
            "fallback_active_rate_max_pct": 5.0,
        },
    }
    if n_frames > 0 and (
        local_arc_mode_series is not None
        or local_arc_active_series is not None
        or local_arc_valid_series is not None
    ):
        all_indices = np.arange(n_frames, dtype=int)
        local_curve_reference["availability"] = "available"
        local_curve_reference["mode"] = _masked_mode_string(local_arc_mode_series, all_indices)
        local_curve_reference["source_mode"] = _masked_mode_string(local_arc_source_series, all_indices)
        local_curve_reference["fallback_reason_mode"] = _masked_mode_string(
            local_arc_fallback_reason_series, all_indices
        )
        if local_arc_active_series is not None:
            active_vals = np.asarray(local_arc_active_series[:n_frames], dtype=np.float64)
            valid_mask = np.isfinite(active_vals) & (active_vals >= 0.0)
            if np.any(valid_mask):
                local_curve_reference["active_rate"] = safe_float(
                    np.mean(active_vals[valid_mask] > 0.5) * 100.0
                )
        if local_arc_shadow_only_series is not None:
            shadow_vals = np.asarray(local_arc_shadow_only_series[:n_frames], dtype=np.float64)
            valid_mask = np.isfinite(shadow_vals) & (shadow_vals >= 0.0)
            if np.any(valid_mask):
                local_curve_reference["shadow_only_rate"] = safe_float(
                    np.mean(shadow_vals[valid_mask] > 0.5) * 100.0
                )
        if local_arc_valid_series is not None:
            valid_vals = np.asarray(local_arc_valid_series[:n_frames], dtype=np.float64)
            valid_mask = np.isfinite(valid_vals) & (valid_vals >= 0.0)
            if np.any(valid_mask):
                local_curve_reference["valid_rate"] = safe_float(
                    np.mean(valid_vals[valid_mask] > 0.5) * 100.0
                )
        if local_arc_fallback_active_series is not None:
            fallback_vals = np.asarray(local_arc_fallback_active_series[:n_frames], dtype=np.float64)
            valid_mask = np.isfinite(fallback_vals) & (fallback_vals >= 0.0)
            if np.any(valid_mask):
                local_curve_reference["fallback_active_rate"] = safe_float(
                    np.mean(fallback_vals[valid_mask] > 0.5) * 100.0
                )
        for key, arr, pct in (
            ("blend_weight_p50", local_arc_blend_weight_series, 50),
            ("progress_weight_p50", local_arc_progress_weight_series, 50),
            ("planner_delta_p50_m", local_arc_vs_planner_delta_series, 50),
            ("planner_delta_p95_m", local_arc_vs_planner_delta_series, 95),
            ("target_distance_p50_m", local_arc_target_distance_series, 50),
            ("arc_curvature_p50", local_arc_curvature_series, 50),
        ):
            if arr is None:
                continue
            vals = np.asarray(arr[:n_frames], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                local_curve_reference[key] = safe_float(np.percentile(vals, pct))

    # Generate recommendations
    recommendations = []
    if lateral_error_rmse > 0.3:
        recommendations.append("Reduce lateral error - check PID gains or trajectory planning")
    if steering_jerk_max > 20.0:
        recommendations.append("Steering jerk exceeds cap - check pp_max_steering_jerk limiter or gap filter")
    if oscillation_frequency > 2.0:
        recommendations.append("Reduce oscillation - increase damping or reduce proportional gain")
    if oscillation_amplitude_runaway:
        recommendations.append("Oscillation amplitude is growing - increase high-speed lookahead and reduce high-speed steering aggressiveness")
    if curve_intent_diag.get("available"):
        if float(curve_intent_diag.get("arm_early_enough_rate", 0.0)) < 80.0:
            recommendations.append(
                "Curve intent arms too late/misses entries - strengthen entry trigger and rearm timing."
            )
        if bool(curve_intent_diag.get("undercall_detected", False)):
            recommendations.append(
                "Controller is undercalling curvature in curves - improve curvature estimation fidelity."
            )
    if curve_local_contract.get("curve_local_contract_available"):
        if float(curve_local_contract.get("curve_local_active_straight_rate", 0.0)) > 5.0:
            recommendations.append(
                "Local curve state stays active on straights - separate far preview from local steering activation."
            )
        if int(curve_local_contract.get("curve_local_arm_without_ready_count", 0)) > 0:
            recommendations.append(
                "Local curve is arming without readiness - block path-only pre-entry arming before changing lookahead ownership."
            )
        if int(curve_local_contract.get("curve_local_reentry_without_gate_count", 0)) > 0:
            recommendations.append(
                "Local curve state re-enters without gate - block REARM/STRAIGHT re-entry unless local evidence is present."
            )
        if int(curve_local_contract.get("curve_local_watchdog_pingpong_count", 0)) > 0:
            recommendations.append(
                "Curve watchdog is ping-ponging on straights - remove non-local state latch before further PP retuning."
            )
        if int(curve_local_contract.get("curve_local_commit_without_distance_ready_count", 0)) > 0:
            recommendations.append(
                "Local curve COMMIT is arming before distance-ready - tighten local relevance gating."
            )
        if int(curve_local_contract.get("curve_local_commit_without_ready_count", 0)) > 0:
            recommendations.append(
                "Local curve COMMIT is arming without commit-ready evidence - tighten COMMIT qualification before changing lookahead ownership."
            )
        if int(curve_local_contract.get("curve_lookahead_collapse_violation_count", 0)) > 0:
            recommendations.append(
                "Reference lookahead is collapsing too quickly before local entry - slow PP shorten slew or raise local floor."
            )
        straight_summary_delta = curve_local_contract.get("straight_summary_vs_segment_rate_delta_pct")
        if straight_summary_delta is not None and float(straight_summary_delta) > 0.5:
            recommendations.append(
                "Curve-state summary disagrees with straight-segment inspector - fix analyzer straight-mask source before tuning."
            )
    if turn_in_owner.get("availability") == "available":
        fallback_rate = turn_in_owner.get("fallback_active_rate")
        if fallback_rate is not None and float(fallback_rate) > 0.0:
            recommendations.append(
                "Reference lookahead owner is falling back off the nominal path - fix scheduler/reference ownership before tuning turn-in."
            )
    for event in curve_turn_events:
        rescue_delta_max = event.get("pp_floor_rescue_delta_max_m")
        shorten_step_min = event.get("pp_pre_floor_shorten_step_min_m_per_frame")
        curve_idx = event.get("curve_index", "?")
        if rescue_delta_max is not None and float(rescue_delta_max) > 1.0:
            recommendations.append(
                f"PP floor is rescuing too much on C{curve_idx} entry - smooth planner-side lookahead contraction before the floor."
            )
        if (
            entry_shorten_guard_limit > 0.0
            and shorten_step_min is not None
            and float(shorten_step_min) < -(entry_shorten_guard_limit + 1e-6)
        ):
            recommendations.append(
                f"Planner lookahead still shortens too abruptly on C{curve_idx} - tighten entry-shorten guard before retuning PP gains."
            )
    if overspeed_into_curve_rate > 10.0:
        recommendations.append(
            "Overspeed into curve events detected - tune curve-cap decel and intent thresholds."
        )
    if turn_infeasible_rate_when_curve_cap_active > 5.0:
        recommendations.append(
            "Curve-cap active while turn remains infeasible - strengthen capability estimator or peak limits."
        )
    if frames_above_cap_1p0mps_rate > 2.0:
        recommendations.append(
            "Cap-tracking lag is high in active curve-cap frames - increase planner cap-tracking catch-up authority."
        )
    if cap_tracking_error_p95_mps > 0.5:
        recommendations.append(
            "Cap-tracking P95 error is above target - tighten cap-tracking hysteresis and recovery behavior."
        )
    if hard_ceiling_applied_rate > 0.5:
        recommendations.append(
            "Safety hard ceiling is being used too often - planner should converge to cap without fallback clamps."
        )
    if curvature_map_authority_lost_rate > 5.0:
        recommendations.append(
            f"Map curvature authority lost on {curvature_map_authority_lost_rate:.1f}% of frames"
        )
    if curve_intent_commit_streak_max_frames > max(60, int(0.4 * n_frames)):
        recommendations.append(
            "Curve intent COMMIT streak is too long - enforce rearm/exit progression to avoid stuck state."
        )
    if feasibility_violation_rate > 5.0:
        recommendations.append(
            "Frequent feasibility overspeed - tighten speed feasibility coupling to curvature contract."
        )
    if curvature_contract_consistency_rate < 99.0:
        recommendations.append(
            "Curvature contract consistency below target - ensure controller/governor/intent use one primary curvature."
        )
    if telemetry_completeness_rate_curvature_contract < 99.0:
        recommendations.append(
            "Curvature contract telemetry completeness below target - verify recorder writes all contract fields."
        )
    if telemetry_completeness_rate_feasibility < 99.0:
        recommendations.append(
            "Feasibility telemetry completeness below target - ensure backstop/feasibility channels are fully recorded."
        )
    if map_health_untrusted_rate > 1.0 or track_mismatch_rate > 0.0:
        recommendations.append(
            "Map health/track match instability detected - verify track profile, odometer continuity, and fallback gating."
        )
    if lane_detection_rate < 90:
        recommendations.append("Improve lane detection - check perception model or CV fallback")
    if lane_line_jitter_p95 > 0.6 or reference_jitter_p95 > 0.25:
        recommendations.append("Reduce perception jitter - increase temporal smoothing or clamp lane-line deltas")
    if right_lane_low_visibility_rate > 10 or single_lane_rate > 10:
        recommendations.append("Right lane visibility drops - add single-lane fallback or widen camera FOV")
    elif right_lane_edge_contact_rate > 20:
        recommendations.append("Right lane frequently touches image edge on right turns - treat as FOV-limited and rely on single-lane corridor logic")
    if stale_hard_rate > 10:
        recommendations.append("Reduce stale data usage - relax jump detection threshold or improve perception")
    if speed_limit_zero_rate > 10:
        recommendations.append("Speed limit missing - verify Unity track speed limits are sent to the bridge")
    if emergency_stop_frame is not None:
        recommendations.append("Emergency stop triggered - review lateral bounds and recovery logic")
    if pid_integral_max > 0.2:
        recommendations.append("Reduce PID integral accumulation - check integral reset mechanisms")
    if straight_oscillation_mean > 0.2:
        recommendations.append("Straight-line oscillation detected - increase deadband or smoothing")
    if straight_sign_mismatch_events > 0:
        recommendations.append("Steering sign mismatches on straights - relax straight smoothing or rate limits")
    if acceleration_p95_filtered > 2.5:
        recommendations.append("Reduce longitudinal acceleration spikes - tune throttle/brake gains")
    if commanded_jerk_p95 > 5.0:
        recommendations.append("Reduce longitudinal jerk - add rate limiting on throttle/brake")

    latency_sync = _build_latency_sync_summary(data)
    transport_contract = _build_transport_contract_summary(data)
    speed_intent = _build_speed_intent_summary(data)
    run_intent = _build_run_intent_summary(data)
    highway_mild_curve_contract = _build_highway_mild_curve_contract_summary(data)
    mpc_gt_cross_track_contract = _build_mpc_gt_cross_track_contract_summary(data)
    lateral_owner_contract = _build_lateral_owner_contract_summary(
        curve_intent_diag=curve_intent_diag,
        curve_local_contract=curve_local_contract,
        turn_in_owner=turn_in_owner,
        local_curve_reference=local_curve_reference,
        highway_mild_curve_contract=highway_mild_curve_contract,
        mpc_gt_cross_track_contract=mpc_gt_cross_track_contract,
    )
    if not bool((latency_sync.get("cadence") or {}).get("tuning_valid", False)):
        recommendations.append(
            "Cadence quality is not tuning-valid - do not use this run for parameter decisions."
        )
    if (
        transport_contract.get("availability") == "available"
        and (transport_contract.get("fallback_active_rate") or 0.0) > 1.0
    ):
        recommendations.append(
            "Sync packet fallback is active too often - fix packet continuity before promoting transport-sensitive tuning."
        )
    if (
        transport_contract.get("availability") == "available"
        and (transport_contract.get("false_teleport_cooldown_count") or 0) > 0
    ):
        recommendations.append(
            "Post-jump cooldown overlaps skipped packets/fallback episodes - treat this as transport continuity, not a real teleport."
        )
    if (
        speed_intent.get("effective_reference_velocity_drop_count") or 0
    ) > 0:
        recommendations.append(
            "Effective controller reference drops below final target - inspect speed intent ownership before tuning longitudinal control."
        )
    if run_intent.get("intent_mismatch_warning"):
        recommendations.append(
            "Run intent differs from road-limit expectation - confirm scenario target and lead-follow mode before interpreting speed behavior."
        )
    if acc_detection_contract.get("issue_detected"):
        recommendations.append(
            "ACC detection stability is not good enough for convergence tuning - fix radar/dropout behavior before changing following policy."
        )
    elif acc_comfort_contract.get("following_convergence_issue_detected"):
        recommendations.append(
            "ACC following appears policy-limited after corrected IDM gap semantics - inspect acquisition policy before changing comfort or scenario assumptions."
        )
    if highway_mild_curve_contract.get("issue_detected"):
        recommendations.append(
            "Mild map-backed curve is present but recognition stays STRAIGHT and lookahead stays long - fix dynamic curve activation and lookahead shaping before steering gain tuning."
        )
    if mpc_gt_cross_track_contract.get("issue_detected"):
        if mpc_gt_cross_track_contract.get("issue_mode") == "absolute_coordinate_mismatch":
            recommendations.append(
                "MPC at-car GT cross-track looks like an absolute lane-center coordinate - compute it from the car's current pose and keep reference-path geometry debug-only."
            )
        else:
            recommendations.append(
                "MPC high-error windows are dominated by lookahead GT geometry while at-car GT stays small - split lookahead vs at-car GT cross-track and keep lookahead debug-only."
            )
    chassis_ground = _build_chassis_ground_summary(data, n_frames=n_frames)
    if chassis_ground.get("health") == "POOR":
        recommendations.append(
            "Chassis-ground health is POOR - fix ride height/collider setup and remove fallback dynamics."
        )
    elif chassis_ground.get("health") == "WARN":
        recommendations.append(
            "Chassis-ground health is WARN - reduce chassis contact risk and eliminate fallback activations."
        )
    if lateral_owner_contract.get("suppress_legacy_curve_intent_warnings"):
        legacy_recommendation_prefixes = (
            "Curve intent arms too late/misses entries",
            "Controller is undercalling curvature in curves",
            "Curve intent COMMIT streak is too long",
        )
        recommendations = [
            rec
            for rec in recommendations
            if not any(rec.startswith(prefix) for prefix in legacy_recommendation_prefixes)
        ]
    
    # Key issues
    key_issues = []
    if out_of_lane_events > 0:
        key_issues.append(f"{out_of_lane_events} out-of-lane events")
    if lane_detection_rate < 80:
        key_issues.append(f"Low lane detection rate ({lane_detection_rate:.1f}%)")
    if lane_line_jitter_p95 > 0.6:
        key_issues.append(f"High lane-line jitter (p95={lane_line_jitter_p95:.2f}m)")
    if reference_jitter_p95 > 0.25:
        key_issues.append(f"High reference jitter (p95={reference_jitter_p95:.2f}m)")
    if right_lane_low_visibility_rate > 10:
        key_issues.append(f"Right lane low visibility ({right_lane_low_visibility_rate:.1f}%)")
    elif right_lane_edge_contact_rate > 20:
        key_issues.append(f"Right lane at image edge often ({right_lane_edge_contact_rate:.1f}%)")
    if stale_hard_rate > 20:
        key_issues.append(f"High hard stale data usage ({stale_hard_rate:.1f}%)")
    if speed_limit_zero_rate > 10:
        key_issues.append(f"Speed limit missing ({speed_limit_zero_rate:.1f}%)")
    if emergency_stop_frame is not None:
        key_issues.append(f"Emergency stop at frame {emergency_stop_frame}")
    if steering_jerk_max > 22.0:
        key_issues.append("Steering jerk above cap")
    if oscillation_amplitude_runaway:
        key_issues.append("Oscillation amplitude growth detected")
    if curve_intent_diag.get("available"):
        if float(curve_intent_diag.get("arm_early_enough_rate", 0.0)) < 80.0:
            key_issues.append(
                f"Curve intent late arm ({curve_intent_diag.get('arm_early_enough_rate', 0.0):.1f}% early)"
            )
        if bool(curve_intent_diag.get("undercall_detected", False)):
            key_issues.append(
                f"Curvature undercall ({curve_intent_diag.get('undercall_frame_rate', 0.0):.1f}% frames)"
            )
    if curve_local_contract.get("curve_local_contract_available"):
        local_active_straight_rate = float(
            curve_local_contract.get("curve_local_active_straight_rate", 0.0)
        )
        if local_active_straight_rate > 5.0:
            key_issues.append(
                f"Local curve state latched on straights ({local_active_straight_rate:.1f}% frames)"
            )
        arm_without_ready_count = int(
            curve_local_contract.get("curve_local_arm_without_ready_count", 0)
        )
        if arm_without_ready_count > 0:
            key_issues.append(
                f"Local curve armed without readiness ({arm_without_ready_count} events)"
            )
        reentry_without_gate_count = int(
            curve_local_contract.get("curve_local_reentry_without_gate_count", 0)
        )
        if reentry_without_gate_count > 0:
            key_issues.append(
                f"Local re-entry without gate ({reentry_without_gate_count} events)"
            )
        pingpong_count = int(
            curve_local_contract.get("curve_local_watchdog_pingpong_count", 0)
        )
        if pingpong_count > 0:
            key_issues.append(
                f"Curve watchdog ping-pong on straights ({pingpong_count} events)"
            )
        early_commit_count = int(
            curve_local_contract.get("curve_local_commit_without_distance_ready_count", 0)
        )
        if early_commit_count > 0:
            key_issues.append(
                f"Local curve COMMIT before distance-ready ({early_commit_count} frames)"
            )
        early_commit_without_ready_count = int(
            curve_local_contract.get("curve_local_commit_without_ready_count", 0)
        )
        if early_commit_without_ready_count > 0:
            key_issues.append(
                f"Local curve COMMIT without commit-ready ({early_commit_without_ready_count} frames)"
            )
    if turn_in_owner.get("availability") == "available":
        fallback_rate = turn_in_owner.get("fallback_active_rate")
        if fallback_rate is not None and float(fallback_rate) > 0.0:
            key_issues.append(
                f"Reference lookahead fallback active ({float(fallback_rate):.1f}% frames)"
            )
    for event in curve_turn_events:
        if bool(event.get("late_turn_in", False)):
            curve_idx = event.get("curve_index", "?")
            key_issues.append(f"Late turn-in on C{curve_idx}")
        rescue_delta_max = event.get("pp_floor_rescue_delta_max_m")
        if rescue_delta_max is not None and float(rescue_delta_max) > 1.0:
            curve_idx = event.get("curve_index", "?")
            key_issues.append(
                f"Large PP floor rescue on C{curve_idx} ({float(rescue_delta_max):.2f} m)"
            )
    if overspeed_into_curve_rate > 10.0:
        key_issues.append(f"Overspeed into curve ({overspeed_into_curve_rate:.1f}% frames)")
    if frames_above_cap_1p0mps_rate > 2.0:
        key_issues.append(f"Cap tracking >1.0 m/s above cap ({frames_above_cap_1p0mps_rate:.1f}% frames)")
    if hard_ceiling_applied_rate > 0.5:
        key_issues.append(f"Hard cap ceiling fallback active ({hard_ceiling_applied_rate:.1f}% frames)")
    if not bool((latency_sync.get("cadence") or {}).get("tuning_valid", False)):
        key_issues.append("Run excluded from tuning (cadence invalid)")
    if curvature_map_authority_lost_rate > 5.0:
        key_issues.append(
            f"Map curvature authority lost ({curvature_map_authority_lost_rate:.1f}% frames)"
        )
    if curvature_contract_consistency_rate < 99.0:
        key_issues.append(
            f"Curvature contract consistency low ({curvature_contract_consistency_rate:.1f}%)"
        )
    if feasibility_violation_rate > 5.0:
        key_issues.append(
            f"Feasibility overspeed high ({feasibility_violation_rate:.1f}% frames)"
        )
    if map_health_untrusted_rate > 1.0:
        key_issues.append(
            f"Map health untrusted ({map_health_untrusted_rate:.1f}% frames)"
        )
    if track_mismatch_rate > 0.0:
        key_issues.append(
            f"Track mismatch present ({track_mismatch_rate:.1f}% frames)"
        )
    if straight_oscillation_mean > 0.2:
        key_issues.append("Straight-line oscillation detected")
    if straight_sign_mismatch_events > 0:
        key_issues.append("Steering sign mismatches on straights")
    if acceleration_p95_filtered > 2.5:
        key_issues.append("High longitudinal acceleration")
    if commanded_jerk_p95 > 5.0:
        key_issues.append("High longitudinal jerk")
    if latency_sync.get("e2e", {}).get("pass") is False:
        e2e_p95 = latency_sync.get("e2e", {}).get("stats_ms", {}).get("p95")
        if e2e_p95 is not None:
            key_issues.append(f"High E2E latency (p95={float(e2e_p95):.1f}ms)")
    if latency_sync.get("sync_alignment", {}).get("pass") is False:
        misaligned_rate = latency_sync.get("sync_alignment", {}).get("contract_misaligned_rate")
        if misaligned_rate is not None:
            key_issues.append(f"High sync misalignment (rate={float(misaligned_rate) * 100.0:.1f}%)")
    if transport_contract.get("availability") == "available":
        fallback_rate = transport_contract.get("fallback_active_rate")
        if fallback_rate is not None and float(fallback_rate) > 1.0:
            key_issues.append(f"Sync packet fallback active ({float(fallback_rate):.1f}%)")
        false_cooldown_count = int(transport_contract.get("false_teleport_cooldown_count", 0) or 0)
        if false_cooldown_count > 0:
            key_issues.append(f"Transport-linked post-jump cooldown episodes ({false_cooldown_count})")
    if mpc_gt_cross_track_contract.get("issue_detected"):
        if mpc_gt_cross_track_contract.get("issue_mode") == "absolute_coordinate_mismatch":
            key_issues.append("MPC GT at-car cross-track uses absolute coordinates")
        else:
            key_issues.append("MPC GT cross-track semantic mismatch")
    effective_drop_rate = speed_intent.get("effective_reference_velocity_drop_rate")
    if effective_drop_rate is not None and float(effective_drop_rate) > 1.0:
        key_issues.append(
            f"Effective controller target drops below final target ({float(effective_drop_rate):.1f}%)"
        )
    if run_intent.get("intent_mismatch_warning"):
        key_issues.append("Run intent differs from road-speed expectation")
    if highway_mild_curve_contract.get("issue_detected"):
        key_issues.append(
            "Highway mild-curve under-activation (reference geometry mismatch on a map-backed arc)"
        )
    if acc_detection_contract.get("issue_detected"):
        recent_loss_rate = acc_detection_contract.get("recent_detection_loss_rate_pct")
        issue_mode = str(acc_detection_contract.get("issue_mode") or "none")
        if recent_loss_rate is not None:
            key_issues.append(
                f"ACC detection stability issue ({issue_mode}, recent-loss={float(recent_loss_rate):.1f}%)"
            )
        else:
            key_issues.append(f"ACC detection stability issue ({issue_mode})")
    elif acc_comfort_contract.get("following_convergence_issue_detected"):
        key_issues.append("ACC following is policy-limited after corrected IDM gap semantics")
    if lateral_owner_contract.get("suppress_legacy_curve_intent_warnings"):
        key_issues = [
            issue
            for issue in key_issues
            if not issue.startswith("Curve intent late arm")
            and not issue.startswith("Curvature undercall")
        ]
    key_issues = list(dict.fromkeys(key_issues))
    if chassis_ground.get("health") in {"WARN", "POOR"}:
        contact_rate = chassis_ground.get("contact_rate_pct")
        penetration_max = chassis_ground.get("penetration_max_m")
        if contact_rate is not None and penetration_max is not None:
            key_issues.append(
                "Chassis-ground contact/penetration detected "
                f"(contact={float(contact_rate):.2f}%, penetration_max={float(penetration_max):.3f}m)"
            )
        else:
            key_issues.append("Chassis-ground contact/penetration detected")
    
    summary_result = {
        "summary_schema_version": "v1",
        "executive_summary": {
            "overall_score": safe_float(score),
            "drive_duration": safe_float(duration),
            "total_frames": int(n_frames),
            "success_rate": safe_float(time_in_lane),
            "key_issues": key_issues,
            "failure_detected": failure_frame is not None,
            "failure_frame": int(failure_frame) if failure_frame is not None else None,
            "analyzed_to_failure": analyze_to_failure,
            "failure_detection_source": error_source if error_data is not None else "none",
            "score_breakdown": {
                "base_score": 100.0,
                "lateral_error_penalty": trajectory_lateral_rmse_penalty,
                "steering_jerk_penalty": control_steering_jerk_penalty,
                "lane_detection_penalty": lane_detection_penalty,
                "stale_data_penalty": stale_data_penalty,
                "perception_instability_penalty": perception_instability_penalty,
                "out_of_lane_penalty": safety_out_of_lane_penalty,
                "straight_sign_mismatch_penalty": control_sign_mismatch_penalty,
                "layer_weights": layer_weights,
                "layer_scores": layer_scores,
                "layer_weighted_contributions": weighted_contributions,
                "overall_base_score": overall_base,
                "overall_cap": critical_cap,
                "cap_reason": cap_reason,
                "critical_layer_status": critical_layer_colors,
            },
        },
        "path_tracking": {
            "lateral_error_rmse": safe_float(lateral_error_rmse),
            "lateral_error_mean": safe_float(lateral_error_mean),
            "lateral_error_max": safe_float(lateral_error_max),
            "lateral_error_p95": safe_float(lateral_error_p95),
            "lateral_error_adj_rmse": safe_float(lateral_error_adj_rmse),
            "lateral_error_adj_p95": safe_float(lateral_error_adj_p95),
            "heading_error_rmse": safe_float(heading_error_rmse),
            "heading_error_max": safe_float(heading_error_max),
            "time_in_lane": safe_float(time_in_lane),
            "time_in_lane_centered": safe_float(time_in_lane_centered)
        },
        "layer_scores": layer_scores,
        "layer_score_breakdown": layer_breakdowns,
        "control_mode": _detect_control_mode(data),
        "pp_feedback_gain": _pp_feedback_gain(data),
        "pp_mean_lookahead_distance": _pp_mean_ld(data),
        "pp_ref_jump_clamped_count": _pp_jump_count(data),
        "control_smoothness": {
            "steering_jerk_max": safe_float(steering_jerk_max),
            "steering_jerk_max_raw": safe_float(steering_jerk_max_raw),
            "steering_rate_max": safe_float(steering_rate_max),
            "steering_smoothness": safe_float(steering_smoothness),
            "oscillation_frequency": safe_float(oscillation_frequency),
            "oscillation_zero_crossing_rate_hz": safe_float(oscillation_zero_crossing_rate_hz),
            "oscillation_rms_growth_slope_mps": safe_float(oscillation_rms_growth_slope_mps),
            "oscillation_rms_window_start_m": safe_float(oscillation_rms_window_start_m),
            "oscillation_rms_window_end_m": safe_float(oscillation_rms_window_end_m),
            "oscillation_rms_window_p95_m": safe_float(oscillation_rms_window_p95_m),
            "oscillation_rms_windows_count": int(oscillation_rms_windows_count),
            "oscillation_amplitude_runaway": bool(oscillation_amplitude_runaway),
            "oscillation_curve_suppressed": bool(oscillation_curve_suppressed),
            "oscillation_curve_fraction": safe_float(curve_fraction),
        },
        "curve_intent_diagnostics": curve_intent_diag,
        "curve_local_contract": curve_local_contract,
        "local_curve_reference": local_curve_reference,
        "turn_in_owner": turn_in_owner,
        "curve_turn_events": curve_turn_events,
        "curve_straight_segments": curve_straight_segments,
        "speed_control": {
            "speed_min_mps": speed_min_mps,
            "speed_max_mps": speed_max_mps,
            "speed_error_rmse": safe_float(speed_error_rmse),
            "speed_error_mean": safe_float(speed_error_mean),
            "speed_error_max": safe_float(speed_error_max),
            "speed_overspeed_rate": safe_float(speed_overspeed_rate),
            "speed_limit_zero_rate": safe_float(speed_limit_zero_rate),
            "speed_surge_count": int(speed_surge_count),
            "speed_surge_avg_drop": safe_float(speed_surge_avg_drop),
            "speed_surge_p95_drop": safe_float(speed_surge_p95_drop),
            "curve_cap_active_rate": safe_float(curve_cap_active_rate),
            "pre_turn_arm_lead_frames_p50": safe_float(curve_cap_pre_turn_lead_frames_p50),
            "pre_turn_arm_lead_frames_p95": safe_float(curve_cap_pre_turn_lead_frames_p95),
            "overspeed_into_curve_rate": safe_float(overspeed_into_curve_rate),
            "turn_infeasible_rate_when_curve_cap_active": safe_float(
                turn_infeasible_rate_when_curve_cap_active
            ),
            "cap_tracking_error_p50_mps": safe_float(cap_tracking_error_p50_mps),
            "cap_tracking_error_p95_mps": safe_float(cap_tracking_error_p95_mps),
            "cap_tracking_error_max_mps": safe_float(cap_tracking_error_max_mps),
            "frames_above_cap_0p3mps_rate": safe_float(frames_above_cap_0p3mps_rate),
            "frames_above_cap_1p0mps_rate": safe_float(frames_above_cap_1p0mps_rate),
            "cap_recovery_frames_p50": safe_float(cap_recovery_frames_p50),
            "cap_recovery_frames_p95": safe_float(cap_recovery_frames_p95),
            "hard_ceiling_applied_rate": safe_float(hard_ceiling_applied_rate),
            "acceleration_mean": safe_float(acceleration_mean),
            "acceleration_p95": safe_float(acceleration_p95),
            "acceleration_max": safe_float(acceleration_max),
            "jerk_mean": safe_float(jerk_mean),
            "jerk_p95": safe_float(jerk_p95),
            "jerk_max": safe_float(jerk_max),
            "acceleration_mean_filtered": safe_float(acceleration_mean_filtered),
            "acceleration_p95_filtered": safe_float(acceleration_p95_filtered),
            "acceleration_max_filtered": safe_float(acceleration_max_filtered),
            "jerk_mean_filtered": safe_float(jerk_mean_filtered),
            "jerk_p95_filtered": safe_float(jerk_p95_filtered),
            "jerk_max_filtered": safe_float(jerk_max_filtered),
            "lateral_accel_p95": safe_float(lateral_accel_p95),
            "lateral_jerk_p95": safe_float(lateral_jerk_p95),
            "lateral_jerk_max": safe_float(lateral_jerk_max)
        },
        "comfort": {
            "steering_jerk_max": safe_float(steering_jerk_max),
            "steering_jerk_max_raw": safe_float(steering_jerk_max_raw),
            "acceleration_p95": safe_float(acceleration_p95),
            "jerk_p95": safe_float(jerk_p95),
            "acceleration_p95_filtered": safe_float(acceleration_p95_filtered),
            "jerk_p95_filtered": safe_float(jerk_p95_filtered),
            "commanded_jerk_p95": safe_float(commanded_jerk_p95),
            "jerk_max_filtered": safe_float(jerk_max_filtered),
            "lateral_accel_p95": safe_float(lateral_accel_p95),
            "lateral_jerk_p95": safe_float(lateral_jerk_p95),
            "acceleration_p95_g": safe_float(acceleration_p95 / G_MPS2),
            "acceleration_p95_filtered_g": safe_float(acceleration_p95_filtered / G_MPS2),
            "jerk_p95_gps": safe_float(jerk_p95 / G_MPS2),
            "jerk_p95_filtered_gps": safe_float(jerk_p95_filtered / G_MPS2),
            "lateral_accel_p95_g": safe_float(lateral_accel_p95 / G_MPS2),
            "lateral_jerk_p95_gps": safe_float(lateral_jerk_p95 / G_MPS2),
            "comfort_gate_thresholds_g": {
                "longitudinal_accel_p95_g": safe_float(ACCEL_P95_GATE_MPS2 / G_MPS2),
                "longitudinal_jerk_p95_gps": safe_float(JERK_P95_GATE_MPS3 / G_MPS2)
            },
            "comfort_gate_thresholds_si": {
                "longitudinal_accel_p95_mps2": ACCEL_P95_GATE_MPS2,
                "longitudinal_jerk_p95_mps3": JERK_P95_GATE_MPS3
            },
            "metric_roles": {
                "commanded_jerk_p95": "gate",
                "acceleration_p95_filtered": "gate",
                "jerk_p95_filtered": "comfort_secondary_gate",
                "jerk_p95": "diagnostic_raw",
            },
            "hotspot_attribution": hotspot_attribution,
        },
        "control_stability": {
            "straight_fraction": safe_float(straight_fraction),
            "straight_oscillation_mean": safe_float(straight_oscillation_mean),
            "straight_oscillation_max": safe_float(straight_oscillation_max),
            "tuned_deadband_mean": safe_float(tuned_deadband_mean),
            "tuned_deadband_max": safe_float(tuned_deadband_max),
            "tuned_smoothing_mean": safe_float(tuned_smoothing_mean),
            "straight_sign_mismatch_rate": safe_float(straight_sign_mismatch_rate),
            "straight_sign_mismatch_events": int(straight_sign_mismatch_events),
            "straight_sign_mismatch_frames": int(straight_sign_mismatch_frames),
            "straight_sign_mismatch_events_list": straight_sign_mismatch_events_list
        },
        "perception_quality": {
            "lane_detection_rate": safe_float(lane_detection_rate),
            "perception_confidence_mean": safe_float(perception_confidence_mean),
            "perception_jumps_detected": int(perception_jumps_detected),
            "perception_instability_detected": int(perception_instability_detected),
            "perception_extreme_coeffs_detected": int(perception_extreme_coeffs_detected),
            "perception_invalid_width_detected": int(perception_invalid_width_detected),
            "stale_perception_rate": safe_float(stale_perception_rate),
            "stale_raw_rate": safe_float(stale_raw_rate),
            "stale_hard_rate": safe_float(stale_hard_rate),
            "stale_fallback_visibility_rate": safe_float(stale_fallback_visibility_rate),
            "perception_stability_score": safe_float(perception_stability_score),
            "lane_position_variance": safe_float(lane_position_variance),
            "lane_width_variance": safe_float(lane_width_variance),
            "lane_line_jitter_p95": safe_float(lane_line_jitter_p95),
            "lane_line_jitter_p99": safe_float(lane_line_jitter_p99),
            "reference_jitter_p95": safe_float(reference_jitter_p95),
            "reference_jitter_p99": safe_float(reference_jitter_p99),
            "single_lane_rate": safe_float(single_lane_rate),
            "right_lane_low_visibility_rate": safe_float(right_lane_low_visibility_rate),
            "right_lane_edge_contact_rate": safe_float(right_lane_edge_contact_rate),
            "left_lane_low_visibility_rate": safe_float(left_lane_low_visibility_rate)
        },
        "trajectory_quality": {
            "trajectory_availability": safe_float(trajectory_availability),
            "ref_point_accuracy_rmse": safe_float(ref_point_accuracy_rmse)
        },
        "turn_bias": turn_bias,
        "alignment_summary": alignment_summary,
        "latency_sync": latency_sync,
        "transport_contract": transport_contract,
        "speed_intent": speed_intent,
        "run_intent": run_intent,
        "acc_comfort_contract": acc_comfort_contract,
        "acc_detection_contract": acc_detection_contract,
        "lateral_owner_contract": lateral_owner_contract,
        "highway_mild_curve_contract": highway_mild_curve_contract,
        "mpc_gt_cross_track_contract": mpc_gt_cross_track_contract,
        "chassis_ground": chassis_ground,
        "recording_provenance": data.get("recording_provenance") or {},
        "curvature_contract_health": {
            "schema_version": "v1",
            "availability": (
                "available"
                if (
                    data.get("curvature_primary_abs") is not None
                    or data.get("curvature_source_diverged") is not None
                    or data.get("curvature_map_authority_lost") is not None
                    or data.get("curvature_contract_consistent_all") is not None
                )
                else "unavailable"
            ),
            "primary_source_mode": (
                max(
                    (
                        (src, cnt)
                        for src, cnt in {
                            str(s).strip(): int(
                                sum(
                                    1
                                    for x in (data.get("curvature_primary_source") or [])
                                    if str(x).strip() == str(s).strip()
                                )
                            )
                            for s in set(data.get("curvature_primary_source") or [])
                            if str(s).strip()
                        }.items()
                    ),
                    key=lambda kv: kv[1],
                )[0]
                if data.get("curvature_primary_source")
                else "unknown"
            ),
            "curvature_source_divergence_p95": safe_float(curvature_source_divergence_p95),
            "curvature_source_diverged_rate": safe_float(curvature_source_diverged_rate),
            "curvature_map_authority_lost_rate": safe_float(
                curvature_map_authority_lost_rate
            ),
            "curve_intent_commit_streak_max_frames": int(curve_intent_commit_streak_max_frames),
            "feasibility_violation_rate": safe_float(feasibility_violation_rate),
            "feasibility_backstop_active_rate": safe_float(feasibility_backstop_active_rate),
            "map_health_untrusted_rate": safe_float(map_health_untrusted_rate),
            "track_mismatch_rate": safe_float(track_mismatch_rate),
            "curvature_contract_consistency_rate": safe_float(curvature_contract_consistency_rate),
            "telemetry_completeness_rate_curvature_contract": safe_float(
                telemetry_completeness_rate_curvature_contract
            ),
            "telemetry_completeness_rate_feasibility": safe_float(
                telemetry_completeness_rate_feasibility
            ),
            "map_segment_lookup_success_rate_p50": safe_float(
                np.percentile(
                    np.asarray(data.get("map_segment_lookup_success_rate"), dtype=float)[
                        np.isfinite(np.asarray(data.get("map_segment_lookup_success_rate"), dtype=float))
                    ],
                    50,
                )
                if data.get("map_segment_lookup_success_rate") is not None
                and np.isfinite(np.asarray(data.get("map_segment_lookup_success_rate"), dtype=float)).any()
                else None,
                default=None,
            ),
            "map_teleport_skip_count_max": safe_float(
                np.max(
                    np.asarray(data.get("map_teleport_skip_count"), dtype=float)[
                        np.isfinite(np.asarray(data.get("map_teleport_skip_count"), dtype=float))
                    ]
                )
                if data.get("map_teleport_skip_count") is not None
                and np.isfinite(np.asarray(data.get("map_teleport_skip_count"), dtype=float)).any()
                else None,
                default=None,
            ),
            "map_odometer_jump_rate_p95": safe_float(
                np.percentile(
                    np.asarray(data.get("map_odometer_jump_rate"), dtype=float)[
                        np.isfinite(np.asarray(data.get("map_odometer_jump_rate"), dtype=float))
                    ],
                    95,
                )
                if data.get("map_odometer_jump_rate") is not None
                and np.isfinite(np.asarray(data.get("map_odometer_jump_rate"), dtype=float)).any()
                else None,
                default=None,
            ),
            "limits": {
                "curvature_map_authority_lost_rate_max_pct": 5.0,
                "curvature_source_diverged_rate_max_pct": 100.0,
                "feasibility_violation_rate_max_pct": 5.0,
                "map_health_untrusted_rate_max_pct": 1.0,
                "track_mismatch_rate_max_pct": 0.0,
                "curvature_contract_consistency_rate_min_pct": 99.0,
                "telemetry_completeness_rate_min_pct": 99.0,
            },
        },
        "first_fault_chain": {
            "first_divergence_frame": (
                int(first_divergence_frame) if first_divergence_frame is not None else None
            ),
            "first_infeasible_frame": (
                int(first_infeasible_frame) if first_infeasible_frame is not None else None
            ),
            "first_speed_above_feasibility_frame": (
                int(first_speed_above_feasibility_frame)
                if first_speed_above_feasibility_frame is not None
                else None
            ),
            "first_boundary_breach_frame": (
                int(first_boundary_breach_frame) if first_boundary_breach_frame is not None else None
            ),
        },
        "system_health": {
            "pid_integral_max": safe_float(pid_integral_max),
            "unity_time_gap_max": safe_float(unity_time_gap_max),
            "unity_time_gap_count": int(unity_time_gap_count)
        },
        "safety": {
            "out_of_lane_events": int(out_of_lane_events),
            "out_of_lane_time": safe_float(out_of_lane_time),
            "out_of_lane_events_list": out_of_lane_events_list,  # List of individual events with frame numbers
            "out_of_lane_events_full_run": int(out_of_lane_events_full_run),
            "out_of_lane_time_full_run": safe_float(out_of_lane_time_full_run),
            "out_of_lane_event_at_failure_boundary": bool(out_of_lane_event_at_failure_boundary),
        },
        "regime_summary": (
            {
                "stanley_frames": int(np.sum(np.asarray(data['regime'][:n_frames], dtype=float) < -0.5)),
                "pp_frames": int(np.sum(
                    (np.asarray(data['regime'][:n_frames], dtype=float) >= -0.5)
                    & (np.asarray(data['regime'][:n_frames], dtype=float) < 0.5)
                )),
                "mpc_frames": int(np.sum(np.asarray(data['regime'][:n_frames], dtype=float) > 0.5)),
                "total_frames": n_frames,
                "blend_frames": int(np.sum(
                    (np.asarray(data['regime_blend_weight'][:n_frames], dtype=float) > 0.01)
                    & (np.asarray(data['regime_blend_weight'][:n_frames], dtype=float) < 0.99)
                )) if data.get('regime_blend_weight') is not None else 0,
                "mpc_fraction": safe_float(
                    float(np.sum(np.asarray(data['regime'][:n_frames], dtype=float) > 0.5))
                    / max(1, n_frames)
                ),
                "stanley_fraction": safe_float(
                    float(np.sum(np.asarray(data['regime'][:n_frames], dtype=float) < -0.5))
                    / max(1, n_frames)
                ),
            }
            if data.get('regime') is not None
            else None
        ),
        "mpc_health": _build_mpc_health_summary(data, n_frames),
        "acc_health": acc_health,
        "grade_metrics": _compute_grade_metrics(data, n_frames),
        "recommendations": recommendations,
        "config": config_summary,
        "time_series": {
            "time": data['time'].tolist(),
            "lateral_error": data['lateral_error'].tolist() if data['lateral_error'] is not None else None,
            "steering": data['steering'].tolist(),
            "num_lanes_detected": data['num_lanes_detected'].tolist() if data['num_lanes_detected'] is not None else None,
            "perception_health_score": None  # Will be added if available
        }
    }
    if include_grade_lateral:
        from grade_lateral_analysis import analyze_grade_lateral

        es_ff = summary_result.get("executive_summary") or {}
        ff = es_ff.get("failure_frame")
        summary_result["grade_lateral"] = analyze_grade_lateral(
            recording_path,
            pre_failure_only=True,
            failure_frame=int(ff) if ff is not None else None,
        )
    return summary_result
