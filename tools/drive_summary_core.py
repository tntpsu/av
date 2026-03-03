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

from trajectory.utils import smooth_curvature_distance

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
    mpc_mask = regime_arr >= 1
    pp_frames = int(np.sum(~mpc_mask))
    mpc_frames = int(np.sum(mpc_mask))
    if mpc_frames == 0:
        return {
            "total_frames": total,
            "pp_frames": pp_frames,
            "pp_rate": 1.0,
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

    return {
        "total_frames": total,
        "pp_frames": pp_frames,
        "pp_rate": round(pp_frames / total, 4),
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

    ctrl_cfg = config.get("control", {}).get("longitudinal", {})
    max_accel_cfg = float(ctrl_cfg.get("max_accel", 2.5))
    max_decel_cfg = float(ctrl_cfg.get("max_decel", 3.0))
    command_accel = None
    if throttle is not None and brake is not None:
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


def _detect_control_mode(data):
    regime = data.get('regime')
    if regime is not None:
        regime_arr = np.asarray(regime, dtype=float)
        mpc_rate = float(np.mean(regime_arr >= 1))
        if mpc_rate > 0.5:
            return 'mpc'
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


def analyze_recording_summary(recording_path: Path, analyze_to_failure: bool = False, h5_file=None) -> Dict:
    """
    Analyze a recording and return summary metrics.
    
    Args:
        recording_path: Path to HDF5 recording file
        analyze_to_failure: If True, only analyze up to the point where car went out of lane and stayed out
        
    Returns:
        Dictionary with summary metrics and recommendations
    """
    data = {}
    
    try:
        with h5py.File(recording_path, 'r') as f:
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
                data['speed'] = np.array(f['vehicle/speed'][:]) if 'vehicle/speed' in f else None
                data['speed_limit'] = (
                    np.array(f['vehicle/speed_limit'][:]) if 'vehicle/speed_limit' in f else None
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
            data['mpc_feasible'] = (
                np.array(f['control/mpc_feasible'][:])
                if 'control/mpc_feasible' in f else None
            )
            data['mpc_solve_time_ms'] = (
                np.array(f['control/mpc_solve_time_ms'][:])
                if 'control/mpc_solve_time_ms' in f else None
            )
            data['mpc_fallback_active'] = (
                np.array(f['control/mpc_fallback_active'][:])
                if 'control/mpc_fallback_active' in f else None
            )
            data['mpc_consecutive_failures'] = (
                np.array(f['control/mpc_consecutive_failures'][:])
                if 'control/mpc_consecutive_failures' in f else None
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
        out_of_lane_mask = np.abs(error_data) > 0.5
    elif data['lateral_error'] is not None:
        error_data = data['lateral_error']
        error_source = "perception"
        out_of_lane_mask = np.abs(error_data) > 0.5
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
        out_of_lane_threshold = 0.5  # meters (half a meter is a reasonable "out of lane" threshold)
        catastrophic_threshold = 2.0  # meters (catastrophic failure - car is way out)
        min_consecutive_out = 10  # frames (must stay out for this long to be considered "stayed out")
        
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
    
    heading_error_rmse = safe_float(np.sqrt(np.mean(data['heading_error']**2)) if data['heading_error'] is not None and len(data['heading_error']) > 0 else 0.0)
    heading_error_max = safe_float(np.max(np.abs(data['heading_error'])) if data['heading_error'] is not None and len(data['heading_error']) > 0 else 0.0)
    
    # Time in Lane:
    # Primary: Use ground truth boundaries if available (car at x=0 is in lane if left <= 0 <= right)
    # Secondary: Centeredness within ±0.5m (for tuning)
    time_in_lane_centered = safe_float(
        np.sum(np.abs(data['lateral_error']) < 0.5) / n_frames * 100
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
        min_consecutive_out = 10
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
        min_consecutive_out = 10
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

    trajectory_lateral_rmse_penalty = safe_float(min(30, lateral_error_rmse * 50))
    trajectory_lateral_p95_penalty = safe_float(min(20, max(0.0, lateral_error_p95 - 0.40) * 35.0))
    trajectory_heading_penalty = safe_float(min(20, max(0.0, np.degrees(heading_error_rmse) - 10.0) * 2.5))

    # Penalty only above the pp_max_steering_jerk cap (18.0 normalized/s²).
    # Operating at-cap is in-spec; exceeding it signals a limiter failure.
    control_steering_jerk_penalty = safe_float(min(20, max(0.0, (steering_jerk_max - 18.0) * 2.0)))
    control_oscillation_penalty = safe_float(
        min(15, max(0.0, oscillation_frequency - 1.0) * 7.0)
    )
    control_sign_mismatch_penalty = safe_float(
        min(12, straight_sign_mismatch_rate * 0.2) if straight_sign_mismatch_rate > 5.0 else 0.0
    )

    # S1-M39: Gates 3.0 m/s², 6.0 m/s³ (0.31g, 0.61 g/s)
    # Scoring uses filtered values to avoid penalising velocity quantisation noise.
    # Raw values are retained in the output dict for diagnostic reference.
    accel_gate_g = 3.0 / G_MPS2
    jerk_gate_gps = 6.0 / G_MPS2
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
        curvature_eps = 1e-6
        abs_curv = np.abs(curvature_arr)
        v_max = np.where(abs_curv > curvature_eps, np.sqrt(2.45 / abs_curv), np.inf)
        speed_feasibility_violation_frames = int(np.sum(speed_arr > v_max))
        if speed_feasibility_violation_frames > 5:
            capped = min(speed_feasibility_violation_frames, 30)
            signal_integrity_speed_feasibility_penalty = safe_float(
                min(15.0, (capped - 5) * 0.6)
            )

    # 9. CURVE INTENT DIAGNOSTICS
    curve_intent_diag = {
        "available": False,
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
                    "name": "Lateral Error RMSE",
                    "value": trajectory_lateral_rmse_penalty,
                    "limit": "<=0.20m",
                },
                {
                    "name": "Lateral Error P95",
                    "value": trajectory_lateral_p95_penalty,
                    "limit": "<=0.40m",
                },
                {
                    "name": "Heading Error RMSE",
                    "value": trajectory_heading_penalty,
                    "limit": "<=10deg",
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
    if not bool((latency_sync.get("cadence") or {}).get("tuning_valid", False)):
        recommendations.append(
            "Cadence quality is not tuning-valid - do not use this run for parameter decisions."
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
    
    return {
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
        },
        "curve_intent_diagnostics": curve_intent_diag,
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
                "longitudinal_accel_p95_g": safe_float(3.0 / G_MPS2),
                "longitudinal_jerk_p95_gps": safe_float(6.0 / G_MPS2)
            },
            "comfort_gate_thresholds_si": {
                "longitudinal_accel_p95_mps2": 3.0,
                "longitudinal_jerk_p95_mps3": 6.0
            },
            "metric_roles": {
                "commanded_jerk_p95": "gate",
                "acceleration_p95_filtered": "gate",
                "jerk_p95_filtered": "diagnostic",
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
        "chassis_ground": chassis_ground,
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
                "pp_frames": int(np.sum(np.asarray(data['regime'][:n_frames], dtype=float) < 0.5)),
                "mpc_frames": int(np.sum(np.asarray(data['regime'][:n_frames], dtype=float) > 0.5)),
                "blend_frames": int(np.sum(
                    (np.asarray(data['regime_blend_weight'][:n_frames], dtype=float) > 0.01)
                    & (np.asarray(data['regime_blend_weight'][:n_frames], dtype=float) < 0.99)
                )) if data.get('regime_blend_weight') is not None else 0,
                "mpc_fraction": safe_float(
                    float(np.sum(np.asarray(data['regime'][:n_frames], dtype=float) > 0.5))
                    / max(1, n_frames)
                ),
            }
            if data.get('regime') is not None
            else None
        ),
        "mpc_health": _build_mpc_health_summary(data, n_frames),
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
