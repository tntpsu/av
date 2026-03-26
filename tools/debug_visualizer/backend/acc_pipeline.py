"""ACC Pipeline diagnostic backend — Step 5.

Exposes four analysis cards for the PhilViz "ACC Pipeline" tab:
  Card 1 — Radar Health (detection rate, SNR, dropout events, raw vs filtered gap)
  Card 2 — Gap Tracking (gap vs target gap, gap error timeline, IDM accel output)
  Card 3 — Safety Timeline (TTC distribution, emergency brake events, ACC state band)
  Card 4 — Worst Frame Inspector (top-N frames by |acc_gap_error_m|)

Returns None for recordings with acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE (safe for
all existing tracks — satisfies the roll-back contract).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

# Add tools/ to path for scoring_registry
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scoring_registry import (
    ACC_TTC_CRITICAL_S,
    ACC_TTC_WARNING_S,
    ACC_TTC_MIN_GATE_S,
    ACC_NEAR_MISS_GAP_M,
    ACC_GAP_RMSE_GATE_M,
    ACC_DETECTION_RATE_GATE,
    ACC_JERK_P95_GATE_MPS3,
    ACC_MIN_ACTIVE_FRAME_RATE,
)


def _safe_float(v, default=0.0):
    if v is None:
        return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _pct(arr, p):
    """Return percentile of finite values in arr."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0
    return float(np.percentile(valid, p))


def get_acc_pipeline_data(recording_path: Path) -> Optional[Dict]:
    """
    Load ACC pipeline data from HDF5 and return JSON-ready dict.
    Returns None if ACC was not active or fields not present.
    """
    try:
        with h5py.File(recording_path, "r") as f:
            if "vehicle/acc_active" not in f:
                return None

            n = len(f["vehicle/acc_active"])

            def arr(key):
                if key not in f:
                    return None
                return np.array(f[key][:n], dtype=float)

            acc_active = arr("vehicle/acc_active")
            if acc_active is None:
                return None
            acc_active_pct = float(np.mean(acc_active > 0.5))
            if acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE:
                return None

            acc_mask = acc_active > 0.5
            timestamps = arr("vehicle/timestamps") or np.arange(n, dtype=float) / 30.0

            detected = arr("vehicle/radar_fwd_detected") or np.zeros(n)
            distance = arr("vehicle/radar_fwd_distance_m")
            range_rate = arr("vehicle/radar_fwd_range_rate_mps")
            snr = arr("vehicle/radar_fwd_snr")
            ttc = arr("vehicle/acc_ttc_s")
            gap_error = arr("vehicle/acc_gap_error_m")
            target_gap = arr("vehicle/acc_target_gap_m")
            speed = arr("vehicle/speed")
            lead_collision_override = arr("vehicle/lead_collision_override_active")
            acc_target_speed = arr("vehicle/acc_target_speed_mps")
            final_target = arr("control/final_longitudinal_target_mps")
            ref_effective = arr("control/reference_velocity_effective")
            acc_state_raw = f["vehicle/acc_state_code"][:] if "vehicle/acc_state_code" in f else None
            owner_raw = f["control/final_longitudinal_owner_code"][:] if "control/final_longitudinal_owner_code" in f else None
            acc_state_mode = None
            if acc_state_raw is not None and len(acc_state_raw) > 0:
                acc_state_strings = [(s.decode("utf-8") if isinstance(s, bytes) else str(s)) for s in acc_state_raw]
                acc_state_mode = max(set(acc_state_strings), key=acc_state_strings.count)
            owner_mode = None
            if owner_raw is not None and len(owner_raw) > 0:
                owner_strings = [(s.decode("utf-8") if isinstance(s, bytes) else str(s)) for s in owner_raw]
                owner_mode = max(set(owner_strings), key=owner_strings.count)

            # ── Card 1 — Radar Health ─────────────────────────────────────────
            detection_rate = float(np.mean(detected > 0.5))
            snr_valid = snr[np.isfinite(snr)] if snr is not None else np.array([])
            snr_hist_counts, snr_hist_edges = (
                np.histogram(snr_valid, bins=20, range=(0.0, 1.0))
                if snr_valid.size > 0 else (np.zeros(20), np.linspace(0, 1, 21))
            )

            dropout_events: List[Dict] = []
            run_start_d = None
            for fi, nd in enumerate(detected < 0.5):
                if nd and run_start_d is None:
                    run_start_d = fi
                elif not nd and run_start_d is not None:
                    if fi - run_start_d >= 3:
                        dropout_events.append({
                            "start_frame": int(run_start_d),
                            "end_frame": int(fi - 1),
                            "duration_frames": int(fi - run_start_d),
                        })
                    run_start_d = None
            if run_start_d is not None and n - run_start_d >= 3:
                dropout_events.append({
                    "start_frame": int(run_start_d),
                    "end_frame": int(n - 1),
                    "duration_frames": int(n - run_start_d),
                })

            card1 = {
                "detection_rate": round(detection_rate, 4),
                "detection_gate_pass": detection_rate >= ACC_DETECTION_RATE_GATE,
                "detection_rate_threshold": ACC_DETECTION_RATE_GATE,
                "snr_p05": round(_pct(snr, 5) if snr is not None else 0.0, 3),
                "snr_p50": round(_pct(snr, 50) if snr is not None else 0.0, 3),
                "snr_p95": round(_pct(snr, 95) if snr is not None else 0.0, 3),
                "snr_histogram": {
                    "counts": snr_hist_counts.tolist(),
                    "edges": snr_hist_edges.tolist(),
                },
                "dropout_events": dropout_events[:20],
                "dropout_event_count": len(dropout_events),
                "raw_gap_series": (
                    [round(_safe_float(v), 2) for v in distance[:500]]
                    if distance is not None else []
                ),
                "time_series": [round(_safe_float(t), 3) for t in timestamps[:500]],
            }

            # ── Card 2 — Gap Tracking ─────────────────────────────────────────
            ge_acc = (
                gap_error[acc_mask & np.isfinite(gap_error)]
                if gap_error is not None else np.array([])
            )
            gap_rmse = float(np.sqrt(np.mean(ge_acc ** 2))) if ge_acc.size > 0 else 0.0

            card2 = {
                "gap_rmse_m": round(gap_rmse, 3),
                "gap_rmse_gate_pass": gap_rmse <= ACC_GAP_RMSE_GATE_M,
                "gap_rmse_threshold": ACC_GAP_RMSE_GATE_M,
                "gap_error_p05": round(_pct(ge_acc, 5) if ge_acc.size > 0 else 0.0, 3),
                "gap_error_p95": round(_pct(ge_acc, 95) if ge_acc.size > 0 else 0.0, 3),
                "gap_series": (
                    [round(_safe_float(v, 0.0), 2) for v in distance[:500]]
                    if distance is not None else []
                ),
                "target_gap_series": (
                    [round(_safe_float(v, 0.0), 2) for v in target_gap[:500]]
                    if target_gap is not None else []
                ),
                "gap_error_series": (
                    [round(_safe_float(v, 0.0), 3) for v in gap_error[:500]]
                    if gap_error is not None else []
                ),
                "acc_active_series": [int(v > 0.5) for v in acc_active[:500]],
                "time_series": [round(_safe_float(t), 3) for t in timestamps[:500]],
            }

            # ── Card 3 — Safety Timeline ──────────────────────────────────────
            ttc_acc = (
                ttc[acc_mask & np.isfinite(ttc)]
                if ttc is not None else np.array([])
            )
            ttc_min = float(np.min(ttc_acc)) if ttc_acc.size > 0 else 999.0
            ttc_p05 = float(np.percentile(ttc_acc, 5)) if ttc_acc.size > 0 else 999.0
            ttc_critical_pct = float(np.mean(ttc_acc < ACC_TTC_CRITICAL_S) * 100.0) if ttc_acc.size > 0 else 0.0
            ttc_warn_pct = float(np.mean(
                (ttc_acc >= ACC_TTC_CRITICAL_S) & (ttc_acc < ACC_TTC_WARNING_S)
            ) * 100.0) if ttc_acc.size > 0 else 0.0

            # Near-miss events
            nm_events = 0
            if distance is not None:
                run_nm = 0
                for nd in acc_mask & (distance > 0.0) & (distance < ACC_NEAR_MISS_GAP_M):
                    if nd:
                        run_nm += 1
                        if run_nm == 3:
                            nm_events += 1
                    else:
                        run_nm = 0

            collision_count = int(np.sum(distance < 0.0)) if distance is not None else 0

            card3 = {
                "ttc_min_s": round(ttc_min, 3),
                "ttc_p05_s": round(ttc_p05, 3),
                "ttc_min_gate_pass": ttc_min >= ACC_TTC_MIN_GATE_S,
                "ttc_p05_gate_pass": ttc_p05 >= ACC_TTC_MIN_GATE_S,
                "ttc_critical_threshold": ACC_TTC_CRITICAL_S,
                "ttc_warning_threshold": ACC_TTC_WARNING_S,
                "ttc_min_gate_threshold": ACC_TTC_MIN_GATE_S,
                "ttc_critical_pct": round(ttc_critical_pct, 2),
                "ttc_warning_pct": round(ttc_warn_pct, 2),
                "near_miss_events": nm_events,
                "lead_collision_override_rate_pct": round(float(np.mean(lead_collision_override > 0.5) * 100.0), 2) if lead_collision_override is not None else 0.0,
                "acc_state_mode": acc_state_mode,
                "final_longitudinal_owner_mode": owner_mode,
                "near_miss_gap_m": ACC_NEAR_MISS_GAP_M,
                "collision_frames": collision_count,
                "ttc_series": (
                    [round(min(_safe_float(v, 999.0), 10.0), 2) for v in ttc[:500]]
                    if ttc is not None else []
                ),
                "acc_active_series": [int(v > 0.5) for v in acc_active[:500]],
                "time_series": [round(_safe_float(t), 3) for t in timestamps[:500]],
            }

            # ── Card 4 — Worst Frame Inspector ────────────────────────────────
            worst_frames: List[Dict] = []
            if gap_error is not None:
                abs_ge = np.abs(gap_error)
                abs_ge[~acc_mask] = 0.0
                abs_ge[~np.isfinite(abs_ge)] = 0.0
                top_idx = np.argsort(abs_ge)[-5:][::-1]
                for fi in top_idx:
                    if abs_ge[fi] < 1e-6:
                        continue
                    worst_frames.append({
                        "frame": int(fi),
                        "timestamp": round(_safe_float(timestamps[fi]), 3),
                        "gap_error_m": round(_safe_float(gap_error[fi]), 3),
                        "gap_m": round(_safe_float(distance[fi] if distance is not None else None), 2),
                        "ttc_s": round(min(_safe_float(ttc[fi] if ttc is not None else None, 999.0), 999.0), 2),
                        "range_rate_mps": round(_safe_float(range_rate[fi] if range_rate is not None else None), 2),
                        "speed_mps": round(_safe_float(speed[fi] if speed is not None else None), 1),
                        "acc_active": int(acc_active[fi] > 0.5),
                        "acc_target_speed_mps": round(_safe_float(acc_target_speed[fi] if acc_target_speed is not None else None), 2),
                        "final_target_speed_mps": round(_safe_float(final_target[fi] if final_target is not None else None), 2),
                        "reference_velocity_effective": round(_safe_float(ref_effective[fi] if ref_effective is not None else None), 2),
                        "lead_collision_override_active": int(lead_collision_override[fi] > 0.5) if lead_collision_override is not None else 0,
                    })

            card4 = {"worst_frames": worst_frames}

    except Exception as exc:
        import traceback
        return {"error": str(exc), "traceback": traceback.format_exc()}

    return {
        "acc_active_pct": round(acc_active_pct * 100.0, 2),
        "n_frames": n,
        "card1_radar_health": card1,
        "card2_gap_tracking": card2,
        "card3_safety_timeline": card3,
        "card4_worst_frames": card4,
    }
