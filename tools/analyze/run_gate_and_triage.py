#!/usr/bin/env python3
"""
Run gate execution, analysis, and triage artifact generation in one pass.

Outputs a schema-versioned bundle at:
  data/reports/gates/<utc_ts>_<label>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.drive_summary_core import analyze_recording_summary

DIAGNOSTICS_IMPORT_ERROR: str | None = None
try:
    from tools.debug_visualizer.backend.issue_detector import detect_issues
    from tools.debug_visualizer.backend.diagnostics import analyze_trajectory_vs_steering
except Exception as exc:  # pragma: no cover - import failure path is environment-specific
    detect_issues = None
    analyze_trajectory_vs_steering = None
    DIAGNOSTICS_IMPORT_ERROR = str(exc)

SCHEMA_VERSION = "v1"
REPORTS_DIR = REPO_ROOT / "data" / "reports" / "gates"
CONFIG_PATH = REPO_ROOT / "config" / "av_stack_config.yaml"
MANDATORY_CONTRACT_FIELDS = (
    "schema_version",
    "git_sha",
    "config_hash",
    "matrix_hash",
    "recording_ids",
    "pass_fail",
    "regression_budget",
)


@dataclass
class RunArtifacts:
    recording_path: Path
    track_id: str
    recording_id: str
    summary: dict[str, Any]
    metrics: dict[str, Any]
    stage6: dict[str, Any]
    trigger_reasons: list[str]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def compute_matrix_hash(matrix_payload: dict[str, Any]) -> str:
    return stable_hash_payload(matrix_payload)


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _diagnostics_preflight(enabled: bool = True) -> dict[str, Any]:
    preflight = {
        "enabled": bool(enabled),
        "available": False,
        "error": None,
        "status": "disabled",
    }
    if not enabled:
        preflight["available"] = True
        return preflight
    if DIAGNOSTICS_IMPORT_ERROR:
        preflight["status"] = "degraded"
        preflight["error"] = DIAGNOSTICS_IMPORT_ERROR
        return preflight
    preflight["available"] = True
    preflight["status"] = "ok"
    return preflight


def _read_recording_metadata(path: Path) -> dict[str, Any]:
    try:
        with h5py.File(path, "r") as f:
            raw = f.attrs.get("metadata")
            if raw is None:
                return {}
            meta_str = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
            parsed = json.loads(meta_str)
            return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _infer_track_id(path: Path) -> str:
    metadata = _read_recording_metadata(path)
    provenance = metadata.get("recording_provenance", {})
    if isinstance(provenance, dict):
        track_id = provenance.get("track_id")
        if isinstance(track_id, str) and track_id.strip():
            return track_id.strip()
    name = path.name.lower()
    if "highway" in name:
        return "highway_65"
    if "s_loop" in name or "sloop" in name:
        return "s_loop"
    return "unknown"


def _recording_provenance(path: Path) -> dict[str, Any]:
    metadata = _read_recording_metadata(path)
    provenance = metadata.get("recording_provenance")
    return provenance if isinstance(provenance, dict) else {}


def _validate_generated_recording_provenance(
    recording_path: Path,
    *,
    expected_track_id: str,
    expected_start_t: float,
) -> None:
    provenance = _recording_provenance(recording_path)
    observed_track_id = str(provenance.get("track_id", "") or "").strip()
    if observed_track_id != expected_track_id:
        raise RuntimeError(
            f"Generated recording provenance track mismatch for {recording_path.name}: "
            f"expected '{expected_track_id}', observed '{observed_track_id or 'missing'}'"
        )
    observed_start_t = provenance.get("start_t")
    try:
        observed_start_t = float(observed_start_t)
    except (TypeError, ValueError):
        observed_start_t = None
    if observed_start_t is None or not math.isfinite(observed_start_t):
        raise RuntimeError(
            f"Generated recording provenance start_t missing/invalid for {recording_path.name}"
        )
    if abs(observed_start_t - float(expected_start_t)) > 1e-6:
        raise RuntimeError(
            f"Generated recording provenance start_t mismatch for {recording_path.name}: "
            f"expected {float(expected_start_t):.6f}, observed {observed_start_t:.6f}"
        )


def _extract_run_metrics(
    recording_path: Path,
    analyze_to_failure: bool,
    *,
    min_control_fps: float,
    max_unity_time_gap_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = analyze_recording_summary(recording_path, analyze_to_failure=analyze_to_failure)
    if not isinstance(summary, dict):
        summary = {"error": "invalid summary payload"}

    with h5py.File(recording_path, "r") as f:
        control_timestamps = (
            np.asarray(f["control/timestamps"][:], dtype=float)
            if "control/timestamps" in f
            else np.array([], dtype=float)
        )
        lateral_error = (
            np.asarray(f["control/lateral_error"][:], dtype=float)
            if "control/lateral_error" in f
            else np.array([], dtype=float)
        )
        emergency_stop = (
            np.asarray(f["control/emergency_stop"][:], dtype=float)
            if "control/emergency_stop" in f
            else np.array([], dtype=float)
        )

    abs_lat = np.abs(lateral_error) if lateral_error.size else np.array([], dtype=float)
    first_300 = abs_lat[:300]
    last_300 = abs_lat[-300:] if abs_lat.size else np.array([], dtype=float)
    emergency_binary = (emergency_stop > 0.5).astype(int)
    emergency_count = int(np.sum(np.diff(emergency_binary) == 1))
    if emergency_binary.size > 0 and emergency_binary[0] == 1:
        emergency_count += 1

    path_tracking = summary.get("path_tracking", {})
    control_smoothness = summary.get("control_smoothness", {})
    safety = summary.get("safety", {})
    executive = summary.get("executive_summary", {})
    system_health = summary.get("system_health", {})
    speed_control = summary.get("speed_control", {})
    contract_health = summary.get("curvature_contract_health", {})
    first_fault_chain = summary.get("first_fault_chain", {})
    curve_intent_diag = summary.get("curve_intent_diagnostics", {})
    latency_sync = summary.get("latency_sync", {})
    cadence = latency_sync.get("cadence", {}) if isinstance(latency_sync, dict) else {}
    cadence_stats = cadence.get("stats_ms", {}) if isinstance(cadence, dict) else {}
    cadence_limits = cadence.get("limits", {}) if isinstance(cadence, dict) else {}
    curve_intent_available = bool(curve_intent_diag.get("available", False))
    curve_intent_arm_signal_available = bool(curve_intent_diag.get("arm_signal_available", False))

    control_frame_count = int(control_timestamps.size)
    control_duration_s = 0.0
    control_fps = 0.0
    if control_timestamps.size >= 2:
        start_t = float(control_timestamps[0])
        end_t = float(control_timestamps[-1])
        control_duration_s = max(0.0, end_t - start_t)
        if control_duration_s > 1e-6:
            control_fps = float(control_timestamps.size / control_duration_s)
    unity_time_gap_max = float(system_health.get("unity_time_gap_max", 0.0) or 0.0)
    run_validity_reasons: list[str] = []
    if control_fps < float(min_control_fps):
        run_validity_reasons.append(
            f"low_control_fps<{float(min_control_fps):.2f} (observed={control_fps:.2f})"
        )
    if unity_time_gap_max > float(max_unity_time_gap_s):
        run_validity_reasons.append(
            f"unity_time_gap_max>{float(max_unity_time_gap_s):.3f}s (observed={unity_time_gap_max:.3f}s)"
        )
    run_validity_pass = len(run_validity_reasons) == 0

    metrics = {
        "lateral_p95": float(path_tracking.get("lateral_error_p95", 0.0)),
        "lateral_p99": float(np.percentile(abs_lat, 99)) if abs_lat.size else 0.0,
        "lateral_max": float(np.max(abs_lat)) if abs_lat.size else 0.0,
        "first_300_lateral_p99": float(np.percentile(first_300, 99)) if first_300.size else 0.0,
        "last_300_lateral_p99": float(np.percentile(last_300, 99)) if last_300.size else 0.0,
        "oscillation_runaway": bool(control_smoothness.get("oscillation_amplitude_runaway", False)),
        "out_of_lane_events": int(safety.get("out_of_lane_events", 0)),
        "out_of_lane_events_full_run": int(
            safety.get("out_of_lane_events_full_run", safety.get("out_of_lane_events", 0))
        ),
        "emergency_stop_count": emergency_count,
        "steering_jerk_max": float(control_smoothness.get("steering_jerk_max", 0.0)),
        "failure_detected": bool(executive.get("failure_detected", False)),
        "failure_frame": executive.get("failure_frame"),
        "control_frame_count": control_frame_count,
        "control_duration_s": float(control_duration_s),
        "control_fps": float(control_fps),
        "unity_time_gap_max": float(unity_time_gap_max),
        "run_validity_pass": bool(run_validity_pass),
        "run_validity_reasons": run_validity_reasons,
        "tuning_valid": bool(cadence.get("tuning_valid", False)),
        "cadence_dt_p95_ms": float(cadence_stats.get("p95", 0.0) or 0.0),
        "cadence_dt_max_ms": float(cadence_stats.get("max", 0.0) or 0.0),
        "cadence_severe_irregular_rate": float(cadence.get("severe_irregular_rate", 0.0) or 0.0),
        "cadence_tuning_dt_p95_ms_max": float(
            cadence_limits.get("dt_p95_ms_max", 80.0) or 80.0
        ),
        "cadence_tuning_dt_max_ms_max": float(
            cadence_limits.get("dt_max_ms_max", 500.0) or 500.0
        ),
        "cadence_tuning_severe_rate_max": float(
            cadence_limits.get("severe_irregular_rate_max", 0.01) or 0.01
        ),
        "curve_intent_available": bool(curve_intent_available),
        "curve_intent_arm_signal_available": bool(curve_intent_arm_signal_available),
        "curve_intent_arm_early_enough_rate": float(
            curve_intent_diag.get("arm_early_enough_rate", 0.0)
        ),
        "curve_intent_undercall_frame_rate": float(
            curve_intent_diag.get("undercall_frame_rate", 0.0)
        ),
        "curve_intent_curvature_ratio_p50": float(
            curve_intent_diag.get("curvature_ratio_p50", 0.0)
        ),
        "curve_intent_curvature_ratio_p95": float(
            curve_intent_diag.get("curvature_ratio_p95", 0.0)
        ),
        "curve_intent_arm_early_enough": bool(
            (not curve_intent_arm_signal_available)
            or float(curve_intent_diag.get("arm_early_enough_rate", 0.0)) >= 80.0
        ),
        "curve_intent_undercall_detected": bool(
            curve_intent_available and curve_intent_diag.get("undercall_detected", False)
        ),
        "curve_cap_active_rate": float(speed_control.get("curve_cap_active_rate", 0.0)),
        "curve_cap_pre_turn_arm_lead_frames_p50": float(
            speed_control.get("pre_turn_arm_lead_frames_p50", 0.0)
        ),
        "curve_cap_pre_turn_arm_lead_frames_p95": float(
            speed_control.get("pre_turn_arm_lead_frames_p95", 0.0)
        ),
        "curve_cap_overspeed_into_curve_rate": float(
            speed_control.get("overspeed_into_curve_rate", 0.0)
        ),
        "curve_cap_turn_infeasible_rate_when_active": float(
            speed_control.get("turn_infeasible_rate_when_curve_cap_active", 0.0)
        ),
        "cap_tracking_error_p95_mps": float(
            speed_control.get("cap_tracking_error_p95_mps", 0.0)
        ),
        "frames_above_cap_1p0mps_rate": float(
            speed_control.get("frames_above_cap_1p0mps_rate", 0.0)
        ),
        "cap_recovery_frames_p95": float(
            speed_control.get("cap_recovery_frames_p95", 0.0)
        ),
        "hard_ceiling_applied_rate": float(
            speed_control.get("hard_ceiling_applied_rate", 0.0)
        ),
        "curvature_source_divergence_p95": float(
            contract_health.get("curvature_source_divergence_p95", 0.0) or 0.0
        ),
        "curvature_source_diverged_rate": float(
            contract_health.get("curvature_source_diverged_rate", 0.0) or 0.0
        ),
        "curvature_map_authority_lost_rate": float(
            contract_health.get("curvature_map_authority_lost_rate", 0.0) or 0.0
        ),
        "curve_intent_commit_streak_max_frames": int(
            contract_health.get("curve_intent_commit_streak_max_frames", 0) or 0
        ),
        "feasibility_violation_rate": float(
            contract_health.get("feasibility_violation_rate", 0.0) or 0.0
        ),
        "feasibility_backstop_active_rate": float(
            contract_health.get("feasibility_backstop_active_rate", 0.0) or 0.0
        ),
        "map_health_untrusted_rate": float(
            contract_health.get("map_health_untrusted_rate", 0.0) or 0.0
        ),
        "track_mismatch_rate": float(
            contract_health.get("track_mismatch_rate", 0.0) or 0.0
        ),
        "curvature_contract_consistency_rate": float(
            contract_health.get("curvature_contract_consistency_rate", 0.0) or 0.0
        ),
        "telemetry_completeness_rate_curvature_contract": float(
            contract_health.get("telemetry_completeness_rate_curvature_contract", 0.0) or 0.0
        ),
        "telemetry_completeness_rate_feasibility": float(
            contract_health.get("telemetry_completeness_rate_feasibility", 0.0) or 0.0
        ),
        "first_divergence_frame": first_fault_chain.get("first_divergence_frame"),
        "first_infeasible_frame": first_fault_chain.get("first_infeasible_frame"),
        "first_speed_above_feasibility_frame": first_fault_chain.get(
            "first_speed_above_feasibility_frame"
        ),
        "first_boundary_breach_frame": first_fault_chain.get("first_boundary_breach_frame"),
        "source_at_failure": str(contract_health.get("primary_source_mode", "unknown") or "unknown"),
    }
    return summary, metrics


def _classify_root_cause_bucket(metrics: dict[str, Any]) -> str:
    if not bool(metrics.get("run_validity_pass", True)):
        return "sampling_invalid"
    if not bool(metrics.get("tuning_valid", True)):
        return "sampling_invalid"
    if (
        float(metrics.get("curvature_contract_consistency_rate", 100.0)) < 99.0
        or float(metrics.get("telemetry_completeness_rate_curvature_contract", 100.0)) < 99.0
        or float(metrics.get("telemetry_completeness_rate_feasibility", 100.0)) < 99.0
        or float(metrics.get("curvature_map_authority_lost_rate", 0.0)) > 5.0
    ):
        return "curvature_contract_failure"
    if float(metrics.get("feasibility_violation_rate", 0.0)) > 5.0:
        return "longitudinal_overspeed"
    return "controller_authority_failure"


def _pick_first_failure(issues_data: dict[str, Any]) -> dict[str, Any] | None:
    timeline = issues_data.get("causal_timeline") or []
    if timeline:
        return timeline[0]
    issues = issues_data.get("issues") or []
    if issues:
        i = issues[0]
        return {
            "phase": "unknown",
            "type": i.get("type"),
            "frame": i.get("frame"),
            "description": i.get("description"),
            "severity": i.get("severity"),
        }
    return None


def _pick_downstream_consequence(
    issues_data: dict[str, Any],
    failure_frame: int | None,
) -> dict[str, Any] | None:
    timeline = issues_data.get("causal_timeline") or []
    if timeline:
        downstream = [e for e in timeline if e.get("phase") == "downstream"]
        if downstream:
            return downstream[0]
    issues = issues_data.get("issues") or []
    for issue in issues:
        if issue.get("type") in {"out_of_lane", "emergency_stop"}:
            return issue
    if failure_frame is not None:
        return {
            "phase": "downstream",
            "type": "failure_frame",
            "frame": int(failure_frame),
            "description": "Failure frame reached.",
            "severity": "critical",
        }
    return None


def _single_next_lever(diagnostics: dict[str, Any]) -> str:
    diagnosis = diagnostics.get("diagnosis") or {}
    primary = diagnosis.get("primary_issue")
    traj = diagnostics.get("trajectory_analysis") or {}
    attr = traj.get("perception_trajectory_attribution") or {}
    attr_label = str(attr.get("attribution_label") or "")
    if "perception-driven-trajectory-error" in attr_label:
        return (
            "Stabilize perception-to-trajectory handoff first (reduce lane-center jitter/stale episodes "
            "during curve entry)."
        )
    if primary == "trajectory":
        return "Improve trajectory/reference quality first (curve-entry continuity and authority)."
    if primary == "control":
        return "Tune control/limiter stack first (rate/jerk/hard-clip bottlenecks)."
    return "Run one isolated A/B pair and apply the strongest single attribution lever."


def _build_stage6_fields(
    recording_path: Path,
    analyze_to_failure: bool,
    curve_entry_start_distance_m: float,
    curve_entry_window_distance_m: float,
    failure_frame: int | None,
    diagnostics_preflight: dict[str, Any],
) -> dict[str, Any]:
    if not bool(diagnostics_preflight.get("available", False)):
        diagnostics_error = str(diagnostics_preflight.get("error") or "diagnostics unavailable")
        return {
            "what_failed_first": None,
            "downstream_consequence": None,
            "single_next_lever": "Restore diagnostics backend imports before using Stage-6 triage decisions.",
            "diagnostics_mode_used": "preflight_unavailable",
            "diagnostics_fallback_used": False,
            "diagnostics_error": diagnostics_error,
            "answered_all_stage6_questions": False,
        }

    try:
        issues = detect_issues(recording_path, analyze_to_failure=analyze_to_failure) if detect_issues else {}
    except Exception as exc:
        issues = {"error": str(exc), "issues": [], "causal_timeline": []}

    try:
        if analyze_trajectory_vs_steering is None:
            raise RuntimeError("diagnostics backend unavailable")
        diagnostics = analyze_trajectory_vs_steering(
            recording_path,
            analyze_to_failure=analyze_to_failure,
            curve_entry_start_distance_m=curve_entry_start_distance_m,
            curve_entry_window_distance_m=curve_entry_window_distance_m,
        )
    except Exception as exc:
        diagnostics = {"error": str(exc), "diagnosis": None}
    diagnostics_mode_used = "to_failure" if analyze_to_failure else "full_run"
    diagnostics_fallback_used = False
    if diagnostics.get("error") or diagnostics.get("diagnosis") is None:
        try:
            if analyze_trajectory_vs_steering is None:
                raise RuntimeError("diagnostics backend unavailable")
            diagnostics = analyze_trajectory_vs_steering(
                recording_path,
                analyze_to_failure=False,
                curve_entry_start_distance_m=curve_entry_start_distance_m,
                curve_entry_window_distance_m=curve_entry_window_distance_m,
            )
        except Exception as exc:
            diagnostics = {"error": str(exc), "diagnosis": None}
        diagnostics_mode_used = "full_run_fallback"
        diagnostics_fallback_used = True

    what_failed_first = _pick_first_failure(issues if isinstance(issues, dict) else {})
    downstream = _pick_downstream_consequence(
        issues if isinstance(issues, dict) else {},
        failure_frame=failure_frame,
    )
    next_lever = _single_next_lever(diagnostics if isinstance(diagnostics, dict) else {})
    return {
        "what_failed_first": what_failed_first,
        "downstream_consequence": downstream,
        "single_next_lever": next_lever,
        "diagnostics_mode_used": diagnostics_mode_used,
        "diagnostics_fallback_used": diagnostics_fallback_used,
        "diagnostics_error": (diagnostics or {}).get("error"),
        "answered_all_stage6_questions": bool(what_failed_first and downstream and next_lever),
    }


def _track_medians(rows: list[RunArtifacts], track_id: str) -> dict[str, float] | None:
    track_rows = [r for r in rows if r.track_id == track_id]
    if not track_rows:
        return None
    track_rows = [r for r in track_rows if bool(r.metrics.get("tuning_valid", True))]
    if not track_rows:
        return None

    def _median(key: str) -> float:
        return float(np.median([float(r.metrics.get(key, 0.0)) for r in track_rows]))

    return {
        "lateral_p95_median": _median("lateral_p95"),
        "first_300_lateral_p99_median": _median("first_300_lateral_p99"),
        "last_300_lateral_p99_median": _median("last_300_lateral_p99"),
        "steering_jerk_max_median": _median("steering_jerk_max"),
        "cap_tracking_error_p95_median_mps": _median("cap_tracking_error_p95_mps"),
        "frames_above_cap_1p0mps_rate_median": _median("frames_above_cap_1p0mps_rate"),
        "cap_recovery_frames_p95_median": _median("cap_recovery_frames_p95"),
        "hard_ceiling_applied_rate_median": _median("hard_ceiling_applied_rate"),
    }


def _build_regression_budget(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "highway_lateral_p95_max_delta_m": float(args.highway_p95_budget_m),
        "sloop_first_300_p99_min_improvement_m": float(args.sloop_first300_improvement_m),
        "sloop_lateral_p95_max_delta_m": float(args.sloop_p95_regression_budget_m),
        "cap_tracking_frames_above_cap_1p0mps_rate_max_pct": float(
            args.cap_tracking_frames_above_cap_1p0mps_rate_max_pct
        ),
        "cap_tracking_error_p95_max_mps": float(args.cap_tracking_error_p95_max_mps),
        "cap_tracking_recovery_frames_p95_max": float(args.cap_tracking_recovery_frames_p95_max),
        "cap_tracking_hard_ceiling_applied_rate_max_pct": float(
            args.cap_tracking_hard_ceiling_applied_rate_max_pct
        ),
    }


def _evaluate_gate(
    rows: list[RunArtifacts],
    baseline_rows: list[RunArtifacts],
    required_track_ids: list[str],
    expected_runs_per_track: int,
    regression_budget: dict[str, Any],
    diagnostics_preflight: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    run_validity_pass = all(bool(r.metrics.get("run_validity_pass", True)) for r in rows)
    tuning_valid_pass = all(bool(r.metrics.get("tuning_valid", True)) for r in rows)
    eligible_rows = [r for r in rows if bool(r.metrics.get("tuning_valid", True))]
    hard_safety_pass = all(
        int(r.metrics.get("emergency_stop_count", 0)) == 0
        and int(r.metrics.get("out_of_lane_events_full_run", 0)) == 0
        and not bool(r.metrics.get("oscillation_runaway", False))
        for r in rows
    )

    run_counts: dict[str, int] = {}
    for r in eligible_rows:
        run_counts[r.track_id] = run_counts.get(r.track_id, 0) + 1
    run_count_gate_pass = all(run_counts.get(track_id, 0) >= expected_runs_per_track for track_id in required_track_ids)

    track_metrics = {track_id: _track_medians(eligible_rows, track_id) for track_id in required_track_ids}
    baseline_metrics = {track_id: _track_medians(baseline_rows, track_id) for track_id in required_track_ids}

    checks = {
        "run_validity_pass": run_validity_pass,
        "tuning_valid_pass": tuning_valid_pass,
        "hard_safety_pass": hard_safety_pass,
        "run_count_gate_pass": run_count_gate_pass,
        "diagnostics_preflight_pass": bool(
            diagnostics_preflight.get("available", False) or str(mode) != "promotion"
        ),
        "curvature_contract_consistency_pass": all(
            float(r.metrics.get("curvature_contract_consistency_rate", 0.0)) >= 99.0
            for r in eligible_rows
        ),
        "curvature_source_divergence_pass": all(
            float(r.metrics.get("curvature_map_authority_lost_rate", 100.0)) <= 5.0
            for r in eligible_rows
        ),
        "feasibility_coherency_pass": all(
            float(r.metrics.get("feasibility_violation_rate", 100.0)) <= 5.0
            for r in eligible_rows
        ),
        "map_health_pass": all(
            float(r.metrics.get("map_health_untrusted_rate", 100.0)) <= 1.0
            and float(r.metrics.get("track_mismatch_rate", 100.0)) <= 0.0
            for r in eligible_rows
        ),
        "telemetry_completeness_pass": all(
            float(r.metrics.get("telemetry_completeness_rate_curvature_contract", 0.0)) >= 99.0
            and float(r.metrics.get("telemetry_completeness_rate_feasibility", 0.0)) >= 99.0
            for r in eligible_rows
        ),
    }

    highway_id = next((t for t in required_track_ids if "highway" in t), "highway_65")
    sloop_id = next((t for t in required_track_ids if "s_loop" in t or "sloop" in t), "s_loop")
    hw = track_metrics.get(highway_id)
    hw_base = baseline_metrics.get(highway_id)
    sl = track_metrics.get(sloop_id)
    sl_base = baseline_metrics.get(sloop_id)

    if hw and hw_base:
        checks["highway_non_regression"] = bool(
            hw["lateral_p95_median"]
            <= (hw_base["lateral_p95_median"] + float(regression_budget["highway_lateral_p95_max_delta_m"]))
        )
    else:
        checks["highway_non_regression"] = None

    if sl and sl_base:
        checks["sloop_first_300_improvement"] = bool(
            sl["first_300_lateral_p99_median"]
            <= (
                sl_base["first_300_lateral_p99_median"]
                - float(regression_budget["sloop_first_300_p99_min_improvement_m"])
            )
        )
        checks["sloop_lateral_p95_non_regression"] = bool(
            sl["lateral_p95_median"]
            <= (
                sl_base["lateral_p95_median"]
                + float(regression_budget["sloop_lateral_p95_max_delta_m"])
            )
        )
    else:
        checks["sloop_first_300_improvement"] = None
        checks["sloop_lateral_p95_non_regression"] = None

    cap_tracking_rows = [
        r
        for r in rows
        if float(r.metrics.get("curve_cap_active_rate", 0.0)) > 0.1
    ]
    if cap_tracking_rows:
        checks["cap_tracking_effectiveness"] = all(
            float(r.metrics.get("frames_above_cap_1p0mps_rate", 0.0))
            <= float(regression_budget["cap_tracking_frames_above_cap_1p0mps_rate_max_pct"])
            and float(r.metrics.get("cap_tracking_error_p95_mps", 0.0))
            <= float(regression_budget["cap_tracking_error_p95_max_mps"])
            and float(r.metrics.get("cap_recovery_frames_p95", 0.0))
            <= float(regression_budget["cap_tracking_recovery_frames_p95_max"])
            and float(r.metrics.get("hard_ceiling_applied_rate", 0.0))
            <= float(regression_budget["cap_tracking_hard_ceiling_applied_rate_max_pct"])
            for r in cap_tracking_rows
        )
    else:
        checks["cap_tracking_effectiveness"] = True

    pass_fail = True
    for value in checks.values():
        if value is False:
            pass_fail = False
            break

    regression_budget_eval = {
        "config": regression_budget,
        "checks": checks,
        "track_metrics": track_metrics,
        "baseline_track_metrics": baseline_metrics,
    }
    return {
        "pass_fail": pass_fail,
        "checks": checks,
        "run_counts": run_counts,
        "regression_budget": regression_budget_eval,
    }


def validate_contract_required_fields(payload: dict[str, Any]) -> None:
    missing = [k for k in MANDATORY_CONTRACT_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"Missing required contract fields: {missing}")
    if not isinstance(payload["recording_ids"], list):
        raise ValueError("recording_ids must be a list")
    if not isinstance(payload["pass_fail"], bool):
        raise ValueError("pass_fail must be a bool")
    if not isinstance(payload["regression_budget"], dict):
        raise ValueError("regression_budget must be an object")


def _latest_recording() -> Path | None:
    recordings = sorted((REPO_ROOT / "data" / "recordings").glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    return recordings[0] if recordings else None


def _execute_gate_runs(
    tracks: list[Path],
    runs_per_track: int,
    duration_s: int,
    *,
    av_start_t: float,
) -> list[tuple[Path, str]]:
    generated: list[tuple[Path, str]] = []
    for track in tracks:
        track_id = _normalize_track_id(str(track))
        for _ in range(runs_per_track):
            before = _latest_recording()
            cmd = [
                "./start_av_stack.sh",
                "--force",
                "--run-unity-player",
                "--track-yaml",
                str(track),
                "--duration",
                str(int(duration_s)),
            ]
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env={**os.environ, "AV_START_T": str(float(av_start_t))},
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Gate execution failed for {track}: rc={completed.returncode}\n"
                    f"{completed.stderr[-2000:]}"
                )
            after = _latest_recording()
            if after is None or (before is not None and after == before):
                raise RuntimeError(f"Gate execution produced no new recording for {track}")
            _validate_generated_recording_provenance(
                after,
                expected_track_id=track_id,
                expected_start_t=float(av_start_t),
            )
            generated.append((after, track_id))
    return generated


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _run_failure_packet(recording_path: Path, packet_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "analyze" / "build_failure_packet.py"),
        str(recording_path),
        "--out-dir",
        str(packet_dir),
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return {
        "command": " ".join(cmd),
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "packet_dir": str(packet_dir),
    }


def _run_optional_counterfactual(
    args: argparse.Namespace,
    failing_rows: list[RunArtifacts],
    bundle_dir: Path,
) -> dict[str, Any] | None:
    if not args.enable_counterfactual_on_failure:
        return None
    if not failing_rows:
        return None

    baseline = Path(args.counterfactual_baseline).resolve() if args.counterfactual_baseline else None
    treatment = Path(args.counterfactual_treatment).resolve() if args.counterfactual_treatment else None
    if baseline is None or treatment is None:
        if len(failing_rows) < 2:
            return {
                "status": "skipped",
                "reason": "Not enough failed/regressed runs to auto-select baseline+treatment.",
            }
        baseline = failing_rows[0].recording_path
        treatment = failing_rows[1].recording_path

    output_json = bundle_dir / "counterfactual_report.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "analyze" / "counterfactual_layer_swap.py"),
        str(baseline),
        str(treatment),
        "--output-prefix",
        f"gate_{bundle_dir.name}",
        "--output-json",
        str(output_json),
        "--use-cv",
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": " ".join(cmd),
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "output_json": str(output_json),
    }


def _load_input_recordings(args: argparse.Namespace) -> list[Path]:
    if args.execute_gates:
        if not args.tracks:
            raise ValueError("--execute-gates requires --tracks")
        tracks = [(REPO_ROOT / t).resolve() if not Path(t).is_absolute() else Path(t) for t in args.tracks]
        missing = [str(t) for t in tracks if not t.exists()]
        if missing:
            raise FileNotFoundError(f"Track YAML files not found: {missing}")
        runs_per_track = args.runs_per_track or (2 if args.mode == "quick" else 10)
        generated = _execute_gate_runs(
            tracks=tracks,
            runs_per_track=runs_per_track,
            duration_s=args.duration,
            av_start_t=float(args.av_start_t),
        )
        args.recording_track_ids = [track_id for _, track_id in generated]
        return [path for path, _ in generated]

    if not args.recordings:
        raise ValueError("Provide --recordings or use --execute-gates.")
    paths = []
    for raw in args.recordings:
        path = (REPO_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Recording not found: {path}")
        paths.append(path)
    return paths


def _resolve_baseline_recordings(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.baseline_recordings:
        path = (REPO_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Baseline recording not found: {path}")
        paths.append(path)
    return paths


def _normalize_track_id(track_yaml: str) -> str:
    return Path(track_yaml).stem


def _decision_markdown(
    decision_payload: dict[str, Any],
    gate_eval: dict[str, Any],
    failing_rows: list[RunArtifacts],
) -> str:
    checks = gate_eval.get("checks", {})
    lines = [
        "# Gate Decision",
        "",
        f"- decision: `{decision_payload.get('decision')}`",
        f"- pass_fail: `{decision_payload.get('pass_fail')}`",
        f"- generated_at_utc: `{decision_payload.get('generated_at_utc')}`",
        "",
        "## Diagnostics Preflight",
        f"- status: `{(decision_payload.get('diagnostics_preflight') or {}).get('status', 'unknown')}`",
        f"- available: `{(decision_payload.get('diagnostics_preflight') or {}).get('available', False)}`",
        f"- error: `{(decision_payload.get('diagnostics_preflight') or {}).get('error')}`",
        "",
        "## Gate Checks",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Actions"])
    for action in decision_payload.get("next_actions", []):
        lines.append(f"- {action}")
    if failing_rows:
        lines.extend(["", "## Failed/Regressed Runs"])
        for row in failing_rows:
            lines.append(
                f"- `{row.recording_id}` ({row.track_id}): {', '.join(row.trigger_reasons) or 'no explicit trigger'}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run gate execution and triage orchestration.")
    parser.add_argument("--label", default="gate")
    parser.add_argument("--mode", choices=["quick", "promotion"], default="quick")
    parser.add_argument("--execute-gates", action="store_true")
    parser.add_argument("--tracks", nargs="*", default=["tracks/highway_65.yml", "tracks/s_loop.yml"])
    parser.add_argument("--runs-per-track", type=int, default=None)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--av-start-t", type=float, default=0.0)
    parser.add_argument("--recordings", nargs="*", default=[])
    parser.add_argument("--recording-track-ids", nargs="*", default=[])
    parser.add_argument("--baseline-recordings", nargs="*", default=[])
    parser.add_argument("--baseline-track-ids", nargs="*", default=[])
    parser.add_argument("--baseline-bundle-path", default="")
    parser.add_argument("--analyze-to-failure", action="store_true", default=True)
    parser.add_argument("--full-run-analysis", action="store_true")
    parser.add_argument("--curve-entry-start-distance-m", type=float, default=34.0)
    parser.add_argument("--curve-entry-window-distance-m", type=float, default=8.0)
    parser.add_argument("--highway-p95-budget-m", type=float, default=0.02)
    parser.add_argument("--sloop-first300-improvement-m", type=float, default=0.03)
    parser.add_argument("--sloop-p95-regression-budget-m", type=float, default=0.0)
    parser.add_argument("--cap-tracking-frames-above-cap-1p0mps-rate-max-pct", type=float, default=2.0)
    parser.add_argument("--cap-tracking-error-p95-max-mps", type=float, default=0.5)
    parser.add_argument("--cap-tracking-recovery-frames-p95-max", type=float, default=30.0)
    parser.add_argument("--cap-tracking-hard-ceiling-applied-rate-max-pct", type=float, default=0.5)
    parser.add_argument("--min-control-fps", type=float, default=15.0)
    parser.add_argument("--max-unity-time-gap-s", type=float, default=0.25)
    parser.add_argument("--enable-counterfactual-on-failure", action="store_true")
    parser.add_argument("--counterfactual-baseline", default=None)
    parser.add_argument("--counterfactual-treatment", default=None)
    args = parser.parse_args()

    analyze_to_failure = False if args.full_run_analysis else bool(args.analyze_to_failure)
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_dir = REPORTS_DIR / f"{timestamp_utc}_{args.label}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    triage_dir = bundle_dir / "triage_packets"
    failure_packets_dir = bundle_dir / "failure_packets"
    triage_dir.mkdir(parents=True, exist_ok=True)
    failure_packets_dir.mkdir(parents=True, exist_ok=True)

    recording_paths = _load_input_recordings(args)
    baseline_paths = _resolve_baseline_recordings(args)

    recording_track_ids_override: dict[str, str] = {}
    if args.recording_track_ids:
        if len(args.recording_track_ids) != len(recording_paths):
            raise ValueError("--recording-track-ids must match --recordings length.")
        recording_track_ids_override = {
            p.name: str(t).strip() for p, t in zip(recording_paths, args.recording_track_ids)
        }

    baseline_track_ids_override: dict[str, str] = {}
    if args.baseline_track_ids:
        if len(args.baseline_track_ids) != len(baseline_paths):
            raise ValueError("--baseline-track-ids must match --baseline-recordings length.")
        baseline_track_ids_override = {
            p.name: str(t).strip() for p, t in zip(baseline_paths, args.baseline_track_ids)
        }

    required_track_ids = [_normalize_track_id(t) for t in args.tracks]
    expected_runs_per_track = args.runs_per_track or (2 if args.mode == "quick" else 10)
    regression_budget = _build_regression_budget(args)

    matrix_payload = {
        "mode": args.mode,
        "execute_gates": bool(args.execute_gates),
        "required_track_ids": required_track_ids,
        "expected_runs_per_track": int(expected_runs_per_track),
        "duration_s": int(args.duration),
        "av_start_t": float(args.av_start_t),
        "min_control_fps": float(args.min_control_fps),
        "max_unity_time_gap_s": float(args.max_unity_time_gap_s),
        "recording_ids": [p.name for p in recording_paths],
        "baseline_recording_ids": [p.name for p in baseline_paths],
        "baseline_bundle_path": str(args.baseline_bundle_path or ""),
        "analyze_to_failure": analyze_to_failure,
    }
    matrix_hash = compute_matrix_hash(matrix_payload)
    config_hash = _sha256_file(CONFIG_PATH)
    git_sha = _git_sha()
    diagnostics_preflight = _diagnostics_preflight(enabled=True)

    run_artifacts: list[RunArtifacts] = []
    for recording_path in recording_paths:
        summary, metrics = _extract_run_metrics(
            recording_path,
            analyze_to_failure=analyze_to_failure,
            min_control_fps=float(args.min_control_fps),
            max_unity_time_gap_s=float(args.max_unity_time_gap_s),
        )
        stage6 = _build_stage6_fields(
            recording_path=recording_path,
            analyze_to_failure=analyze_to_failure,
            curve_entry_start_distance_m=float(args.curve_entry_start_distance_m),
            curve_entry_window_distance_m=float(args.curve_entry_window_distance_m),
            failure_frame=metrics.get("failure_frame"),
            diagnostics_preflight=diagnostics_preflight,
        )
        run_artifacts.append(
            RunArtifacts(
                recording_path=recording_path,
                track_id=recording_track_ids_override.get(recording_path.name) or _infer_track_id(recording_path),
                recording_id=recording_path.name,
                summary=summary,
                metrics=metrics,
                stage6=stage6,
                trigger_reasons=[],
            )
        )

    baseline_artifacts: list[RunArtifacts] = []
    for recording_path in baseline_paths:
        summary, metrics = _extract_run_metrics(
            recording_path,
            analyze_to_failure=analyze_to_failure,
            min_control_fps=float(args.min_control_fps),
            max_unity_time_gap_s=float(args.max_unity_time_gap_s),
        )
        baseline_artifacts.append(
            RunArtifacts(
                recording_path=recording_path,
                track_id=baseline_track_ids_override.get(recording_path.name) or _infer_track_id(recording_path),
                recording_id=recording_path.name,
                summary=summary,
                metrics=metrics,
                stage6={},
                trigger_reasons=[],
            )
        )

    if baseline_artifacts:
        baseline_payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": git_sha,
            "config_hash": config_hash,
            "matrix_hash": matrix_hash,
            "recording_ids": [row.recording_id for row in baseline_artifacts],
            "pass_fail": True,
            "regression_budget": regression_budget,
            "diagnostics_preflight": diagnostics_preflight,
            "baseline_bundle_path": str(args.baseline_bundle_path or ""),
            "runs": [
                {
                    "recording_id": row.recording_id,
                    "recording_path": str(row.recording_path),
                    "track_id": row.track_id,
                    "provenance": _recording_provenance(row.recording_path),
                    "metrics": row.metrics,
                }
                for row in baseline_artifacts
            ],
        }
        validate_contract_required_fields(baseline_payload)
        _write_json(bundle_dir / "baseline" / "baseline_report.json", baseline_payload)

    gate_eval = _evaluate_gate(
        rows=run_artifacts,
        baseline_rows=baseline_artifacts,
        required_track_ids=required_track_ids,
        expected_runs_per_track=int(expected_runs_per_track),
        regression_budget=regression_budget,
        diagnostics_preflight=diagnostics_preflight,
        mode=args.mode,
    )

    highway_id = next((t for t in required_track_ids if "highway" in t), "highway_65")
    sloop_id = next((t for t in required_track_ids if "s_loop" in t or "sloop" in t), "s_loop")
    baseline_track_metrics = gate_eval["regression_budget"].get("baseline_track_metrics", {})
    baseline_highway_p95 = (baseline_track_metrics.get(highway_id) or {}).get("lateral_p95_median")
    baseline_sloop_first300 = (baseline_track_metrics.get(sloop_id) or {}).get("first_300_lateral_p99_median")
    baseline_sloop_p95 = (baseline_track_metrics.get(sloop_id) or {}).get("lateral_p95_median")

    for row in run_artifacts:
        reasons: list[str] = []
        if args.mode == "promotion" and not bool(diagnostics_preflight.get("available", False)):
            reasons.append("diagnostics_preflight_degraded")
        if int(row.metrics.get("emergency_stop_count", 0)) > 0:
            reasons.append("emergency_stop")
        if int(row.metrics.get("out_of_lane_events_full_run", 0)) > 0:
            reasons.append("out_of_lane")
        if bool(row.metrics.get("oscillation_runaway", False)):
            reasons.append("oscillation_runaway")
        if not bool(row.metrics.get("run_validity_pass", True)):
            reasons.append("run_invalid_sampling")
        if not bool(row.metrics.get("tuning_valid", True)):
            reasons.append("tuning_invalid")
        if not bool(row.metrics.get("curve_intent_arm_early_enough", True)):
            reasons.append("curve_intent_late_arm")
        if bool(row.metrics.get("curve_intent_undercall_detected", False)):
            reasons.append("curve_intent_undercall")
        if float(row.metrics.get("frames_above_cap_1p0mps_rate", 0.0)) > float(
            regression_budget["cap_tracking_frames_above_cap_1p0mps_rate_max_pct"]
        ):
            reasons.append("cap_tracking_above_cap_rate")
        if float(row.metrics.get("cap_tracking_error_p95_mps", 0.0)) > float(
            regression_budget["cap_tracking_error_p95_max_mps"]
        ):
            reasons.append("cap_tracking_error_p95")
        if float(row.metrics.get("cap_recovery_frames_p95", 0.0)) > float(
            regression_budget["cap_tracking_recovery_frames_p95_max"]
        ):
            reasons.append("cap_tracking_recovery_lag")
        if float(row.metrics.get("hard_ceiling_applied_rate", 0.0)) > float(
            regression_budget["cap_tracking_hard_ceiling_applied_rate_max_pct"]
        ):
            reasons.append("cap_tracking_hard_ceiling_rate")
        if baseline_highway_p95 is not None and "highway" in row.track_id:
            if float(row.metrics.get("lateral_p95", 0.0)) > (
                float(baseline_highway_p95) + float(regression_budget["highway_lateral_p95_max_delta_m"])
            ):
                reasons.append("highway_lateral_p95_regression")
        if baseline_sloop_first300 is not None and ("s_loop" in row.track_id or "sloop" in row.track_id):
            if float(row.metrics.get("first_300_lateral_p99", 0.0)) > (
                float(baseline_sloop_first300) - float(regression_budget["sloop_first_300_p99_min_improvement_m"])
            ):
                reasons.append("sloop_first_300_no_improvement")
        if baseline_sloop_p95 is not None and ("s_loop" in row.track_id or "sloop" in row.track_id):
            if float(row.metrics.get("lateral_p95", 0.0)) > (
                float(baseline_sloop_p95) + float(regression_budget["sloop_lateral_p95_max_delta_m"])
            ):
                reasons.append("sloop_lateral_p95_regression")
        row.trigger_reasons = reasons

    gate_report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle_dir": str(bundle_dir),
        "label": args.label,
        "mode": args.mode,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "matrix_hash": matrix_hash,
        "recording_ids": [r.recording_id for r in run_artifacts],
        "pass_fail": bool(gate_eval["pass_fail"]),
        "regression_budget": gate_eval["regression_budget"],
        "run_counts": gate_eval["run_counts"],
        "required_track_ids": required_track_ids,
        "expected_runs_per_track": int(expected_runs_per_track),
        "analyze_to_failure_default": bool(analyze_to_failure),
        "diagnostics_preflight": diagnostics_preflight,
        "baseline_bundle_path": str(args.baseline_bundle_path or ""),
        "runs": [
            {
                "recording_id": row.recording_id,
                "recording_path": str(row.recording_path),
                "track_id": row.track_id,
                "metrics": row.metrics,
                "stage6": row.stage6,
                "trigger_reasons": row.trigger_reasons,
            }
            for row in run_artifacts
        ],
    }
    validate_contract_required_fields(gate_report)
    _write_json(bundle_dir / "gate_report.json", gate_report)

    failing_rows = [r for r in run_artifacts if r.trigger_reasons]
    triage_packets: list[dict[str, Any]] = []
    for row in failing_rows:
        packet_dir = failure_packets_dir / row.recording_id.replace(".h5", "")
        failure_packet = _run_failure_packet(row.recording_path, packet_dir)
        root_cause_bucket = _classify_root_cause_bucket(row.metrics)
        packet = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": git_sha,
            "config_hash": config_hash,
            "matrix_hash": matrix_hash,
            "recording_ids": [row.recording_id],
            "pass_fail": False,
            "regression_budget": gate_eval["regression_budget"],
            "recording_id": row.recording_id,
            "recording_path": str(row.recording_path),
            "track_id": row.track_id,
            "trigger_reasons": row.trigger_reasons,
            "what_failed_first": row.stage6.get("what_failed_first"),
            "downstream_consequence": row.stage6.get("downstream_consequence"),
            "single_next_lever": row.stage6.get("single_next_lever"),
            "diagnostics_mode_used": row.stage6.get("diagnostics_mode_used"),
            "diagnostics_fallback_used": bool(row.stage6.get("diagnostics_fallback_used", False)),
            "failure_packet_artifact": failure_packet,
            "metrics": row.metrics,
            "root_cause_bucket": root_cause_bucket,
            "first_divergence_frame": row.metrics.get("first_divergence_frame"),
            "first_infeasible_frame": row.metrics.get("first_infeasible_frame"),
            "first_speed_above_feasibility_frame": row.metrics.get(
                "first_speed_above_feasibility_frame"
            ),
            "source_at_failure": row.metrics.get("source_at_failure"),
            "telemetry_completeness": {
                "curvature_contract": row.metrics.get(
                    "telemetry_completeness_rate_curvature_contract"
                ),
                "feasibility": row.metrics.get("telemetry_completeness_rate_feasibility"),
            },
        }
        validate_contract_required_fields(packet)
        out_path = triage_dir / f"{row.recording_id.replace('.h5', '')}.json"
        _write_json(out_path, packet)
        triage_packets.append(packet)

    counterfactual = _run_optional_counterfactual(args=args, failing_rows=failing_rows, bundle_dir=bundle_dir)
    next_actions: list[str] = []
    if gate_eval["pass_fail"]:
        decision = "promote"
        next_actions.append("Run promotion gate (10+10) before release.")
    else:
        decision = "reject"
        if failing_rows:
            next_actions.extend([row.stage6.get("single_next_lever", "Review run diagnostics.") for row in failing_rows[:3]])
        else:
            next_actions.append("Fill missing track/run coverage to satisfy gate protocol.")

    decision_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "config_hash": config_hash,
        "matrix_hash": matrix_hash,
        "recording_ids": [r.recording_id for r in run_artifacts],
        "pass_fail": bool(gate_eval["pass_fail"]),
        "regression_budget": gate_eval["regression_budget"],
        "decision": decision,
        "rationale": gate_eval.get("checks", {}),
        "next_actions": next_actions,
        "triage_packet_count": len(triage_packets),
        "counterfactual": counterfactual,
        "diagnostics_preflight": diagnostics_preflight,
        "baseline_bundle_path": str(args.baseline_bundle_path or ""),
    }
    validate_contract_required_fields(decision_payload)
    _write_json(bundle_dir / "decision.json", decision_payload)
    (bundle_dir / "decision.md").write_text(_decision_markdown(decision_payload, gate_eval, failing_rows))

    print(json.dumps(
        {
            "bundle_dir": str(bundle_dir),
            "pass_fail": gate_eval["pass_fail"],
            "recording_count": len(run_artifacts),
            "triage_packet_count": len(triage_packets),
            "decision": decision,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
