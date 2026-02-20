#!/usr/bin/env python3
"""Freeze canonical baseline gates and verify analyzer consistency."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.debug_visualizer.backend.summary_analyzer import analyze_recording_summary
from tools.debug_visualizer.backend.diagnostics import analyze_trajectory_vs_steering


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _run_turn_authority_root_cause(recording_path: Path) -> dict[str, Any]:
    out_path = REPO_ROOT / "tmp" / "analysis" / f"rootcause_{recording_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "analyze" / "analyze_trajectory_turn_authority_root_cause.py"),
        str(recording_path),
        "--output-json",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), capture_output=True, text=True)
        return json.loads(out_path.read_text())
    except Exception:
        return {"underturn_rate": None, "error": "root_cause_not_available"}


def _summarize_recording(recording_path: Path, analyze_to_failure: bool) -> dict[str, Any]:
    summary = analyze_recording_summary(recording_path, analyze_to_failure=analyze_to_failure)
    diagnostics = analyze_trajectory_vs_steering(
        recording_path,
        analyze_to_failure=analyze_to_failure,
        curve_entry_start_distance_m=34.0,
        curve_entry_window_distance_m=8.0,
    )
    root_cause = _run_turn_authority_root_cause(recording_path)

    exec_summary = summary.get("executive_summary", {})
    perception = summary.get("perception_quality", {})
    comfort = summary.get("comfort", {})
    safety = summary.get("safety", {})
    feas = (
        (diagnostics.get("control_analysis") or {})
        .get("curve_entry_feasibility", {})
    )
    steering_authority = feas.get("steering_authority", {}) if isinstance(feas, dict) else {}

    stale_hard_rate = _safe_float(perception.get("stale_hard_rate"))
    stale_pct = _safe_float(feas.get("stale_perception_pct"))
    consistency_delta = abs(stale_hard_rate - stale_pct)
    return {
        "recording": str(recording_path),
        "overall_score": _safe_float(exec_summary.get("overall_score")),
        "failure_detected": bool(exec_summary.get("failure_detected", False)),
        "stale_hard_rate_summary_pct": stale_hard_rate,
        "stale_pct_diagnostics_pct": stale_pct,
        "stale_cross_analyzer_delta_pct": consistency_delta,
        "authority_gap_mean": _safe_float(steering_authority.get("authority_gap_mean")),
        "transfer_ratio_mean": _safe_float(steering_authority.get("transfer_ratio_mean")),
        "comfort_accel_p95_g": _safe_float(comfort.get("acceleration_p95_g")),
        "comfort_jerk_p95_gps": _safe_float(comfort.get("jerk_p95_gps")),
        "safety_out_of_lane_events": int(safety.get("out_of_lane_events", 0)),
        "root_cause_underturn_rate": _safe_float(root_cause.get("underturn_rate")),
        "consistency": {
            "stale_rate_alignment_pass": consistency_delta <= 5.0,
            "stale_rate_alignment_threshold_pct": 5.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze canonical baseline matrix and gate consistency.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "tools" / "analyze" / "canonical_e2e_matrix.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "reports" / "baseline_freeze_report.json",
    )
    args = parser.parse_args()

    matrix = yaml.safe_load(args.matrix.read_text())
    analyze_to_failure = bool((matrix.get("defaults") or {}).get("analyze_to_failure", True))
    rows: list[dict[str, Any]] = []
    for item in matrix.get("recordings", []):
        rec_path = REPO_ROOT / item["path"]
        if not rec_path.exists():
            rows.append(
                {
                    "recording": str(rec_path),
                    "error": "missing_recording",
                }
            )
            continue
        row = _summarize_recording(rec_path, analyze_to_failure=analyze_to_failure)
        row["recording_id"] = item.get("id", "unknown")
        row["track_id"] = item.get("track_id", "unknown")
        row["duration_target_s"] = _safe_float(item.get("duration_target_s"), default=0.0)
        rows.append(row)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matrix_path": str(args.matrix),
        "rows": rows,
        "all_consistency_pass": all(
            bool(r.get("consistency", {}).get("stale_rate_alignment_pass", False))
            for r in rows
            if "consistency" in r
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
