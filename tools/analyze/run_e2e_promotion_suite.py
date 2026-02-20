#!/usr/bin/env python3
"""Run mandatory E2E promotion suite and emit a gate report."""

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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _evaluate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    exec_summary = summary.get("executive_summary", {})
    perception = summary.get("perception_quality", {})
    comfort = summary.get("comfort", {})
    safety = summary.get("safety", {})
    overall_score = _safe_float(exec_summary.get("overall_score"))
    fail_detected = bool(exec_summary.get("failure_detected", False))
    stale_hard_rate = _safe_float(perception.get("stale_hard_rate"))
    accel_p95_g = _safe_float(comfort.get("acceleration_p95_g"))
    jerk_p95_gps = _safe_float(comfort.get("jerk_p95_gps"))
    out_of_lane_events = int(safety.get("out_of_lane_events", 0))

    gates = {
        "safety_no_failure": not fail_detected and out_of_lane_events == 0,
        "stale_hard_rate_ok": stale_hard_rate <= 10.0,
        "comfort_accel_ok": accel_p95_g <= 0.25,
        "comfort_jerk_ok": jerk_p95_gps <= 0.51,
        "overall_score_ok": overall_score >= 60.0,
    }
    return {
        "overall_score": overall_score,
        "failure_detected": fail_detected,
        "stale_hard_rate": stale_hard_rate,
        "accel_p95_g": accel_p95_g,
        "jerk_p95_gps": jerk_p95_gps,
        "out_of_lane_events": out_of_lane_events,
        "gates": gates,
        "gate_pass": all(gates.values()),
    }


def _run_ab_repeat3(track_yaml: Path, duration_s: float) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "analyze" / "run_ab_batch.py"),
        "--param",
        "trajectory.dynamic_effective_horizon_enabled",
        "--a",
        "false",
        "--b",
        "true",
        "--repeats",
        "3",
        "--duration",
        str(int(duration_s)),
        "--track-yaml",
        str(track_yaml),
        "--fixed-start-t",
        "0.0",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": " ".join(cmd),
        "return_code": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
        "ok": completed.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical E2E promotion suite.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "tools" / "analyze" / "canonical_e2e_matrix.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "reports" / "e2e_promotion_suite_report.json",
    )
    parser.add_argument(
        "--run-ab-repeat3",
        action="store_true",
        help="Run canonical repeated A/B package as part of suite.",
    )
    args = parser.parse_args()

    matrix = yaml.safe_load(args.matrix.read_text())
    analyze_to_failure = bool((matrix.get("defaults") or {}).get("analyze_to_failure", True))
    rows = []
    for item in matrix.get("recordings", []):
        rec_path = REPO_ROOT / item["path"]
        if not rec_path.exists():
            rows.append({"recording_id": item.get("id"), "error": "missing_recording", "path": str(rec_path)})
            continue
        summary = analyze_recording_summary(rec_path, analyze_to_failure=analyze_to_failure)
        result = _evaluate_summary(summary if isinstance(summary, dict) else {})
        rows.append(
            {
                "recording_id": item.get("id", "unknown"),
                "path": str(rec_path),
                "track_id": item.get("track_id", "unknown"),
                "duration_target_s": _safe_float(item.get("duration_target_s")),
                **result,
            }
        )

    ab_result = None
    if args.run_ab_repeat3:
        # Default to s-loop canonical track.
        ab_result = _run_ab_repeat3(REPO_ROOT / "tracks" / "s_loop.yml", 45.0)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matrix_path": str(args.matrix),
        "rows": rows,
        "aggregate_gate_pass": all(bool(r.get("gate_pass", False)) for r in rows if "gate_pass" in r),
        "ab_repeat3": ab_result,
        "verification_scope_note": "This report is E2E promotion evidence and is separate from unit/integration verification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
