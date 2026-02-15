#!/usr/bin/env python3
"""
Run Stage 6 acceptance evidence checks on selected recordings.

For each recording, this script answers:
1) What failed first?
2) What was the downstream consequence?
3) Which single next lever is justified?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_func(module_relpath: str, func_name: str):
    module_path = REPO_ROOT / module_relpath
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise RuntimeError(f"Function {func_name} not found in {module_path}")
    return fn


analyze_recording_summary = _load_func(
    "tools/debug_visualizer/backend/summary_analyzer.py",
    "analyze_recording_summary",
)
detect_issues = _load_func(
    "tools/debug_visualizer/backend/issue_detector.py",
    "detect_issues",
)
analyze_trajectory_vs_steering = _load_func(
    "tools/debug_visualizer/backend/diagnostics.py",
    "analyze_trajectory_vs_steering",
)


def _pick_first_failure(issues_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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


def _pick_downstream_consequence(issues_data: Dict[str, Any], failure_frame: Optional[int]) -> Optional[Dict[str, Any]]:
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


def _single_next_lever(diag: Dict[str, Any]) -> str:
    diagnosis = diag.get("diagnosis") or {}
    primary = diagnosis.get("primary_issue")
    traj = diag.get("trajectory_analysis") or {}
    attr = traj.get("perception_trajectory_attribution") or {}
    attr_label = str(attr.get("attribution_label") or "")

    if "perception-driven-trajectory-error" in attr_label:
        return "Stabilize perception-to-trajectory handoff first (reduce lane-center jitter/stale episodes during curve entry)."
    if primary == "trajectory":
        return "Improve trajectory/reference quality first (curve-entry reference authority and continuity) before control retuning."
    if primary == "control":
        return "Tune control/limiter stack first (steering authority through rate/jerk/hard-clip bottlenecks)."
    return "Run another isolated pair and choose one lever with strongest attribution signal."


def _evaluate_one(recording_path: Path) -> Dict[str, Any]:
    summary = analyze_recording_summary(recording_path, analyze_to_failure=True)
    issues = detect_issues(recording_path, analyze_to_failure=True)
    diagnostics = analyze_trajectory_vs_steering(
        recording_path,
        analyze_to_failure=True,
        curve_entry_start_distance_m=34.0,
        curve_entry_window_distance_m=8.0,
    )
    diagnostics_mode = "to_failure"
    if diagnostics.get("error") or diagnostics.get("diagnosis") is None:
        diagnostics = analyze_trajectory_vs_steering(
            recording_path,
            analyze_to_failure=False,
            curve_entry_start_distance_m=34.0,
            curve_entry_window_distance_m=8.0,
        )
        diagnostics_mode = "full_run_fallback"

    failure_frame = ((summary.get("executive_summary") or {}).get("failure_frame"))
    failed_first = _pick_first_failure(issues)
    downstream = _pick_downstream_consequence(issues, failure_frame=failure_frame)
    next_lever = _single_next_lever(diagnostics)

    answered_all = bool(failed_first and downstream and next_lever)
    return {
        "recording": str(recording_path),
        "failure_frame": int(failure_frame) if failure_frame is not None else None,
        "what_failed_first": failed_first,
        "downstream_consequence": downstream,
        "single_next_lever": next_lever,
        "diagnostics_mode_used": diagnostics_mode,
        "diagnostics_error": diagnostics.get("error"),
        "diagnosis": diagnostics.get("diagnosis"),
        "trajectory_attribution": (diagnostics.get("trajectory_analysis") or {}).get(
            "perception_trajectory_attribution"
        ),
        "answered_all_stage6_questions": answered_all,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 6 acceptance evidence checks.")
    parser.add_argument("recordings", nargs="+", help="Recording .h5 paths")
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "tmp" / "analysis" / "stage6_acceptance_report.json"),
    )
    args = parser.parse_args()

    runs: List[Dict[str, Any]] = []
    for rec in args.recordings:
        path = Path(rec)
        if not path.exists():
            raise FileNotFoundError(f"Recording not found: {path}")
        runs.append(_evaluate_one(path))

    all_answered = all(bool(r.get("answered_all_stage6_questions")) for r in runs) if runs else False
    out = {
        "recording_count": len(runs),
        "all_runs_answer_stage6_questions": all_answered,
        "runs": runs,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

