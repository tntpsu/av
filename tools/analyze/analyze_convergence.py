#!/usr/bin/env python3
"""
Experiment Convergence Detector

Reads the experiment journal and detects "spinning wheels" patterns:
- Multiple attempts targeting the same metric/track with no improvement
- Root cause convergence (different approaches failing for the same reason)
- Diminishing returns (metric plateau despite engineering effort)

Inspired by how top AV companies track engineering efficiency:
- Waymo: intervention root cause clustering
- Tesla: shadow mode regression detection
- ML industry: experiment tracking (MLflow/W&B) with outcome logging

Usage:
    python tools/analyze/analyze_convergence.py           # full analysis
    python tools/analyze/analyze_convergence.py --metric lateral_rmse
    python tools/analyze/analyze_convergence.py --track s_loop
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional

project_root = Path(__file__).resolve().parents[2]
JOURNAL_PATH = project_root / "data" / "reports" / "experiment_journal.json"


def load_journal() -> List[dict]:
    if not JOURNAL_PATH.exists():
        print(f"  No experiment journal found at {JOURNAL_PATH}")
        print(f"  Start logging experiments with /log-experiment")
        return []
    with open(JOURNAL_PATH) as f:
        return json.load(f)


def detect_spinning_wheels(experiments: List[dict],
                           metric_filter: Optional[str] = None,
                           track_filter: Optional[str] = None) -> List[dict]:
    """Detect patterns of repeated failure on the same target."""
    # Group by (target_metric, target_track)
    groups = defaultdict(list)
    for exp in experiments:
        if metric_filter and exp.get("target_metric") != metric_filter:
            continue
        if track_filter and exp.get("target_track") != track_filter:
            continue
        key = (exp.get("target_metric", "unknown"),
               exp.get("target_track", "unknown"))
        groups[key].append(exp)

    alerts = []
    for (metric, track), exps in groups.items():
        failed = [e for e in exps if e.get("outcome") in
                  ("NO_IMPROVEMENT", "REGRESSION")]
        succeeded = [e for e in exps if e.get("outcome") == "IMPROVEMENT"]

        if len(failed) < 2:
            continue

        # Root cause convergence: do failures share a root cause?
        root_causes = Counter(
            e.get("root_cause_of_failure", "unknown") for e in failed
        )
        dominant_cause, dominant_count = root_causes.most_common(1)[0]

        # Total iterations spent on failed attempts
        total_iterations = sum(e.get("iterations_spent", 1) for e in failed)

        # Fix levels attempted
        fix_levels = Counter(e.get("fix_level", "unknown") for e in failed)

        alert = {
            "metric": metric,
            "track": track,
            "failed_attempts": len(failed),
            "successful_attempts": len(succeeded),
            "total_iterations_wasted": total_iterations,
            "dominant_root_cause": dominant_cause,
            "dominant_root_cause_count": dominant_count,
            "root_cause_convergence": dominant_count / len(failed),
            "fix_levels_tried": dict(fix_levels),
            "approaches_tried": [e.get("approach", "?") for e in failed],
            "severity": "CRITICAL" if dominant_count >= 3 else
                        "WARNING" if dominant_count >= 2 else "INFO",
        }

        # Determine recommendation
        if dominant_count >= 3:
            alert["recommendation"] = (
                f"STOP tuning {metric} on {track}. "
                f"{dominant_count} attempts all failed due to '{dominant_cause}'. "
                f"This is an ARCHITECTURAL ceiling, not a tuning problem. "
                f"Escalate to /plan-feature with a different controller architecture."
            )
        elif dominant_count >= 2:
            alert["recommendation"] = (
                f"Likely ceiling on {metric}/{track}: "
                f"{dominant_count} attempts failed for '{dominant_cause}'. "
                f"Before trying again, verify the root cause is addressable "
                f"at the fix level you're planning. Check /process-health."
            )

        alerts.append(alert)

    return alerts


def detect_root_cause_clusters(experiments: List[dict]) -> Dict[str, dict]:
    """Find root causes that block multiple metrics/tracks."""
    cause_map = defaultdict(lambda: {
        "count": 0, "metrics": set(), "tracks": set(),
        "approaches": [], "iterations_total": 0,
    })

    for exp in experiments:
        if exp.get("outcome") not in ("NO_IMPROVEMENT", "REGRESSION"):
            continue
        cause = exp.get("root_cause_of_failure", "unknown")
        info = cause_map[cause]
        info["count"] += 1
        info["metrics"].add(exp.get("target_metric", "?"))
        info["tracks"].add(exp.get("target_track", "?"))
        info["approaches"].append(exp.get("approach", "?"))
        info["iterations_total"] += exp.get("iterations_spent", 1)

    # Convert sets to lists for display
    for info in cause_map.values():
        info["metrics"] = sorted(info["metrics"])
        info["tracks"] = sorted(info["tracks"])

    return dict(cause_map)


def compute_efficiency(experiments: List[dict]) -> dict:
    """Compute engineering efficiency metrics."""
    if not experiments:
        return {}

    total = len(experiments)
    succeeded = sum(1 for e in experiments
                    if e.get("outcome") == "IMPROVEMENT")
    failed = sum(1 for e in experiments
                 if e.get("outcome") in ("NO_IMPROVEMENT", "REGRESSION"))
    regressed = sum(1 for e in experiments
                    if e.get("outcome") == "REGRESSION")

    total_iterations = sum(e.get("iterations_spent", 1) for e in experiments)
    wasted_iterations = sum(e.get("iterations_spent", 1) for e in experiments
                           if e.get("outcome") != "IMPROVEMENT")

    # Fix level distribution
    level_dist = Counter(e.get("fix_level", "?") for e in experiments)
    level_success = Counter(e.get("fix_level", "?") for e in experiments
                            if e.get("outcome") == "IMPROVEMENT")

    return {
        "total_experiments": total,
        "success_rate": succeeded / total if total else 0,
        "regression_rate": regressed / total if total else 0,
        "total_iterations": total_iterations,
        "wasted_iterations": wasted_iterations,
        "efficiency": (total_iterations - wasted_iterations) / total_iterations
        if total_iterations else 0,
        "fix_level_distribution": dict(level_dist),
        "fix_level_success_rate": {
            level: level_success.get(level, 0) / count
            for level, count in level_dist.items()
        },
    }


def print_report(experiments: List[dict],
                 metric_filter: Optional[str] = None,
                 track_filter: Optional[str] = None):
    """Print the full convergence analysis report."""
    print(f"\n{'='*72}")
    print(f"  EXPERIMENT CONVERGENCE ANALYSIS")
    print(f"  ═══════════════════════════════")
    print(f"  Journal: {JOURNAL_PATH}")
    print(f"  Experiments: {len(experiments)}")
    if metric_filter:
        print(f"  Filter: metric={metric_filter}")
    if track_filter:
        print(f"  Filter: track={track_filter}")
    print(f"{'='*72}")

    # --- Efficiency ---
    eff = compute_efficiency(experiments)
    if eff:
        print(f"\n  ENGINEERING EFFICIENCY")
        print(f"  ─────────────────────")
        print(f"  Success rate:      {eff['success_rate']:.0%} "
              f"({eff['total_experiments'] - int(eff['total_experiments'] * (1-eff['success_rate']))} of {eff['total_experiments']})")
        print(f"  Regression rate:   {eff['regression_rate']:.0%}")
        print(f"  Iterations total:  {eff['total_iterations']}")
        print(f"  Iterations wasted: {eff['wasted_iterations']} "
              f"({eff['efficiency']:.0%} efficient)")

        print(f"\n  Fix Level Success Rates:")
        for level, rate in sorted(eff['fix_level_success_rate'].items()):
            count = eff['fix_level_distribution'][level]
            print(f"    {level:>15}: {rate:.0%} ({count} attempts)")

    # --- Spinning Wheels ---
    alerts = detect_spinning_wheels(experiments, metric_filter, track_filter)
    if alerts:
        print(f"\n  SPINNING WHEELS DETECTED")
        print(f"  ───────────────────────")
        for alert in sorted(alerts, key=lambda a: -a["failed_attempts"]):
            severity = alert["severity"]
            icon = "■" if severity == "CRITICAL" else "▲" if severity == "WARNING" else "●"
            print(f"\n  {icon} [{severity}] {alert['metric']} on {alert['track']}")
            print(f"    Failed attempts: {alert['failed_attempts']} "
                  f"(+{alert['successful_attempts']} successful)")
            print(f"    Iterations wasted: {alert['total_iterations_wasted']}")
            print(f"    Root cause convergence: {alert['root_cause_convergence']:.0%} "
                  f"→ '{alert['dominant_root_cause']}' "
                  f"({alert['dominant_root_cause_count']}× dominant)")
            print(f"    Fix levels tried: {alert['fix_levels_tried']}")
            print(f"    Approaches:")
            for approach in alert["approaches_tried"]:
                print(f"      - {approach}")
            if alert.get("recommendation"):
                print(f"    ▶ {alert['recommendation']}")
    else:
        print(f"\n  No spinning wheels patterns detected ✓")

    # --- Root Cause Clusters ---
    clusters = detect_root_cause_clusters(experiments)
    if clusters:
        print(f"\n  ROOT CAUSE CLUSTERS (cross-cutting blockers)")
        print(f"  ─────────────────────────────────────────────")
        for cause, info in sorted(clusters.items(),
                                  key=lambda x: -x[1]["count"]):
            print(f"\n  '{cause}' — {info['count']} failures, "
                  f"{info['iterations_total']} iterations")
            print(f"    Metrics: {', '.join(info['metrics'])}")
            print(f"    Tracks:  {', '.join(info['tracks'])}")
            print(f"    Approaches that failed:")
            for approach in info["approaches"]:
                print(f"      - {approach}")

    # --- Recommendation ---
    if alerts:
        critical = [a for a in alerts if a["severity"] == "CRITICAL"]
        if critical:
            print(f"\n  {'='*72}")
            print(f"  ACTION REQUIRED — {len(critical)} critical ceiling(s) detected")
            print(f"  {'='*72}")
            for a in critical:
                print(f"\n  Stop tuning {a['metric']} on {a['track']}.")
                print(f"  Root cause '{a['dominant_root_cause']}' is architectural.")
                print(f"  Next step: /plan-feature to design a replacement,")
                print(f"  not another iteration on the same architecture.")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment convergence detector")
    parser.add_argument("--metric", default=None,
                        help="Filter by target metric")
    parser.add_argument("--track", default=None,
                        help="Filter by target track")
    args = parser.parse_args()

    experiments = load_journal()
    if not experiments:
        return

    print_report(experiments, args.metric, args.track)


if __name__ == "__main__":
    main()
