#!/usr/bin/env python3
"""
T-033: Config change detector for CI.

Detects when scoring-critical config parameters change between commits and
reports a structured diff. Annotates PRs with config deltas (informational).

Usage:
    # Compare against previous commit (push)
    python tools/ci/check_config_regression.py

    # Compare against a specific base ref (PR)
    python tools/ci/check_config_regression.py --base origin/main

    # Show only scoring-critical changes
    python tools/ci/check_config_regression.py --critical-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Config files to monitor for changes.
CONFIG_FILES = [
    "config/av_stack_config.yaml",
    "config/mpc_sloop.yaml",
    "config/mpc_highway.yaml",
    "config/mpc_mixed.yaml",
    "config/mpc_hairpin.yaml",
]

# Parameters that affect scoring-critical paths.
# Organized by category for clearer reporting.
SCORING_CRITICAL_PARAMS: dict[str, str] = {
    # Lateral control
    "pp_lookahead_distance": "lateral_control",
    "pp_min_lookahead": "lateral_control",
    "pp_max_lookahead": "lateral_control",
    "pp_feedback_gain": "lateral_control",
    "stanley_k": "lateral_control",
    "stanley_speed_gain": "lateral_control",
    "mpc_q_lat": "lateral_control",
    "mpc_q_heading": "lateral_control",
    "mpc_r_steer_rate": "lateral_control",
    "mpc_horizon": "lateral_control",
    # Speed governor
    "pp_max_speed_mps": "speed_governor",
    "curve_cap_peak_lat_accel_g": "speed_governor",
    "target_speed_mps": "speed_governor",
    "max_speed_mps": "speed_governor",
    # Comfort thresholds
    "accel_limit": "comfort",
    "jerk_limit": "comfort",
    "steering_rate_limit": "comfort",
    "steering_jerk_limit": "comfort",
    # Regime selection
    "regime_enabled": "regime",
    "regime_mpc_downshift_speed": "regime",
    "regime_mpc_upshift_speed": "regime",
    "regime_blend_duration_s": "regime",
    # Curve scheduling
    "curve_scheduler_mode": "curve_scheduling",
    "curve_local_commit_distance_ready_m": "curve_scheduling",
    "curve_local_phase_time_start_tight_s": "curve_scheduling",
}


def _git_show_file(ref: str, filepath: str) -> str | None:
    """Get file contents at a specific git ref."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{filepath}"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None


def _flatten_yaml(d: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a nested YAML dict into dot-separated keys."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_yaml(v, key))
        else:
            flat[key] = v
    return flat


def _diff_configs(old: dict, new: dict) -> list[dict]:
    """Diff two flattened config dicts. Returns list of change records."""
    old_flat = _flatten_yaml(old)
    new_flat = _flatten_yaml(new)
    all_keys = sorted(set(old_flat) | set(new_flat))
    changes = []
    for key in all_keys:
        old_val = old_flat.get(key)
        new_val = new_flat.get(key)
        if old_val == new_val:
            continue
        # Check if any suffix of the key matches a scoring-critical param
        leaf = key.split(".")[-1]
        category = SCORING_CRITICAL_PARAMS.get(leaf)
        changes.append({
            "param": key,
            "old": old_val,
            "new": new_val,
            "scoring_critical": category is not None,
            "category": category or "other",
        })
    return changes


def check_config_changes(base_ref: str) -> dict:
    """Check all monitored config files for changes against base_ref."""
    report = {
        "base_ref": base_ref,
        "files_checked": [],
        "files_changed": [],
        "changes": [],
        "scoring_critical_count": 0,
        "total_change_count": 0,
    }

    for filepath in CONFIG_FILES:
        full_path = REPO_ROOT / filepath
        report["files_checked"].append(filepath)

        old_content = _git_show_file(base_ref, filepath)
        if full_path.exists():
            new_content = full_path.read_text()
        else:
            new_content = None

        if old_content is None and new_content is None:
            continue
        if old_content == new_content:
            continue

        report["files_changed"].append(filepath)

        old_yaml = yaml.safe_load(old_content) if old_content else {}
        new_yaml = yaml.safe_load(new_content) if new_content else {}
        if not isinstance(old_yaml, dict):
            old_yaml = {}
        if not isinstance(new_yaml, dict):
            new_yaml = {}

        changes = _diff_configs(old_yaml, new_yaml)
        for c in changes:
            c["file"] = filepath
        report["changes"].extend(changes)

    report["total_change_count"] = len(report["changes"])
    report["scoring_critical_count"] = sum(
        1 for c in report["changes"] if c["scoring_critical"]
    )
    return report


def _format_report(report: dict, critical_only: bool = False) -> str:
    """Format the config change report for terminal output."""
    lines = []
    lines.append(f"Config change report (base: {report['base_ref']})")
    lines.append(f"Files checked: {len(report['files_checked'])}")
    lines.append(f"Files changed: {len(report['files_changed'])}")
    lines.append(
        f"Total changes: {report['total_change_count']} "
        f"({report['scoring_critical_count']} scoring-critical)"
    )

    if not report["changes"]:
        lines.append("\nNo config changes detected.")
        return "\n".join(lines)

    lines.append("")
    changes = report["changes"]
    if critical_only:
        changes = [c for c in changes if c["scoring_critical"]]
        if not changes:
            lines.append("No scoring-critical changes.")
            return "\n".join(lines)

    # Group by file
    by_file: dict[str, list] = {}
    for c in changes:
        by_file.setdefault(c["file"], []).append(c)

    for filepath, file_changes in by_file.items():
        lines.append(f"  {filepath}:")
        for c in file_changes:
            marker = " [SCORING-CRITICAL]" if c["scoring_critical"] else ""
            lines.append(
                f"    {c['param']}: {c['old']} -> {c['new']}{marker}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect scoring-critical config changes."
    )
    parser.add_argument(
        "--base",
        default="HEAD~1",
        help="Git ref to compare against (default: HEAD~1)",
    )
    parser.add_argument(
        "--critical-only",
        action="store_true",
        help="Only show scoring-critical changes",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output as JSON instead of human-readable",
    )
    args = parser.parse_args()

    report = check_config_changes(args.base)

    if args.output_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_format_report(report, critical_only=args.critical_only))

    # Exit 0 always — this is informational, not blocking
    return 0


if __name__ == "__main__":
    sys.exit(main())
