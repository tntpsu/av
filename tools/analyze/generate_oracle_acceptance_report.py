#!/usr/bin/env python3
"""
Generate pass/fail acceptance report for oracle trajectory rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import h5py
import numpy as np
import yaml


def _safe_mean_bool(arr: np.ndarray) -> float | None:
    if arr.size == 0:
        return None
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    return float(np.mean(arr[finite] > 0.5))


def _fetch_json(url: str) -> dict[str, Any] | None:
    try:
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def analyze_recording(recording_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "recording": str(recording_path),
        "exists": recording_path.exists(),
        "checks": {},
        "details": {},
    }
    if not recording_path.exists():
        return report

    required_datasets = [
        "trajectory/oracle_points",
        "trajectory/oracle_point_count",
        "trajectory/oracle_horizon_meters",
        "trajectory/oracle_point_spacing_meters",
        "trajectory/oracle_samples_enabled",
    ]

    with h5py.File(recording_path, "r") as f:
        dataset_presence = {k: (k in f) for k in required_datasets}
        report["details"]["dataset_presence"] = dataset_presence
        report["checks"]["step2_dataset_presence"] = all(dataset_presence.values())

        if "trajectory/oracle_point_count" in f:
            count = np.array(f["trajectory/oracle_point_count"][:], dtype=float)
            availability_ratio = _safe_mean_bool(count)
            report["details"]["oracle_point_count_nonzero_ratio"] = availability_ratio
            report["checks"]["step1_oracle_telemetry_availability"] = (
                availability_ratio is not None and availability_ratio >= 0.99
            )
        else:
            report["details"]["oracle_point_count_nonzero_ratio"] = None
            report["checks"]["step1_oracle_telemetry_availability"] = False

        if "trajectory/oracle_points" in f:
            oracle_ds = f["trajectory/oracle_points"]
            first_points = []
            monotonic_flags = []
            max_n = min(len(oracle_ds), 300)
            for i in range(max_n):
                row = np.array(oracle_ds[i], dtype=float).reshape(-1)
                if row.size < 4 or row.size % 2 != 0:
                    continue
                pts = row.reshape(-1, 2)
                x = pts[:, 0]
                y = pts[:, 1]
                finite = np.isfinite(x) & np.isfinite(y)
                if np.sum(finite) < 2:
                    continue
                x = x[finite]
                y = y[finite]
                first_points.append(float(np.hypot(x[0], y[0])))
                monotonic_flags.append(float(np.all(np.diff(y) >= -1e-6)))
            first_dist_mean = float(np.mean(first_points)) if first_points else None
            monotonic_ratio = float(np.mean(monotonic_flags)) if monotonic_flags else None
            report["details"]["oracle_first_point_dist_mean"] = first_dist_mean
            report["details"]["oracle_y_monotonic_ratio"] = monotonic_ratio
            report["checks"]["step1_first_point_near_vehicle"] = (
                first_dist_mean is not None and first_dist_mean <= 1.0
            )
            report["checks"]["step1_y_monotonic_majority"] = (
                monotonic_ratio is not None and monotonic_ratio >= 0.95
            )
        else:
            report["checks"]["step1_first_point_near_vehicle"] = False
            report["checks"]["step1_y_monotonic_majority"] = False

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate oracle rollout acceptance report")
    parser.add_argument("--before", required=True, help="Before-change recording path or filename")
    parser.add_argument("--after", required=True, help="After-change recording path or filename")
    parser.add_argument(
        "--recordings-dir",
        default="data/recordings",
        help="Base recordings directory when filenames are provided",
    )
    parser.add_argument(
        "--visualizer-base-url",
        default="http://localhost:5001",
        help="PhilViz backend URL for diagnostics/frame checks",
    )
    parser.add_argument(
        "--output",
        default="tmp/analysis/oracle_acceptance_report.json",
        help="Output report JSON path",
    )
    args = parser.parse_args()

    recordings_dir = Path(args.recordings_dir)

    def resolve_path(value: str) -> Path:
        p = Path(value)
        if p.exists():
            return p
        return recordings_dir / value

    before_path = resolve_path(args.before)
    after_path = resolve_path(args.after)

    report: dict[str, Any] = {
        "before": analyze_recording(before_path),
        "after": analyze_recording(after_path),
        "checks": {},
        "artifacts": {
            "before_recording": str(before_path),
            "after_recording": str(after_path),
        },
    }

    # Step 3 + Step 4 checks from backend (after recording).
    after_name = after_path.name
    topdown_url = f"{args.visualizer_base_url}/api/recording/{quote(after_name)}/topdown-diagnostics"
    topdown = _fetch_json(topdown_url)
    report["after"]["details"]["topdown_diagnostics_url"] = topdown_url
    if topdown is not None:
        iso = topdown.get("trajectory_source_isolation", {})
        gap = iso.get("planner_vs_oracle_gap_metrics", {})
        report["after"]["details"]["planner_vs_oracle_gap_metrics"] = gap
        report["checks"]["step4_gap_metrics_present"] = (
            isinstance(gap, dict)
            and (gap.get("frames_with_oracle_overlap", 0) > 0)
            and (gap.get("gap_at_lookahead_8m_stats", {}).get("count", 0) > 0)
            and (gap.get("max_lateral_gap_0_h_stats", {}).get("count", 0) > 0)
        )
    else:
        report["checks"]["step4_gap_metrics_present"] = False

    # Step 3: compare oracle point count from frame API vs dataset for sampled frames.
    frame_match_count = 0
    frame_checked = 0
    if after_path.exists():
        with h5py.File(after_path, "r") as f:
            ds = f["trajectory/oracle_point_count"] if "trajectory/oracle_point_count" in f else None
            if ds is not None:
                sample_idx = list(range(0, min(len(ds), 120), 12))
                for idx in sample_idx:
                    frame_url = f"{args.visualizer_base_url}/api/recording/{quote(after_name)}/frame/{idx}"
                    frame_data = _fetch_json(frame_url)
                    if not frame_data:
                        continue
                    backend_count = int(
                        frame_data.get("trajectory", {}).get("oracle_point_count", -1)
                    )
                    file_count = int(ds[idx])
                    frame_checked += 1
                    if backend_count == file_count:
                        frame_match_count += 1
    ratio = (float(frame_match_count) / float(frame_checked)) if frame_checked > 0 else 0.0
    report["after"]["details"]["oracle_frame_count_match_ratio"] = ratio
    report["checks"]["step3_overlay_data_integrity"] = ratio >= 0.95

    # Step 5 checks from config and before/after default behavior.
    cfg_path = Path("config/av_stack_config.yaml")
    trajectory_source = None
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        trajectory_source = (
            (cfg.get("trajectory") or {}).get("trajectory_source")
            if isinstance(cfg, dict)
            else None
        )
    report["details"] = {"trajectory_source_config": trajectory_source}
    report["checks"]["step5_default_mode_is_planner"] = (trajectory_source == "planner")
    report["checks"]["step5_oracle_mode_switch_available"] = trajectory_source in {"planner", "oracle"}

    # Aggregate by planned step labels.
    report["checks"]["step1_pass"] = (
        bool(report["after"]["checks"].get("step1_oracle_telemetry_availability"))
        and bool(report["after"]["checks"].get("step1_first_point_near_vehicle"))
        and bool(report["after"]["checks"].get("step1_y_monotonic_majority"))
    )
    report["checks"]["step2_pass"] = bool(report["after"]["checks"].get("step2_dataset_presence"))
    report["checks"]["step3_pass"] = bool(report["checks"].get("step3_overlay_data_integrity"))
    report["checks"]["step4_pass"] = bool(report["checks"].get("step4_gap_metrics_present"))
    report["checks"]["step5_pass"] = (
        bool(report["checks"].get("step5_default_mode_is_planner"))
        and bool(report["checks"].get("step5_oracle_mode_switch_available"))
    )

    all_steps = ["step1_pass", "step2_pass", "step3_pass", "step4_pass", "step5_pass"]
    report["checks"]["all_pass"] = all(bool(report["checks"].get(k, False)) for k in all_steps)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()

