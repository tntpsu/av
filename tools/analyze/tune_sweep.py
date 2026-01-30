#!/usr/bin/env python3
"""
Lightweight sweep harness for AV stack tuning.

Runs a small grid over config parameters, executes a short simulation,
and reports key metrics using the existing analyzer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.util
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_drive_analyzer():
    module_path = REPO_ROOT / "tools" / "analyze" / "analyze_drive_overall.py"
    spec = importlib.util.spec_from_file_location("analyze_drive_overall", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load analyzer module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DriveAnalyzer


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _latest_recording(recordings_dir: Path, previous: set[Path]) -> Path | None:
    candidates = [p for p in recordings_dir.glob("*.h5") if p not in previous]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _analyze(recording_path: Path) -> Tuple[object, Dict[str, float]]:
    analyzer_cls = _load_drive_analyzer()
    analyzer = analyzer_cls(recording_path, stop_on_emergency=True)
    if not analyzer.load_data():
        raise RuntimeError(f"Failed to load recording: {recording_path}")
    metrics = analyzer.calculate_metrics()
    summary = {
        "total_frames": metrics.total_frames,
        "lateral_rmse": metrics.lateral_error_rmse,
        "centeredness": metrics.time_in_lane_centered,
        "steering_jerk_max": metrics.steering_jerk_max,
        "accel_p95": metrics.acceleration_p95,
        "jerk_p95": metrics.jerk_p95,
        "lateral_jerk_p95": metrics.lateral_jerk_p95,
        "speed_error_rmse": metrics.speed_error_rmse,
        "emergency_stop_frame": analyzer.emergency_stop_frame or 0,
    }
    return analyzer, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small tuning sweep.")
    parser.add_argument("--track-yaml", default="tracks/s_loop.yml")
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--gains", default="1.15,1.25")
    parser.add_argument("--lookaheads", default="8.0,9.0")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "av_stack_config.yaml"
    recordings_dir = repo_root / "data" / "recordings"
    track_path = Path(args.track_yaml)

    gains = [float(v.strip()) for v in args.gains.split(",") if v.strip()]
    lookaheads = [float(v.strip()) for v in args.lookaheads.split(",") if v.strip()]

    original_cfg_text = config_path.read_text()
    results = []

    try:
        for gain in gains:
            for lookahead in lookaheads:
                cfg = _load_yaml(config_path)
                cfg.setdefault("control", {}).setdefault("lateral", {})["curve_feedforward_gain"] = gain
                cfg.setdefault("trajectory", {})["reference_lookahead"] = lookahead
                _write_yaml(config_path, cfg)

                print(f"\n=== Sweep: curve_feedforward_gain={gain} reference_lookahead={lookahead} ===")

                # Ensure port 8000 is free between runs.
                subprocess.run(
                    "lsof -ti tcp:8000 | xargs -r kill -9",
                    shell=True,
                    cwd=repo_root,
                    check=False,
                )

                before = set(recordings_dir.glob("*.h5"))
                subprocess.run(
                    [
                        str(repo_root / "start_av_stack.sh"),
                        "--run-unity-player",
                        "--track-yaml",
                        str(track_path),
                        "--duration",
                        str(args.duration),
                    ],
                    cwd=repo_root,
                    check=True,
                )

                # Give filesystem a moment to flush recording.
                time.sleep(1.0)
                recording = _latest_recording(recordings_dir, before)
                if recording is None:
                    print("No new recording found; skipping analysis.")
                    continue

                _, summary = _analyze(recording)
                summary.update({
                    "recording": recording.name,
                    "gain": gain,
                    "lookahead": lookahead,
                })
                results.append(summary)

                print(
                    f"  recording={recording.name} "
                    f"lateral_rmse={summary['lateral_rmse']:.3f} "
                    f"centered={summary['centeredness']:.1f}% "
                    f"steer_jerk_max={summary['steering_jerk_max']:.1f} "
                    f"accel_p95={summary['accel_p95']:.2f} "
                    f"jerk_p95={summary['jerk_p95']:.1f} "
                    f"lat_jerk_p95={summary['lateral_jerk_p95']:.1f} "
                    f"speed_rmse={summary['speed_error_rmse']:.2f} "
                    f"emergency_stop_frame={summary['emergency_stop_frame']}"
                )
    finally:
        config_path.write_text(original_cfg_text)

    if results:
        best = sorted(
            results,
            key=lambda r: (
                r["emergency_stop_frame"] > 0,
                r["lateral_rmse"],
                r["steering_jerk_max"],
                r["accel_p95"],
                r["jerk_p95"],
            ),
        )[0]
        print("\n=== Best candidate ===")
        print(
            f"gain={best['gain']} lookahead={best['lookahead']} "
            f"recording={best['recording']} "
            f"lateral_rmse={best['lateral_rmse']:.3f} "
            f"centered={best['centeredness']:.1f}% "
            f"steer_jerk_max={best['steering_jerk_max']:.1f} "
            f"accel_p95={best['accel_p95']:.2f} "
            f"jerk_p95={best['jerk_p95']:.1f} "
            f"lat_jerk_p95={best['lateral_jerk_p95']:.1f} "
            f"speed_rmse={best['speed_error_rmse']:.2f} "
            f"emergency_stop_frame={best['emergency_stop_frame']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
