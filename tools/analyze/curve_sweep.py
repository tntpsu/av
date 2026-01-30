#!/usr/bin/env python3
"""
Sweep curve radii by generating temporary tracks and analyzing recordings.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

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


def _update_arc_radii(track: dict, radius: float) -> dict:
    updated = yaml.safe_load(yaml.safe_dump(track))
    for segment in updated.get("segments", []):
        if segment.get("type") == "arc":
            segment["radius"] = radius
    updated["name"] = f"{updated.get('name', 'track')}_r{int(radius)}"
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep curve radii using generated tracks.")
    parser.add_argument("--base-track", default="tracks/s_loop.yml")
    parser.add_argument("--arc-radii", default="20,30,40,60")
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--output-dir", default="tracks/generated")
    args = parser.parse_args()

    base_track_path = (REPO_ROOT / args.base_track).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    track_data = yaml.safe_load(base_track_path.read_text())
    radii = [float(v.strip()) for v in args.arc_radii.split(",") if v.strip()]

    recordings_dir = REPO_ROOT / "data" / "recordings"
    results = []
    generated_tracks = []

    try:
        for radius in radii:
            updated_track = _update_arc_radii(track_data, radius)
            track_path = output_dir / f"{updated_track['name']}.yml"
            track_path.write_text(yaml.safe_dump(updated_track, sort_keys=False))
            generated_tracks.append(track_path)

            print(f"\n=== Curve sweep: radius={radius} track={track_path.name} ===")

            subprocess.run(
                "lsof -ti tcp:8000 | xargs -r kill -9",
                shell=True,
                cwd=REPO_ROOT,
                check=False,
            )

            before = set(recordings_dir.glob("*.h5"))
            subprocess.run(
                [
                    str(REPO_ROOT / "start_av_stack.sh"),
                    "--run-unity-player",
                    "--track-yaml",
                    str(track_path),
                    "--duration",
                    str(args.duration),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            time.sleep(1.0)
            recording = _latest_recording(recordings_dir, before)
            if recording is None:
                print("No new recording found; skipping analysis.")
                continue

            _, summary = _analyze(recording)
            if summary.get("total_frames", 0) == 0:
                print("No frames recorded; skipping analysis.")
                continue
            summary.update({
                "recording": recording.name,
                "radius": radius,
                "track": track_path.name,
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
        for track_path in generated_tracks:
            if track_path.exists():
                track_path.unlink()
        if output_dir.exists() and not any(output_dir.iterdir()):
            shutil.rmtree(output_dir, ignore_errors=True)

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
            f"radius={best['radius']} track={best['track']} "
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
