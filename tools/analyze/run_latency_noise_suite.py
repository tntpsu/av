#!/usr/bin/env python3
"""
Run deterministic latency/noise stress suite for trajectory-locked replay.

Profiles are executed offline on recordings and summarized into one JSON report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_replay_func():
    module_path = REPO_ROOT / "tools" / "analyze" / "replay_trajectory_locked.py"
    spec = importlib.util.spec_from_file_location("replay_trajectory_locked", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.replay_trajectory_locked


replay_trajectory_locked = _load_replay_func()

DEFAULT_SEGMENTATION_CHECKPOINT = (
    REPO_ROOT / "data" / "segmentation_dataset" / "checkpoints" / "segnet_best.pt"
)


def _pick_latest_recording() -> Path:
    recordings = sorted(
        (REPO_ROOT / "data" / "recordings").glob("*.h5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not recordings:
        raise FileNotFoundError("No recordings found in data/recordings")
    return recordings[0]


def _default_profiles() -> List[Dict[str, Any]]:
    # Deterministic timing + perturbation ladder for Stage 4.
    return [
        {
            "name": "baseline",
            "latency_ms": 0.0,
            "noise_std_x": 0.0,
            "noise_std_heading": 0.0,
            "noise_std_velocity": 0.0,
            "noise_std_curvature": 0.0,
        },
        {
            "name": "latency_50ms",
            "latency_ms": 50.0,
            "noise_std_x": 0.0,
            "noise_std_heading": 0.0,
            "noise_std_velocity": 0.0,
            "noise_std_curvature": 0.0,
        },
        {
            "name": "latency_100ms",
            "latency_ms": 100.0,
            "noise_std_x": 0.0,
            "noise_std_heading": 0.0,
            "noise_std_velocity": 0.0,
            "noise_std_curvature": 0.0,
        },
        {
            "name": "noise_mild",
            "latency_ms": 0.0,
            "noise_std_x": 0.03,
            "noise_std_heading": 0.02,
            "noise_std_velocity": 0.15,
            "noise_std_curvature": 0.001,
        },
        {
            "name": "latency_50ms_noise_mild",
            "latency_ms": 50.0,
            "noise_std_x": 0.03,
            "noise_std_heading": 0.02,
            "noise_std_velocity": 0.15,
            "noise_std_curvature": 0.001,
        },
    ]


def _classify(
    results: List[Dict[str, Any]],
    degrade_threshold: float,
    waived_profiles: Set[str] | None = None,
) -> str:
    waived_profiles = waived_profiles or set()
    baseline = next((r for r in results if r["profile"]["name"] == "baseline"), None)
    if baseline is None:
        return "unknown"
    b = baseline.get("steering_abs_diff_mean_vs_source")
    if b is None:
        return "unknown"
    b = float(b)
    considered = [r for r in results if r["profile"]["name"] not in waived_profiles]
    if not considered:
        return "unknown"
    worst = max(float(r.get("steering_abs_diff_mean_vs_source") or 0.0) for r in considered)
    ratio = worst / max(1e-6, b)
    if ratio >= degrade_threshold:
        return "fails-latency-noise-gate"
    if ratio >= 1.2:
        return "degraded-but-pass"
    return "passes-latency-noise-gate"


def _worst_ratio(results: List[Dict[str, Any]]) -> float | None:
    baseline = next((r for r in results if r["profile"]["name"] == "baseline"), None)
    if baseline is None or baseline.get("steering_abs_diff_mean_vs_source") is None:
        return None
    b = float(baseline["steering_abs_diff_mean_vs_source"])
    worst = max(float(r.get("steering_abs_diff_mean_vs_source") or 0.0) for r in results)
    return worst / max(1e-6, b)


def _run_suite_for_seed(
    input_recording: Path,
    lock_recording: Path,
    output_prefix: str,
    noise_seed: int,
    degrade_threshold: float,
    waived_profiles: Set[str],
    disable_vehicle_frame_lookahead_ref: bool,
    use_segmentation: bool,
    segmentation_checkpoint: str | None,
) -> Dict[str, Any]:
    profiles = _default_profiles()
    results: List[Dict[str, Any]] = []
    for p in profiles:
        out_name = f"{output_prefix}_seed{noise_seed}_{p['name']}"
        print(f"\n=== Running profile: {p['name']} (seed={noise_seed}) ===")
        summary = replay_trajectory_locked(
            input_recording=input_recording,
            lock_recording=lock_recording,
            output_name=out_name,
            use_segmentation=bool(use_segmentation),
            segmentation_checkpoint=segmentation_checkpoint,
            disable_vehicle_frame_lookahead_ref=bool(disable_vehicle_frame_lookahead_ref),
            lock_latency_ms=float(p["latency_ms"]),
            lock_latency_frames=0,
            noise_seed=int(noise_seed),
            noise_std_x=float(p["noise_std_x"]),
            noise_std_heading=float(p["noise_std_heading"]),
            noise_std_velocity=float(p["noise_std_velocity"]),
            noise_std_curvature=float(p["noise_std_curvature"]),
        )
        results.append(
            {
                "profile": p,
                "output_recording": summary.get("output_recording"),
                "summary_json": summary.get("summary_json"),
                "steering_abs_diff_mean_vs_source": summary.get("steering_abs_diff_mean_vs_source"),
                "steering_abs_diff_p95_vs_source": summary.get("steering_abs_diff_p95_vs_source"),
                "throttle_abs_diff_mean_vs_source": summary.get("throttle_abs_diff_mean_vs_source"),
                "brake_abs_diff_mean_vs_source": summary.get("brake_abs_diff_mean_vs_source"),
                "latency_total_frames": ((summary.get("latency") or {}).get("total_latency_frames")),
            }
        )
    baseline = next((r for r in results if r["profile"]["name"] == "baseline"), None)
    b = float(baseline["steering_abs_diff_mean_vs_source"]) if baseline and baseline.get("steering_abs_diff_mean_vs_source") is not None else None
    ratio_by_profile = {}
    if b is not None:
        for r in results:
            name = r["profile"]["name"]
            v = r.get("steering_abs_diff_mean_vs_source")
            ratio_by_profile[name] = (float(v) / max(1e-6, b)) if v is not None else None
    gate = _classify(
        results=results,
        degrade_threshold=float(degrade_threshold),
        waived_profiles=waived_profiles,
    )
    considered_ratios = [
        float(v)
        for k, v in ratio_by_profile.items()
        if v is not None and k not in waived_profiles
    ]
    worst_considered_ratio = max(considered_ratios) if considered_ratios else None
    return {
        "noise_seed": int(noise_seed),
        "profiles": results,
        "gate_result": gate,
        "worst_to_baseline_ratio": _worst_ratio(results),
        "worst_considered_to_baseline_ratio": worst_considered_ratio,
        "ratio_by_profile": ratio_by_profile,
        "waived_profiles": sorted(list(waived_profiles)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic latency/noise stress suite.")
    parser.add_argument("input_recording", nargs="?", default=None)
    parser.add_argument("--lock-recording", default=None)
    parser.add_argument("--output-prefix", default="latnoise_suite")
    parser.add_argument("--noise-seed", type=int, default=1337)
    parser.add_argument(
        "--noise-seeds",
        default=None,
        help="Comma-separated seeds for cross-seed robustness (overrides --noise-seed).",
    )
    parser.add_argument(
        "--degrade-threshold",
        type=float,
        default=1.5,
        help="Worst/baseline steering mean ratio threshold for fail gate.",
    )
    parser.add_argument(
        "--keep-vehicle-frame-lookahead-ref",
        action="store_true",
        help="Do not disable vehicle-frame lookahead reference override.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path; default under tmp/analysis",
    )
    parser.add_argument(
        "--min-seed-pass-rate",
        type=float,
        default=1.0,
        help="Required fraction of seeds that must pass gate.",
    )
    parser.add_argument(
        "--waive-profiles",
        default=None,
        help="Comma-separated profile names to waive from fail threshold checks.",
    )
    parser.add_argument(
        "--waive-rationale",
        default=None,
        help="Required rationale when --waive-profiles is provided.",
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Force CV-based perception (override default segmentation).",
    )
    parser.add_argument(
        "--segmentation-checkpoint",
        default=str(DEFAULT_SEGMENTATION_CHECKPOINT),
        help="Segmentation checkpoint path (default: segnet_best.pt).",
    )
    args = parser.parse_args()

    input_recording = Path(args.input_recording) if args.input_recording else _pick_latest_recording()
    lock_recording = Path(args.lock_recording) if args.lock_recording else input_recording
    if not input_recording.exists():
        raise FileNotFoundError(f"Input recording not found: {input_recording}")
    if not lock_recording.exists():
        raise FileNotFoundError(f"Lock recording not found: {lock_recording}")
    use_segmentation = not bool(args.use_cv)
    if use_segmentation and not Path(args.segmentation_checkpoint).exists():
        raise FileNotFoundError(
            f"Segmentation checkpoint not found: {args.segmentation_checkpoint}. "
            "Provide --segmentation-checkpoint or run with --use-cv."
        )

    disable_vehicle_frame_lookahead_ref = not bool(args.keep_vehicle_frame_lookahead_ref)
    if args.noise_seeds:
        seeds = [int(s.strip()) for s in args.noise_seeds.split(",") if s.strip()]
    else:
        seeds = [int(args.noise_seed)]
    if not seeds:
        raise ValueError("No valid seeds supplied.")
    if not (0.0 <= float(args.min_seed_pass_rate) <= 1.0):
        raise ValueError("--min-seed-pass-rate must be in [0, 1].")

    waived_profiles: Set[str] = set()
    if args.waive_profiles:
        waived_profiles = {s.strip() for s in args.waive_profiles.split(",") if s.strip()}
        if not args.waive_rationale or not str(args.waive_rationale).strip():
            raise ValueError("--waive-rationale is required when --waive-profiles is set.")

    seed_runs: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_runs.append(
            _run_suite_for_seed(
                input_recording=input_recording,
                lock_recording=lock_recording,
                output_prefix=args.output_prefix,
                noise_seed=seed,
                degrade_threshold=float(args.degrade_threshold),
                waived_profiles=waived_profiles,
                disable_vehicle_frame_lookahead_ref=bool(disable_vehicle_frame_lookahead_ref),
                use_segmentation=use_segmentation,
                segmentation_checkpoint=args.segmentation_checkpoint,
            )
        )

    pass_states = {"passes-latency-noise-gate", "degraded-but-pass"}
    pass_count = sum(1 for r in seed_runs if str(r.get("gate_result", "")) in pass_states)
    pass_rate = float(pass_count) / float(len(seed_runs))
    worst_ratios = [float(r["worst_to_baseline_ratio"]) for r in seed_runs if r.get("worst_to_baseline_ratio") is not None]
    aggregate_worst_ratio = max(worst_ratios) if worst_ratios else None
    cross_seed_gate_pass = pass_rate >= float(args.min_seed_pass_rate)
    if cross_seed_gate_pass:
        aggregate_gate = "passes-latency-noise-gate"
    else:
        aggregate_gate = "fails-cross-seed-robustness-gate"

    out = {
        "input_recording": str(input_recording),
        "lock_recording": str(lock_recording),
        "noise_seeds": seeds,
        "degrade_threshold": float(args.degrade_threshold),
        "min_seed_pass_rate": float(args.min_seed_pass_rate),
        "waive_profiles": sorted(list(waived_profiles)),
        "waive_rationale": args.waive_rationale,
        "seed_runs": seed_runs,
        "cross_seed": {
            "pass_count": int(pass_count),
            "seed_count": int(len(seed_runs)),
            "pass_rate": float(pass_rate),
            "cross_seed_gate_pass": bool(cross_seed_gate_pass),
            "aggregate_worst_to_baseline_ratio": aggregate_worst_ratio,
        },
        "gate_result": aggregate_gate,
    }
    print("\n=== Stage 4 suite result ===")
    print(json.dumps(out, indent=2))

    output_json = (
        Path(args.output_json)
        if args.output_json
        else REPO_ROOT / "tmp" / "analysis" / f"{args.output_prefix}_report.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
