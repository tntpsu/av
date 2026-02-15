#!/usr/bin/env python3
"""
Compare control-lock self and cross results to quantify sensitivity.

Unlike trajectory cross-lock, control-lock self cases are expected to be exactly
zero vs input source, so this comparator focuses on:
- lock fidelity (vs_lock deltas)
- cross-shift magnitude (vs_input_source deltas)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load(path: Path) -> Dict[str, Any]:
    d = json.loads(path.read_text())
    needed = [
        "input_recording",
        "lock_recording",
        "steering_abs_diff_mean_vs_lock",
        "steering_abs_diff_p95_vs_lock",
        "steering_abs_diff_mean_vs_input_source",
        "steering_abs_diff_p95_vs_input_source",
    ]
    missing = [k for k in needed if k not in d]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    return d


def _f(x: Any) -> float:
    return float(x) if x is not None else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="Compare control-lock self/cross summaries.")
    p.add_argument("--self-a", required=True)
    p.add_argument("--self-b", required=True)
    p.add_argument("--cross-a-lock-b", required=True)
    p.add_argument("--cross-b-lock-a", required=True)
    p.add_argument("--fidelity-threshold", type=float, default=1e-4)
    p.add_argument("--steer-mean-strong", type=float, default=0.25)
    p.add_argument("--steer-p95-strong", type=float, default=0.8)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    sa = _load(Path(args.self_a))
    sb = _load(Path(args.self_b))
    cab = _load(Path(args.cross_a_lock_b))
    cba = _load(Path(args.cross_b_lock_a))

    fidelity_ok = all(
        _f(v) <= args.fidelity_threshold
        for v in [
            sa["steering_abs_diff_mean_vs_lock"],
            sa["throttle_abs_diff_mean_vs_lock"],
            sa["brake_abs_diff_mean_vs_lock"],
            sb["steering_abs_diff_mean_vs_lock"],
            sb["throttle_abs_diff_mean_vs_lock"],
            sb["brake_abs_diff_mean_vs_lock"],
            cab["steering_abs_diff_mean_vs_lock"],
            cab["throttle_abs_diff_mean_vs_lock"],
            cab["brake_abs_diff_mean_vs_lock"],
            cba["steering_abs_diff_mean_vs_lock"],
            cba["throttle_abs_diff_mean_vs_lock"],
            cba["brake_abs_diff_mean_vs_lock"],
        ]
    )

    cross_steer_mean = (_f(cab["steering_abs_diff_mean_vs_input_source"]) + _f(cba["steering_abs_diff_mean_vs_input_source"])) / 2.0
    cross_steer_p95 = (_f(cab["steering_abs_diff_p95_vs_input_source"]) + _f(cba["steering_abs_diff_p95_vs_input_source"])) / 2.0
    cross_throttle_mean = (_f(cab.get("throttle_abs_diff_mean_vs_input_source")) + _f(cba.get("throttle_abs_diff_mean_vs_input_source"))) / 2.0
    cross_brake_mean = (_f(cab.get("brake_abs_diff_mean_vs_input_source")) + _f(cba.get("brake_abs_diff_mean_vs_input_source"))) / 2.0

    if cross_steer_mean >= args.steer_mean_strong and cross_steer_p95 >= args.steer_p95_strong:
        cls = "strong-control-sensitivity"
    elif cross_steer_mean >= (0.5 * args.steer_mean_strong):
        cls = "moderate-control-sensitivity"
    else:
        cls = "low-control-sensitivity"

    out = {
        "fidelity_ok": bool(fidelity_ok),
        "fidelity_threshold": float(args.fidelity_threshold),
        "cross_shift": {
            "steering_mean_vs_input_source_avg": cross_steer_mean,
            "steering_p95_vs_input_source_avg": cross_steer_p95,
            "throttle_mean_vs_input_source_avg": cross_throttle_mean,
            "brake_mean_vs_input_source_avg": cross_brake_mean,
        },
        "classification": cls,
        "inputs": {
            "self_a": args.self_a,
            "self_b": args.self_b,
            "cross_a_lock_b": args.cross_a_lock_b,
            "cross_b_lock_a": args.cross_b_lock_a,
        },
    }

    print(json.dumps(out, indent=2))
    if args.output_json:
        op = Path(args.output_json)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2))
        print(f"\nSaved: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
