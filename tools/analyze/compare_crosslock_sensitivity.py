#!/usr/bin/env python3
"""
Compare self-lock vs cross-lock trajectory replay summaries.

This helper quantifies how much control output changes when trajectory
reference is swapped across runs, normalized by self-lock baseline drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    required = [
        "input_recording",
        "lock_recording",
        "steering_abs_diff_mean_vs_source",
        "steering_abs_diff_p95_vs_source",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing keys in {path}: {missing}")
    return data


def _safe_ratio(num: float, den: float, eps: float = 1e-6) -> float:
    return float(num) / max(float(den), eps)


def _classify(
    amp_a_mean: float,
    amp_b_mean: float,
    amp_a_p95: float,
    amp_b_p95: float,
    strong_threshold: float,
    moderate_threshold: float,
) -> str:
    if (
        amp_a_mean >= strong_threshold
        and amp_b_mean >= strong_threshold
        and amp_a_p95 >= strong_threshold
        and amp_b_p95 >= strong_threshold
    ):
        return "strong-trajectory-sensitivity"
    if (
        amp_a_mean >= moderate_threshold
        and amp_b_mean >= moderate_threshold
    ) or (
        amp_a_p95 >= strong_threshold
        or amp_b_p95 >= strong_threshold
    ):
        return "moderate-trajectory-sensitivity"
    return "low-trajectory-sensitivity"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare self-lock and cross-lock sensitivity summaries."
    )
    parser.add_argument("--self-a", required=True, help="Summary JSON for A->A self-lock")
    parser.add_argument("--self-b", required=True, help="Summary JSON for B->B self-lock")
    parser.add_argument(
        "--cross-a-lock-b",
        required=True,
        help="Summary JSON for A input locked to B trajectory",
    )
    parser.add_argument(
        "--cross-b-lock-a",
        required=True,
        help="Summary JSON for B input locked to A trajectory",
    )
    parser.add_argument(
        "--strong-threshold",
        type=float,
        default=2.0,
        help="Amplification ratio threshold for strong sensitivity (default: 2.0)",
    )
    parser.add_argument(
        "--moderate-threshold",
        type=float,
        default=1.5,
        help="Amplification ratio threshold for moderate sensitivity (default: 1.5)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write result JSON",
    )
    args = parser.parse_args()

    self_a = _load_summary(Path(args.self_a))
    self_b = _load_summary(Path(args.self_b))
    cross_ab = _load_summary(Path(args.cross_a_lock_b))
    cross_ba = _load_summary(Path(args.cross_b_lock_a))

    self_a_mean = float(self_a["steering_abs_diff_mean_vs_source"] or 0.0)
    self_b_mean = float(self_b["steering_abs_diff_mean_vs_source"] or 0.0)
    self_a_p95 = float(self_a["steering_abs_diff_p95_vs_source"] or 0.0)
    self_b_p95 = float(self_b["steering_abs_diff_p95_vs_source"] or 0.0)
    cross_ab_mean = float(cross_ab["steering_abs_diff_mean_vs_source"] or 0.0)
    cross_ba_mean = float(cross_ba["steering_abs_diff_mean_vs_source"] or 0.0)
    cross_ab_p95 = float(cross_ab["steering_abs_diff_p95_vs_source"] or 0.0)
    cross_ba_p95 = float(cross_ba["steering_abs_diff_p95_vs_source"] or 0.0)

    amp_a_mean = _safe_ratio(cross_ab_mean, self_a_mean)
    amp_b_mean = _safe_ratio(cross_ba_mean, self_b_mean)
    amp_a_p95 = _safe_ratio(cross_ab_p95, self_a_p95)
    amp_b_p95 = _safe_ratio(cross_ba_p95, self_b_p95)
    amp_mean_avg = (amp_a_mean + amp_b_mean) / 2.0
    amp_p95_avg = (amp_a_p95 + amp_b_p95) / 2.0

    classification = _classify(
        amp_a_mean=amp_a_mean,
        amp_b_mean=amp_b_mean,
        amp_a_p95=amp_a_p95,
        amp_b_p95=amp_b_p95,
        strong_threshold=float(args.strong_threshold),
        moderate_threshold=float(args.moderate_threshold),
    )

    result = {
        "self_lock": {
            "a_input": {
                "input_recording": self_a["input_recording"],
                "lock_recording": self_a["lock_recording"],
                "steering_abs_diff_mean_vs_source": self_a_mean,
                "steering_abs_diff_p95_vs_source": self_a_p95,
            },
            "b_input": {
                "input_recording": self_b["input_recording"],
                "lock_recording": self_b["lock_recording"],
                "steering_abs_diff_mean_vs_source": self_b_mean,
                "steering_abs_diff_p95_vs_source": self_b_p95,
            },
        },
        "cross_lock": {
            "a_input_lock_b": {
                "input_recording": cross_ab["input_recording"],
                "lock_recording": cross_ab["lock_recording"],
                "steering_abs_diff_mean_vs_source": cross_ab_mean,
                "steering_abs_diff_p95_vs_source": cross_ab_p95,
            },
            "b_input_lock_a": {
                "input_recording": cross_ba["input_recording"],
                "lock_recording": cross_ba["lock_recording"],
                "steering_abs_diff_mean_vs_source": cross_ba_mean,
                "steering_abs_diff_p95_vs_source": cross_ba_p95,
            },
        },
        "amplification": {
            "a_mean_ratio": amp_a_mean,
            "b_mean_ratio": amp_b_mean,
            "mean_ratio_avg": amp_mean_avg,
            "a_p95_ratio": amp_a_p95,
            "b_p95_ratio": amp_b_p95,
            "p95_ratio_avg": amp_p95_avg,
        },
        "thresholds": {
            "strong": float(args.strong_threshold),
            "moderate": float(args.moderate_threshold),
        },
        "classification": classification,
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
