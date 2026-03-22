#!/usr/bin/env python3
"""CLI: grade vs lateral breakdown (parity with PhilViz ``/grade-lateral``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from grade_lateral_analysis import analyze_grade_lateral  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Grade-lateral analysis (grade_lateral_v1 JSON).")
    p.add_argument("recording", type=Path, help="Path to recording .h5")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print compact JSON to stdout",
    )
    p.add_argument(
        "--pre-failure-only",
        dest="pre_failure_only",
        action="store_true",
        default=True,
        help="Clip to executive_summary failure_frame (default: on)",
    )
    p.add_argument(
        "--no-pre-failure-only",
        dest="pre_failure_only",
        action="store_false",
        help="Use full recording length",
    )
    p.add_argument(
        "--grade-threshold",
        type=float,
        default=0.02,
        help="Rise/run threshold for flat band (default: 0.02)",
    )
    args = p.parse_args()
    out = analyze_grade_lateral(
        args.recording,
        pre_failure_only=bool(args.pre_failure_only),
        grade_threshold=float(args.grade_threshold),
    )
    print(json.dumps(out, indent=2))
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
