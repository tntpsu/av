#!/usr/bin/env python3
"""Update a track's score floor in tests/scores.json after a validated run.

Usage:
    python tools/update_score_floor.py --track sloop --score 92.5
    python tools/update_score_floor.py --track hill_highway --score 94.0 \
        --notes "After oscillation fix 2026-02-17"

Rules enforced by this script:
  - You can only RAISE floors, never lower them (use --force to override).
  - The script reads the latest recording for the given track to get layer scores.
  - Floors are updated to 95% of the new score (5% headroom).
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_ROOT   = Path(__file__).resolve().parent.parent
_FLOORS = _ROOT / "tests" / "scores.json"

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))


def _load() -> dict:
    with open(_FLOORS) as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(_FLOORS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[update_score_floor] Saved {_FLOORS}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track",   required=True, help="Track name (e.g. sloop, hill_highway)")
    p.add_argument("--score",   required=True, type=float, help="New overall score")
    p.add_argument("--notes",   default="",    help="Optional annotation")
    p.add_argument("--force",   action="store_true",
                   help="Allow lowering a floor (use with caution)")
    p.add_argument("--headroom", type=float, default=0.95,
                   help="Fraction of score to use as floor (default 0.95 = 5%% headroom)")
    args = p.parse_args()

    data   = _load()
    floors = data.setdefault("floors", {})
    track  = args.track

    new_floor = round(args.score * args.headroom, 1)

    existing = floors.get(track, {}).get("overall", 0.0)
    if new_floor < existing and not args.force:
        print(f"[update_score_floor] ERROR: new floor {new_floor} < existing {existing}.")
        print(f"  Use --force to lower the floor (requires team review).")
        sys.exit(1)

    if track not in floors:
        floors[track] = {}

    floors[track]["overall"]  = new_floor
    floors[track]["notes"]    = args.notes or f"Updated {date.today()}, score={args.score}"
    data["_last_updated"]     = str(date.today())

    # Optionally read layer breakdown from latest recording
    try:
        from drive_summary_core import analyze_recording_summary  # type: ignore
        from test_score_floors import _find_latest_recording      # type: ignore
        recording = _find_latest_recording(track)
        if recording:
            result = analyze_recording_summary(recording)
            layers = result.get("layer_breakdowns", {})
            for layer_name, layer_data in layers.items():
                layer_score = float(layer_data.get("final_score", 0.0))
                layer_floor = round(layer_score * args.headroom, 1)
                old_floor   = float(floors[track].get(layer_name, 0.0))
                if layer_floor >= old_floor or args.force:
                    floors[track][layer_name] = layer_floor
            print(f"[update_score_floor] Layer floors updated from {recording.name}")
    except Exception as e:
        print(f"[update_score_floor] Skipping layer update: {e}")

    _save(data)
    print(f"[update_score_floor] Track '{track}': overall floor → {new_floor}")


if __name__ == "__main__":
    main()
