#!/usr/bin/env python3
"""
Verify deterministic timestamp pairing parity for replay analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def find_latest_recording() -> Path | None:
    candidates = sorted(Path("data/recordings").glob("recording_*.h5"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def select_indices(
    camera_ts: np.ndarray,
    series_ts: np.ndarray,
    max_diff_s: float,
) -> list[int | None]:
    selected: list[int | None] = []
    if camera_ts.size == 0:
        return selected
    if series_ts.size == 0:
        return [None for _ in range(int(camera_ts.size))]
    for ts in camera_ts:
        if not np.isfinite(ts):
            selected.append(None)
            continue
        diffs = np.abs(series_ts - ts)
        idx = int(np.argmin(diffs))
        if float(diffs[idx]) <= max_diff_s:
            selected.append(idx)
        else:
            selected.append(None)
    return selected


def digest_selection(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check deterministic replay pairing parity")
    parser.add_argument("recording", nargs="?", help="Path to recording .h5")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    parser.add_argument("--max-diff-ms", type=float, default=200.0, help="Pairing window in milliseconds")
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    rec_path: Path | None
    if args.latest:
        rec_path = find_latest_recording()
    elif args.recording:
        rec_path = Path(args.recording)
    else:
        rec_path = find_latest_recording()
    if rec_path is None or not rec_path.exists():
        raise SystemExit("No recording found.")

    max_diff_s = float(args.max_diff_ms) / 1000.0
    with h5py.File(rec_path, "r") as f:
        camera_ts = np.asarray(f["camera/timestamps"][:], dtype=np.float64) if "camera/timestamps" in f else np.array([])
        traj_ts = np.asarray(f["trajectory/timestamps"][:], dtype=np.float64) if "trajectory/timestamps" in f else np.array([])
        ctrl_ts = np.asarray(f["control/timestamps"][:], dtype=np.float64) if "control/timestamps" in f else np.array([])
        veh_ts = np.asarray(f["vehicle/timestamps"][:], dtype=np.float64) if "vehicle/timestamps" in f else np.array([])

    pass_a = {
        "trajectory": select_indices(camera_ts, traj_ts, max_diff_s),
        "control": select_indices(camera_ts, ctrl_ts, max_diff_s),
        "vehicle": select_indices(camera_ts, veh_ts, max_diff_s),
    }
    pass_b = {
        "trajectory": select_indices(camera_ts, traj_ts, max_diff_s),
        "control": select_indices(camera_ts, ctrl_ts, max_diff_s),
        "vehicle": select_indices(camera_ts, veh_ts, max_diff_s),
    }
    parity_ok = pass_a == pass_b
    out = {
        "recording": str(rec_path),
        "camera_frames": int(camera_ts.size),
        "max_diff_ms": float(args.max_diff_ms),
        "parity_ok": bool(parity_ok),
        "selection_hash_a": digest_selection(pass_a),
        "selection_hash_b": digest_selection(pass_b),
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("tmp/analysis") / f"sync_replay_parity_{rec_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
