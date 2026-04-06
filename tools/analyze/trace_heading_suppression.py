#!/usr/bin/env python3
"""
trace_heading_suppression.py — Trace heading zero gate activation/release events.

Reads an HDF5 recording and identifies frames where the heading suppression
gate changes state (active<->released), printing the signals that drive the
gate to diagnose late release or incorrect suppression during curves.

Usage:
    python tools/analyze/trace_heading_suppression.py --latest
    python tools/analyze/trace_heading_suppression.py data/recordings/some_run.h5
"""

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── HDF5 field map ──────────────────────────────────────────────────
FIELD_CANDIDATES = {
    "heading_zero_gate_active": [
        "control/diag_heading_zero_gate_active",
        "trajectory/heading_zero_gate_active",
        "control/heading_zero_gate_active",
    ],
    "curve_local_phase": [
        "control/curve_local_phase",
        "control/curve_phase",
    ],
    "curve_local_state": [
        "control/curve_local_state",
    ],
    "local_gate_weight": [
        "control/reference_lookahead_local_gate_weight",
    ],
    "curvature_primary_abs": [
        "control/curvature_primary_abs",
        "control/path_curvature_primary_abs",
    ],
    "curvature_preview_abs": [
        "control/curvature_preview_abs",
        "control/path_curvature_preview_abs",
    ],
    "far_preview_phase": [
        "control/far_preview_phase",
        "trajectory/curve_preview_far_phase",
    ],
    "speed": [
        "vehicle_state/speed",
        "vehicle_state/velocity",
        "vehicle/speed",
    ],
}

# Thresholds
CURVATURE_STRAIGHT_THRESHOLD = 0.001
FAR_PREVIEW_RELEASE_THRESHOLD = 0.06


def _resolve_field(h5f, name):
    """Return the first existing dataset path for *name*, or None."""
    for candidate in FIELD_CANDIDATES.get(name, []):
        if candidate in h5f:
            return candidate
    return None


def _read_scalar(h5f, field_path, frame_idx):
    """Read a single float from an HDF5 dataset at *frame_idx*."""
    if field_path is None:
        return None
    ds = h5f[field_path]
    if frame_idx < 0 or frame_idx >= ds.shape[0]:
        return None
    val = float(ds[frame_idx])
    return val if math.isfinite(val) else None


def _read_string(h5f, field_path, frame_idx):
    """Read a single string from an HDF5 dataset at *frame_idx*."""
    if field_path is None:
        return None
    ds = h5f[field_path]
    if frame_idx < 0 or frame_idx >= ds.shape[0]:
        return None
    raw = ds[frame_idx]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def _fmt(val, decimals=4, width=8):
    """Format a numeric value for display."""
    if val is None:
        return "N/A".rjust(width)
    return f"{val:.{decimals}f}".rjust(width)


def _detect_gate_events(h5f, gate_path):
    """
    Detect heading suppression state changes.

    Returns list of dicts: frame, direction ('ON' or 'OFF'), duration (frames until next change).
    """
    if gate_path is None:
        return []

    ds = h5f[gate_path]
    n = ds.shape[0]
    arr = ds[:]

    # Convert to boolean active state
    active = np.zeros(n, dtype=bool)
    for i in range(n):
        val = arr[i]
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace").strip().lower()
            active[i] = val in ("1", "true", "yes")
        elif isinstance(val, (float, np.floating, int, np.integer)):
            active[i] = float(val) > 0.5
        else:
            active[i] = str(val).strip().lower() in ("1", "true", "yes")

    events = []
    for i in range(1, n):
        if active[i] != active[i - 1]:
            direction = "ON" if active[i] else "OFF"
            events.append({
                "frame": i,
                "direction": direction,
            })

    # Compute duration for each event (frames until next state change)
    for j in range(len(events)):
        if j + 1 < len(events):
            events[j]["duration"] = events[j + 1]["frame"] - events[j]["frame"]
        else:
            events[j]["duration"] = n - events[j]["frame"]

    return events, active


def _print_gate_event(idx, event, h5f, paths):
    """Print the signal trace for one heading suppression state change."""
    frame = event["frame"]
    direction = event["direction"]
    duration = event["duration"]
    gate_state = "active" if direction == "ON" else "released"

    curv_at_car = _read_scalar(h5f, paths["curvature_primary_abs"], frame)
    curv_preview = _read_scalar(h5f, paths["curvature_preview_abs"], frame)
    far_preview = _read_scalar(h5f, paths["far_preview_phase"], frame)
    curve_state = _read_string(h5f, paths["curve_local_state"], frame)
    lgw = _read_scalar(h5f, paths["local_gate_weight"], frame)
    speed = _read_scalar(h5f, paths["speed"], frame)
    curve_phase = _read_scalar(h5f, paths["curve_local_phase"], frame)

    # Status helpers
    def _curv_status(val):
        if val is None:
            return "N/A"
        return "curve" if val > CURVATURE_STRAIGHT_THRESHOLD else "straight"

    def _far_preview_status(val):
        if val is None:
            return "N/A"
        return f"above release threshold {FAR_PREVIEW_RELEASE_THRESHOLD}" if val >= FAR_PREVIEW_RELEASE_THRESHOLD else f"below release threshold {FAR_PREVIEW_RELEASE_THRESHOLD}"

    print(f"\nHEADING SUPPRESSION {direction} #{idx}")
    print("\u2550" * 40)
    print(f"Frame: {frame}  |  gate: {gate_state}  |  duration: {duration} frames")
    print()
    print(f"  {'Signal':<25s} {'Value':>8s}    Status")
    print(f"  {'─' * 25} {'─' * 8}    {'─' * 30}")
    print(f"  {'Curvature (at-car)':<25s} {_fmt(curv_at_car, 5)}    {_curv_status(curv_at_car)}")
    print(f"  {'Curvature (preview)':<25s} {_fmt(curv_preview, 5)}    {_curv_status(curv_preview)}")
    print(f"  {'Far preview phase':<25s} {_fmt(far_preview)}    {_far_preview_status(far_preview)}")
    print(f"  {'Curve local state':<25s} {(curve_state or 'N/A'):>8s}")
    print(f"  {'local_gate_weight':<25s} {_fmt(lgw)}")
    print(f"  {'Curve local phase':<25s} {_fmt(curve_phase)}")
    print(f"  {'Speed':<25s} {_fmt(speed, 1)}    m/s")

    # Release reason analysis (only for OFF events)
    if direction == "OFF":
        print()
        print("  RELEASE REASON:")
        far_preview_ok = far_preview is not None and far_preview >= FAR_PREVIEW_RELEASE_THRESHOLD
        print(f"    - far_preview_phase >= {FAR_PREVIEW_RELEASE_THRESHOLD}: {'yes' if far_preview_ok else 'no'}")

        state_entry_commit = curve_state is not None and curve_state.upper() in ("ENTRY", "COMMIT")
        print(f"    - scheduler state ENTRY/COMMIT: {'yes' if state_entry_commit else 'no'}")

        # time_to_curve approximation: if speed > 0 and we have curvature preview data
        ttc_ok = False
        if speed is not None and speed > 0.5 and curv_preview is not None and curv_preview > CURVATURE_STRAIGHT_THRESHOLD:
            # Rough heuristic -- can't compute exact TTC without distance, but flag if curve is near
            ttc_ok = True  # preview curvature present suggests curve is near
        print(f"    - time_to_curve <= 1.35s: {'yes (curve preview active)' if ttc_ok else 'no'}")

        map_curv_ok = curv_at_car is not None and curv_at_car > CURVATURE_STRAIGHT_THRESHOLD
        print(f"    - map curvature > {CURVATURE_STRAIGHT_THRESHOLD}: {'yes' if map_curv_ok else 'no'}")
    print()


def _print_summary(events_and_active, h5f, paths):
    """Print the heading suppression summary."""
    events, active = events_and_active
    n = len(active)

    active_count = int(np.sum(active))
    active_pct = 100.0 * active_count / n if n > 0 else 0

    # Classify frames as curve or straight based on curvature
    curv_path = paths["curvature_primary_abs"]
    curve_frames = 0
    straight_frames = 0
    suppressed_on_curve = 0
    suppressed_on_straight = 0

    if curv_path is not None:
        curv_arr = h5f[curv_path][:]
        for i in range(n):
            val = float(curv_arr[i]) if i < len(curv_arr) else 0.0
            if not math.isfinite(val):
                val = 0.0
            if val > CURVATURE_STRAIGHT_THRESHOLD:
                curve_frames += 1
                if active[i]:
                    suppressed_on_curve += 1
            else:
                straight_frames += 1
                if active[i]:
                    suppressed_on_straight += 1

    print("\nHEADING SUPPRESSION SUMMARY")
    print("\u2550" * 40)
    print(f"Total frames: {n}")
    print(f"Suppression active: {active_count} frames ({active_pct:.1f}%)")

    if curve_frames > 0:
        pct = 100.0 * suppressed_on_curve / curve_frames
        print(f"On curves: {suppressed_on_curve} frames ({pct:.1f}% of {curve_frames} curve frames)")
    else:
        print("On curves: 0 frames (no curve frames detected)")

    if straight_frames > 0:
        pct = 100.0 * suppressed_on_straight / straight_frames
        print(f"On straights: {suppressed_on_straight} frames ({pct:.1f}% of {straight_frames} straight frames)")
    else:
        print("On straights: 0 frames (no straight frames detected)")

    # Blame
    if curve_frames > 0 and (100.0 * suppressed_on_curve / curve_frames) > 10.0:
        print()
        print("  BLAME: Heading suppression active during curves -- check release thresholds")
        print("    - far_preview_phase release threshold may be too high")
        print("    - curve scheduler state transitions may be delayed")
    print()


# ── Main ────────────────────────────────────────────────────────────

def resolve_recording(args):
    """Return a Path to the recording file."""
    if args.latest:
        recordings_dir = Path("data/recordings")
        if not recordings_dir.exists():
            print("No recordings directory found")
            sys.exit(1)
        recordings = sorted(
            recordings_dir.glob("*.h5"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not recordings:
            print("No recordings found")
            sys.exit(1)
        return recordings[0]
    elif args.recording:
        p = Path(args.recording)
        if not p.exists():
            print(f"Error: Recording file not found: {p}")
            sys.exit(1)
        return p
    else:
        print("Error: Please provide a recording file or use --latest")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Trace heading zero gate activation/release in an HDF5 recording",
    )
    parser.add_argument("recording", nargs="?", help="Path to recording file")
    parser.add_argument("--latest", action="store_true", help="Use the latest recording")
    args = parser.parse_args()

    recording_path = resolve_recording(args)
    print(f"Recording: {recording_path}")
    print()

    with h5py.File(str(recording_path), "r") as h5f:
        # Resolve field paths
        paths = {name: _resolve_field(h5f, name) for name in FIELD_CANDIDATES}

        if paths["heading_zero_gate_active"] is None:
            print("No heading_zero_gate_active field found in this recording.")
            print("Checked candidates:")
            for c in FIELD_CANDIDATES["heading_zero_gate_active"]:
                print(f"  - {c}")
            return

        result = _detect_gate_events(h5f, paths["heading_zero_gate_active"])
        events, active = result

        if not events:
            print("No heading suppression state changes found in this recording.")
            # Still print summary
            _print_summary((events, active), h5f, paths)
            return

        print(f"Found {len(events)} heading suppression state change(s)\n")

        for i, event in enumerate(events, 1):
            _print_gate_event(i, event, h5f, paths)

        _print_summary((events, active), h5f, paths)


if __name__ == "__main__":
    main()
