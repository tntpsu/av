#!/usr/bin/env python3
"""
trace_regime_transition.py — Signal-chain blame trace for PP↔MPC regime transitions.

Reads an HDF5 recording and identifies frames where the regime/controller mode
changes, then prints steering, blend weight, jerk, and curvature around each
transition to diagnose blend asymmetry, jerk spikes, and steering discontinuities.

Usage:
    python tools/analyze/trace_regime_transition.py --latest
    python tools/analyze/trace_regime_transition.py data/recordings/some_run.h5
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
    "blend_weight": [
        "control/regime_blend_weight",
        "control/blend_weight",
    ],
    "active_regime": [
        "control/active_regime",
        "control/controller_mode",
        "control/regime",
    ],
    "steering_before_limits": [
        "control/steering_before_limits",
    ],
    "steering": [
        "control/steering",
        "control/steering_command",
    ],
    "steering_jerk": [
        "control/steering_jerk",
    ],
    "pp_geometric_steering": [
        "control/pp_geometric_steering",
    ],
    "mpc_steering": [
        "control/mpc_steering",
        "control/mpc_steer_cmd",
    ],
    "speed": [
        "vehicle_state/speed",
        "vehicle_state/velocity",
        "vehicle/speed",
    ],
    "curvature_primary_abs": [
        "control/curvature_primary_abs",
        "control/path_curvature_primary_abs",
    ],
}


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


def _compute_jerk(h5f, steering_path, frame_idx):
    """Compute steering jerk as second difference of steering at frame_idx."""
    if steering_path is None:
        return None
    ds = h5f[steering_path]
    n = ds.shape[0]
    if frame_idx < 2 or frame_idx >= n:
        return None
    s0 = float(ds[frame_idx - 2])
    s1 = float(ds[frame_idx - 1])
    s2 = float(ds[frame_idx])
    if not (math.isfinite(s0) and math.isfinite(s1) and math.isfinite(s2)):
        return None
    return s2 - 2.0 * s1 + s0


def _detect_transitions(h5f, regime_path, blend_path):
    """
    Detect regime transition frames.

    Returns list of dicts with keys: frame, from_regime, to_regime.
    """
    transitions = []

    # Strategy 1: use active_regime string/numeric field
    if regime_path is not None:
        ds = h5f[regime_path]
        n = ds.shape[0]
        prev = None
        for i in range(n):
            raw = ds[i]
            if isinstance(raw, bytes):
                val = raw.decode("utf-8", errors="replace").strip()
            else:
                val = str(raw).strip() if not isinstance(raw, (float, np.floating)) else str(raw)
            if prev is not None and val != prev:
                transitions.append({
                    "frame": i,
                    "from_regime": prev,
                    "to_regime": val,
                })
            prev = val
        if transitions:
            return transitions

    # Strategy 2: use blend_weight crossing 0.5
    if blend_path is not None:
        ds = h5f[blend_path]
        n = ds.shape[0]
        arr = ds[:]
        for i in range(1, n):
            v_prev = float(arr[i - 1])
            v_curr = float(arr[i])
            if not (math.isfinite(v_prev) and math.isfinite(v_curr)):
                continue
            # Detect crossing through 0.5
            if (v_prev < 0.5 and v_curr >= 0.5):
                transitions.append({
                    "frame": i,
                    "from_regime": "PP",
                    "to_regime": "MPC",
                })
            elif (v_prev >= 0.5 and v_curr < 0.5):
                transitions.append({
                    "frame": i,
                    "from_regime": "MPC",
                    "to_regime": "PP",
                })

    return transitions


def _print_transition_trace(idx, transition, h5f, paths):
    """Print the signal-chain blame block for one regime transition."""
    frame = transition["frame"]
    from_regime = transition["from_regime"]
    to_regime = transition["to_regime"]

    # Read before/after values (frame-1 and frame)
    before_frame = max(0, frame - 1)
    after_frame = frame

    speed = _read_scalar(h5f, paths["speed"], frame)

    steer_before = _read_scalar(h5f, paths["steering"], before_frame)
    steer_after = _read_scalar(h5f, paths["steering"], after_frame)

    blend_before = _read_scalar(h5f, paths["blend_weight"], before_frame)
    blend_after = _read_scalar(h5f, paths["blend_weight"], after_frame)

    # Jerk: try dedicated field first, then compute from steering
    jerk_before = _read_scalar(h5f, paths["steering_jerk"], before_frame)
    jerk_after = _read_scalar(h5f, paths["steering_jerk"], after_frame)
    if jerk_after is None:
        jerk_after = _compute_jerk(h5f, paths["steering"], after_frame)
    if jerk_before is None:
        jerk_before = _compute_jerk(h5f, paths["steering"], before_frame)

    curv_before = _read_scalar(h5f, paths["curvature_primary_abs"], before_frame)
    curv_after = _read_scalar(h5f, paths["curvature_primary_abs"], after_frame)

    pp_steer = _read_scalar(h5f, paths["pp_geometric_steering"], frame)
    mpc_steer = _read_scalar(h5f, paths["mpc_steering"], frame)

    # Deltas
    def _delta(a, b):
        if a is not None and b is not None:
            return b - a
        return None

    steer_delta = _delta(steer_before, steer_after)
    blend_delta = _delta(blend_before, blend_after)
    jerk_delta = _delta(jerk_before, jerk_after)
    curv_delta = _delta(curv_before, curv_after)

    speed_str = f"{speed:.1f}" if speed is not None else "N/A"

    print(f"\nREGIME TRANSITION #{idx}")
    print("\u2550" * 40)
    print(f"Frame: {frame}  |  {from_regime} -> {to_regime}  |  speed: {speed_str} m/s")
    if pp_steer is not None or mpc_steer is not None:
        print(f"  PP steer: {_fmt(pp_steer, 5)}  |  MPC steer: {_fmt(mpc_steer, 5)}")
    print()
    print(f"  {'Signal':<25s} {'Before':>8s}   {'After':>8s}   {'Delta':>8s}")
    print(f"  {'─' * 25} {'─' * 8}   {'─' * 8}   {'─' * 8}")
    print(f"  {'Steering':<25s} {_fmt(steer_before, 5)}   {_fmt(steer_after, 5)}   {_fmt(steer_delta, 5)}")
    print(f"  {'Blend weight':<25s} {_fmt(blend_before)}   {_fmt(blend_after)}   {_fmt(blend_delta)}")
    print(f"  {'Steering jerk':<25s} {_fmt(jerk_before, 5)}   {_fmt(jerk_after, 5)}   {_fmt(jerk_delta, 5)}")
    print(f"  {'Curvature':<25s} {_fmt(curv_before, 5)}   {_fmt(curv_after, 5)}   {_fmt(curv_delta, 5)}")
    print()

    # Jerk check
    jerk_val = jerk_after if jerk_after is not None else jerk_before
    if jerk_val is not None and abs(jerk_val) >= 6.0:
        print(f"  JERK CHECK: SPIKE (|jerk| = {abs(jerk_val):.3f} >= 6.0)")
    elif jerk_val is not None:
        print(f"  JERK CHECK: OK (|jerk| = {abs(jerk_val):.3f} < 6.0)")
    else:
        print("  JERK CHECK: N/A (no jerk data)")

    # Blend check
    if blend_delta is not None and abs(blend_delta) > 0.5:
        print(f"  BLEND CHECK: DISCONTINUOUS (step of {blend_delta:.3f})")
    elif blend_delta is not None:
        print(f"  BLEND CHECK: OK (delta {blend_delta:.3f})")
    else:
        print("  BLEND CHECK: N/A (no blend data)")

    # Blame
    if jerk_val is not None and abs(jerk_val) >= 6.0:
        print(f"  BLAME: blend asymmetry -- {from_regime}->{to_regime} path missing blend logic")
    elif blend_delta is not None and abs(blend_delta) > 0.5:
        print(f"  BLAME: abrupt blend step -- {from_regime}->{to_regime} blend ramp too fast")
    elif steer_delta is not None and abs(steer_delta) > 0.05:
        print(f"  BLAME: steering discontinuity ({steer_delta:+.5f}) at handoff")
    else:
        print("  BLAME: transition appears smooth -- no dominant issue")
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
        description="Trace PP<->MPC regime transitions in an HDF5 recording",
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

        # Detect transitions
        transitions = _detect_transitions(h5f, paths["active_regime"], paths["blend_weight"])

        if not transitions:
            print("No regime transitions found in this recording.")
            return

        print(f"Found {len(transitions)} regime transition(s)\n")

        for i, t in enumerate(transitions, 1):
            _print_transition_trace(i, t, h5f, paths)

    # Summary
    print("REGIME TRANSITION SUMMARY")
    print("\u2550" * 40)
    print(f"Total transitions: {len(transitions)}")
    directions = {}
    for t in transitions:
        key = f"{t['from_regime']}->{t['to_regime']}"
        directions[key] = directions.get(key, 0) + 1
    for key, count in sorted(directions.items()):
        print(f"  {key}: {count}")
    print()


if __name__ == "__main__":
    main()
