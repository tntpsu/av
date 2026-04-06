#!/usr/bin/env python3
"""
trace_curve_entry.py — Signal-chain blame trace for each curve event.

Reads an HDF5 recording, extracts per-curve event data via
analyze_recording_summary(), then reads raw HDF5 signals at each
curve_start_frame to produce a structured blame report.

Usage:
    python tools/analyze/trace_curve_entry.py --latest
    python tools/analyze/trace_curve_entry.py data/recordings/some_run.h5
"""

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.drive_summary_core import analyze_recording_summary


# ── HDF5 field map ──────────────────────────────────────────────────
# Each key is the logical name used in this script; the value is a list
# of candidate HDF5 dataset paths (first match wins).
FIELD_CANDIDATES = {
    "curve_local_phase": [
        "control/curve_local_phase",
        "control/curve_local_phase_weight",
    ],
    "local_gate_weight": [
        "control/reference_lookahead_local_gate_weight",
    ],
    "entry_severity": [
        "control/curve_local_entry_severity",
    ],
    "entry_weight_source": [
        "control/reference_lookahead_entry_weight_source",
    ],
    "ld_target": [
        "control/reference_lookahead_target",
        "control/reference_lookahead_active",
    ],
    "ld_after_slew": [
        "control/reference_lookahead_after_slew",
    ],
    "ld_final": [
        "control/pp_lookahead_distance",
    ],
    "pp_floor_active": [
        "control/pp_curve_local_floor_active",
    ],
    "pp_floor_m": [
        "control/pp_curve_local_floor_m",
    ],
    "pp_geometric_steering": [
        "control/pp_geometric_steering",
    ],
    "pp_map_ff_applied": [
        "control/pp_map_ff_applied",
    ],
    "steering_before_limits": [
        "control/steering_before_limits",
    ],
    "curvature_primary_abs": [
        "control/curvature_primary_abs",
        "control/path_curvature_primary_abs",
        "control/path_curvature_for_map_ff",
    ],
}


def _resolve_field(h5f, name):
    """Return the first existing dataset for *name*, or None."""
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


def _status(val, thresholds):
    """
    Return a status string based on threshold dict.
    thresholds is a list of (test_fn, label) checked in order; first True wins.
    Falls back to the last label.
    """
    if val is None:
        return "N/A"
    for test_fn, label in thresholds:
        if test_fn(val):
            return label
    return thresholds[-1][1] if thresholds else "?"


# ── Blame logic ─────────────────────────────────────────────────────

def _compute_blame(signals):
    """Return (blame_str, suggestion_str) from signal dict."""
    ld_final = signals.get("ld_final")
    floor_rescue = signals.get("floor_rescue_max")
    lgw = signals.get("local_gate_weight")
    severity = signals.get("entry_severity")
    map_ff = signals.get("pp_map_ff_applied")

    if ld_final is not None and ld_final > 10.0:
        return (
            "Ld_target too long -- entry table too conservative or lgw too low",
            "Lower reference_lookahead entry table targets or increase lgw sensitivity",
        )
    if floor_rescue is not None and floor_rescue > 1.0:
        return (
            "PP floor doing heavy rescue -- planner contraction insufficient",
            "Increase planner Ld contraction rate or lower floor rescue ceiling",
        )
    if lgw is not None and lgw < 0.3:
        return (
            "Proximity signal too low -- arm distance or time threshold too conservative",
            "Lower curve_local arm distance/time thresholds to fire earlier",
        )
    if severity is not None and severity < 0.2:
        return (
            "Curve too gentle for severity scaling -- consider lowering severity threshold",
            "Lower entry_severity_threshold in config to allow gentler curves to trigger",
        )
    if map_ff is not None and abs(map_ff) < 0.02:
        return (
            "FF contribution negligible -- curvature below FF minimum threshold",
            "Lower map_ff_curvature_min or increase map_ff_gain",
        )
    return ("No dominant constraint identified", "Inspect frame-level traces for subtle interactions")


# ── Per-curve report ────────────────────────────────────────────────

def _print_curve_trace(curve_idx_label, event, h5f):
    """Print the signal-chain blame block for one curve event."""
    curve_start = event.get("curve_start_frame")
    if curve_start is None:
        # Fall back to entry_frame if curve_start_frame unavailable
        curve_start = event.get("entry_frame")
    if curve_start is None:
        print(f"\nCURVE {curve_idx_label}: no curve_start_frame — skipped\n")
        return

    frame = int(curve_start)

    # Resolve HDF5 paths
    paths = {name: _resolve_field(h5f, name) for name in FIELD_CANDIDATES}

    # Read signals
    map_kappa = _read_scalar(h5f, paths["curvature_primary_abs"], frame)
    lgw = _read_scalar(h5f, paths["local_gate_weight"], frame)
    curve_phase = _read_scalar(h5f, paths["curve_local_phase"], frame)
    ews = _read_string(h5f, paths["entry_weight_source"], frame)
    severity = _read_scalar(h5f, paths["entry_severity"], frame)

    ld_target = _read_scalar(h5f, paths["ld_target"], frame)
    ld_slew = _read_scalar(h5f, paths["ld_after_slew"], frame)
    ld_final = _read_scalar(h5f, paths["ld_final"], frame)
    pp_floor_val = _read_scalar(h5f, paths["pp_floor_m"], frame)

    pp_geom = _read_scalar(h5f, paths["pp_geometric_steering"], frame)
    map_ff = _read_scalar(h5f, paths["pp_map_ff_applied"], frame)
    steer_before = _read_scalar(h5f, paths["steering_before_limits"], frame)

    # Derived: floor rescue max from event summary (pre-computed)
    floor_rescue_max = event.get("pp_floor_rescue_delta_max_m")

    # Onset delay
    onset_frames = event.get("steering_onset_minus_curve_start_frames")
    peak_lat = event.get("peak_lateral_error_m")

    # Header
    onset_str = f"+{onset_frames} frames late" if onset_frames is not None and onset_frames >= 0 else (
        f"{onset_frames} frames early" if onset_frames is not None else "N/A"
    )
    peak_str = f"{peak_lat:.3f}m" if peak_lat is not None else "N/A"

    print(f"\nCURVE {curve_idx_label} SIGNAL CHAIN BLAME")
    print("\u2550" * 40)
    print(f"Frame: {frame}  |  onset: {onset_str}  |  peak|lat|: {peak_str}")
    print()
    print(f"  {'Signal':<28s} {'Value':>8s}    Status")
    print(f"  {'─' * 28} {'─' * 8}    {'─' * 25}")

    # Map preview kappa
    kappa_status = _status(map_kappa, [
        (lambda v: v > 0.005, "OK"),
        (lambda v: True, "LOW"),
    ])
    print(f"  {'Map preview kappa':<28s} {_fmt(map_kappa)}    {kappa_status}")

    # local_gate_weight
    lgw_status = _status(lgw, [
        (lambda v: v > 0.5, "OK"),
        (lambda v: v >= 0.1, "PARTIAL"),
        (lambda v: True, "LOW"),
    ])
    print(f"  {'local_gate_weight':<28s} {_fmt(lgw)}    {lgw_status}")

    # curve_local_phase (EMA)
    phase_status = _status(curve_phase, [
        (lambda v: v > 0.45, "OK"),
        (lambda v: True, "BELOW THRESHOLD"),
    ])
    print(f"  {'curve_local_phase (EMA)':<28s} {_fmt(curve_phase)}    {phase_status}")

    # entry_weight_source
    ews_display = ews if ews else "N/A"
    ews_status = "good" if ews and "local_gate_weight" in ews else (
        "EMA-lagged" if ews and "phase" in ews.lower() else ews_display
    )
    print(f"  {'entry_weight_source':<28s} {ews_display:>8s}    {ews_status}")

    # entry_severity
    sev_status = _status(severity, [
        (lambda v: v > 0.3, "OK"),
        (lambda v: True, "LOW"),
    ])
    print(f"  {'entry_severity':<28s} {_fmt(severity)}    {sev_status}")

    # Lookahead chain
    print()
    print("  Lookahead Chain:")

    ld_tgt_status = _status(ld_target, [
        (lambda v: v < 8.0, "SHORT"),
        (lambda v: v <= 12.0, "MODERATE"),
        (lambda v: True, "LONG"),
    ])
    print(f"  {'Ld_target (planner)':<28s} {_fmt(ld_target, 2)}m   {ld_tgt_status}")

    slew_status = "SLEW ACTIVE" if (
        ld_slew is not None and ld_target is not None and abs(ld_slew - ld_target) > 0.05
    ) else "OK"
    print(f"  {'Ld_after_slew':<28s} {_fmt(ld_slew, 2)}m   {slew_status}")

    floor_bound = "FLOOR BOUND" if (
        ld_final is not None and pp_floor_val is not None and abs(ld_final - pp_floor_val) < 0.05
    ) else "OK"
    print(f"  {'Ld_final (controller)':<28s} {_fmt(ld_final, 2)}m   {floor_bound}")

    print(f"  {'PP floor value':<28s} {_fmt(pp_floor_val, 2)}m")

    rescue_status = _status(floor_rescue_max, [
        (lambda v: v >= 1.0, "HEAVY"),
        (lambda v: v >= 0.5, "MODERATE"),
        (lambda v: True, "NONE"),
    ])
    print(f"  {'floor_rescue_max':<28s} {_fmt(floor_rescue_max, 2)}m   {rescue_status}")

    # Steering components
    print()
    print("  Steering Components:")
    print(f"  {'PP geometric':<28s} {_fmt(pp_geom, 5)}    magnitude")
    print(f"  {'Map FF':<28s} {_fmt(map_ff, 5)}    magnitude")
    print(f"  {'Total before limits':<28s} {_fmt(steer_before, 5)}")

    # Blame
    blame_signals = {
        "ld_final": ld_final,
        "floor_rescue_max": floor_rescue_max,
        "local_gate_weight": lgw,
        "entry_severity": severity,
        "pp_map_ff_applied": map_ff,
    }
    blame, suggestion = _compute_blame(blame_signals)
    print()
    print(f"  BLAME: {blame}")
    print(f"  SUGGESTION: {suggestion}")
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
        description="Signal-chain blame trace for each curve event in an HDF5 recording",
    )
    parser.add_argument("recording", nargs="?", help="Path to recording file")
    parser.add_argument("--latest", action="store_true", help="Use the latest recording")
    args = parser.parse_args()

    recording_path = resolve_recording(args)
    print(f"Recording: {recording_path}")
    print()

    # Get the summary which includes curve_turn_events
    summary = analyze_recording_summary(str(recording_path), analyze_to_failure=True)

    curve_events = summary.get("curve_turn_events", [])
    if not curve_events:
        print("No curve events found in this recording.")
        return

    print(f"Found {len(curve_events)} curve event(s)\n")

    # Open HDF5 once for all curves
    with h5py.File(str(recording_path), "r") as h5f:
        for event in curve_events:
            idx = event.get("curve_index", "?")
            label = f"C{idx}"
            _print_curve_trace(label, event, h5f)


if __name__ == "__main__":
    main()
