"""Compute approximate track length from a track YAML file."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml


def compute_track_length(track_yaml: Path) -> float:
    """Return total track length in meters from a track YAML."""
    with open(track_yaml) as f:
        data = yaml.safe_load(f)

    total = 0.0
    for seg in data.get("segments", []):
        seg_type = seg.get("type", "straight")
        if seg_type == "straight":
            total += seg.get("length", 0.0)
        elif seg_type == "arc":
            radius = seg.get("radius", 100.0)
            angle_deg = seg.get("angle_deg", 90.0)
            total += radius * math.radians(angle_deg)
    return total


def one_lap_duration(track_yaml: Path, speed_mps: float, margin_pct: float = 0.05) -> float:
    """Return duration in seconds for one lap minus a safety margin.

    Args:
        track_yaml: Path to track YAML file.
        speed_mps: Ground truth speed in m/s.
        margin_pct: Fraction of lap to skip at the end (default 5%).

    Returns:
        Duration in seconds that covers (1 - margin_pct) of one lap.
    """
    length = compute_track_length(track_yaml)
    if speed_mps <= 0:
        return 60.0
    full_lap_s = length / speed_mps
    return full_lap_s * (1.0 - margin_pct)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python track_length.py <track.yml> [speed_mps]")
        sys.exit(1)
    track = Path(sys.argv[1])
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    length = compute_track_length(track)
    dur = one_lap_duration(track, speed)
    print(f"Track: {track.name}")
    print(f"Length: {length:.1f}m")
    print(f"At {speed} m/s: full lap = {length/speed:.1f}s, safe duration = {dur:.1f}s")
