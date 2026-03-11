"""
track_geometry.py — 2D centerline builder and self-intersection checker.

Converts track YAML segment descriptions into 2D (x, y) coordinates via
dead-reckoning heading integration, then detects:
  1. Exact centerline self-intersections (road crosses itself).
  2. Road-clearance violations (parallel sections too close for road_width).

No external dependencies beyond math.

Usage (runtime):
    from av_stack.track_geometry import validate_track_geometry
    errors = validate_track_geometry(track_cfg_dict)
    for msg in errors:
        logger.warning("[TRACK GEOMETRY] %s", msg)

Usage (tests):
    from av_stack.track_geometry import build_centerline_points, find_geometry_violations
"""
from __future__ import annotations

import math
from typing import NamedTuple


class GeometryViolation(NamedTuple):
    kind: str           # "self_intersection" | "road_clearance"
    s1_m: float         # arc-length of first location (metres)
    s2_m: float         # arc-length of second location (metres)
    detail: str         # human-readable description


# ── 2D centerline builder ─────────────────────────────────────────────────────

def build_centerline_points(
    segments: list[dict],
    spacing_m: float = 0.5,
) -> list[tuple[float, float]]:
    """Dead-reckon 2D centerline (x, y) from track segment descriptions.

    Coordinate convention:
      - Origin (0, 0) at the track start.
      - Heading 0 = east (+X); positive heading = counter-clockwise (left).
      - Left turns increase heading; right turns decrease it.

    Args:
        segments: list of segment dicts with keys:
            type: "straight" | "arc"
            length: float  (straight only)
            radius: float  (arc only)
            angle_deg: float  (arc only)
            direction: "left" | "right"  (arc only)
        spacing_m: sample spacing in metres (default 0.5 m).

    Returns:
        List of (x, y) tuples sampled along the centerline.
    """
    x, y = 0.0, 0.0
    heading = 0.0  # radians
    pts: list[tuple[float, float]] = [(x, y)]

    for seg in (segments or []):
        seg_type = str(seg.get("type", "straight")).strip().lower()

        if seg_type == "straight":
            length = float(seg.get("length", 0.0) or 0.0)
            if length <= 0.0:
                continue
            n = max(1, round(length / spacing_m))
            dl = length / n
            for _ in range(n):
                x += dl * math.cos(heading)
                y += dl * math.sin(heading)
                pts.append((x, y))

        elif seg_type == "arc":
            radius = float(seg.get("radius", 0.0) or 0.0)
            angle_deg = float(seg.get("angle_deg", seg.get("angle", 0.0)) or 0.0)
            if radius <= 0.0 or angle_deg <= 0.0:
                continue
            direction = str(seg.get("direction", "left")).strip().lower()
            sweep = math.radians(angle_deg)
            if direction == "right":
                sweep = -sweep  # CW → negative sweep

            arc_len = abs(radius * sweep)
            n = max(1, round(arc_len / spacing_m))

            # Centre of curvature (perpendicular to current heading).
            # Left turn: centre is heading+90° (north if heading east)
            #   cx = x - R·sin(h),  cy = y + R·cos(h)
            # Right turn: centre is heading-90°
            #   cx = x + R·sin(h),  cy = y - R·cos(h)
            if direction == "left":
                cx = x - radius * math.sin(heading)
                cy = y + radius * math.cos(heading)
            else:
                cx = x + radius * math.sin(heading)
                cy = y - radius * math.cos(heading)

            cur_angle = math.atan2(y - cy, x - cx)
            d_angle = sweep / n
            for _ in range(n):
                cur_angle += d_angle
                x = cx + radius * math.cos(cur_angle)
                y = cy + radius * math.sin(cur_angle)
                pts.append((x, y))
            heading += sweep

    return pts


# ── Segment–segment intersection ──────────────────────────────────────────────

def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    """Counter-clockwise orientation test."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Return True iff line segments p1–p2 and p3–p4 properly cross each other.

    Uses the CCW cross-product test.  Collinear / endpoint-touching segments
    are not flagged (avoids false positives at segment junctions).
    """
    return (
        _ccw(p1, p3, p4) != _ccw(p2, p3, p4)
        and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)
    )


# ── Violation detectors ───────────────────────────────────────────────────────

def find_geometry_violations(
    pts: list[tuple[float, float]],
    road_width_m: float = 7.0,
    spacing_m: float = 0.5,
    is_loop: bool = False,
) -> list[GeometryViolation]:
    """Detect self-intersections and road-clearance violations.

    Args:
        pts: Centerline points from build_centerline_points().
        road_width_m: Road width in metres (from track YAML road_width key).
        spacing_m: The same spacing_m used when building pts (for arc-length estimates).

    Returns:
        List of GeometryViolation; empty means geometry is valid.
    """
    n = len(pts)
    violations: list[GeometryViolation] = []

    # ── 1. Centerline self-intersection ──────────────────────────────────────
    # Skip segment i+1 (shares an endpoint with segment i).
    # O(n²) but n ~ track_length / 0.5 ≈ a few hundred — fast for any real track.
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if _segments_cross(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                violations.append(GeometryViolation(
                    kind="self_intersection",
                    s1_m=i * spacing_m,
                    s2_m=j * spacing_m,
                    detail=(
                        f"centerline crosses itself: segment at s≈{i * spacing_m:.1f}m "
                        f"intersects segment at s≈{j * spacing_m:.1f}m"
                    ),
                ))
                # Report first crossing only — subsequent ones are usually downstream artefacts.
                return violations

    # ── 2. Road-clearance violation ───────────────────────────────────────────
    # Two non-adjacent centerline sections within road_width of each other means
    # their road edges overlap in Unity.
    #
    # Skip window: points within road_width*4 of arc distance are considered
    # "adjacent" geometry (e.g. S-curve with parallel legs that are legitimately
    # close).  Tighter spacing would cause false positives on tight-radius tracks.
    #
    # For loop tracks the last sample rejoins the first — skip wrap-around pairs
    # within skip points of the closure seam to avoid false positives there.
    skip = max(10, round(road_width_m * 4.0 / spacing_m))

    for i in range(n):
        for j in range(i + skip, n):
            # For loop tracks: also treat j close to the end as "adjacent" to i
            # close to the start (circular distance ≤ skip).
            if is_loop and (n - j) + i < skip:
                continue
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            if math.hypot(dx, dy) < road_width_m:
                violations.append(GeometryViolation(
                    kind="road_clearance",
                    s1_m=i * spacing_m,
                    s2_m=j * spacing_m,
                    detail=(
                        f"road sections too close: s≈{i * spacing_m:.1f}m and "
                        f"s≈{j * spacing_m:.1f}m are within road_width={road_width_m:.1f}m "
                        "— road edges will overlap in Unity"
                    ),
                ))
                return violations  # First violation is enough; stop early.

    return violations


# ── Top-level validator (used by orchestrator + tests) ────────────────────────

def validate_track_geometry(track_cfg: dict) -> list[str]:
    """Validate track YAML geometry.  Returns list of error strings (empty = ok).

    Args:
        track_cfg: Parsed track YAML dict (output of yaml.safe_load).

    Returns:
        List of human-readable error strings.  Empty list means no problems found.
    """
    segments = track_cfg.get("segments") or []
    road_width_m = float(track_cfg.get("road_width", 7.0) or 7.0)
    track_name = str(track_cfg.get("name", "unknown"))
    spacing_m = 0.5

    if not segments:
        return [f"{track_name}: no segments defined"]

    pts = build_centerline_points(segments, spacing_m=spacing_m)
    is_loop = bool(track_cfg.get("loop", False))
    violations = find_geometry_violations(pts, road_width_m=road_width_m, spacing_m=spacing_m, is_loop=is_loop)

    return [f"{track_name}: {v.detail}" for v in violations]
