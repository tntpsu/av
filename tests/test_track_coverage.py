"""
Track coverage tests — Step 1 (S2-M5).

Validates that all track YAMLs are self-consistent without requiring Unity.
Tests geometry (loop closure, curvature range, speed limits, self-intersection)
and registration (golden recordings, config mapping).

Run:
    pytest tests/test_track_coverage.py -v
"""
import math
from pathlib import Path
import pytest
import yaml

from av_stack.track_geometry import (
    build_centerline_points,
    find_geometry_violations,
    validate_track_geometry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = REPO_ROOT / "tracks"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_track(track_name: str) -> dict:
    path = TRACKS_DIR / f"{track_name}.yml"
    with open(path) as f:
        return yaml.safe_load(f)


def segment_heading_delta(seg: dict) -> float:
    """Heading change for a segment in degrees. Left = positive, right = negative."""
    if seg["type"] == "straight":
        return 0.0
    angle = float(seg["angle_deg"])
    return angle if seg["direction"] == "left" else -angle


def segment_arc_length(seg: dict) -> float:
    """Physical arc length in metres."""
    if seg["type"] == "straight":
        return float(seg["length"])
    return float(seg["radius"]) * math.radians(float(seg["angle_deg"]))


# ── Track names ──────────────────────────────────────────────────────────────

ALL_TRACKS = [
    "s_loop",
    "highway_65",
    "hairpin_15",
    "mixed_radius",
    "sweeping_highway",
]

NEW_TRACKS = [
    "hairpin_15",
    "mixed_radius",
    "sweeping_highway",
]

# Supported curvature range: R < 10m (κ > 0.1) is below PP minimum viable radius.
# R > 1000m (κ < 0.001) is effectively straight — should be a straight segment.
CURVATURE_MIN = 0.001   # 1/1000
CURVATURE_MAX = 0.10    # 1/10  (R=10m)


# ── Tests: geometry ──────────────────────────────────────────────────────────

class TestTrackGeometry:
    """Validate track YAML geometry without running Unity."""

    @pytest.mark.parametrize("track_name", ALL_TRACKS)
    def test_track_yaml_loads(self, track_name):
        """Track YAML exists and parses without error."""
        track = load_track(track_name)
        assert "segments" in track, f"{track_name}: missing 'segments' key"
        assert len(track["segments"]) > 0, f"{track_name}: no segments"

    @pytest.mark.parametrize("track_name", ALL_TRACKS)
    def test_track_loop_closes(self, track_name):
        """For loop tracks, total heading change must be a multiple of 360°."""
        track = load_track(track_name)
        if not track.get("loop", False):
            pytest.skip(f"{track_name} is not a loop track")

        total_heading = sum(segment_heading_delta(s) for s in track["segments"])
        # Accept heading sum within 1° of any multiple of 360°
        nearest_multiple = round(total_heading / 360.0) * 360.0
        error = abs(total_heading - nearest_multiple)
        assert error < 1.0, (
            f"{track_name}: heading sum {total_heading:.1f}° is not a multiple of 360° "
            f"(nearest: {nearest_multiple:.0f}°, error: {error:.1f}°). "
            "The loop will not close in Unity — add a correcting arc/straight."
        )

    @pytest.mark.parametrize("track_name", ALL_TRACKS)
    def test_arc_curvatures_in_range(self, track_name):
        """All arc radii produce curvatures within the supported range."""
        track = load_track(track_name)
        for i, seg in enumerate(track["segments"]):
            if seg["type"] != "arc":
                continue
            radius = float(seg["radius"])
            curvature = 1.0 / radius
            assert curvature >= CURVATURE_MIN, (
                f"{track_name} segment {i}: radius={radius}m → κ={curvature:.4f} "
                f"is below minimum κ={CURVATURE_MIN} (R > 1000m — use a straight)"
            )
            assert curvature <= CURVATURE_MAX, (
                f"{track_name} segment {i}: radius={radius}m → κ={curvature:.4f} "
                f"exceeds maximum κ={CURVATURE_MAX} (R < 10m — below PP minimum)"
            )

    @pytest.mark.parametrize("track_name", ALL_TRACKS)
    def test_arc_segments_have_speed_limits(self, track_name):
        """Every arc segment must have a speed_limit_mph override.

        Missing speed limits on arcs fall back silently to the track-level default,
        which may be dangerously fast for tight curves.
        """
        track = load_track(track_name)
        top_speed = track.get("speed_limit_mph", None)
        for i, seg in enumerate(track["segments"]):
            if seg["type"] != "arc":
                continue
            seg_speed = seg.get("speed_limit_mph", None)
            assert seg_speed is not None, (
                f"{track_name} segment {i} (arc R={seg['radius']}): "
                "missing speed_limit_mph — set an explicit limit for each arc"
            )
            if top_speed is not None:
                assert seg_speed <= top_speed, (
                    f"{track_name} segment {i}: arc speed {seg_speed} mph "
                    f"exceeds track-level {top_speed} mph"
                )

    @pytest.mark.parametrize("track_name", ALL_TRACKS)
    def test_track_total_length_positive(self, track_name):
        """Track total length is positive."""
        track = load_track(track_name)
        total = sum(segment_arc_length(s) for s in track["segments"])
        assert total > 0, f"{track_name}: total length {total}m is not positive"

    @pytest.mark.parametrize("track_name", NEW_TRACKS)
    def test_new_tracks_have_start_distance(self, track_name):
        """New tracks must set start_distance to avoid edge-straddle at spawn."""
        track = load_track(track_name)
        assert "start_distance" in track, (
            f"{track_name}: missing start_distance — vehicle may spawn straddling "
            "edge geometry. Set start_distance: 3.0 (matches s_loop/highway_65)"
        )
        assert float(track["start_distance"]) >= 1.0, (
            f"{track_name}: start_distance={track['start_distance']} is too small"
        )

    def test_hairpin_15_tighter_than_sloop(self):
        """hairpin_15 must have at least one arc tighter than s_loop's minimum (R40)."""
        track = load_track("hairpin_15")
        min_radius = min(
            float(s["radius"]) for s in track["segments"] if s["type"] == "arc"
        )
        assert min_radius < 40.0, (
            f"hairpin_15 min radius={min_radius}m is not tighter than s_loop "
            "(R40). Purpose of this track is to stress PP below s_loop geometry."
        )

    def test_mixed_radius_has_regime_spanning_curvatures(self):
        """mixed_radius must have arcs spanning both PP and MPC speed regimes.

        PP territory: R < 100m (cap-tracked speed < 10 m/s at κ > 0.01)
        MPC territory: R > 100m (speed stays above 10 m/s)
        """
        track = load_track("mixed_radius")
        arcs = [float(s["radius"]) for s in track["segments"] if s["type"] == "arc"]
        has_pp_arc = any(r < 100 for r in arcs)
        has_mpc_arc = any(r > 100 for r in arcs)
        assert has_pp_arc, f"mixed_radius has no tight arc (R<100m) to trigger PP mode. Arcs: {arcs}"
        assert has_mpc_arc, f"mixed_radius has no gentle arc (R>100m) to keep MPC active. Arcs: {arcs}"

    def test_sweeping_highway_all_arcs_gentle(self):
        """sweeping_highway arcs must all be gentle (R >= 250m) for sustained MPC."""
        track = load_track("sweeping_highway")
        for i, seg in enumerate(track["segments"]):
            if seg["type"] != "arc":
                continue
            r = float(seg["radius"])
            assert r >= 250.0, (
                f"sweeping_highway segment {i}: R={r}m is too tight "
                "(want R >= 250m for sustained MPC at 15 m/s)"
            )


# ── Tests: registration ───────────────────────────────────────────────────────

class TestTrackRegistration:
    """Validate that new tracks are registered correctly in the CI system."""

    # Config overlay expected for each track (mirrors run_live_gate.py TRACK_CONFIG_MAP)
    TRACK_CONFIG_MAP = {
        "s_loop":           "config/mpc_sloop.yaml",
        "highway_65":       "config/mpc_highway.yaml",
        "hairpin_15":       "config/mpc_sloop.yaml",
        "mixed_radius":     "config/mpc_mixed.yaml",
        "sweeping_highway": "config/mpc_highway.yaml",
    }

    def test_all_track_configs_exist(self):
        """Every config overlay in TRACK_CONFIG_MAP must exist on disk."""
        for track_name, config_path in self.TRACK_CONFIG_MAP.items():
            full_path = REPO_ROOT / config_path
            assert full_path.exists(), (
                f"Track '{track_name}' references config '{config_path}' which does not exist"
            )

    @pytest.mark.parametrize("track_name", NEW_TRACKS)
    def test_new_track_in_config_map(self, track_name):
        """Each new track has a config mapping."""
        assert track_name in self.TRACK_CONFIG_MAP, (
            f"'{track_name}' not in TRACK_CONFIG_MAP — add it to "
            "tests/test_track_coverage.py and tools/ci/run_live_gate.py"
        )

    @pytest.mark.parametrize("track_name", NEW_TRACKS)
    def test_new_track_golden_recording_registered(self, track_name):
        """New tracks should have golden recordings registered after Phase 1D.

        This test is skipped until the golden recording is registered.
        After live validation, update tests/fixtures/golden_recordings.json
        and this test will enforce the file exists.
        """
        import json
        manifest_path = FIXTURES_DIR / "golden_recordings.json"
        if not manifest_path.exists():
            pytest.skip("No golden recordings manifest found")

        manifest = json.loads(manifest_path.read_text())
        tracks = manifest.get("tracks", {})

        if track_name not in tracks:
            pytest.skip(
                f"'{track_name}' not yet registered in golden_recordings.json. "
                "Run Phase 1D: validate 3× runs, pick median, register golden recording."
            )

        recording_path = REPO_ROOT / tracks[track_name]
        assert recording_path.exists(), (
            f"Golden recording for '{track_name}' registered but file missing: "
            f"{recording_path}"
        )


# ── Tests: 2D geometry (self-intersection, road clearance) ────────────────────

class TestTrackSelfIntersection:
    """Detect track geometry that would produce invalid Unity road meshes.

    These checks run purely from the YAML segments — no Unity required.
    A failed test means the track will cross itself in simulation, causing
    spurious out-of-lane events even when the AV is driving correctly.
    """

    @pytest.mark.parametrize("track_name", ALL_TRACKS)
    def test_centerline_no_self_intersection(self, track_name):
        """Track centerline must not cross itself.

        Dead-reckons the 2D centerline from segments and checks for proper
        segment–segment crossings using the CCW orientation test.
        """
        track = load_track(track_name)
        road_width = float(track.get("road_width", 7.0))
        is_loop = bool(track.get("loop", False))
        pts = build_centerline_points(track["segments"], spacing_m=0.5)
        violations = [
            v for v in find_geometry_violations(
                pts, road_width_m=road_width, spacing_m=0.5, is_loop=is_loop
            )
            if v.kind == "self_intersection"
        ]
        assert not violations, (
            f"{track_name}: {violations[0].detail}. "
            "The road geometry crosses itself in Unity — adjust segment "
            "angles/lengths/radii so the track does not self-intersect."
        )

    @pytest.mark.parametrize("track_name", ALL_TRACKS)
    def test_road_sections_no_edge_overlap(self, track_name):
        """Non-adjacent road sections must have > road_width centre-to-centre gap.

        If two sections of road are closer than road_width their edges overlap
        in Unity, which creates invalid mesh geometry and confuses perception.
        """
        track = load_track(track_name)
        road_width = float(track.get("road_width", 7.0))
        is_loop = bool(track.get("loop", False))
        pts = build_centerline_points(track["segments"], spacing_m=0.5)
        violations = [
            v for v in find_geometry_violations(
                pts, road_width_m=road_width, spacing_m=0.5, is_loop=is_loop
            )
            if v.kind == "road_clearance"
        ]
        assert not violations, (
            f"{track_name}: {violations[0].detail}. "
            "Increase the separation between road sections by adjusting "
            "straight lengths or arc radii."
        )

    def test_build_centerline_straight_only(self):
        """Sanity check: a 100m straight ends ~100m from origin."""
        segs = [{"type": "straight", "length": 100.0}]
        pts = build_centerline_points(segs, spacing_m=1.0)
        assert abs(pts[-1][0] - 100.0) < 0.5
        assert abs(pts[-1][1]) < 0.5

    def test_build_centerline_left_arc_90deg(self):
        """A 90° left arc of radius R ends at (R, R) from origin (heading east)."""
        R = 40.0
        segs = [{"type": "arc", "radius": R, "angle_deg": 90.0, "direction": "left"}]
        pts = build_centerline_points(segs, spacing_m=0.5)
        end_x, end_y = pts[-1]
        assert abs(end_x - R) < 1.0, f"Expected end_x≈{R}, got {end_x:.2f}"
        assert abs(end_y - R) < 1.0, f"Expected end_y≈{R}, got {end_y:.2f}"

    def test_build_centerline_right_arc_90deg(self):
        """A 90° right arc of radius R ends at (R, -R) from origin (heading east)."""
        R = 40.0
        segs = [{"type": "arc", "radius": R, "angle_deg": 90.0, "direction": "right"}]
        pts = build_centerline_points(segs, spacing_m=0.5)
        end_x, end_y = pts[-1]
        assert abs(end_x - R) < 1.0, f"Expected end_x≈{R}, got {end_x:.2f}"
        assert abs(end_y - (-R)) < 1.0, f"Expected end_y≈{-R}, got {end_y:.2f}"

    def test_validate_track_geometry_detects_crossing(self):
        """validate_track_geometry returns errors for a track that crosses itself."""
        # A figure-8: two 360° loops sharing a start point self-intersect.
        crossing_cfg = {
            "name": "test_crossing",
            "road_width": 7.0,
            "segments": [
                {"type": "arc", "radius": 30.0, "angle_deg": 360.0, "direction": "left"},
                # After one full loop we're back at origin — a second loop with a
                # different radius will cause a crossing between the two circles.
                {"type": "arc", "radius": 15.0, "angle_deg": 360.0, "direction": "right"},
            ],
        }
        errors = validate_track_geometry(crossing_cfg)
        assert errors, "Expected geometry errors for a self-crossing track, got none"

    def test_validate_track_geometry_clean_for_sloop(self):
        """s_loop track has no geometry violations."""
        track = load_track("s_loop")
        errors = validate_track_geometry(track)
        assert not errors, f"s_loop has unexpected geometry violations: {errors}"
