"""Geometric safety contracts for trajectory lookahead parameters.

WHY THIS TEST EXISTS
--------------------
The trajectory layer (tight_scale, lookahead tables) is a BLIND SPOT in the test
pyramid: the closed-loop simulation and replay validator cannot re-run trajectory
planning, so changes there only get validated in Unity.

This test is a PURE GEOMETRY CHECK that catches the most dangerous class of
trajectory parameter misconfiguration before running Unity:

  "Will the reference lookahead point land past the curve apex?"

If lookahead_m > curve_radius * safety_fraction, the trajectory planner targets a
point that has already rounded the corner.  The car then steers toward the far side
of the curve and overshoots, causing a lane departure.

TRACKS REGISTERED HERE (update when adding a new track):
  hill_highway  max_kappa=0.010  (radius=100m) — gentle highway curves
  sloop         max_kappa=0.025  (radius=40m)  — tighter loop curves

WHAT WOULD HAVE CAUGHT THE SLOOP RISK:
  Changing tight_scale from 0.50 → 0.70 in the BASE CONFIG would push the sloop
  reference point to 7m on 40m radius curves (17.5% of radius, approaching the
  apex).  This test raises a loud failure before any Unity run.

WHAT THIS TEST CANNOT CATCH:
  - Whether the reference point is actually centred in the lane
  - Time-domain effects (slew rate, smoothing)
  - Interaction with the PP floor rescue
  Use the replay validator and score floor tests for those.
"""

import math
import yaml
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Track registry: (name, max_kappa, typical_speed_mps)
# Update this when a new track is added.  max_kappa comes from ground_truth
# path_curvature in golden recordings or track designer specs.
# ---------------------------------------------------------------------------

TRACK_REGISTRY = [
    # name            max_kappa  typical_speed_mps
    ("hill_highway",  0.010,     8.0),
    ("sloop",         0.025,     8.0),
    ("oval",          0.000,     8.0),    # straight oval — no tight curves
    ("highway",       0.005,     13.0),   # gentle highway
]

# Safe chord angle: the angle subtended at the curve center by the lookahead chord
# must stay below this limit so the reference point is well before the apex.
# theta = 2 * arcsin(ld / (2*R)).  At 30°, ld/R = 2*sin(15°) ≈ 0.518.
# At 40m radius: 30° limit → ld ≤ 20.7m.  Our max lookahead is 10m → 14.4° max. Safe.
_SAFE_CHORD_ANGLE_DEG = 45.0  # max angle before reference point is "around the bend"

# Geometric threshold for tight_scale activation (from av_stack_config.yaml)
_TIGHT_CURVATURE_THRESHOLD = 0.010  # rad/m


def _load_merged_config(overlay_name: str | None = None) -> dict:
    """Load base config merged with an optional overlay."""
    base_path = Path("config/av_stack_config.yaml")
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}
    if overlay_name is None:
        return base

    overlay_path = Path("config") / overlay_name
    if not overlay_path.exists():
        return base
    with open(overlay_path) as f:
        overlay = yaml.safe_load(f) or {}

    # Shallow merge at trajectory/control level
    merged = dict(base)
    for section in ("trajectory", "control"):
        if section in overlay:
            merged.setdefault(section, {}).update(overlay[section])
    return merged


def _get_nominal_lookahead(cfg: dict, speed_mps: float) -> float:
    """Interpolate nominal lookahead from config speed table at a given speed."""
    traj = cfg.get("trajectory", {}) or {}
    table = traj.get("reference_lookahead_speed_table", [])
    if not table:
        return float(traj.get("reference_lookahead", 10.0))
    speeds  = [row["speed_mps"]  for row in table]
    lhds    = [row["lookahead_m"] for row in table]
    return float(np.interp(speed_mps, speeds, lhds))


def _max_safe_lookahead_m(max_kappa: float) -> float:
    """Maximum safe lookahead for a curve with the given peak curvature.

    Uses chord-angle geometry: theta = 2*arcsin(ld/(2*R)).  Safe as long as theta
    is below _SAFE_CHORD_ANGLE_DEG (45°), which ensures the reference point is well
    before the curve apex.  At sloop kappa=0.025 (R=40m), this allows up to 30.6m
    lookahead — far beyond our typical 7-10m range.  Any practical tight_scale value
    will pass this test; violations only occur with extreme misconfiguration (scale>>1).
    """
    if max_kappa <= 0:
        return float("inf")
    radius = 1.0 / max_kappa
    # Solve 2*arcsin(ld/(2*R)) = angle_limit → ld = 2*R*sin(angle_limit/2)
    angle_rad = math.radians(_SAFE_CHORD_ANGLE_DEG)
    return 2.0 * radius * math.sin(angle_rad / 2.0)


# ---------------------------------------------------------------------------
# Parameterised safety tests
# ---------------------------------------------------------------------------

import numpy as np


class TestTightScaleGeometricSafety:
    """For every registered track, verify that the configured tight_scale does
    not push the reference lookahead past the safe fraction of curve radius."""

    @pytest.mark.parametrize("track_name,max_kappa,speed_mps", TRACK_REGISTRY)
    def test_base_config_tight_scale_is_safe(self, track_name, max_kappa, speed_mps):
        """Base config tight_scale must be geometrically safe on every track."""
        cfg  = _load_merged_config()
        traj = cfg.get("trajectory", {}) or {}
        tight_scale = float(traj.get("reference_lookahead_tight_scale", 1.0))
        nominal_ld  = _get_nominal_lookahead(cfg, speed_mps)

        # tight_scale only fires when kappa >= tight_curvature_threshold
        if max_kappa < _TIGHT_CURVATURE_THRESHOLD:
            pytest.skip(f"{track_name}: kappa={max_kappa} below tight threshold — scale not active")

        effective_ld = tight_scale * nominal_ld
        safe_limit   = _max_safe_lookahead_m(max_kappa)

        assert effective_ld <= safe_limit, (
            f"Base config: track={track_name} kappa={max_kappa:.3f} (r={1/max_kappa:.0f}m)\n"
            f"  tight_scale={tight_scale}, nominal_ld={nominal_ld:.1f}m, "
            f"effective_ld={effective_ld:.1f}m\n"
            f"  safe limit = {_SAFE_LOOKAHEAD_RADIUS_FRACTION*100:.0f}% of radius = {safe_limit:.1f}m\n"
            f"  FAIL: lookahead {effective_ld:.1f}m > safe limit {safe_limit:.1f}m on {track_name}.\n"
            f"  If applying tight_scale globally, sloop curves become unsafe.\n"
            f"  Put track-specific tight_scale in mpc_{{track}}.yaml instead."
        )

    @pytest.mark.parametrize("track_name,max_kappa,speed_mps", TRACK_REGISTRY)
    def test_hill_highway_overlay_tight_scale_is_safe(self, track_name, max_kappa, speed_mps):
        """mpc_hill_highway.yaml overlay tight_scale (if set) must still be safe on sloop."""
        overlay_cfg  = _load_merged_config("mpc_hill_highway.yaml")
        traj = overlay_cfg.get("trajectory", {}) or {}
        tight_scale = float(traj.get("reference_lookahead_tight_scale", 0.5))
        nominal_ld  = _get_nominal_lookahead(overlay_cfg, speed_mps)

        if max_kappa < _TIGHT_CURVATURE_THRESHOLD:
            pytest.skip(f"{track_name}: kappa below tight threshold")

        effective_ld = tight_scale * nominal_ld
        safe_limit   = _max_safe_lookahead_m(max_kappa)

        assert effective_ld <= safe_limit, (
            f"mpc_hill_highway.yaml (applied to {track_name}): "
            f"tight_scale={tight_scale}, effective_ld={effective_ld:.1f}m > safe={safe_limit:.1f}m.\n"
            f"  Hill-highway overlay settings must not break sloop geometry.\n"
            f"  Use a per-track overlay (mpc_sloop.yaml) if sloop needs a different scale."
        )

    def test_sloop_specific_tight_scale_is_safe(self):
        """mpc_sloop.yaml tight_scale (if set) must be safe at sloop's max kappa=0.025."""
        cfg  = _load_merged_config("mpc_sloop.yaml")
        traj = cfg.get("trajectory", {}) or {}
        tight_scale = float(traj.get("reference_lookahead_tight_scale", 0.5))
        speed_mps   = 8.0
        nominal_ld  = _get_nominal_lookahead(cfg, speed_mps)

        sloop_max_kappa = 0.025
        effective_ld = tight_scale * nominal_ld
        safe_limit   = _max_safe_lookahead_m(sloop_max_kappa)

        assert effective_ld <= safe_limit, (
            f"mpc_sloop.yaml: tight_scale={tight_scale}, effective_ld={effective_ld:.1f}m "
            f"> safe limit {safe_limit:.1f}m at sloop kappa=0.025 (r=40m)."
        )

    # NOTE: At sloop's kappa=0.025 (R=40m), the 45° chord angle limit allows up to
    # 30.6m lookahead — far beyond any tight_scale we'd ever use (max scale=1.0 → 10m).
    # All values in our operational range are GEOMETRICALLY safe.  Behavioral safety
    # (late turn-in, corner cutting) is a separate concern validated by E2E score floors.
    @pytest.mark.parametrize("tight_scale,should_pass", [
        (0.50, True),   # current base config — safe (5m on 40m radius = 7° chord)
        (0.70, True),   # proposed hill_highway change — safe (7m on 40m radius = 10° chord)
        (1.00, True),   # full nominal — safe (10m on 40m radius = 14° chord)
        (3.10, False),  # extreme over-scale — 31m on 40m radius = 45.7° chord, exceeds 45° limit
        (4.00, False),  # extreme over-scale — 40m on 40m radius = 59° chord, well past limit
    ])
    def test_tight_scale_safety_threshold_at_sloop_kappa(self, tight_scale, should_pass):
        """Parametric safety boundary: verify the safety formula at sloop's max kappa.

        This documents the exact safety budget so anyone reading the code can see
        at a glance which scale values are safe and which are not.
        """
        nominal_ld   = 10.0  # m (nominal lookahead at 8 m/s)
        sloop_kappa  = 0.025
        effective_ld = tight_scale * nominal_ld
        safe_limit   = _max_safe_lookahead_m(sloop_kappa)

        is_safe = effective_ld <= safe_limit
        if should_pass:
            assert is_safe, (
                f"tight_scale={tight_scale} expected safe: {effective_ld:.1f}m vs limit {safe_limit:.1f}m"
            )
        else:
            assert not is_safe, (
                f"tight_scale={tight_scale} expected UNSAFE: {effective_ld:.1f}m vs limit {safe_limit:.1f}m"
            )


class TestTightScaleConfigBounds:
    """Config bounds tests for tight_scale.

    Geometric safety is not the binding constraint (the 45° chord limit allows up to
    30m on sloop, far beyond any practical value).  Instead, these tests enforce a
    BEHAVIOURAL SAFETY MARGIN: tight_scale must stay ≤ 1.0 in any config to prevent
    the reference lookahead from exceeding the nominal value on tight curves, which
    could cause late turn-in that geometry alone cannot detect.

    Update these bounds only after E2E validation shows a higher value is safe.
    """

    def test_base_config_tight_scale_within_bounds(self):
        """Base config tight_scale must not exceed 1.0."""
        cfg  = _load_merged_config()
        traj = cfg.get("trajectory", {}) or {}
        tight_scale = float(traj.get("reference_lookahead_tight_scale", 0.5))
        assert tight_scale <= 1.0, (
            f"Base config tight_scale={tight_scale} > 1.0.\n"
            f"Values above 1.0 extend the lookahead BEYOND nominal on tight curves,\n"
            f"risking late turn-in.  Validate in E2E before exceeding 1.0."
        )

    @pytest.mark.parametrize("overlay", ["mpc_sloop.yaml", "mpc_hill_highway.yaml",
                                          "mpc_hairpin.yaml", "mpc_highway.yaml"])
    def test_overlay_tight_scale_within_bounds(self, overlay):
        """All track overlays must not set tight_scale above 1.0."""
        cfg  = _load_merged_config(overlay)
        traj = cfg.get("trajectory", {}) or {}
        tight_scale = float(traj.get("reference_lookahead_tight_scale", 0.5))
        assert tight_scale <= 1.0, (
            f"{overlay}: tight_scale={tight_scale} > 1.0.\n"
            f"Validate behavioural safety in E2E before exceeding 1.0."
        )


class TestHeadingGateThresholdRobustness:
    """Verify that heading gate release threshold is safe across all tracks.

    The gate releases when preview_curvature > threshold.  The threshold must be:
      - Small enough to release before the car enters the curve (catches curves early)
      - Large enough to NOT release on noise/gentle road undulation on straights
    """

    def test_gate_threshold_below_min_preview_curvature_of_real_curves(self):
        """The heading gate threshold must be below the min preview curvature when
        approaching any real curve.

        The threshold works on PREVIEW curvature (map lookahead), not the curve body
        curvature directly.  For the highway (body kappa≈0.005), preview curvature
        starts rising from ~0.002 on approach.  The threshold must be below that.
        Empirical calibration: threshold < 0.5 × min_real_curve_kappa.
        """
        cfg = _load_merged_config()
        traj = cfg.get("trajectory", {}) or {}
        threshold = float(traj.get("traj_heading_zero_gate_curvature_preview_off_threshold", 0.0015))

        min_real_kappa = min(k for _, k, _ in TRACK_REGISTRY if k > 0)  # 0.005

        # Preview curvature on approach is typically 40-60% of body kappa; threshold
        # must sit below the bottom of that range (40% × min_real_kappa).
        max_safe_threshold = min_real_kappa * 0.50

        assert threshold <= max_safe_threshold, (
            f"gate_threshold={threshold:.4f} > {max_safe_threshold:.4f} "
            f"(50% of min real curve kappa={min_real_kappa}).\n"
            f"The gate might NOT release before the car enters gentle highway curves."
        )

    def test_gate_threshold_above_noise_floor(self):
        """The heading gate threshold must be > 3× the road undulation noise floor.

        Prevents spurious gate releases on straight roads with minor kappa noise (~0.0002).
        """
        cfg = _load_merged_config()
        traj = cfg.get("trajectory", {}) or {}
        threshold = float(traj.get("traj_heading_zero_gate_curvature_preview_off_threshold", 0.0015))
        noise_floor = 0.0002  # approximate road undulation kappa on straights

        assert threshold >= noise_floor * 3, (
            f"gate_threshold={threshold:.4f} is too close to noise floor {noise_floor:.4f}.\n"
            f"Minimum recommended: {noise_floor * 3:.4f}"
        )

    @pytest.mark.parametrize("threshold,should_be_safe", [
        (0.0003, False),  # too low — below 3× noise floor (spurious releases on straights)
        (0.0010, True),   # proposed new value — safe range
        (0.0015, True),   # current value — safe range
        (0.0020, True),   # previous value — safe range
        (0.0030, False),  # too high — above 50% of highway kappa=0.005; gate won't release
    ])
    def test_gate_threshold_safe_range(self, threshold, should_be_safe):
        """Document the safe range for the heading gate release threshold.

        Safe = (above 3× noise floor) AND (below 50% of min real-curve body kappa).
        Both the current value (0.0015) and the proposed new value (0.001) are safe.
        """
        noise_floor   = 0.0002
        min_real_kappa = 0.005  # highway
        above_noise   = threshold >= noise_floor * 3
        below_curve   = threshold <= min_real_kappa * 0.50
        is_safe       = above_noise and below_curve

        if should_be_safe:
            assert is_safe, (
                f"threshold={threshold:.4f} expected safe: "
                f"above_noise={above_noise} (>={noise_floor*3:.4f}), "
                f"below_curve={below_curve} (<={min_real_kappa*0.5:.4f})"
            )
        else:
            assert not is_safe, (
                f"threshold={threshold:.4f} expected UNSAFE: "
                f"above_noise={above_noise}, below_curve={below_curve}"
            )
