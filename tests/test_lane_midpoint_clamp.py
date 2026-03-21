"""Tests for the lane-midpoint curve-contamination clamp.

Root cause: when ``coord_conversion_distance`` (the lookahead used to evaluate the
lane-centre pixel) extends past the start of an upcoming curve, the sampled
x-position acquires a lateral bias of approximately κ·Δd²/2, where Δd is the
distance past the curve start.  ``compute_lane_midpoint_clamp`` caps the
evaluation distance so this contamination stays below a configurable threshold.
"""

import math

import pytest

from av_stack.lane_gating import compute_lane_midpoint_clamp


# ---------------------------------------------------------------------------
# Test 1: clamp fires when lookahead reaches past curve start
# ---------------------------------------------------------------------------

def test_clamp_fires_when_lookahead_exceeds_safe_distance() -> None:
    """Clamp should reduce lookahead when it overshoots the safe distance."""
    # sloop-like scenario: kappa=0.01, 2 m to curve start, 8 m lookahead
    # delta_d_max = sqrt(2 * 0.05 / 0.01) = sqrt(10) ≈ 3.16 m
    # max_safe = 2.0 + 3.16 = 5.16 m  → lookahead 8.0 should be clamped
    clamped_dist, was_clamped = compute_lane_midpoint_clamp(
        lookahead_m=8.0,
        kappa_preview=0.01,
        dist_to_curve_m=2.0,
        in_curve=False,
        threshold_m=0.05,
        min_distance_m=3.0,
    )
    assert was_clamped, "Clamp should have fired"
    expected_max = 2.0 + math.sqrt(2.0 * 0.05 / 0.01)
    assert abs(clamped_dist - expected_max) < 1e-9
    assert clamped_dist < 8.0


# ---------------------------------------------------------------------------
# Test 2: no clamp when lookahead is already within safe distance
# ---------------------------------------------------------------------------

def test_no_clamp_when_lookahead_within_safe_distance() -> None:
    """Lookahead that does not over-reach the curve should pass through unchanged."""
    # kappa=0.01, 20 m to curve → max_safe ≈ 23.16 m; lookahead 8 m is safe
    clamped_dist, was_clamped = compute_lane_midpoint_clamp(
        lookahead_m=8.0,
        kappa_preview=0.01,
        dist_to_curve_m=20.0,
        in_curve=False,
        threshold_m=0.05,
        min_distance_m=3.0,
    )
    assert not was_clamped
    assert clamped_dist == 8.0


# ---------------------------------------------------------------------------
# Test 3: no clamp when already inside a curve
# ---------------------------------------------------------------------------

def test_no_clamp_when_already_in_curve() -> None:
    """Clamp must be skipped when the vehicle is currently cornering."""
    clamped_dist, was_clamped = compute_lane_midpoint_clamp(
        lookahead_m=8.0,
        kappa_preview=0.02,
        dist_to_curve_m=0.5,
        in_curve=True,           # currently cornering → skip clamp
        threshold_m=0.05,
        min_distance_m=3.0,
    )
    assert not was_clamped
    assert clamped_dist == 8.0


# ---------------------------------------------------------------------------
# Test 4: no clamp when kappa_preview is negligible (straight road ahead)
# ---------------------------------------------------------------------------

def test_no_clamp_when_kappa_negligible() -> None:
    """Very small kappa (straight ahead) must not trigger the clamp."""
    clamped_dist, was_clamped = compute_lane_midpoint_clamp(
        lookahead_m=8.0,
        kappa_preview=1e-5,       # below 1e-4 threshold
        dist_to_curve_m=1.0,
        in_curve=False,
        threshold_m=0.05,
        min_distance_m=3.0,
    )
    assert not was_clamped
    assert clamped_dist == 8.0


# ---------------------------------------------------------------------------
# Test 5: dist_to_curve_m=None (unknown) leaves distance unchanged
# ---------------------------------------------------------------------------

def test_no_clamp_when_dist_to_curve_unknown() -> None:
    """If distance to curve is unavailable the clamp must not fire."""
    clamped_dist, was_clamped = compute_lane_midpoint_clamp(
        lookahead_m=8.0,
        kappa_preview=0.02,
        dist_to_curve_m=None,
        in_curve=False,
        threshold_m=0.05,
        min_distance_m=3.0,
    )
    assert not was_clamped
    assert clamped_dist == 8.0


# ---------------------------------------------------------------------------
# Test 6: min_distance floor is respected
# ---------------------------------------------------------------------------

def test_min_distance_floor_is_respected() -> None:
    """Clamped distance must never fall below min_distance_m."""
    # Very tight curve immediately ahead: kappa=0.2, dist=0.0
    # delta_d_max = sqrt(2*0.05/0.2) = sqrt(0.5) ≈ 0.707 m
    # raw max_safe = 0.0 + 0.707 = 0.707 m < min_distance_m (3.0)
    # → result should be pinned at 3.0 m
    clamped_dist, was_clamped = compute_lane_midpoint_clamp(
        lookahead_m=8.0,
        kappa_preview=0.2,
        dist_to_curve_m=0.0,
        in_curve=False,
        threshold_m=0.05,
        min_distance_m=3.0,
    )
    assert was_clamped
    assert clamped_dist == pytest.approx(3.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 7: threshold=0 disables the clamp entirely
# ---------------------------------------------------------------------------

def test_threshold_zero_disables_clamp() -> None:
    """Setting threshold_m=0 must disable the clamp even in adverse conditions."""
    clamped_dist, was_clamped = compute_lane_midpoint_clamp(
        lookahead_m=8.0,
        kappa_preview=0.05,
        dist_to_curve_m=0.5,
        in_curve=False,
        threshold_m=0.0,          # disabled
        min_distance_m=3.0,
    )
    assert not was_clamped
    assert clamped_dist == 8.0


# ---------------------------------------------------------------------------
# Test 8: tighter threshold reduces clamp distance further
# ---------------------------------------------------------------------------

def test_tighter_threshold_clamps_more_aggressively() -> None:
    """A tighter contamination threshold should produce a shorter clamped distance."""
    kwargs = dict(
        lookahead_m=10.0,
        kappa_preview=0.01,
        dist_to_curve_m=2.0,
        in_curve=False,
        min_distance_m=3.0,
    )
    dist_loose, _ = compute_lane_midpoint_clamp(threshold_m=0.10, **kwargs)
    dist_tight, _ = compute_lane_midpoint_clamp(threshold_m=0.02, **kwargs)
    assert dist_tight < dist_loose, "Tighter threshold should yield a shorter max distance"


# ---------------------------------------------------------------------------
# Test 9: contamination formula validation
# ---------------------------------------------------------------------------

def test_contamination_at_clamped_boundary_is_within_threshold() -> None:
    """At the clamped boundary the expected lateral contamination ≤ threshold_m."""
    kappa = 0.015
    dist_to_curve = 3.0
    threshold = 0.05

    clamped_dist, was_clamped = compute_lane_midpoint_clamp(
        lookahead_m=12.0,
        kappa_preview=kappa,
        dist_to_curve_m=dist_to_curve,
        in_curve=False,
        threshold_m=threshold,
        min_distance_m=3.0,
    )
    assert was_clamped
    # Δd = how far past curve start the clamped evaluation point sits
    delta_d = max(0.0, clamped_dist - dist_to_curve)
    contamination = 0.5 * kappa * delta_d ** 2
    assert contamination <= threshold + 1e-9, (
        f"contamination={contamination:.4f} exceeds threshold={threshold}"
    )
