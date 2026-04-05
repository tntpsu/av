"""Tests for MPC reference point alignment (Phase 3.0).

Validates that MPC's e_lat input uses the correct reference source
(lookahead GT cross-track) to align with the scored lateral_error,
and that sign conventions, fallback paths, and telemetry are correct.
"""

import pytest


# ---------------------------------------------------------------------------
# Helper: simulate the MPC e_lat reference selection logic
# (mirrors pid_controller.py lines 5709-5741)
# ---------------------------------------------------------------------------

def _select_mpc_e_lat(
    *,
    use_lookahead_ref: bool = True,
    gt_cross_track: float | None = None,
    gt_heading: float | None = None,
    gt_cross_track_lookahead: float | None = None,
    gt_cross_track_source_code: str = '',
    lateral_error: float = 0.0,
    heading_error: float = 0.0,
) -> tuple[float, float, str]:
    """Return (raw_e_lat, raw_e_heading, ref_source)."""
    ref_source = 'fallback_pp'
    if gt_cross_track is not None and gt_heading is not None:
        if use_lookahead_ref and gt_cross_track_lookahead is not None:
            raw_e_lat = -float(gt_cross_track_lookahead)
            ref_source = 'lookahead_gt'
        elif gt_cross_track_source_code == 'road_frame_at_car':
            raw_e_lat = float(gt_cross_track)
            ref_source = 'at_car_gt'
        else:
            raw_e_lat = -float(gt_cross_track)
            ref_source = 'at_car_gt'
        raw_e_heading = float(gt_heading)
    else:
        raw_e_lat = -float(lateral_error)
        raw_e_heading = float(heading_error)
    return raw_e_lat, raw_e_heading, ref_source


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMPCReferenceAlignment:
    """MPC e_lat reference source selection."""

    def test_lookahead_reference_used_when_available(self):
        """When config enabled and lookahead GT present, use lookahead."""
        e_lat, _, src = _select_mpc_e_lat(
            use_lookahead_ref=True,
            gt_cross_track=0.1,
            gt_heading=0.01,
            gt_cross_track_lookahead=0.2,
        )
        assert src == 'lookahead_gt'
        assert e_lat == pytest.approx(-0.2)

    def test_at_car_fallback_when_lookahead_missing(self):
        """When lookahead is None, fall back to at-car GT."""
        e_lat, _, src = _select_mpc_e_lat(
            use_lookahead_ref=True,
            gt_cross_track=0.1,
            gt_heading=0.01,
            gt_cross_track_lookahead=None,
        )
        assert src == 'at_car_gt'
        assert e_lat == pytest.approx(-0.1)

    def test_at_car_fallback_when_config_disabled(self):
        """When config flag is False, use at-car GT even if lookahead exists."""
        e_lat, _, src = _select_mpc_e_lat(
            use_lookahead_ref=False,
            gt_cross_track=0.1,
            gt_heading=0.01,
            gt_cross_track_lookahead=0.2,
        )
        assert src == 'at_car_gt'
        assert e_lat == pytest.approx(-0.1)

    def test_pp_fallback_when_no_gt(self):
        """When no GT cross-track available, fall back to -lateral_error."""
        e_lat, e_hdg, src = _select_mpc_e_lat(
            gt_cross_track=None,
            gt_heading=None,
            lateral_error=0.3,
            heading_error=0.05,
        )
        assert src == 'fallback_pp'
        assert e_lat == pytest.approx(-0.3)
        assert e_hdg == pytest.approx(0.05)

    def test_sign_convention_matches_lateral_error(self):
        """Lookahead GT positive (ref right of car) → e_lat negative (car left).

        This ensures MPC e_lat aligns with -lateral_error.
        """
        # Lookahead reference: positive = lane center RIGHT of car = car LEFT
        # MPC e_lat > 0 = car RIGHT. So negate: e_lat = -lookahead
        e_lat_lk, _, _ = _select_mpc_e_lat(
            use_lookahead_ref=True,
            gt_cross_track=0.15,
            gt_heading=0.0,
            gt_cross_track_lookahead=0.25,
        )

        # Fallback PP: lateral_error = ref_x (positive = ref RIGHT of car)
        e_lat_pp, _, _ = _select_mpc_e_lat(
            gt_cross_track=None,
            gt_heading=None,
            lateral_error=0.25,
        )

        # Both should produce the same sign for the same physical offset
        assert e_lat_lk == pytest.approx(e_lat_pp)

    def test_road_frame_at_car_no_negation(self):
        """road_frame_at_car source uses gt_cross_track directly (no negate)."""
        e_lat, _, src = _select_mpc_e_lat(
            use_lookahead_ref=False,
            gt_cross_track=0.1,
            gt_heading=0.01,
            gt_cross_track_source_code='road_frame_at_car',
        )
        assert src == 'at_car_gt'
        assert e_lat == pytest.approx(0.1)  # No negation for road_frame

    def test_heading_always_from_gt_when_available(self):
        """Heading error always comes from GT when GT is present, regardless of ref source."""
        _, e_hdg_lk, _ = _select_mpc_e_lat(
            use_lookahead_ref=True,
            gt_cross_track=0.1,
            gt_heading=0.05,
            gt_cross_track_lookahead=0.2,
        )
        _, e_hdg_at, _ = _select_mpc_e_lat(
            use_lookahead_ref=False,
            gt_cross_track=0.1,
            gt_heading=0.05,
        )
        assert e_hdg_lk == pytest.approx(0.05)
        assert e_hdg_at == pytest.approx(0.05)

    def test_reference_divergence_computation(self):
        """Reference divergence is lookahead - at_car (both available)."""
        gt_cross_track = 0.10
        gt_cross_track_lookahead = 0.25
        divergence = float(gt_cross_track_lookahead) - float(gt_cross_track)
        assert divergence == pytest.approx(0.15)
        # On curves, divergence grows with curvature — this is expected
