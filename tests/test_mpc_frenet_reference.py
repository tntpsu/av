"""Unit tests for Frenet-frame MPC reference (greedy-swimming-naur.md Phase F).

Pins the sign convention discovered in Phase 0 against 2026-04-18 recordings:

    d_frenet_linearized = gt_cross_track_lookahead + Ld · sin(gt_heading_error)

The PLUS follows from this codebase's Unity vehicle-frame convention
(lookahead offset and Ld·sin(e_h) carry opposite signs relative to at-car d
during oscillation). Verified on 2152 H2 frames with +0.9872 correlation.

These tests operate on the module-level helpers
`_compute_frenet_d_linearized` and `_select_mpc_e_lat_reference` so they
do not require instantiating a full PIDController.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from control.pid_controller import (
    _compute_frenet_d_linearized,
    _select_mpc_e_lat_reference,
)


def _cfg(mode: str = "frenet_linearized", shadow: bool = False,
         use_lookahead: bool = True) -> dict:
    return {
        "mpc_e_lat_reference_mode": mode,
        "mpc_e_lat_frenet_shadow_mode": shadow,
        "mpc_e_lat_use_lookahead_reference": use_lookahead,
    }


# ─── Core math ───────────────────────────────────────────────────────────────


def test_compute_frenet_matches_lookahead_on_zero_heading():
    """On a straight road with zero heading error, d_frenet == lookahead_offset.

    This is the base identity: when Ld·sin(e_h) = 0, the correction term
    vanishes and the linearized d equals the raw lookahead signal.
    """
    d = _compute_frenet_d_linearized(
        gt_cross_track_lookahead=0.25,
        gt_heading_error_rad=0.0,
        lookahead_distance=8.0,
    )
    assert d == pytest.approx(0.25, abs=1e-12)


def test_compute_frenet_removes_pure_heading_leakage():
    """Pure heading error with zero Frenet d → d_linearized ≈ 0 after correction.

    If a car sits on the centerline but heading-offset by 0.1 rad, the
    lookahead sensor reads Ld·sin(0.1) ≈ 0.799 m. The linearized formula
    adds Ld·sin(−0.1) = −0.799 (wait — no, lookahead_offset ALREADY carries
    -0.799 in this vehicle-frame convention, and we ADD Ld·sin(+0.1) = +0.799
    to recover d ≈ 0). This is the sign correction Phase 0 discovered.
    """
    Ld = 8.0
    e_h = 0.1
    # The raw lookahead sensor reads −Ld·sin(e_h) when d_frenet = 0 in the
    # codebase's convention — tested via a synthesized pair that Phase 0
    # showed matches at-car measurements.
    lookahead_offset = -Ld * math.sin(e_h)
    d = _compute_frenet_d_linearized(lookahead_offset, e_h, Ld)
    assert d == pytest.approx(0.0, abs=1e-12)


def test_frenet_sign_convention_matches_at_car():
    """Regression pin: the PLUS sign (not MINUS) matches at-car d.

    Phase 0 empirical result on recording_20260418_195233.h5: the
    variant B formula lookahead + Ld·sin(e_h) matched
    gt_cross_track_at_car_m to 0.051 m RMS across 2152 frames with
    +0.9872 correlation. The MINUS variant gave 1.50 m P95 and +0.22 corr.

    This test builds a synthetic triplet that encodes the discovered
    geometric relationship and asserts the implementation uses the
    correct sign. If someone flips the sign "to fix a bug", this test
    fails immediately.
    """
    # Synthetic frame: car 0.05 m right of centerline, heading 0.1 rad
    # to the right; path straight. Then:
    #   at_car_d      = +0.05     (physical Frenet d)
    #   lookahead_off = at_car_d - Ld·sin(e_h)   [Unity convention]
    Ld = 8.0
    e_h = 0.1
    at_car_d = 0.05
    lookahead_offset = at_car_d - Ld * math.sin(e_h)

    d = _compute_frenet_d_linearized(lookahead_offset, e_h, Ld)
    # The CORRECT sign restores at_car_d. The WRONG sign would give
    # at_car_d − 2·Ld·sin(e_h) ≈ −1.547 — far from the true +0.05.
    assert d == pytest.approx(at_car_d, abs=1e-9), (
        "Frenet sign is wrong. Phase 0 verified PLUS on 2152 H2 frames."
    )


def test_compute_frenet_returns_none_when_inputs_missing():
    """Defensive: missing GT → None (caller falls back, no NaN crash)."""
    assert _compute_frenet_d_linearized(None, 0.1, 8.0) is None
    assert _compute_frenet_d_linearized(0.1, None, 8.0) is None
    assert _compute_frenet_d_linearized(0.1, 0.1, 0.0) is None
    assert _compute_frenet_d_linearized(0.1, 0.1, -1.0) is None


# ─── Reference selection ─────────────────────────────────────────────────────


def test_active_frenet_feeds_negated_d():
    """mode=frenet_linearized, shadow=False → raw_e_lat == -d_frenet.

    The negation matches the legacy convention: MPC e_lat > 0 = car RIGHT,
    vehicle-frame lookahead > 0 = lane center RIGHT of car = car LEFT.
    """
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(mode="frenet_linearized", shadow=False),
        gt_cross_track=0.05,
        gt_cross_track_lookahead=0.1,
        gt_cross_track_source_code="road_frame_at_car",
        gt_heading_error_rad=0.2,
        lookahead_distance=8.0,
        fallback_lateral_error=0.0,
        fallback_heading_error=0.0,
    )
    expected_d = 0.1 + 8.0 * math.sin(0.2)
    assert r["raw_e_lat"] == pytest.approx(-expected_d, abs=1e-9)
    assert r["mpc_e_lat_reference_source"] == "frenet_linearized"
    assert r["mpc_e_lat_frenet_shadow_mode"] == 0


def test_shadow_mode_feeds_legacy_lookahead_but_logs_frenet():
    """shadow=True → MPC fed legacy -lookahead; telemetry still populated.

    This is the rollout-safety precedent from project_velocity_profile.md.
    The candidate Frenet value goes to HDF5; MPC unaffected.
    """
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(mode="frenet_linearized", shadow=True),
        gt_cross_track=0.05,
        gt_cross_track_lookahead=0.8,
        gt_cross_track_source_code="road_frame_at_car",
        gt_heading_error_rad=0.1,
        lookahead_distance=8.0,
        fallback_lateral_error=0.0,
        fallback_heading_error=0.0,
    )
    assert r["raw_e_lat"] == pytest.approx(-0.8, abs=1e-12)
    assert r["mpc_e_lat_reference_source"] == "lookahead_gt_shadow_frenet"
    assert r["mpc_e_lat_frenet_shadow_mode"] == 1
    # Telemetry is populated even in shadow
    assert math.isfinite(r["mpc_e_lat_frenet_linearized_m"])
    assert math.isfinite(r["mpc_e_lat_shadow_delta_m"])
    # Shadow delta equals +Ld·sin(e_h) by construction
    assert r["mpc_e_lat_shadow_delta_m"] == pytest.approx(
        8.0 * math.sin(0.1), abs=1e-9
    )


def test_legacy_mode_lookahead_gt():
    """Back-compat: explicit legacy lookahead reproduces 2026-04-05 behavior."""
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(mode="lookahead_gt_legacy"),
        gt_cross_track=0.05,
        gt_cross_track_lookahead=0.8,
        gt_cross_track_source_code="road_frame_at_car",
        gt_heading_error_rad=0.1,
        lookahead_distance=8.0,
        fallback_lateral_error=0.0,
        fallback_heading_error=0.0,
    )
    assert r["raw_e_lat"] == pytest.approx(-0.8, abs=1e-12)
    assert r["mpc_e_lat_reference_source"] == "lookahead_gt"


def test_legacy_mode_at_car_gt():
    """Explicit at-car legacy forces the at-car branch."""
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(mode="at_car_gt_legacy"),
        gt_cross_track=0.05,
        gt_cross_track_lookahead=0.8,
        gt_cross_track_source_code="road_frame_at_car",
        gt_heading_error_rad=0.1,
        lookahead_distance=8.0,
        fallback_lateral_error=0.0,
        fallback_heading_error=0.0,
    )
    assert r["mpc_e_lat_reference_source"] == "at_car_gt"
    assert r["raw_e_lat"] == pytest.approx(0.05, abs=1e-12)


def test_frenet_map_exact_falls_through_to_legacy():
    """Phase 2 placeholder: frenet_map_exact not implemented → legacy path."""
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(mode="frenet_map_exact"),
        gt_cross_track=0.05,
        gt_cross_track_lookahead=0.8,
        gt_cross_track_source_code="road_frame_at_car",
        gt_heading_error_rad=0.1,
        lookahead_distance=8.0,
        fallback_lateral_error=0.0,
        fallback_heading_error=0.0,
    )
    # Falls through to legacy lookahead (use_lookahead_reference=True default)
    assert r["mpc_e_lat_reference_source"] in ("lookahead_gt", "at_car_gt")


def test_fallback_to_pp_when_no_gt():
    """No GT at all → PP lateral error fallback (last-resort path)."""
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(),
        gt_cross_track=None,
        gt_cross_track_lookahead=None,
        gt_cross_track_source_code="",
        gt_heading_error_rad=None,
        lookahead_distance=8.0,
        fallback_lateral_error=0.3,
        fallback_heading_error=0.05,
    )
    assert r["mpc_e_lat_reference_source"] == "fallback_pp"
    assert r["raw_e_lat"] == pytest.approx(-0.3, abs=1e-12)
    assert r["raw_e_heading"] == pytest.approx(0.05, abs=1e-12)


def test_zero_lookahead_distance_falls_back_gracefully():
    """Ld=0 → Frenet candidate unavailable, fall back to legacy chain."""
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(mode="frenet_linearized"),
        gt_cross_track=0.05,
        gt_cross_track_lookahead=0.8,
        gt_cross_track_source_code="road_frame_at_car",
        gt_heading_error_rad=0.1,
        lookahead_distance=0.0,
        fallback_lateral_error=0.0,
        fallback_heading_error=0.0,
    )
    # No NaN / exception; falls through to legacy
    assert r["mpc_e_lat_reference_source"] in ("lookahead_gt", "at_car_gt")
    assert math.isnan(r["mpc_e_lat_frenet_linearized_m"])


def test_force_lookahead_off_for_nmpc_cross_term_cost():
    """NMPC path sets force_lookahead_off=True when mpc_lookahead_cost_enabled.

    The QP cross-term already encodes the lookahead relationship, so the
    reference should drop to at_car_gt rather than doubling up.
    """
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(mode="at_car_gt_legacy"),
        gt_cross_track=0.05,
        gt_cross_track_lookahead=0.8,
        gt_cross_track_source_code="road_frame_at_car",
        gt_heading_error_rad=0.1,
        lookahead_distance=8.0,
        fallback_lateral_error=0.0,
        fallback_heading_error=0.0,
        force_lookahead_off=True,
    )
    # Legacy at_car_gt path only (lookahead override disabled)
    assert r["mpc_e_lat_reference_source"] == "at_car_gt"


def test_telemetry_populated_in_all_modes():
    """All three telemetry fields present in output regardless of mode."""
    for mode in ("frenet_linearized", "lookahead_gt_legacy", "at_car_gt_legacy"):
        for shadow in (True, False):
            r = _select_mpc_e_lat_reference(
                mpc_cfg=_cfg(mode=mode, shadow=shadow),
                gt_cross_track=0.05,
                gt_cross_track_lookahead=0.8,
                gt_cross_track_source_code="road_frame_at_car",
                gt_heading_error_rad=0.1,
                lookahead_distance=8.0,
                fallback_lateral_error=0.0,
                fallback_heading_error=0.0,
            )
            assert "mpc_e_lat_frenet_linearized_m" in r, (mode, shadow)
            assert "mpc_e_lat_shadow_delta_m" in r, (mode, shadow)
            assert "mpc_e_lat_frenet_shadow_mode" in r, (mode, shadow)


# ─── Guardrail for future refactor ───────────────────────────────────────────


def test_shadow_delta_equals_heading_leak_term():
    """shadow_delta is exactly +Ld·sin(e_h) — this equivalence is what the
    pipeline-analysis §N printout relies on.
    """
    Ld = 8.0
    e_h = 0.15
    r = _select_mpc_e_lat_reference(
        mpc_cfg=_cfg(mode="frenet_linearized", shadow=True),
        gt_cross_track=0.0,
        gt_cross_track_lookahead=0.4,
        gt_cross_track_source_code="road_frame_at_car",
        gt_heading_error_rad=e_h,
        lookahead_distance=Ld,
        fallback_lateral_error=0.0,
        fallback_heading_error=0.0,
    )
    assert r["mpc_e_lat_shadow_delta_m"] == pytest.approx(
        Ld * math.sin(e_h), abs=1e-9
    )
