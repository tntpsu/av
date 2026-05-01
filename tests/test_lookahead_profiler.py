"""Tests for the map-predictive lookahead profiler."""

import math
import os
import time

import numpy as np
import pytest

_IN_CI = os.environ.get("CI") == "true" or os.environ.get("AV_NIGHTLY_RUN") == "1"

from trajectory.lookahead_profiler import (
    LookaheadProfile,
    LookaheadProfiler,
    LookaheadProfilerConfig,
)


def _make_config(**overrides) -> LookaheadProfilerConfig:
    defaults = dict(
        preview_horizon_m=40.0,
        e_target_m=0.04,
        k_speed_min=0.35,
        ld_max_m=10.0,
        ld_min_m=2.5,
        contraction_rate=0.30,
        extension_rate=0.10,
        sample_spacing_m=1.0,
        curvature_min=0.001,
    )
    defaults.update(overrides)
    return LookaheadProfilerConfig(**defaults)


def _straight_track(length_m: float = 500.0) -> list[dict]:
    """No curves — pure straight."""
    return []


def _single_curve_track(
    straight_before: float = 200.0,
    radius: float = 40.0,
    angle_deg: float = 90.0,
    straight_after: float = 200.0,
) -> tuple[list[dict], float]:
    """Track: straight → arc → straight."""
    arc_len = radius * math.radians(angle_deg)
    curves = [
        {
            "start_m": straight_before,
            "end_m": straight_before + arc_len,
            "curvature_abs": 1.0 / radius,
            "direction_sign": 1.0,
        }
    ]
    total = straight_before + arc_len + straight_after
    return curves, total


def _hairpin_track() -> tuple[list[dict], float]:
    """Simplified hairpin: straight → R15 → straight → R20 → straight."""
    curves = [
        {"start_m": 80.0, "end_m": 80 + 15 * math.radians(180), "curvature_abs": 1.0 / 15, "direction_sign": 1.0},
        {"start_m": 80 + 15 * math.radians(180) + 40, "end_m": 80 + 15 * math.radians(180) + 40 + 20 * math.radians(90), "curvature_abs": 1.0 / 20, "direction_sign": -1.0},
    ]
    total = curves[-1]["end_m"] + 80
    return curves, total


class TestStraightTrack:
    def test_all_ld_equals_max(self):
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile([], 500.0, is_loop=False)
        assert np.allclose(profile.lookaheads_m, 10.0)

    def test_curvature_is_zero(self):
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile([], 500.0, is_loop=False)
        assert np.allclose(profile.ld_curvature, 10.0)


class TestSingleCurve:
    def test_ld_contracts_before_curve(self):
        curves, total = _single_curve_track(radius=40)
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile(curves, total, is_loop=False)

        # 20m before curve start, Ld should already be below max
        curve_start = curves[0]["start_m"]
        ld_before = profiler.lookup_lookahead(profile, curve_start - 20)
        assert ld_before < 10.0, f"Ld should contract before curve: {ld_before:.2f}"

    def test_ld_minimum_at_curve(self):
        curves, total = _single_curve_track(radius=40)
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile(curves, total, is_loop=False)

        # At curve midpoint, Ld should be close to sqrt(8R×e)
        curve_mid = (curves[0]["start_m"] + curves[0]["end_m"]) / 2
        ld_curve = profiler.lookup_lookahead(profile, curve_mid)
        ld_physics = math.sqrt(8 * 40 * 0.04)
        assert abs(ld_curve - ld_physics) < 1.0, f"Ld at curve: {ld_curve:.2f}, expected ~{ld_physics:.2f}"

    def test_contraction_starts_early_enough(self):
        curves, total = _single_curve_track(radius=40)
        cfg = _make_config(contraction_rate=0.30, preview_horizon_m=40.0)
        profiler = LookaheadProfiler(cfg)
        profile = profiler.build_profile(curves, total, is_loop=False)

        # The preview horizon (40m) + contraction rate together determine when
        # Ld starts dropping. At 60m before the curve, Ld should still be near max.
        curve_start = curves[0]["start_m"]
        ld_far = profiler.lookup_lookahead(profile, curve_start - 60)
        assert ld_far > cfg.ld_max_m - 1.0, f"Ld 60m before curve should be near max: {ld_far:.2f}"

        # At 10m before the curve, Ld should be significantly contracted
        ld_near = profiler.lookup_lookahead(profile, curve_start - 10)
        assert ld_near < cfg.ld_max_m - 2.0, f"Ld 10m before curve should be contracted: {ld_near:.2f}"

    def test_ld_extends_gradually_after_curve(self):
        curves, total = _single_curve_track(radius=40)
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile(curves, total, is_loop=False)

        curve_end = curves[0]["end_m"]
        ld_at_exit = profiler.lookup_lookahead(profile, curve_end)
        ld_10m_after = profiler.lookup_lookahead(profile, curve_end + 10)
        ld_50m_after = profiler.lookup_lookahead(profile, curve_end + 50)

        # Extension is gradual (0.10 m/m)
        assert ld_10m_after > ld_at_exit, "Ld should extend after curve"
        assert ld_50m_after > ld_10m_after, "Ld should keep extending"
        assert ld_10m_after - ld_at_exit <= 0.10 * 10 + 0.1, "Extension should respect rate"


class TestHairpinTrack:
    def test_ld_stays_low_between_tight_curves(self):
        curves, total = _hairpin_track()
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile(curves, total, is_loop=False)

        # Between the two curves, Ld should stay relatively low (not jump to max)
        gap_mid = (curves[0]["end_m"] + curves[1]["start_m"]) / 2
        ld_gap = profiler.lookup_lookahead(profile, gap_mid)
        assert ld_gap < 8.0, f"Ld between tight curves should stay low: {ld_gap:.2f}"


class TestMixedRadius:
    def test_tighter_curve_gets_lower_ld(self):
        # R40 and R150 on same track
        curves = [
            {"start_m": 100.0, "end_m": 163.0, "curvature_abs": 1.0 / 40, "direction_sign": 1.0},
            {"start_m": 300.0, "end_m": 535.0, "curvature_abs": 1.0 / 150, "direction_sign": -1.0},
        ]
        total = 700.0
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile(curves, total, is_loop=False)

        ld_r40 = profiler.lookup_lookahead(profile, 130.0)
        ld_r150 = profiler.lookup_lookahead(profile, 420.0)
        assert ld_r40 < ld_r150, f"R40 Ld ({ld_r40:.2f}) should be < R150 Ld ({ld_r150:.2f})"


class TestLoopWrap:
    def test_contraction_wraps_across_boundary(self):
        """Curve near station 0 → contraction should propagate backward across loop boundary."""
        curves = [
            {"start_m": 10.0, "end_m": 73.0, "curvature_abs": 1.0 / 40, "direction_sign": 1.0},
        ]
        total = 500.0
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile(curves, total, is_loop=True)

        # Near end of track (station 490), should already be contracting for the curve at 10m
        ld_before_wrap = profiler.lookup_lookahead(profile, 490.0)
        assert ld_before_wrap < 10.0, f"Ld should contract before loop wrap: {ld_before_wrap:.2f}"


class TestNonLoop:
    def test_no_wrap_propagation(self):
        """Non-loop track: end of track should not be affected by curve at start."""
        curves = [
            {"start_m": 10.0, "end_m": 73.0, "curvature_abs": 1.0 / 40, "direction_sign": 1.0},
        ]
        total = 500.0
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile(curves, total, is_loop=False)

        ld_end = profiler.lookup_lookahead(profile, 490.0)
        assert ld_end == pytest.approx(10.0, abs=0.5), f"Non-loop end should be near max: {ld_end:.2f}"


class TestSpeedFloor:
    def test_ld_floored_by_speed_profile(self):
        curves, total = _single_curve_track(radius=15)  # very tight
        stations = np.linspace(0, total, 500)
        speeds = np.full(500, 4.0)  # 4 m/s everywhere
        profiler = LookaheadProfiler(_make_config(k_speed_min=0.35))
        profile = profiler.build_profile(
            curves, total, is_loop=False,
            v_profile_speeds=speeds, v_profile_stations=stations,
        )

        # sqrt(8×15×0.04) = 2.19m, but speed floor = 0.35 × 4 = 1.4m → floor is below physics
        # On straights with no curve ahead, Ld should be at ld_max
        ld_straight = profiler.lookup_lookahead(profile, 10.0)
        assert ld_straight >= 10.0 - 0.5


class TestMonotonicRates:
    def test_rate_within_bounds(self):
        curves, total = _single_curve_track(radius=40)
        cfg = _make_config(contraction_rate=0.30, extension_rate=0.10)
        profiler = LookaheadProfiler(cfg)
        profile = profiler.build_profile(curves, total, is_loop=False)

        ld = profile.lookaheads_m
        ds = profile.stations_m[1] - profile.stations_m[0]
        max_rate = max(cfg.contraction_rate, cfg.extension_rate)
        for i in range(1, len(ld)):
            rate = abs(ld[i] - ld[i - 1]) / ds
            assert rate <= max_rate + 0.01, f"Rate {rate:.3f} exceeds max {max_rate} at station {i}"


class TestLookup:
    def test_interpolation_midpoint(self):
        profiler = LookaheadProfiler(_make_config(sample_spacing_m=10.0))
        profile = profiler.build_profile([], 100.0, is_loop=False)
        # Straight track, all values = 10.0 → interpolation should also give 10.0
        assert profiler.lookup_lookahead(profile, 55.0) == pytest.approx(10.0)

    def test_loop_wrap(self):
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile([], 500.0, is_loop=True)
        # Station 600 on a 500m loop = station 100
        ld_100 = profiler.lookup_lookahead(profile, 100.0)
        ld_600 = profiler.lookup_lookahead(profile, 600.0)
        assert abs(ld_100 - ld_600) < 0.01


class TestPerformance:
    @pytest.mark.skipif(
        _IN_CI,
        reason="Hardware-sensitive perf budget; GitHub Actions runners are too slow to honor a 20ms target. Run locally to verify.",
    )
    def test_build_under_10ms(self):
        curves = [
            {"start_m": 250.0 + i * 670, "end_m": 250.0 + i * 670 + 471, "curvature_abs": 1.0 / 300, "direction_sign": 1.0}
            for i in range(4)
        ]
        total = 2700.0
        profiler = LookaheadProfiler(_make_config())
        t0 = time.perf_counter()
        profiler.build_profile(curves, total, is_loop=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 20.0, f"build_profile took {elapsed_ms:.1f}ms (limit 20ms)"

    def test_lookup_under_0_1ms(self):
        profiler = LookaheadProfiler(_make_config())
        profile = profiler.build_profile([], 2700.0, is_loop=True)
        t0 = time.perf_counter()
        for _ in range(1000):
            profiler.lookup_lookahead(profile, 1350.0)
        elapsed_us = (time.perf_counter() - t0) * 1e6 / 1000
        assert elapsed_us < 100, f"lookup took {elapsed_us:.1f}µs (limit 100µs)"
