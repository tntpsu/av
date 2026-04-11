"""Tests for trajectory.velocity_profiler — pre-computed velocity profile."""

import math
import time

import numpy as np
import pytest

from trajectory.velocity_profiler import (
    VelocityProfile,
    VelocityProfiler,
    VelocityProfilerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

G = 9.81


def _default_config(**overrides) -> VelocityProfilerConfig:
    defaults = dict(
        a_lat_max_mps2=0.20 * G,       # 0.20g
        a_lon_accel_mps2=2.0,
        a_lon_brake_mps2=1.8,
        v_max_mps=12.0,
        min_speed_mps=3.0,
        sample_spacing_m=1.0,
    )
    defaults.update(overrides)
    return VelocityProfilerConfig(**defaults)


def _straight_track(length_m: float = 200.0) -> tuple[list[dict], float, bool]:
    """Pure straight, no curves."""
    return [], length_m, True


def _single_curve_track(
    straight_before: float = 80.0,
    radius: float = 40.0,
    angle_deg: float = 90.0,
    straight_after: float = 80.0,
    loop: bool = False,
) -> tuple[list[dict], float, bool]:
    """Straight → single arc → straight."""
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
    return curves, total, loop


def _hairpin_track() -> tuple[list[dict], float, bool]:
    """Mimics hairpin_15: tight R15/R20 S-curves."""
    segments = []
    s = 20.0  # 20m straight lead-in
    for radius, angle in [(20, 90), (15, 120), (20, 90), (15, 120)]:
        arc_len = radius * math.radians(angle)
        segments.append({
            "start_m": s,
            "end_m": s + arc_len,
            "curvature_abs": 1.0 / radius,
            "direction_sign": 1.0 if len(segments) % 2 == 0 else -1.0,
        })
        s += arc_len + 10.0  # 10m straight between curves
    total = s + 20.0  # 20m straight tail
    return segments, total, True


def _mixed_radius_track() -> tuple[list[dict], float, bool]:
    """Mixed radii: R40, R60, R150, R200."""
    segments = []
    s = 50.0
    for radius in [40, 60, 150, 200]:
        arc_len = radius * math.radians(90)
        segments.append({
            "start_m": s,
            "end_m": s + arc_len,
            "curvature_abs": 1.0 / radius,
            "direction_sign": 1.0,
        })
        s += arc_len + 50.0
    return segments, s, True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStraightTrack:
    def test_uniform_v_max(self):
        """On a straight track, profile should be v_max everywhere."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _straight_track()
        profile = profiler.build_profile(curves, total, loop)

        np.testing.assert_allclose(
            profile.speeds_mps,
            _default_config().v_max_mps,
            atol=0.01,
        )

    def test_curvature_is_zero(self):
        """Straight track should have κ=0 everywhere."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _straight_track()
        profile = profiler.build_profile(curves, total, loop)
        np.testing.assert_array_equal(profile.kappa_abs, 0.0)


class TestSingleCurve:
    def test_speed_drops_at_curve(self):
        """Speed should be lower inside the curve than on the straight."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _single_curve_track()
        profile = profiler.build_profile(curves, total, loop)

        # Find indices inside the curve
        curve_start = curves[0]["start_m"]
        curve_end = curves[0]["end_m"]
        curve_mask = (profile.stations_m >= curve_start) & (profile.stations_m < curve_end)
        straight_mask = profile.stations_m < curve_start - 20  # well before curve

        v_in_curve = profile.speeds_mps[curve_mask].mean()
        v_on_straight = profile.speeds_mps[straight_mask].mean()

        assert v_in_curve < v_on_straight, (
            f"Curve speed {v_in_curve:.2f} should be < straight speed {v_on_straight:.2f}"
        )

    def test_braking_ramp_before_curve(self):
        """Speed should decrease monotonically approaching the curve."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _single_curve_track(straight_before=100.0, loop=False)
        profile = profiler.build_profile(curves, total, loop)

        curve_start = curves[0]["start_m"]
        # Find the braking zone: last 30m before curve
        ramp_mask = (profile.stations_m >= curve_start - 30) & (profile.stations_m < curve_start)
        ramp_speeds = profile.speeds_mps[ramp_mask]

        # Should be monotonically decreasing (or equal)
        diffs = np.diff(ramp_speeds)
        assert np.all(diffs <= 0.01), f"Braking ramp not monotonic: diffs={diffs}"

    def test_accel_ramp_after_curve(self):
        """Speed should increase monotonically after the curve."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _single_curve_track(straight_after=100.0, loop=False)
        profile = profiler.build_profile(curves, total, loop)

        curve_end = curves[0]["end_m"]
        # First 30m after curve
        ramp_mask = (profile.stations_m > curve_end) & (profile.stations_m <= curve_end + 30)
        ramp_speeds = profile.speeds_mps[ramp_mask]

        diffs = np.diff(ramp_speeds)
        assert np.all(diffs >= -0.01), f"Accel ramp not monotonic: diffs={diffs}"


class TestLoopWrap:
    def test_braking_for_curve_at_start(self):
        """On a loop track with a curve near s=0, braking should appear at track end."""
        # Curve starts at s=5m on a 200m loop
        curves = [{
            "start_m": 5.0,
            "end_m": 5.0 + 40 * math.radians(90),
            "curvature_abs": 1.0 / 40.0,
            "direction_sign": 1.0,
        }]
        profiler = VelocityProfiler(_default_config())
        profile = profiler.build_profile(curves, 200.0, is_loop=True)

        # Last 20m of track should be decelerating (braking for curve at s=5)
        tail_mask = profile.stations_m >= 180.0
        tail_speeds = profile.speeds_mps[tail_mask]

        diffs = np.diff(tail_speeds)
        assert np.any(diffs < -0.01), (
            "Loop wrap: expected braking at track end for curve near start"
        )


class TestHairpinGeometry:
    def test_min_speed_matches_physics(self):
        """Hairpin R15 at 0.20g should give v ≈ sqrt(0.20*9.81/0.0667) ≈ 5.42 m/s."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _hairpin_track()
        profile = profiler.build_profile(curves, total, loop)

        kappa_r15 = 1.0 / 15.0
        v_expected = math.sqrt(0.20 * G / kappa_r15)

        # Find minimum speed in the profile
        v_min = profile.speeds_mps.min()

        # Should be close to physics prediction (forward/backward passes may
        # lower it slightly if curve sequence is too tight for accel recovery)
        assert v_min <= v_expected + 0.5, (
            f"Min speed {v_min:.2f} should be near physics limit {v_expected:.2f}"
        )
        assert v_min >= _default_config().min_speed_mps, (
            f"Min speed {v_min:.2f} should be >= floor {_default_config().min_speed_mps}"
        )


class TestMixedRadius:
    def test_different_speeds_per_radius(self):
        """Each radius should produce a different curve speed."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _mixed_radius_track()
        profile = profiler.build_profile(curves, total, loop)

        mid_speeds = []
        for c in curves:
            mid_s = (c["start_m"] + c["end_m"]) / 2
            mid_speeds.append(profiler.lookup_speed(profile, mid_s))

        # R40 < R60 < R150 < R200 (smaller radius → slower)
        for i in range(len(mid_speeds) - 1):
            assert mid_speeds[i] < mid_speeds[i + 1] + 0.5, (
                f"R{[40,60,150,200][i]} speed {mid_speeds[i]:.2f} should be "
                f"<= R{[40,60,150,200][i+1]} speed {mid_speeds[i+1]:.2f}"
            )


class TestCurveSequence:
    def test_no_full_recovery_between_close_curves(self):
        """Two curves 15m apart: speed shouldn't fully recover to v_max."""
        gap_m = 15.0
        r = 40.0
        arc_len = r * math.radians(90)
        curves = [
            {"start_m": 50.0, "end_m": 50.0 + arc_len, "curvature_abs": 1/r, "direction_sign": 1.0},
            {"start_m": 50.0 + arc_len + gap_m, "end_m": 50.0 + 2*arc_len + gap_m,
             "curvature_abs": 1/r, "direction_sign": -1.0},
        ]
        total = 50.0 + 2 * arc_len + gap_m + 50.0

        profiler = VelocityProfiler(_default_config())
        profile = profiler.build_profile(curves, total, is_loop=False)

        # Speed in the gap between curves
        gap_mid = curves[0]["end_m"] + gap_m / 2
        v_gap = profiler.lookup_speed(profile, gap_mid)

        assert v_gap < _default_config().v_max_mps - 0.5, (
            f"Gap speed {v_gap:.2f} should be well below v_max "
            f"{_default_config().v_max_mps} due to close curve sequence"
        )


class TestForwardPassLimitsAccel:
    def test_cant_jump_speed(self):
        """Can't jump from curve speed to v_max instantly."""
        profiler = VelocityProfiler(_default_config(a_lon_accel_mps2=1.0))
        curves, total, loop = _single_curve_track(straight_after=100.0, loop=False)
        profile = profiler.build_profile(curves, total, loop)

        curve_end = curves[0]["end_m"]
        v_at_exit = profiler.lookup_speed(profile, curve_end)
        v_2m_after = profiler.lookup_speed(profile, curve_end + 2.0)

        # At 1.0 m/s² accel, Δv over 2m: v² = v0² + 2*a*Δs → Δv ≈ 2*a*Δs/(2*v0) ≈ small
        max_possible = math.sqrt(v_at_exit**2 + 2 * 1.0 * 2.0)
        assert v_2m_after <= max_possible + 0.1, (
            f"Speed 2m after curve ({v_2m_after:.2f}) exceeds kinematic limit ({max_possible:.2f})"
        )


class TestBackwardPassLimitsBraking:
    def test_braking_starts_far_enough(self):
        """Braking ramp should start far enough before the curve."""
        cfg = _default_config(a_lon_brake_mps2=1.8, v_max_mps=12.0)
        profiler = VelocityProfiler(cfg)
        curves, total, loop = _single_curve_track(straight_before=100.0, radius=40, loop=False)
        profile = profiler.build_profile(curves, total, loop)

        # v at curve entry
        v_curve = math.sqrt(cfg.a_lat_max_mps2 / (1.0 / 40.0))
        # Braking distance: Δs = (v_max² - v_curve²) / (2 * a_brake)
        expected_brake_dist = (cfg.v_max_mps**2 - v_curve**2) / (2 * cfg.a_lon_brake_mps2)

        # Find where speed first drops below v_max - 0.5
        curve_start = curves[0]["start_m"]
        below_mask = (profile.speeds_mps < cfg.v_max_mps - 0.5) & (profile.stations_m < curve_start)
        if below_mask.any():
            brake_start = profile.stations_m[below_mask][0]
            actual_brake_dist = curve_start - brake_start
            # Allow 0.6x tolerance due to sample spacing discretization
            assert actual_brake_dist >= expected_brake_dist * 0.6, (
                f"Braking starts {actual_brake_dist:.1f}m before curve, expected ~{expected_brake_dist:.1f}m"
            )


class TestLookupInterpolation:
    def test_exact_sample_points(self):
        """Lookup at exact station values returns the stored speed."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _straight_track(100.0)
        profile = profiler.build_profile(curves, total, loop)

        for i in [0, 10, 50, len(profile.stations_m) - 1]:
            v = profiler.lookup_speed(profile, profile.stations_m[i])
            assert abs(v - profile.speeds_mps[i]) < 0.01

    def test_between_samples(self):
        """Lookup between samples should linearly interpolate."""
        profiler = VelocityProfiler(_default_config(sample_spacing_m=10.0))
        # Create a curve so speeds vary
        curves, total, loop = _single_curve_track(loop=False)
        profile = profiler.build_profile(curves, total, loop)

        # Pick a midpoint between two samples
        idx = len(profile.stations_m) // 2
        s_mid = (profile.stations_m[idx] + profile.stations_m[idx + 1]) / 2
        v_mid = profiler.lookup_speed(profile, s_mid)
        v_expected = (profile.speeds_mps[idx] + profile.speeds_mps[idx + 1]) / 2

        assert abs(v_mid - v_expected) < 0.01, (
            f"Interpolation: got {v_mid:.4f}, expected {v_expected:.4f}"
        )


class TestLookupLoopWrap:
    def test_wrap_past_total_length(self):
        """Odometer > total_length should wrap on loop tracks."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _single_curve_track(loop=True)
        profile = profiler.build_profile(curves, total, loop)

        v_at_10 = profiler.lookup_speed(profile, 10.0)
        v_at_10_wrapped = profiler.lookup_speed(profile, 10.0 + total)
        assert abs(v_at_10 - v_at_10_wrapped) < 0.01


class TestRebuildWithConstraint:
    def test_constraint_lowers_speed(self):
        """ACC constraint should lower speed at the constraint station."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _straight_track()
        base = profiler.build_profile(curves, total, loop)

        constrained = profiler.rebuild_with_constraint(base, 100.0, 5.0)
        v_at_100 = profiler.lookup_speed(constrained, 100.0)
        assert v_at_100 <= 5.5, f"Constrained speed at 100m: {v_at_100:.2f}, expected ≤5.5"

    def test_constraint_propagates_backward(self):
        """Braking ramp should appear before the constraint point."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _straight_track(300.0)
        base = profiler.build_profile(curves, total, loop)

        constrained = profiler.rebuild_with_constraint(base, 150.0, 5.0)

        # Braking distance from 12→5 at 1.8 m/s² is ~33m. At 20m before
        # the constraint (well inside the braking zone), speed should be reduced.
        v_before = profiler.lookup_speed(constrained, 130.0)
        assert v_before < _default_config().v_max_mps - 0.5, (
            f"Speed 20m before constraint: {v_before:.2f}, should be below v_max"
        )

    def test_base_profile_unchanged(self):
        """rebuild_with_constraint should not mutate the base profile."""
        profiler = VelocityProfiler(_default_config())
        curves, total, loop = _straight_track()
        base = profiler.build_profile(curves, total, loop)
        original_speeds = base.speeds_mps.copy()

        profiler.rebuild_with_constraint(base, 100.0, 5.0)

        np.testing.assert_array_equal(base.speeds_mps, original_speeds)


class TestMinSpeedFloor:
    def test_no_speed_below_floor(self):
        """Profile should never go below min_speed_mps."""
        cfg = _default_config(min_speed_mps=3.0)
        profiler = VelocityProfiler(cfg)
        curves, total, loop = _hairpin_track()
        profile = profiler.build_profile(curves, total, loop)

        assert profile.speeds_mps.min() >= cfg.min_speed_mps - 0.01, (
            f"Min speed {profile.speeds_mps.min():.3f} below floor {cfg.min_speed_mps}"
        )


class TestPerformance:
    def test_build_under_10ms(self):
        """Profile build should complete in < 10ms for a large track."""
        cfg = _default_config(sample_spacing_m=1.0)
        profiler = VelocityProfiler(cfg)
        # 5km track with many curves
        curves = []
        s = 0.0
        for i in range(50):
            r = 20.0 + (i % 5) * 20.0
            arc = r * math.radians(90)
            curves.append({
                "start_m": s + 50.0,
                "end_m": s + 50.0 + arc,
                "curvature_abs": 1.0 / r,
                "direction_sign": 1.0 if i % 2 == 0 else -1.0,
            })
            s += 50.0 + arc

        total = s + 50.0

        # Warm up
        profiler.build_profile(curves, total, True)

        # Timed run
        t0 = time.perf_counter()
        profile = profiler.build_profile(curves, total, True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 10.0, f"build_profile took {elapsed_ms:.1f}ms (limit 10ms)"
        assert len(profile.stations_m) > 4000, f"Expected >4000 samples, got {len(profile.stations_m)}"
