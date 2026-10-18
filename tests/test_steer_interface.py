"""Steer-command interface (T-STANLEY-FF-SPEED-MAP, 2026-10-17): physical wheel angle → Unity's normalized command.

Unity: desiredSteerAngle = steerInput × Lerp(30°, 16°, v/12 m/s). Verified on recordings 2026-10-17
(steering_angle_actual / steering_input): 26.1° at 3 m/s (formula 26.5), 23.8 at 5.5 (23.6), 20.6 at 7 (21.8), 18.3 at 9 (19.5).
"""
from __future__ import annotations

import math
import pytest

from control.steer_interface import (unity_max_steer_deg, unity_max_steer_rad, resolve_max_steer_rad,
                                     steer_rad_to_normalized)


class TestMirror:
    @pytest.mark.parametrize("v,deg", [(0.0, 30.0), (3.0, 26.5), (5.3, 23.8167), (6.0, 23.0), (8.0, 20.6667), (12.0, 16.0), (15.0, 16.0)])
    def test_lerp_matches_car_controller(self, v, deg):
        assert unity_max_steer_deg(v) == pytest.approx(deg, abs=1e-3)

    def test_recorded_delivery_ratios_match_within_1_5_deg(self):
        for v, measured in [(3.0, 26.1), (5.5, 23.8), (7.0, 20.6), (9.0, 18.3)]:
            assert abs(unity_max_steer_deg(v) - measured) < 1.5

    def test_bad_speed_falls_back_to_low_speed_value(self):
        assert unity_max_steer_deg(float('nan')) == 30.0 and unity_max_steer_deg(None) == 30.0


class TestResolve:
    def test_plant_value_preferred_when_plausible(self):
        assert resolve_max_steer_rad(5.3, 24.0) == pytest.approx(math.radians(24.0))

    @pytest.mark.parametrize("bad", [None, float('nan'), 0.0, 90.0, 'x'])
    def test_falls_back_to_mirror(self, bad):
        assert resolve_max_steer_rad(5.3, bad) == pytest.approx(unity_max_steer_rad(5.3))


class TestNormalize:
    def test_radians_over_full_scale(self):
        assert steer_rad_to_normalized(0.1, math.radians(25.0)) == pytest.approx(0.1 / math.radians(25.0))

    def test_clips_to_unit(self):
        assert steer_rad_to_normalized(1.0, math.radians(20.0)) == 1.0 and steer_rad_to_normalized(-1.0, 0.3) == -1.0

    def test_delivered_angle_equals_commanded_radians(self):
        # Unity delivers norm × max_steer(v): with the interface the round trip is the identity at every speed
        for v in (3.2, 5.3, 8.0, 11.6):
            ms = unity_max_steer_rad(v); n = steer_rad_to_normalized(0.15, ms)
            assert n * ms == pytest.approx(0.15)
        # the fixed-30° normalization under-delivers by r(v)
        assert (0.15 / math.radians(30.0)) * unity_max_steer_rad(11.6) == pytest.approx(0.15 * (1 - 14 / 30 * 11.6 / 12), abs=1e-9)
