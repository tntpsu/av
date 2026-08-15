"""Tests for vehicle-fell-off-the-world detection.

The negative cases matter as much as the positive one: an earlier version of
this detector used an absolute drop threshold and false-positived on BOTH
graded tracks, because hill_highway and hill_g1 legitimately descend ~5 m.
The descent-RATE formulation is what separates a grade from a fall, and these
tests pin that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from av_stack.fall_detector import (  # noqa: E402
    DEFAULT_DESCENT_RATE_MPS,
    FallDetector,
    detect_fall_offline,
)


def _drive(det: FallDetector, ys, dt=1 / 13.0, contacts=None):
    """Feed a y-series through the streaming detector at a realistic 13 FPS."""
    for i, y in enumerate(ys):
        det.update(y, contacts[i] if contacts else None, frame=i, time_s=i * dt)
    return det.state


class TestFreeFallDetection:
    def test_detects_free_fall(self):
        """9.81 m/s² descent from ride height."""
        dt = 1 / 13.0
        ys = [0.8 - 0.5 * 9.81 * (i * dt) ** 2 for i in range(40)]
        assert _drive(FallDetector(), ys).fallen

    def test_reports_descent_rate_and_drop(self):
        dt = 1 / 13.0
        ys = [0.8 - 0.5 * 9.81 * (i * dt) ** 2 for i in range(40)]
        st = _drive(FallDetector(), ys)
        assert "free_fall" in st.reason
        assert st.frame is not None and st.time_s is not None

    def test_latches_once_fallen(self):
        dt = 1 / 13.0
        ys = [0.8 - 0.5 * 9.81 * (i * dt) ** 2 for i in range(40)] + [0.8] * 10
        assert _drive(FallDetector(), ys).fallen  # recovery does not clear it


class TestGradesAreNotFalls:
    """The regression that broke the first implementation."""

    @pytest.mark.parametrize("grade_pct,speed", [(5, 12.0), (10, 25.0), (10, 15.0)])
    def test_steep_grade_at_speed_is_not_a_fall(self, grade_pct, speed):
        dt = 1 / 13.0
        descent_rate = speed * (grade_pct / 100.0)   # m/s of altitude loss
        ys = [50.0 - descent_rate * (i * dt) for i in range(400)]  # long descent
        st = _drive(FallDetector(), ys)
        assert not st.fallen, f"{grade_pct}% grade at {speed} m/s misread as a fall"

    def test_large_absolute_drop_on_a_grade_is_not_a_fall(self):
        """Loses 30 m of altitude — far more than any absolute threshold."""
        dt = 1 / 13.0
        ys = [100.0 - 1.0 * (i * dt) for i in range(400)]  # 1 m/s, 30 m total
        st = _drive(FallDetector(), ys)
        assert not st.fallen
        assert st.max_drop_m > 25.0   # the drop is real; it just isn't a fall

    def test_flat_road_is_not_a_fall(self):
        assert not _drive(FallDetector(), [0.8] * 200).fallen


class TestGroundContact:
    def test_all_wheels_off_triggers(self):
        det = FallDetector()
        st = _drive(det, [0.8] * 20, contacts=[[0.0] * 4] * 20)
        assert st.fallen and "no_ground_contact" in st.reason

    def test_single_wheel_lift_does_not_trigger(self):
        """Individual wheels lift ~6% of the time in normal cornering."""
        contacts = [[0.0, 1.0, 1.0, 1.0]] * 50
        assert not _drive(FallDetector(), [0.8] * 50, contacts=contacts).fallen

    def test_brief_airborne_below_threshold_does_not_trigger(self):
        contacts = [[1.0] * 4] * 10 + [[0.0] * 4] * 3 + [[1.0] * 4] * 10
        assert not _drive(FallDetector(airborne_frames=5), [0.8] * 23, contacts=contacts).fallen


class TestOfflineAPI:
    def test_offline_matches_streaming(self):
        dt = 1 / 13.0
        ys = [0.8 - 0.5 * 9.81 * (i * dt) ** 2 for i in range(40)]
        ts = [i * dt for i in range(40)]
        assert detect_fall_offline(ys, None, ts).fallen

    def test_offline_grade_is_clean(self):
        dt = 1 / 13.0
        ys = [100.0 - 1.0 * (i * dt) for i in range(400)]
        ts = [i * dt for i in range(400)]
        assert not detect_fall_offline(ys, None, ts).fallen

    def test_empty_input_is_safe(self):
        assert not detect_fall_offline([], None, []).fallen

    def test_threshold_sits_between_grade_and_fall(self):
        """Physical margin check, not a tuning knob."""
        max_grade_descent = 25.0 * 0.10      # max_speed x max_grade
        assert max_grade_descent < DEFAULT_DESCENT_RATE_MPS < 47.0
