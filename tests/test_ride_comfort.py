"""Unit tests for ISO 2631-1 frequency-weighted ride comfort metrics.

Added 2026-08-15. These exist because every pre-existing comfort gate is an
amplitude percentile and therefore frequency-blind: a 0.24 Hz weave and a 3 Hz
shake with identical P95 score the same. The tests below pin the property that
motivated the metric — that it separates those two cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "analyze"))

from analyze_ride_comfort import (  # noqa: E402
    MSDV_BAND_HZ,
    compute_msdv,
    comfort_verdict,
    dominant_frequency,
    wf_weighting,
)


def _sine(freq_hz: float, amp: float, dur_s: float, fs: float) -> np.ndarray:
    t = np.arange(0.0, dur_s, 1.0 / fs)
    return amp * np.sin(2 * np.pi * freq_hz * t)


class TestWfWeighting:
    def test_peaks_inside_motion_sickness_band(self):
        f = np.linspace(0.01, 5.0, 2000)
        w = wf_weighting(f)
        peak_f = f[int(np.argmax(w))]
        assert MSDV_BAND_HZ[0] <= peak_f <= MSDV_BAND_HZ[1]

    def test_attenuates_well_above_the_band(self):
        assert wf_weighting(np.array([3.0]))[0] < 0.25 * wf_weighting(np.array([0.2]))[0]

    def test_attenuates_well_below_the_band(self):
        assert wf_weighting(np.array([0.01]))[0] < 0.25 * wf_weighting(np.array([0.2]))[0]

    def test_zero_frequency_is_finite(self):
        assert np.isfinite(wf_weighting(np.array([0.0]))).all()


class TestMSDV:
    def test_sickness_band_dominates_equal_amplitude_high_frequency(self):
        """THE point of the metric: same amplitude, very different sickness dose."""
        fs, dur, amp = 50.0, 120.0, 1.0
        msdv_sick = compute_msdv(_sine(0.2, amp, dur, fs), fs)
        msdv_shake = compute_msdv(_sine(3.0, amp, dur, fs), fs)
        assert msdv_sick > 3 * msdv_shake

    def test_amplitude_percentile_cannot_tell_them_apart(self):
        """Guards the premise — P95 is identical for both signals above."""
        fs, dur, amp = 50.0, 120.0, 1.0
        a = _sine(0.2, amp, dur, fs)
        b = _sine(3.0, amp, dur, fs)
        assert np.percentile(np.abs(a), 95) == pytest.approx(
            np.percentile(np.abs(b), 95), rel=0.02
        )

    def test_scales_with_amplitude(self):
        fs, dur = 50.0, 60.0
        assert compute_msdv(_sine(0.2, 2.0, dur, fs), fs) > \
               1.8 * compute_msdv(_sine(0.2, 1.0, dur, fs), fs)

    def test_zero_signal_is_zero(self):
        assert compute_msdv(np.zeros(500), 50.0) == pytest.approx(0.0, abs=1e-9)

    def test_too_short_returns_nan(self):
        assert np.isnan(compute_msdv(np.array([1.0, 2.0]), 50.0))


class TestDominantFrequency:
    def test_recovers_known_frequency(self):
        fs = 50.0
        f, _ = dominant_frequency(_sine(0.25, 1.0, 200.0, fs), fs)
        assert f == pytest.approx(0.25, abs=0.02)

    def test_reports_peak_to_peak_amplitude(self):
        fs = 50.0
        _, pk = dominant_frequency(_sine(0.25, 1.5, 200.0, fs), fs)
        assert pk == pytest.approx(3.0, rel=0.05)

    def test_ignores_dc_offset(self):
        fs = 50.0
        f, _ = dominant_frequency(_sine(0.25, 1.0, 200.0, fs) + 10.0, fs)
        assert f == pytest.approx(0.25, abs=0.02)


class TestComfortVerdict:
    @pytest.mark.parametrize("aw,expected", [
        (0.1, "not uncomfortable"),
        (0.5, "a little uncomfortable"),
        (0.8, "fairly uncomfortable"),
        (1.3, "uncomfortable"),
        (2.0, "very uncomfortable"),
        (9.0, "extremely uncomfortable"),
    ])
    def test_iso_2631_reaction_scale(self, aw, expected):
        assert comfort_verdict(aw) == expected
