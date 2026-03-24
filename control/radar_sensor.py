"""
Forward radar sensor with EMA filter.

Provides the RadarSensor ABC and ForwardRadarSensor implementation used by
the ACC controller. Designed for Step 7 extension — RearCornerRadarSensor
will implement the same RadarSensor interface without touching ACCController.

HDF5 field names (radar_fwd_* prefix — forward-specific, Step 7 compatible):
  vehicle/radar_fwd_detected       — bool (0/1)
  vehicle/radar_fwd_distance_m     — float32, raw range from SphereCast
  vehicle/radar_fwd_range_rate_mps — float32, Doppler range rate (+ = closing)
  vehicle/radar_fwd_snr            — float32, signal-to-noise proxy [0, 1]

EMA cold-start rule: when detection resumes after any gap, filtered state is
seeded to the first raw reading rather than continuing from the stale (zero)
value. Without this, EMA ramps from 0 → actual gap over ~3 frames, causing
IDM to see a falsely small gap and command unnecessary hard braking.

See docs/plans/step5_acc_plan.md Phase A — Python Side.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RadarReading:
    """Processed radar measurement — EMA-filtered values exposed to controller."""
    detected: bool
    gap_m: float           # EMA-filtered gap (m); 0.0 when not detected
    range_rate_mps: float  # EMA-filtered range rate (+ = closing); 0.0 when not detected
    snr: float             # signal-to-noise ratio [0, 1]; 0.0 when not detected
    gap_raw: float         # pre-filter range (m); for HDF5 diagnostics
    range_rate_raw: float  # pre-filter range rate (m/s); for HDF5 diagnostics


class RadarSensor(ABC):
    """
    Base interface for all radar sensors. Swappable without touching ACCController.

    Step 5: ForwardRadarSensor (60m, ±5° FOV)
    Step 7: RearCornerRadarSensor (30m, ±60° FOV) — same interface
    """

    @abstractmethod
    def read_frame(self, raw: dict) -> RadarReading:
        """Process one bridge frame dict and return a RadarReading."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal filter state (called on scene reset)."""
        ...


class ForwardRadarSensor(RadarSensor):
    """
    Step 5 forward radar: 60m range, ±5° FOV, EMA filter.

    Reads radar_fwd_* keys from the Unity bridge JSON dict.
    EMA cold-start: re-initializes filtered state to raw on first detection
    after any detection gap to avoid falsely-small gap estimates.
    """

    def __init__(self, gap_alpha: float = 0.30, rate_alpha: float = 0.20) -> None:
        self._gap_alpha = gap_alpha
        self._rate_alpha = rate_alpha
        self._gap_filtered: float = 0.0
        self._rate_filtered: float = 0.0
        self._last_detected: bool = False

    def reset(self) -> None:
        self._gap_filtered = 0.0
        self._rate_filtered = 0.0
        self._last_detected = False

    def read_frame(self, raw: dict) -> RadarReading:
        detected = bool(raw.get("radar_fwd_detected", False))

        if not detected:
            self._last_detected = False
            return RadarReading(
                detected=False,
                gap_m=0.0,
                range_rate_mps=0.0,
                snr=0.0,
                gap_raw=0.0,
                range_rate_raw=0.0,
            )

        gap_raw = float(raw.get("radar_fwd_distance_m", 0.0))
        rate_raw = float(raw.get("radar_fwd_range_rate_mps", 0.0))
        snr_raw = float(raw.get("radar_fwd_snr", 0.0))

        # EMA cold-start: seed filter state on first detection after any gap
        if not self._last_detected:
            self._gap_filtered = gap_raw
            self._rate_filtered = rate_raw
        else:
            self._gap_filtered = (
                self._gap_alpha * gap_raw
                + (1.0 - self._gap_alpha) * self._gap_filtered
            )
            self._rate_filtered = (
                self._rate_alpha * rate_raw
                + (1.0 - self._rate_alpha) * self._rate_filtered
            )

        self._last_detected = True

        return RadarReading(
            detected=True,
            gap_m=max(0.0, self._gap_filtered),
            range_rate_mps=self._rate_filtered,
            snr=max(0.0, min(1.0, snr_raw)),
            gap_raw=max(0.0, gap_raw),
            range_rate_raw=rate_raw,
        )
