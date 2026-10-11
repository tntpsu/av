"""Deterministic steering excitation for closed-loop plant identification (T-LATERAL-LMPC-SPEED-MARGIN).

r(t) is a Schroeder-phased multisine on log-spaced tones, added to the steering command AFTER the
controller (orchestrator `_pf_send_control`), so the controller keeps the car in lane while the plant
sees a known external disturbance. Because r is a function of time and the config only, the analysis
regenerates it exactly from the recording's `vehicle/unity_time`; `vehicle/steering_input` (what Unity
received) minus `control/steering` (what the controller commanded) is the recorded cross-check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class InjectionConfig:
    enabled: bool = False
    amplitude: float = 0.10          # normalised steering, peak of the summed multisine (crest-factor limited)
    f_min_hz: float = 0.15
    f_max_hz: float = 1.50
    n_tones: int = 12
    start_s: float = 12.0            # seconds after the first control frame
    duration_s: float = 50.0
    min_speed_mps: float = 3.0       # no excitation while launching

    @classmethod
    def from_config(cls, cfg: dict) -> "InjectionConfig":
        sec = (cfg or {}).get("experiment", {}).get("steering_injection", {}) if isinstance(cfg, dict) else {}
        kw = {f: sec[f] for f in cls.__dataclass_fields__ if f in sec}
        return cls(**kw)

    def tones_hz(self) -> np.ndarray:
        return np.geomspace(self.f_min_hz, self.f_max_hz, self.n_tones)

    def phases(self) -> np.ndarray:
        k = np.arange(self.n_tones)
        return math.pi * k * (k - 1) / self.n_tones          # Schroeder phases → low crest factor

    def value(self, tau_s: float, speed_mps: Optional[float] = None) -> float:
        """Excitation at `tau_s` seconds after the first control frame (0 outside the window)."""
        if not self.enabled or tau_s < self.start_s or tau_s > self.start_s + self.duration_s:
            return 0.0
        if speed_mps is not None and speed_mps < self.min_speed_mps:
            return 0.0
        return float(self.amplitude * self._unit(np.array([tau_s - self.start_s]))[0])

    def series(self, tau_s: np.ndarray, speed_mps: Optional[np.ndarray] = None) -> np.ndarray:
        tau_s = np.asarray(tau_s, dtype=float)
        r = self.amplitude * self._unit(tau_s - self.start_s)
        r[(tau_s < self.start_s) | (tau_s > self.start_s + self.duration_s)] = 0.0
        if speed_mps is not None:
            r[np.asarray(speed_mps, dtype=float) < self.min_speed_mps] = 0.0
        return r if self.enabled else np.zeros_like(r)

    def _unit(self, t: np.ndarray) -> np.ndarray:
        f = self.tones_hz(); ph = self.phases()
        x = np.sum([np.sin(2 * math.pi * fk * t + pk) for fk, pk in zip(f, ph)], axis=0)
        return x / max(self._peak(), 1e-9)

    def _peak(self) -> float:
        # Normalise so the summed waveform peaks at 1.0 over the whole excitation window (the tones
        # are log-spaced, hence not periodic over any single tone's period).
        t = np.linspace(0.0, self.duration_s, int(self.duration_s * 200) + 1)
        f = self.tones_hz(); ph = self.phases()
        return float(np.max(np.abs(np.sum([np.sin(2 * math.pi * fk * t + pk) for fk, pk in zip(f, ph)], axis=0))))
