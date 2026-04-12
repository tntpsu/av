"""
Lookahead profiler: pre-computed PP lookahead distance profile over a reference path.

Replaces the reactive smoothstep lookahead chain with a physics-based
pre-computation from track geometry. Acts as a ceiling on the reactive Ld
so the controller starts contracting BEFORE curve entry.

Algorithm (3-pass, mirrors velocity_profiler):
  1. Curvature pass: for each station, scan κ ahead over a preview horizon
     and compute Ld = sqrt(8R × e_target) for the tightest upcoming curve.
  2. Contraction pass (backward): Ld can only drop at contraction_rate m per m.
  3. Extension pass (forward): Ld can only rise at extension_rate m per m.

The result is a dense (station, lookahead) array that the orchestrator
indexes by odometer each frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LookaheadProfilerConfig:
    """Physics parameters for lookahead profile construction."""

    preview_horizon_m: float      # how far ahead to scan curvature (m)
    e_target_m: float             # tracking error budget for sqrt(8R×e)
    k_speed_min: float            # Ld floor = k × v_profile(s)
    ld_max_m: float               # ceiling on straights
    ld_min_m: float               # absolute floor
    contraction_rate: float       # m/m of travel (Ld drops at most this rate)
    extension_rate: float         # m/m of travel (Ld rises at most this rate)
    sample_spacing_m: float       # profile resolution
    curvature_min: float          # ignore κ below this


@dataclass
class LookaheadProfile:
    """Pre-computed lookahead distance profile over a reference path."""

    stations_m: np.ndarray        # (N,) monotonic station values
    lookaheads_m: np.ndarray      # (N,) pre-computed Ld at each station
    total_length_m: float
    is_loop: bool

    # Diagnostics
    ld_curvature: np.ndarray      # (N,) curvature-only Ld (before smoothing passes)


class LookaheadProfiler:
    """Builds and queries a lookahead distance profile from track geometry."""

    def __init__(self, config: LookaheadProfilerConfig) -> None:
        self.config = config

    def build_profile(
        self,
        curves: list[dict],
        total_length_m: float,
        is_loop: bool,
        v_profile_speeds: Optional[np.ndarray] = None,
        v_profile_stations: Optional[np.ndarray] = None,
    ) -> LookaheadProfile:
        """Build a lookahead profile from track curve segments.

        Args:
            curves: list of dicts with keys {start_m, end_m, curvature_abs}.
            total_length_m: total track length in metres.
            is_loop: whether the track loops.
            v_profile_speeds: optional velocity profile for speed-floor integration.
            v_profile_stations: station array corresponding to v_profile_speeds.
        """
        cfg = self.config
        ds = cfg.sample_spacing_m

        # --- Station + curvature arrays ---
        n = max(2, int(math.ceil(total_length_m / ds)) + 1)
        stations = np.linspace(0.0, total_length_m, n)
        kappa = self._sample_curvature(stations, curves)

        # --- Pass 1: curvature-limited Ld with preview horizon ---
        ld = np.full(n, cfg.ld_max_m, dtype=np.float64)
        preview_samples = max(1, int(cfg.preview_horizon_m / ds))

        for i in range(n):
            # Scan ahead over preview horizon
            kappa_max = 0.0
            for j in range(1, preview_samples + 1):
                if is_loop:
                    idx = (i + j) % n
                else:
                    idx = min(i + j, n - 1)
                kappa_max = max(kappa_max, kappa[idx])
            # Also include curvature AT this station
            kappa_max = max(kappa_max, kappa[i])

            if kappa_max >= cfg.curvature_min:
                R = 1.0 / kappa_max
                ld_curve = math.sqrt(8.0 * R * cfg.e_target_m)
                ld[i] = min(cfg.ld_max_m, max(cfg.ld_min_m, ld_curve))

        # Apply speed floor if velocity profile available
        if v_profile_speeds is not None and v_profile_stations is not None:
            for i in range(n):
                v = float(np.interp(stations[i], v_profile_stations, v_profile_speeds))
                ld_speed_floor = cfg.k_speed_min * v
                ld[i] = max(ld[i], ld_speed_floor)

        ld_curvature = ld.copy()

        # --- Pass 2: contraction pass (backward) ---
        ld = self._contraction_pass(ld, stations, cfg.contraction_rate, is_loop)

        # --- Pass 3: extension pass (forward) ---
        ld = self._extension_pass(ld, stations, cfg.extension_rate, is_loop)

        # Floor enforcement
        ld = np.maximum(ld, cfg.ld_min_m)

        return LookaheadProfile(
            stations_m=stations,
            lookaheads_m=ld,
            total_length_m=total_length_m,
            is_loop=is_loop,
            ld_curvature=ld_curvature,
        )

    def lookup_lookahead(self, profile: LookaheadProfile, odometer_m: float) -> float:
        """Interpolate Ld at a given station. O(log N).

        Loop tracks wrap via modulo.
        """
        s = odometer_m
        if profile.is_loop and profile.total_length_m > 0:
            s = s % profile.total_length_m

        stations = profile.stations_m
        lds = profile.lookaheads_m

        if s <= stations[0]:
            return float(lds[0])
        if s >= stations[-1]:
            return float(lds[-1])

        idx = int(np.searchsorted(stations, s, side="right")) - 1
        s0, s1 = stations[idx], stations[idx + 1]
        ld0, ld1 = lds[idx], lds[idx + 1]
        t = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
        return float(ld0 + t * (ld1 - ld0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_curvature(stations: np.ndarray, curves: list[dict]) -> np.ndarray:
        """Sample absolute curvature at each station from segment list."""
        kappa = np.zeros(len(stations), dtype=np.float64)
        if not curves:
            return kappa

        starts = np.array([c["start_m"] for c in curves], dtype=np.float64)
        ends = np.array([c["end_m"] for c in curves], dtype=np.float64)
        kappas = np.array([c["curvature_abs"] for c in curves], dtype=np.float64)

        seg_idx = np.searchsorted(starts, stations, side="right") - 1
        for i in range(len(stations)):
            si = seg_idx[i]
            if 0 <= si < len(curves):
                if stations[i] < ends[si]:
                    kappa[i] = kappas[si]
        return kappa

    @staticmethod
    def _contraction_pass(
        ld: np.ndarray,
        stations: np.ndarray,
        rate: float,
        is_loop: bool,
    ) -> np.ndarray:
        """Backward pass: Ld can only drop at rate m per m of travel.

        ld[i] = min(ld[i], ld[i+1] + rate × Δs)
        Walking backward: "given the Ld needed at the curve, how far back
        do we need to start contracting?"
        """
        n = len(ld)
        for i in range(n - 2, -1, -1):
            ds = stations[i + 1] - stations[i]
            ld[i] = min(ld[i], ld[i + 1] + rate * ds)

        if is_loop and n >= 2:
            ds_wrap = stations[1] - stations[0]
            ld_from_wrap = ld[0] + rate * ds_wrap
            if ld_from_wrap < ld[-1]:
                ld[-1] = ld_from_wrap
                for i in range(n - 2, -1, -1):
                    ds = stations[i + 1] - stations[i]
                    ld[i] = min(ld[i], ld[i + 1] + rate * ds)
        return ld

    @staticmethod
    def _extension_pass(
        ld: np.ndarray,
        stations: np.ndarray,
        rate: float,
        is_loop: bool,
    ) -> np.ndarray:
        """Forward pass: Ld can only rise at rate m per m of travel.

        ld[i] = min(ld[i], ld[i-1] + rate × Δs)
        Walking forward: ensures Ld extends gradually after curve exit.
        """
        n = len(ld)
        for i in range(1, n):
            ds = stations[i] - stations[i - 1]
            ld[i] = min(ld[i], ld[i - 1] + rate * ds)

        if is_loop and n >= 2:
            ds_wrap = stations[1] - stations[0]
            ld_from_wrap = ld[-1] + rate * ds_wrap
            if ld_from_wrap < ld[0]:
                ld[0] = ld_from_wrap
                for i in range(1, n):
                    ds = stations[i] - stations[i - 1]
                    ld[i] = min(ld[i], ld[i - 1] + rate * ds)
        return ld
