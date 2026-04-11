"""
Velocity profiler: pre-computed speed profile over a reference path.

Replaces the reactive curve preview / curve cap / anticipation score chain
with a physics-based forward+backward kinematic pass over track geometry.

Algorithm (3-pass):
  1. Curvature-limited: v(s) = sqrt(a_lat_max / κ(s)) at each station
  2. Backward pass: enforce braking feasibility from upcoming constraints
  3. Forward pass: enforce acceleration feasibility from current speed

The result is a dense (station, speed) array that the controller indexes
by odometer each frame. ACC / lead-vehicle constraints are injected as
additional v_max(s) bounds and the profile is rebuilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


G = 9.81  # m/s²


@dataclass
class VelocityProfilerConfig:
    """Physics parameters for profile construction."""

    a_lat_max_mps2: float      # lateral accel budget (m/s²), e.g. 0.20g * 9.81
    a_lon_accel_mps2: float    # max longitudinal acceleration (forward pass)
    a_lon_brake_mps2: float    # max comfortable braking (backward pass)
    v_max_mps: float           # global speed limit (from target_speed)
    min_speed_mps: float       # floor speed (comfort_governor_min_speed)
    sample_spacing_m: float    # profile resolution (from track YAML)


@dataclass
class VelocityProfile:
    """Pre-computed speed profile over a reference path."""

    stations_m: np.ndarray     # (N,) monotonic station values
    speeds_mps: np.ndarray     # (N,) target speed at each station
    total_length_m: float
    is_loop: bool

    # Diagnostics (same shape as stations_m)
    kappa_abs: np.ndarray      # (N,) absolute curvature at each station
    v_curvature: np.ndarray    # (N,) curvature-only speed (before kinematic passes)


class VelocityProfiler:
    """Builds and queries a velocity profile from track geometry."""

    def __init__(self, config: VelocityProfilerConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------

    def build_profile(
        self,
        curves: list[dict],
        total_length_m: float,
        is_loop: bool,
    ) -> VelocityProfile:
        """Build a velocity profile from track curve segments.

        Args:
            curves: list of dicts with keys {start_m, end_m, curvature_abs, direction_sign}.
                    Only arc segments; straights are implicit (κ=0).
            total_length_m: total track length in metres.
            is_loop: whether the track loops back to the start.

        Returns:
            VelocityProfile with dense (station, speed) arrays.
        """
        cfg = self.config
        ds = cfg.sample_spacing_m

        # --- Pass 0: build station + curvature arrays ---
        n = max(2, int(math.ceil(total_length_m / ds)) + 1)
        stations = np.linspace(0.0, total_length_m, n)
        kappa = self._sample_curvature(stations, curves)

        # --- Pass 1: curvature-limited speed ---
        # v = sqrt(a_lat / κ), clamped to [min_speed, v_max]
        safe_kappa = np.maximum(kappa, 1e-6)
        v_curv = np.sqrt(cfg.a_lat_max_mps2 / safe_kappa)
        v_curv = np.clip(v_curv, cfg.min_speed_mps, cfg.v_max_mps)
        # On straights (κ ≈ 0) the sqrt blows up → already clamped to v_max
        v = v_curv.copy()

        # --- Pass 2: backward pass (braking feasibility) ---
        v = self._backward_pass(v, stations, cfg.a_lon_brake_mps2, is_loop)

        # --- Pass 3: forward pass (acceleration feasibility) ---
        v = self._forward_pass(v, stations, cfg.a_lon_accel_mps2, is_loop)

        return VelocityProfile(
            stations_m=stations,
            speeds_mps=v,
            total_length_m=total_length_m,
            is_loop=is_loop,
            kappa_abs=kappa,
            v_curvature=v_curv,
        )

    # ------------------------------------------------------------------
    # Runtime lookup
    # ------------------------------------------------------------------

    def lookup_speed(self, profile: VelocityProfile, odometer_m: float) -> float:
        """Interpolate v_target at a given station.

        Uses np.searchsorted + linear interpolation. O(log N).
        Loop tracks wrap via modulo.
        """
        s = odometer_m
        if profile.is_loop and profile.total_length_m > 0:
            s = s % profile.total_length_m

        stations = profile.stations_m
        speeds = profile.speeds_mps

        # Clamp to profile range
        if s <= stations[0]:
            return float(speeds[0])
        if s >= stations[-1]:
            return float(speeds[-1])

        # Binary search + linear interpolation
        idx = int(np.searchsorted(stations, s, side="right")) - 1
        s0, s1 = stations[idx], stations[idx + 1]
        v0, v1 = speeds[idx], speeds[idx + 1]
        t = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
        return float(v0 + t * (v1 - v0))

    # ------------------------------------------------------------------
    # Dynamic constraint (ACC / lead vehicle)
    # ------------------------------------------------------------------

    def rebuild_with_constraint(
        self,
        base_profile: VelocityProfile,
        constraint_station_m: float,
        constraint_speed_mps: float,
    ) -> VelocityProfile:
        """Rebuild profile with an additional speed constraint (e.g. lead vehicle).

        Inserts v_max = min(v_existing, constraint_speed) at the constraint
        station, then re-runs backward+forward passes.
        """
        cfg = self.config
        v = base_profile.speeds_mps.copy()
        stations = base_profile.stations_m

        # Apply point constraint (spread over a small window for smoothness)
        s = constraint_station_m
        if base_profile.is_loop and base_profile.total_length_m > 0:
            s = s % base_profile.total_length_m
        idx = int(np.searchsorted(stations, s, side="right")) - 1
        idx = max(0, min(idx, len(v) - 1))
        v[idx] = min(v[idx], max(0.0, constraint_speed_mps))

        # Re-run kinematic passes
        v = self._backward_pass(v, stations, cfg.a_lon_brake_mps2, base_profile.is_loop)
        v = self._forward_pass(v, stations, cfg.a_lon_accel_mps2, base_profile.is_loop)

        return VelocityProfile(
            stations_m=stations.copy(),
            speeds_mps=v,
            total_length_m=base_profile.total_length_m,
            is_loop=base_profile.is_loop,
            kappa_abs=base_profile.kappa_abs,
            v_curvature=base_profile.v_curvature,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_curvature(stations: np.ndarray, curves: list[dict]) -> np.ndarray:
        """Sample absolute curvature at each station from segment list.

        Segments are assumed non-overlapping and sorted by start_m.
        Stations not inside any arc segment get κ=0 (straight).
        """
        kappa = np.zeros(len(stations), dtype=np.float64)
        if not curves:
            return kappa

        # Build sorted arrays for vectorised lookup
        starts = np.array([c["start_m"] for c in curves], dtype=np.float64)
        ends = np.array([c["end_m"] for c in curves], dtype=np.float64)
        kappas = np.array([c["curvature_abs"] for c in curves], dtype=np.float64)

        # For each station, find which segment (if any) it falls in
        # searchsorted gives the index where station would be inserted into starts
        seg_idx = np.searchsorted(starts, stations, side="right") - 1

        for i in range(len(stations)):
            si = seg_idx[i]
            if 0 <= si < len(curves):
                if stations[i] < ends[si]:
                    kappa[i] = kappas[si]

        return kappa

    @staticmethod
    def _backward_pass(
        v: np.ndarray,
        stations: np.ndarray,
        a_brake: float,
        is_loop: bool,
    ) -> np.ndarray:
        """Backward kinematic pass: enforce braking feasibility.

        v(i) = min(v(i), sqrt(v(i+1)² + 2·a_brake·Δs))
        """
        n = len(v)
        # Single backward sweep
        for i in range(n - 2, -1, -1):
            ds = stations[i + 1] - stations[i]
            v_brake_sq = v[i + 1] ** 2 + 2.0 * a_brake * ds
            if v_brake_sq > 0:
                v[i] = min(v[i], math.sqrt(v_brake_sq))

        # For loop tracks: one extra wrap (last → first boundary)
        if is_loop and n >= 2:
            ds_wrap = stations[-1] - stations[-2]  # approximate wrap spacing
            v_brake_sq = v[0] ** 2 + 2.0 * a_brake * ds_wrap
            if v_brake_sq > 0:
                v_from_wrap = math.sqrt(v_brake_sq)
                if v_from_wrap < v[-1]:
                    v[-1] = v_from_wrap
                    # Propagate backward one more full sweep
                    for i in range(n - 2, -1, -1):
                        ds = stations[i + 1] - stations[i]
                        v_brake_sq = v[i + 1] ** 2 + 2.0 * a_brake * ds
                        if v_brake_sq > 0:
                            v[i] = min(v[i], math.sqrt(v_brake_sq))

        return v

    @staticmethod
    def _forward_pass(
        v: np.ndarray,
        stations: np.ndarray,
        a_accel: float,
        is_loop: bool,
    ) -> np.ndarray:
        """Forward kinematic pass: enforce acceleration feasibility.

        v(i) = min(v(i), sqrt(v(i-1)² + 2·a_accel·Δs))
        """
        n = len(v)
        # Single forward sweep
        for i in range(1, n):
            ds = stations[i] - stations[i - 1]
            v_accel_sq = v[i - 1] ** 2 + 2.0 * a_accel * ds
            if v_accel_sq > 0:
                v[i] = min(v[i], math.sqrt(v_accel_sq))

        # For loop tracks: wrap from last → first
        if is_loop and n >= 2:
            ds_wrap = stations[1] - stations[0]  # approximate wrap spacing
            v_accel_sq = v[-1] ** 2 + 2.0 * a_accel * ds_wrap
            if v_accel_sq > 0:
                v_from_wrap = math.sqrt(v_accel_sq)
                if v_from_wrap < v[0]:
                    v[0] = v_from_wrap
                    # Propagate forward one more full sweep
                    for i in range(1, n):
                        ds = stations[i] - stations[i - 1]
                        v_accel_sq = v[i - 1] ** 2 + 2.0 * a_accel * ds
                        if v_accel_sq > 0:
                            v[i] = min(v[i], math.sqrt(v_accel_sq))

        return v
