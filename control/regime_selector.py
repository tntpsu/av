"""
MPC-primary hierarchical control regime selector.

Routes lateral control to MPC by default, with PP as fallback only (solver failure
or sub-minimum-speed). LMPC handles moderate curvature; NMPC handles tight curves
where LMPC's linearization degrades.

Regime hierarchy (MPC-primary):
  v < stanley_max_speed_mps              → STANLEY (very low speed / parking)
  v < mpc_min_speed_absolute_mps (2.0)   → PURE_PURSUIT (MPC needs nonzero speed)
  κ_map ≥ nmpc_curvature_threshold       → NONLINEAR_MPC (tight curves, R ≤ 67m)
  v > lmpc_max_speed_mps (20 m/s)        → NONLINEAR_MPC (high speed)
  |heading_error| > threshold            → NONLINEAR_MPC (large heading error)
  otherwise                              → LINEAR_MPC (default)

PP is NOT a desired regime above min_speed — it only activates via mpc_fallback_active
(when the MPC solver fails consecutively).

Hysteresis prevents chatter near threshold.
min_hold_frames prevents transient spikes from triggering switches.
blend_frames gives a smooth linear blend at transition.

HDF5 encoding (float): STANLEY=-1.0, PURE_PURSUIT=0.0, LINEAR_MPC=1.0, NONLINEAR_MPC=2.0
  Backward-compatible: old mask (regime >= 0.5) still correctly identifies MPC frames.
  New Stanley mask: regime < -0.5
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple


class ControlRegime(IntEnum):
    STANLEY = -1             # Low-speed precise tracking (heading + crosstrack) — Phase 2.9
    PURE_PURSUIT = 0
    LINEAR_MPC = 1
    NONLINEAR_MPC = 2        # Future — Phase 2.7


@dataclass
class RegimeConfig:
    """Configuration for the regime selector. Loaded from YAML control.regime.*"""
    enabled: bool = False                          # Master switch — False = always PP
    # ── Lateral acceleration budget (primary PP↔MPC decision) ──────────────────
    mpc_max_lateral_accel_mps2: float = 1.5        # κ×v² budget — MPC→PP above this (m/s²)
    mpc_lateral_accel_hysteresis_mps2: float = 0.3 # Dead band ±0.3 around threshold (m/s²)
    mpc_min_speed_absolute_mps: float = 2.0        # Hard floor — MPC needs nonzero speed (m/s)
    # ── Map curvature guard (linearization quality) ────────────────────────────
    mpc_max_map_curvature: float = 0.020           # Map κ above this → PP (MPC linearization degrades)
    # ── Transition smoothing ───────────────────────────────────────────────────
    blend_frames: int = 15                         # Frames to linearly blend PP↔LMPC transition
    min_hold_frames: int = 30                      # Frames desired must be stable before switch
    # ── LMPC↔NMPC (independent of PP↔LMPC) ─────────────────────────────────────
    lmpc_max_speed_mps: float = 20.0               # Above this → NMPC
    lmpc_max_heading_error_rad: float = 0.25       # Above this → NMPC
    nmpc_curvature_threshold: float = 0.015        # Map κ above this → NMPC (tight curves need nonlinear model)
    lmpc_nmpc_blend_frames: int = 30               # Blend frames for LMPC↔NMPC (2× default)
    lmpc_nmpc_min_hold_frames: int = 40            # Hold frames for LMPC↔NMPC (longer = less chatter)
    # ── Deprecated (kept for backward compat; lateral accel budget replaces) ────
    pp_max_speed_mps: float = 10.0                 # DEPRECATED — unused
    upshift_hysteresis_mps: float = 1.0            # DEPRECATED — replaced by lateral accel hysteresis
    downshift_hysteresis_mps: float = 1.0          # DEPRECATED — replaced by lateral accel hysteresis
    mpc_min_speed_mps: float = 3.0                 # DEPRECATED — replaced by lateral accel budget
    mpc_max_curvature: float = 0.020               # DEPRECATED — replaced by lateral accel budget
    mpc_curvature_hysteresis: float = 0.003        # DEPRECATED — replaced by lateral accel budget
    # Stanley low-speed regime (Phase 2.9)
    stanley_enabled: bool = True                   # Independent switch; requires enabled=True
    stanley_max_speed_mps: float = 5.0             # PP→Stanley: downshift below this − hysteresis
    stanley_upshift_hysteresis_mps: float = 0.5    # Stanley→PP: must exceed max + this
    stanley_downshift_hysteresis_mps: float = 0.5  # PP→Stanley: must drop below max − this
    stanley_blend_frames: int = 10                 # Shorter blend (low speed = less risk)
    stanley_min_hold_frames: int = 15              # Shorter hold (low speed = faster transitions)
    # Curvature inhibit: do not use Stanley when |κ| exceeds this (1/m).
    # Prevents MPC→Stanley handoff during curve braking (e.g. v dips to ~4.5 m/s on
    # C2 while κ≈0.01), which combined with perception dropout caused large steering
    # jerk.  0.0 = disabled (legacy behaviour).  ~0.004–0.005 blocks R≤250 m curves.
    stanley_inhibit_curvature_abs: float = 0.004

    @classmethod
    def from_config(cls, cfg: dict) -> "RegimeConfig":
        """Load from nested config dict: cfg['control']['regime']['<field>']."""
        rc = cfg.get("control", {}).get("regime", {})
        kwargs = {}
        # Warn if deprecated params are set in config
        _deprecated = [
            "mpc_min_speed_mps", "mpc_max_curvature", "mpc_curvature_hysteresis",
            "upshift_hysteresis_mps", "downshift_hysteresis_mps", "pp_max_speed_mps",
        ]
        for dep_key in _deprecated:
            if dep_key in rc:
                import logging
                logging.getLogger(__name__).warning(
                    "regime config '%s' is deprecated — lateral accel budget "
                    "(mpc_max_lateral_accel_mps2) replaces speed/curvature proxies",
                    dep_key,
                )
        for key in [
            "enabled",
            "mpc_max_lateral_accel_mps2", "mpc_lateral_accel_hysteresis_mps2",
            "mpc_min_speed_absolute_mps", "mpc_max_map_curvature",
            "blend_frames", "min_hold_frames",
            "lmpc_max_speed_mps", "lmpc_max_heading_error_rad",
            "nmpc_curvature_threshold",
            "lmpc_nmpc_blend_frames", "lmpc_nmpc_min_hold_frames",
            "stanley_enabled", "stanley_max_speed_mps",
            "stanley_upshift_hysteresis_mps", "stanley_downshift_hysteresis_mps",
            "stanley_blend_frames", "stanley_min_hold_frames",
            "stanley_inhibit_curvature_abs",
            # Deprecated — still loaded if present for backward compat
            "pp_max_speed_mps", "upshift_hysteresis_mps", "downshift_hysteresis_mps",
            "mpc_min_speed_mps", "mpc_max_curvature", "mpc_curvature_hysteresis",
        ]:
            if key in rc:
                kwargs[key] = rc[key]
        return cls(**kwargs)


class RegimeSelector:
    """
    Selects lateral control algorithm based on speed regime with hysteresis.

    Usage:
        selector = RegimeSelector(RegimeConfig(enabled=True))
        regime, blend = selector.update(speed=12.0)
        # regime = ControlRegime.LINEAR_MPC, blend = 1.0 (fully settled in MPC)

    During blending (blend < 1.0):
        final_steer = (1 − blend) * previous_steer + blend * new_steer

    When enabled=False, always returns (PURE_PURSUIT, 1.0) — zero overhead.
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self._active_regime = ControlRegime.LINEAR_MPC
        self._target_regime = ControlRegime.LINEAR_MPC
        self._blend_progress = 1.0    # 1.0 = fully settled in active regime
        self._hold_counter = 0
        self._last_a_lat = 0.0        # κ×v² from last update (for HDF5 recording)

    @property
    def active_regime(self) -> ControlRegime:
        return self._active_regime

    @property
    def blend_progress(self) -> float:
        return self._blend_progress

    @property
    def last_lateral_accel(self) -> float:
        """κ×v² from the most recent update() call (m/s²)."""
        return self._last_a_lat

    def peek(self) -> Tuple[ControlRegime, float]:
        """Return current regime and blend weight WITHOUT advancing state.

        Used by inter-frame control extrapolation to check regime without
        triggering hysteresis transitions (those only happen on camera frames).
        """
        return self._active_regime, self._blend_progress

    def update(
        self,
        speed: float,
        curvature_abs: float = 0.0,
        map_curvature_abs: float = 0.0,
        lat_error: float = 0.0,
        heading_error: float = 0.0,
        mpc_fallback_active: bool = False,
    ) -> Tuple[ControlRegime, float]:
        """
        Update regime selection for this frame.

        Args:
            speed: Current vehicle speed (m/s)
            curvature_abs: Absolute path curvature at car (1/m) — used for a_lat budget
            map_curvature_abs: Map/track geometry curvature (1/m) — used for linearization guard
            lat_error: Lateral error magnitude (m, unused for now)
            heading_error: Heading error (rad) — used for NMPC upshift guard
            mpc_fallback_active: True if MPC solver has failed consecutively

        Returns:
            (active_regime, blend_weight)
            blend_weight is 0.0→1.0 during transition, 1.0 when fully settled.
        """
        if not self.config.enabled:
            return ControlRegime.PURE_PURSUIT, 1.0

        # Force PP when MPC solver is in fallback
        if mpc_fallback_active:
            desired = ControlRegime.PURE_PURSUIT
        else:
            desired = self._compute_desired(
                speed, heading_error,
                curvature_abs=curvature_abs,
                map_curvature_abs=map_curvature_abs,
            )

        # Hysteresis hold: desired must be stable for min_hold_frames before switching
        if desired != self._target_regime:
            self._target_regime = desired
            self._hold_counter = 0

        self._hold_counter += 1

        # Determine hold and blend frame counts.
        # Stanley: fastest transitions (low-speed, low-risk).
        # LMPC↔NMPC: slowest transitions (reduce chatter, smooth high-speed handoff).
        # PP↔LMPC: default.
        stanley_involved = (
            self._active_regime == ControlRegime.STANLEY
            or self._target_regime == ControlRegime.STANLEY
        )
        nmpc_involved = (
            self._active_regime == ControlRegime.NONLINEAR_MPC
            or self._target_regime == ControlRegime.NONLINEAR_MPC
        )
        if stanley_involved:
            effective_hold_frames = self.config.stanley_min_hold_frames
            effective_blend_frames = self.config.stanley_blend_frames
        elif nmpc_involved:
            effective_hold_frames = self.config.lmpc_nmpc_min_hold_frames
            effective_blend_frames = self.config.lmpc_nmpc_blend_frames
        else:
            effective_hold_frames = self.config.min_hold_frames
            effective_blend_frames = self.config.blend_frames

        # Trigger switch once hold counter reaches threshold
        if (
            self._target_regime != self._active_regime
            and self._hold_counter >= effective_hold_frames
        ):
            self._active_regime = self._target_regime
            self._blend_progress = 0.0

        # Advance blend toward 1.0
        if self._blend_progress < 1.0:
            self._blend_progress = min(
                1.0,
                self._blend_progress + 1.0 / max(1, effective_blend_frames),
            )

        return self._active_regime, self._blend_progress

    def reset(self) -> None:
        """Reset to LMPC (default), no blend in progress. Call after e-stop or teleport."""
        self._active_regime = ControlRegime.LINEAR_MPC
        self._target_regime = ControlRegime.LINEAR_MPC
        self._blend_progress = 1.0
        self._hold_counter = 0
        self._last_a_lat = 0.0

    def _compute_desired(self, speed: float, heading_error: float,
                         curvature_abs: float = 0.0,
                         map_curvature_abs: float = 0.0) -> ControlRegime:
        """Compute desired regime. MPC-primary architecture — PP is fallback only.

        Hierarchy (MPC-primary):
          Stanley: v < stanley_max_speed_mps (very low speed / parking)
          PP:      v < mpc_min_speed_absolute_mps (MPC needs nonzero speed)
          NMPC:    κ_map ≥ nmpc_curvature_threshold OR v > lmpc_max_speed OR
                   |heading_error| > lmpc_max_heading_error_rad
          LMPC:    default for everything else

        PP is never a "desired" regime above min_speed — it only activates via
        mpc_fallback_active (solver failure) handled in update().
        """
        # Record lateral accel for HDF5 telemetry (informational only)
        self._last_a_lat = curvature_abs * speed * speed

        min_speed = self.config.mpc_min_speed_absolute_mps

        # --- Stanley low-speed regime ---
        if self.config.stanley_enabled:
            thr_inhibit = float(self.config.stanley_inhibit_curvature_abs)
            inhibit_stanley_on_curve = thr_inhibit > 0.0 and curvature_abs > thr_inhibit

            stanley_up = (
                self.config.stanley_max_speed_mps + self.config.stanley_upshift_hysteresis_mps
            )
            stanley_down = (
                self.config.stanley_max_speed_mps - self.config.stanley_downshift_hysteresis_mps
            )
            if self._active_regime == ControlRegime.STANLEY:
                if not inhibit_stanley_on_curve and speed <= stanley_up:
                    return ControlRegime.STANLEY
            else:
                if not inhibit_stanley_on_curve and speed < stanley_down:
                    return ControlRegime.STANLEY

        # --- Speed floor: MPC needs nonzero speed ---
        if speed < min_speed:
            return ControlRegime.PURE_PURSUIT

        # --- NMPC for tight curves, high speed, or large heading error ---
        nmpc_curv_threshold = self.config.nmpc_curvature_threshold
        nmpc_curvature_triggered = nmpc_curv_threshold > 0.0 and map_curvature_abs >= nmpc_curv_threshold

        if (
            nmpc_curvature_triggered
            or speed > self.config.lmpc_max_speed_mps
            or abs(heading_error) > self.config.lmpc_max_heading_error_rad
        ):
            return ControlRegime.NONLINEAR_MPC

        # --- Default: LMPC ---
        return ControlRegime.LINEAR_MPC
