"""
Hierarchical control regime selector.

Routes lateral control to the appropriate algorithm based on speed, curvature,
and error magnitude. Provides hysteresis and smooth blending during transitions.

Regime thresholds (defaults, all configurable via YAML control.regime.*):
  v < stanley_max_speed_mps (5.0 m/s) − hysteresis → STANLEY (low-speed precise)
  v < pp_max_speed_mps (10 m/s)                     → PURE_PURSUIT
  v > pp_max_speed_mps + hysteresis                 → LINEAR_MPC
  v > lmpc_max_speed_mps (20 m/s)                   → NONLINEAR_MPC (future — Phase 2.7)

Hysteresis prevents chatter near threshold.
min_hold_frames prevents transient speed spikes from triggering switches.
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
    pp_max_speed_mps: float = 10.0                 # Below this → PP; above → LMPC
    lmpc_max_speed_mps: float = 20.0               # Above this → NMPC (future)
    lmpc_max_heading_error_rad: float = 0.25       # Above this → NMPC (future)
    upshift_hysteresis_mps: float = 1.0            # PP→LMPC: must exceed threshold + this
    downshift_hysteresis_mps: float = 1.0          # LMPC→PP: must drop below threshold − this
    blend_frames: int = 15                         # Frames to linearly blend PP↔LMPC transition
    min_hold_frames: int = 30                      # Frames desired must be stable before switch
    # Stanley low-speed regime (Phase 2.9)
    stanley_enabled: bool = True                   # Independent switch; requires enabled=True
    stanley_max_speed_mps: float = 5.0             # PP→Stanley: downshift below this − hysteresis
    stanley_upshift_hysteresis_mps: float = 0.5    # Stanley→PP: must exceed max + this
    stanley_downshift_hysteresis_mps: float = 0.5  # PP→Stanley: must drop below max − this
    stanley_blend_frames: int = 10                 # Shorter blend (low speed = less risk)
    stanley_min_hold_frames: int = 15              # Shorter hold (low speed = faster transitions)

    @classmethod
    def from_config(cls, cfg: dict) -> "RegimeConfig":
        """Load from nested config dict: cfg['control']['regime']['<field>']."""
        rc = cfg.get("control", {}).get("regime", {})
        kwargs = {}
        for key in [
            "enabled", "pp_max_speed_mps", "lmpc_max_speed_mps",
            "lmpc_max_heading_error_rad", "upshift_hysteresis_mps",
            "downshift_hysteresis_mps", "blend_frames", "min_hold_frames",
            "stanley_enabled", "stanley_max_speed_mps",
            "stanley_upshift_hysteresis_mps", "stanley_downshift_hysteresis_mps",
            "stanley_blend_frames", "stanley_min_hold_frames",
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
        self._active_regime = ControlRegime.PURE_PURSUIT
        self._target_regime = ControlRegime.PURE_PURSUIT
        self._blend_progress = 1.0    # 1.0 = fully settled in active regime
        self._hold_counter = 0

    @property
    def active_regime(self) -> ControlRegime:
        return self._active_regime

    @property
    def blend_progress(self) -> float:
        return self._blend_progress

    def update(
        self,
        speed: float,
        curvature_abs: float = 0.0,
        lat_error: float = 0.0,
        heading_error: float = 0.0,
        mpc_fallback_active: bool = False,
    ) -> Tuple[ControlRegime, float]:
        """
        Update regime selection for this frame.

        Args:
            speed: Current vehicle speed (m/s)
            curvature_abs: Absolute path curvature (1/m, unused for now)
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
            desired = self._compute_desired(speed, heading_error)

        # Hysteresis hold: desired must be stable for min_hold_frames before switching
        if desired != self._target_regime:
            self._target_regime = desired
            self._hold_counter = 0

        self._hold_counter += 1

        # Determine hold and blend frame counts (Stanley transitions are faster)
        stanley_involved = (
            self._active_regime == ControlRegime.STANLEY
            or self._target_regime == ControlRegime.STANLEY
        )
        effective_hold_frames = (
            self.config.stanley_min_hold_frames if stanley_involved
            else self.config.min_hold_frames
        )
        effective_blend_frames = (
            self.config.stanley_blend_frames if stanley_involved
            else self.config.blend_frames
        )

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
        """Reset to PP, no blend in progress. Call after e-stop or teleport."""
        self._active_regime = ControlRegime.PURE_PURSUIT
        self._target_regime = ControlRegime.PURE_PURSUIT
        self._blend_progress = 1.0
        self._hold_counter = 0

    def _compute_desired(self, speed: float, heading_error: float) -> ControlRegime:
        """Compute desired regime based on current speed and heading error."""
        pp_thresh = self.config.pp_max_speed_mps
        up_hyst = self.config.upshift_hysteresis_mps
        down_hyst = self.config.downshift_hysteresis_mps

        # --- Stanley low-speed regime (Phase 2.9) ---
        if self.config.stanley_enabled:
            stanley_up = (
                self.config.stanley_max_speed_mps + self.config.stanley_upshift_hysteresis_mps
            )
            stanley_down = (
                self.config.stanley_max_speed_mps - self.config.stanley_downshift_hysteresis_mps
            )
            if self._active_regime == ControlRegime.STANLEY:
                # Stay in Stanley unless speed rises above upper threshold
                if speed <= stanley_up:
                    return ControlRegime.STANLEY
                # Fall through to PP / higher regime logic below
            else:
                # Downshift to Stanley if speed drops below lower threshold
                if speed < stanley_down:
                    return ControlRegime.STANLEY

        # --- PP ↔ LMPC ---
        if self._active_regime == ControlRegime.PURE_PURSUIT:
            # Upshift to LMPC: must exceed pp_max + hysteresis
            if speed > pp_thresh + up_hyst:
                return ControlRegime.LINEAR_MPC
            return ControlRegime.PURE_PURSUIT
        else:
            # Downshift to PP: must drop below pp_max - hysteresis
            if speed < pp_thresh - down_hyst:
                return ControlRegime.PURE_PURSUIT
            # Upshift to NMPC (future): above lmpc_max or large heading error
            if (
                speed > self.config.lmpc_max_speed_mps
                or abs(heading_error) > self.config.lmpc_max_heading_error_rad
            ):
                return ControlRegime.NONLINEAR_MPC
            return ControlRegime.LINEAR_MPC
