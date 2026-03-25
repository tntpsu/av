"""
Adaptive Cruise Control (ACC) longitudinal controller.

Uses the Intelligent Driver Model (IDM) to maintain a safe following gap
at the ego vehicle's target speed.

State machine (6 states, evaluated in priority order each frame):
  1. TTC_ESTOP       — TTC < 1.5s → fires emergency_stop(), overrides everything
  2. EMERGENCY_BRAKE — gap < 1.5×v AND closing → hard -4.0 m/s² override
  3. CUTOUT          — ego_speed < cutout_speed_mps → disengage ramp to free-flow
  4. DETECTION_LOSS  — ≥ fallback_frames with no detection → disengage ramp
  5. ACC_ACTIVE      — normal IDM following
  6. FREE_FLOW       — default, no lead detected

Bumpless transfer:
  Disengage: v_target ramps toward free_flow_target at disengage_ramp_mps2 (m/s²).
  Engage: v_target_prev seeded to ego_speed on first ACC_ACTIVE frame after
  a non-ACC state (soft-engage) — prevents an IDM step-command on detection.

IDM formula:
  a_IDM = a_max × [1 - (v/v0)^4 - (s*(v,Δv)/gap)^2]
  s*(v,Δv) = s0 + max(0, v×T + v×Δv / (2×√(a_max×b)))
  where Δv = range_rate_mps (+ = closing)

Output is a target_speed, not a direct acceleration. The existing longitudinal
PID + jerk limiter chain is the sole authority on actual commanded acceleration.

See docs/plans/step5_acc_plan.md ## ACC Authority, State Machine & Control Integration
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from control.radar_sensor import RadarReading


class ACCState(Enum):
    FREE_FLOW       = "FREE_FLOW"
    ACC_ACTIVE      = "ACC_ACTIVE"
    DETECTION_LOSS  = "DETECTION_LOSS"
    CUTOUT          = "CUTOUT"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    TTC_ESTOP       = "TTC_ESTOP"


@dataclass
class ACCParams:
    """ACC controller parameters. All fields map to acc: block in config YAML."""
    enabled: bool = False
    min_gap_m: float = 2.0               # s0 — standstill gap (m)
    time_headway_s: float = 1.5          # T  — time headway (s)
    max_accel_mps2: float = 2.0          # a_max — IDM max acceleration (m/s²)
    comfortable_decel_mps2: float = 2.5  # b  — IDM comfortable deceleration (m/s²)
    detection_range_m: float = 60.0      # forward radar max range (m)
    cutout_speed_mps: float = 5.0        # ACC disengages below this ego speed (m/s)
    fallback_frames: int = 5             # consecutive no-detect frames before DETECTION_LOSS
    reengage_frames: int = 3             # consecutive detected frames to re-arm from DETECTION_LOSS
    disengage_ramp_mps2: float = 2.0    # ramp rate on disengage, m/s² (matches max_accel_mps2)

    @classmethod
    def from_config(cls, cfg: dict) -> "ACCParams":
        """Construct from the `acc:` YAML block. Keys follow YAML naming conventions."""
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            min_gap_m=float(cfg.get("min_gap_s0_m", 2.0)),
            time_headway_s=float(cfg.get("target_gap_time_headway_s", 1.5)),
            max_accel_mps2=float(cfg.get("idm_max_accel_mps2", 2.0)),
            comfortable_decel_mps2=float(cfg.get("idm_comfortable_decel_mps2", 2.5)),
            detection_range_m=float(cfg.get("detection_range_m", 60.0)),
            cutout_speed_mps=float(cfg.get("cutout_speed_mps", 5.0)),
            fallback_frames=int(cfg.get("fallback_frames", 5)),
            reengage_frames=int(cfg.get("reengage_frames", 3)),
            disengage_ramp_mps2=float(cfg.get("disengage_ramp_mps2", 2.0)),
        )


@dataclass
class ACCOutput:
    """
    All outputs produced by one call to ACCController.compute_target_speed().

    target_speed   — longitudinal speed setpoint to hand to the PID chain (m/s)
    acc_active     — 1.0 when IDM is controlling; 0.0 when free-flow/ramp/cutout
    target_gap_m   — desired gap = s0 + v×T (m); useful for diagnostics
    gap_error_m    — filtered_gap − target_gap (m); + = too close, − = too far
    ttc_s          — time-to-contact (s); capped at 999.0 when not applicable
    request_estop  — True if TTC guard fired; orchestrator MUST call emergency_stop()
    state          — current ACCState enum value for logging / HDF5
    """
    target_speed: float
    acc_active: float
    target_gap_m: float
    gap_error_m: float
    ttc_s: float
    request_estop: bool
    state: ACCState


# ── Module-level constants (used before scoring_registry is available) ────────
_TTC_ESTOP_THRESHOLD_S: float = 1.5      # Phase D wires to ACC_TTC_CRITICAL_S in registry
_EMERGENCY_BRAKE_ACCEL_MPS2: float = -4.0
_EMERGENCY_BRAKE_GAP_FACTOR: float = 1.5  # gap < factor × ego_speed → emergency brake
_TTC_CAP: float = 999.0                   # sentinel: TTC not meaningful (no lead / pulling away)


class ACCController:
    """
    ACC longitudinal controller implementing the 6-state priority state machine.

    Usage:
        params = ACCParams(enabled=True, ...)
        ctrl = ACCController(params)
        output = ctrl.compute_target_speed(ego_speed, free_flow_target, reading, dt)
        if output.request_estop:
            orchestrator.emergency_stop("acc_ttc_violation")
        target_speed = output.target_speed
    """

    def __init__(self, params: ACCParams) -> None:
        self._p = params
        # Detection-loss state
        self._detection_loss_count: int = 0
        self._rearm_count: int = 0
        self._in_loss: bool = False
        # Bumpless transfer state
        self._was_acc_active: bool = False
        self._v_target_prev: float = 0.0

    def reset(self) -> None:
        """Reset all state (call on Unity scene reset between runs)."""
        self._detection_loss_count = 0
        self._rearm_count = 0
        self._in_loss = False
        self._was_acc_active = False
        self._v_target_prev = 0.0

    def compute_target_speed(
        self,
        ego_speed: float,
        free_flow_target: float,
        reading: RadarReading,
        dt: float,
    ) -> ACCOutput:
        """
        Evaluate state machine and return ACCOutput for this frame.

        Parameters
        ----------
        ego_speed        Current vehicle speed (m/s)
        free_flow_target Speed target when ACC is inactive (m/s)
        reading          Processed RadarReading from ForwardRadarSensor
        dt               Frame time step (s); typically 1/30 s
        """
        p = self._p

        # ── TTC — always computed (guard evaluates in ALL states) ─────────────
        ttc_s = _TTC_CAP
        if reading.detected and reading.range_rate_mps > 0.0:
            ttc_s = reading.gap_m / max(reading.range_rate_mps, 1e-6)

        # ── Detection counters ────────────────────────────────────────────────
        if reading.detected:
            self._detection_loss_count = 0
            if self._in_loss:
                self._rearm_count += 1
                # Re-arm when enough consecutive detected frames AND above cutout speed
                if (self._rearm_count >= p.reengage_frames
                        and ego_speed >= p.cutout_speed_mps):
                    self._in_loss = False
                    self._rearm_count = 0
        else:
            self._rearm_count = 0
            self._detection_loss_count += 1
            if self._detection_loss_count >= p.fallback_frames:
                self._in_loss = True

        # ── Desired gap and gap error ─────────────────────────────────────────
        target_gap_m = p.min_gap_m + ego_speed * p.time_headway_s
        gap_error_m = (reading.gap_m - target_gap_m) if reading.detected else 0.0

        # ── State machine — highest priority first ────────────────────────────

        # 1. TTC_ESTOP: fires emergency_stop — overrides everything
        if reading.detected and ttc_s < _TTC_ESTOP_THRESHOLD_S:
            self._was_acc_active = False
            return ACCOutput(
                target_speed=0.0,
                acc_active=0.0,
                target_gap_m=target_gap_m,
                gap_error_m=gap_error_m,
                ttc_s=ttc_s,
                request_estop=True,
                state=ACCState.TTC_ESTOP,
            )

        # 2. EMERGENCY_BRAKE: hard deceleration, no e-stop
        if (reading.detected
                and reading.gap_m < _EMERGENCY_BRAKE_GAP_FACTOR * ego_speed
                and reading.range_rate_mps > 0.0):
            v_emergency = max(0.0, ego_speed + _EMERGENCY_BRAKE_ACCEL_MPS2 * dt)
            self._v_target_prev = v_emergency
            self._was_acc_active = False
            return ACCOutput(
                target_speed=v_emergency,
                acc_active=1.0,
                target_gap_m=target_gap_m,
                gap_error_m=gap_error_m,
                ttc_s=ttc_s,
                request_estop=False,
                state=ACCState.EMERGENCY_BRAKE,
            )

        # 3. CUTOUT: ego too slow — disengage ramp
        # Note: detection_loss_count is NOT reset; hysteresis state preserved.
        if ego_speed < p.cutout_speed_mps:
            v_ramped = self._ramp(self._v_target_prev, free_flow_target, dt)
            self._v_target_prev = v_ramped
            self._was_acc_active = False
            return ACCOutput(
                target_speed=v_ramped,
                acc_active=0.0,
                target_gap_m=target_gap_m,
                gap_error_m=gap_error_m,
                ttc_s=ttc_s,
                request_estop=False,
                state=ACCState.CUTOUT,
            )

        # 4. DETECTION_LOSS: ramp back to free-flow
        if self._in_loss:
            v_ramped = self._ramp(self._v_target_prev, free_flow_target, dt)
            self._v_target_prev = v_ramped
            self._was_acc_active = False
            return ACCOutput(
                target_speed=v_ramped,
                acc_active=0.0,
                target_gap_m=target_gap_m,
                gap_error_m=gap_error_m,
                ttc_s=ttc_s,
                request_estop=False,
                state=ACCState.DETECTION_LOSS,
            )

        # 5. ACC_ACTIVE: lead detected, all guards clear
        if reading.detected:
            # Soft-engage: seed v_target_prev at ego_speed on first active frame
            if not self._was_acc_active:
                self._v_target_prev = ego_speed

            a_idm = self._idm(
                ego_speed=ego_speed,
                free_flow_speed=free_flow_target,
                gap=reading.gap_m,
                delta_v=reading.range_rate_mps,
            )
            v_target = float(
                max(0.0, min(free_flow_target, self._v_target_prev + a_idm * dt))
            )
            self._v_target_prev = v_target
            self._was_acc_active = True
            return ACCOutput(
                target_speed=v_target,
                acc_active=1.0,
                target_gap_m=target_gap_m,
                gap_error_m=gap_error_m,
                ttc_s=ttc_s,
                request_estop=False,
                state=ACCState.ACC_ACTIVE,
            )

        # 6. FREE_FLOW: default
        # If we just left ACC_ACTIVE (e.g. first no-detect frame before fallback_frames),
        # ramp v_target toward free_flow_target rather than stepping immediately.
        # This covers the "lead temporarily occluded" window before DETECTION_LOSS.
        if self._was_acc_active and self._v_target_prev != free_flow_target:
            v_ramped = self._ramp(self._v_target_prev, free_flow_target, dt)
            self._v_target_prev = v_ramped
            # Keep _was_acc_active=True while still ramping (ramp must reach free_flow_target
            # before we snap to True free-flow — avoids re-triggering ramp next call)
            self._was_acc_active = v_ramped < free_flow_target - 1e-4
            return ACCOutput(
                target_speed=v_ramped,
                acc_active=0.0,
                target_gap_m=target_gap_m,
                gap_error_m=0.0,
                ttc_s=_TTC_CAP,
                request_estop=False,
                state=ACCState.FREE_FLOW,
            )

        self._was_acc_active = False
        self._v_target_prev = free_flow_target
        return ACCOutput(
            target_speed=free_flow_target,
            acc_active=0.0,
            target_gap_m=target_gap_m,
            gap_error_m=0.0,
            ttc_s=_TTC_CAP,
            request_estop=False,
            state=ACCState.FREE_FLOW,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ramp(self, v_current: float, v_target: float, dt: float) -> float:
        """Ramp v_current toward v_target at disengage_ramp_mps2 per second."""
        step = self._p.disengage_ramp_mps2 * dt
        if v_target > v_current:
            return min(v_current + step, v_target)
        if v_target < v_current:
            return max(v_current - step, v_target)
        return v_target

    def _idm(
        self,
        ego_speed: float,
        free_flow_speed: float,
        gap: float,
        delta_v: float,
    ) -> float:
        """
        IDM acceleration (m/s²).

        delta_v: + = closing (ego faster than lead), − = pulling away.
        Returns a_IDM clamped to [−4b, a_max].
        """
        p = self._p
        v = ego_speed
        v0 = max(free_flow_speed, 1e-6)

        # Desired gap  s*(v, Δv)
        sqrt_ab = math.sqrt(p.max_accel_mps2 * p.comfortable_decel_mps2)
        s_star = p.min_gap_m + max(
            0.0,
            v * p.time_headway_s + v * delta_v / (2.0 * sqrt_ab),
        )

        gap_safe = max(gap, 1e-3)
        ratio_v = (v / v0) ** 4
        ratio_s = (s_star / gap_safe) ** 2

        a_idm = p.max_accel_mps2 * (1.0 - ratio_v - ratio_s)

        # Clamp: no harder than 4b deceleration, no more than a_max acceleration
        return float(
            max(-4.0 * p.comfortable_decel_mps2, min(p.max_accel_mps2, a_idm))
        )
