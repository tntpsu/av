"""Unity-free closed-loop harness for the ACC longitudinal stack.

Closes the loop  radar → ACCController → owner-resolver → LongitudinalController
→ safety clip → point-mass plant  for N frames, with a scripted lead vehicle and
a configurable grade.  This is the longitudinal counterpart of
``test_closedloop_stability.py`` (which does the same for the MPC lateral loop
with a bicycle model).  It exists because every ACC failure in ``docs/agent/
tasks.md`` could only be reproduced by a 90 s Unity run producing a 600 MB HDF5.

Every controller in the loop is the PRODUCTION object, constructed from the real
merged YAML (``av_stack.config.load_config``), so the harness tracks config drift
for free.  Only two things are modelled:

  * the plant — a point mass on a grade with a 1-frame actuator lag
  * the radar — perfect detection inside ``acc.detection_range_m``, reporting the
    CENTRE-TO-CENTRE range like Unity does (``radar_range_offset_m``, default the
    measured 4.43 m) and the 0.1 m collision override on contact. The production
    ``ForwardRadarSensor`` subtracts ``acc.radar_range_offset_m`` from it.

Orchestrator glue that sits between the controllers is reproduced verbatim from
``av_stack/orchestrator.py`` and cited by line so it can be re-checked:

  * ACC is stepped with a HARDCODED dt of 1/30 s  (orchestrator.py:9864) while
    the plant and the longitudinal controller see the real frame period.  The
    measured frame period on the Mac mini is 76.9 ms (13 FPS), so the IDM
    target-speed integration runs at 43 % of its designed rate.  ``acc_dt`` is
    exposed so tests can show the effect.
  * ``_pf_resolve_longitudinal_target`` (orchestrator.py:9927) — ACC owns the
    target only in states outside {FREE_FLOW, DETECTION_LOSS, CUTOUT}.  In CUTOUT
    the governor's free-flow target is applied unmodified.
  * IDM→reference_accel routing (orchestrator.py:8560) — IDM accel reaches the
    controller as ``reference_accel`` only when acc_active AND idm < 0.
  * Post-controller safety clip (orchestrator.py:9489) — latched e-stop forces
    brake=1.0 until speed ≤ ``safety.emergency_stop_release_speed``; the B1
    EMERGENCY_BRAKE bypass forces brake=1.0 non-latched.

Plant calibration (recording_20260908_041941.h5 — hill_g2, 2026-09-08 04:19,
5 % uphill, frames 204–406):
  throttle 0.24 → +0.98 m/s² net uphill   ⇒  K_THROTTLE ≈ 7 m/s² per unit
  brake    1.00 → −5.0  m/s² net uphill   ⇒  K_BRAKE    ≈ 4.5 m/s² per unit
  cross-correlation of accel_cmd vs measured accel peaks at lag = 1 frame
Gravity is physical (g·sin θ).  Note the production ``grade_ff_gain`` is 1.8,
"calibrated for Unity WheelCollider physics", so on a 5 % grade the controller
adds +0.88 m/s² of feedforward against a −0.49 m/s² physical load — a standing
+0.39 m/s² propulsive bias.  That is preserved here deliberately; it is part of
the system under test.

The per-frame trace uses the HDF5 field names (``acc_state_code``,
``radar_fwd_distance_m``, ``target_speed_final`` …) so a harness run can be
compared column-for-column against a real recording.
"""
from __future__ import annotations

import collections
import inspect
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:      # pytest puts rootdir on the path; `python3 tests/…` does not
    sys.path.insert(0, str(REPO_ROOT))

from av_stack.config import load_config                        # noqa: E402
from control.acc_controller import ACCController, ACCParams    # noqa: E402
from control.pid_controller import LongitudinalController      # noqa: E402
from control.radar_sensor import ForwardRadarSensor            # noqa: E402

# Measured, not assumed — see feedback_recording_dependent_tests_cycle / cadence memos.
FRAME_DT_MEASURED_S = 0.0769          # 13 FPS Mac mini, median of 1785 frames
ACC_DT_PRODUCTION_S = 1.0 / 30.0      # orchestrator.py:9864 hardcode

G_MPS2 = 9.81

# Radar range FRAME. AVBridge.cs:3048-3051 measures |leadCenter - egoTransform| —
# centre-to-centre, not bumper-to-bumper. LeadVehicle.OnTriggerEnter (physical
# contact) fired at a reported 4.34-4.64 m in all 11 hill_g2 recordings examined
# on 2026-09-22 (median 4.43). Every gap threshold in ACCController and
# scoring_registry (s0 2.0, EB floor 3.0, collapsed 0.5, near-miss 2.0) is
# therefore INSIDE the lead vehicle's body. ``run_closedloop(radar_range_offset_m=
# RADAR_RANGE_OFFSET_MEASURED_M)`` models the real sensor; the default 0.0 keeps
# the idealised bumper-gap radar the earlier tests were written against.
RADAR_RANGE_OFFSET_MEASURED_M = 4.43
COLLISION_OVERRIDE_RANGE_M = 0.1      # AVBridge.cs:3070 — what the radar reports on contact

_ACC_OWNER_INACTIVE_STATES = {"", "FREE_FLOW", "DETECTION_LOSS", "CUTOUT"}
_ESTOP_STATES = {"TTC_ESTOP", "COLLAPSED_GAP_STOP"}


# ---------------------------------------------------------------------------
# Config → production controllers
# ---------------------------------------------------------------------------

def load_scenario_config(overlay: str = "config/acc_hill_highway.yaml") -> dict:
    """Merged config exactly as ``start_av_stack.sh --config <overlay>`` sees it."""
    return load_config(str(REPO_ROOT / overlay))


def build_acc_controller(cfg: dict, **param_overrides) -> ACCController:
    params = ACCParams.from_config(dict(cfg.get("acc", {})))
    for k, v in param_overrides.items():
        setattr(params, k, v)
    return ACCController(params)


def build_longitudinal_controller(cfg: dict, **overrides) -> LongitudinalController:
    """Construct LongitudinalController the way orchestrator.py:440-580 does.

    Ctor parameter names equal the YAML keys except three the orchestrator
    remaps: ``target_speed`` (longitudinal.target_speed), ``max_speed``
    (safety.max_speed) and ``speed_smoothing_alpha`` (longitudinal.speed_smoothing).
    ``max_accel/max_decel/max_jerk`` fall back to trajectory.speed_planner.
    """
    lon = dict(cfg.get("control", {}).get("longitudinal", {}))
    safety = cfg.get("safety", {})
    planner = cfg.get("trajectory", {}).get("speed_planner", {})

    sig = inspect.signature(LongitudinalController.__init__)
    kwargs: Dict[str, object] = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if name in lon:
            kwargs[name] = lon[name]
    kwargs["target_speed"] = lon.get("target_speed", 8.0)
    kwargs["max_speed"] = safety.get("max_speed", 10.0)
    kwargs["speed_smoothing_alpha"] = lon.get("speed_smoothing", 0.6)
    kwargs["max_accel"] = float(lon.get("max_accel", planner.get("max_accel", 2.5)))
    kwargs["max_decel"] = float(lon.get("max_decel", planner.get("max_decel", 3.0)))
    kwargs["max_jerk"] = float(lon.get("max_jerk", planner.get("max_jerk", 2.0)))
    kwargs["max_jerk_min"] = lon.get("max_jerk_min", kwargs["max_jerk"])
    kwargs["max_jerk_max"] = lon.get("max_jerk_max", max(kwargs["max_jerk"], 6.0))
    kwargs.update(overrides)
    return LongitudinalController(**kwargs)


# ---------------------------------------------------------------------------
# Scripted lead vehicle  (mirrors unity/.../SpeedProfiler.cs)
# ---------------------------------------------------------------------------

@dataclass
class LeadProfile:
    """Scripted lead speed v(t), mirroring SpeedProfiler.cs profiles.

    ``hard_brake`` (default): hold ``speed`` until ``brake_at_s``, then decelerate
    at ``decel`` toward ``brake_to``.  SpeedProfiler.cs declares 3.0 m/s²; the
    2026-09-08 G2 recording shows the lead closing at ~1.5 m/s² equivalent
    (range-rate ramp 0.6→9.4 m/s over 6.5 s).  Default to the measured value —
    the milder lead makes any reproduction here conservative.

    ``stop_go``: SpeedProfiler's sinusoid, speed = top·½·(1 − cos 2πt/T).
    """
    speed: float = 10.0
    brake_at_s: float = 0.0
    decel: float = 1.5
    brake_to: float = 0.0
    kind: str = "hard_brake"
    stop_go_top_speed: float = 8.0
    stop_go_period_s: float = 20.0

    def speed_at(self, t: float) -> float:
        if self.kind == "stop_go":
            phase = (t % self.stop_go_period_s) / self.stop_go_period_s
            return self.stop_go_top_speed * 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
        if self.kind == "constant":
            return self.speed
        if t < self.brake_at_s:
            return self.speed
        v = self.speed - self.decel * (t - self.brake_at_s)
        return max(self.brake_to, v)


# ---------------------------------------------------------------------------
# Plant
# ---------------------------------------------------------------------------

@dataclass
class PointMassPlant:
    k_throttle: float = 7.0        # m/s² per unit throttle (calibrated, see module doc)
    k_brake: float = 4.5           # m/s² per unit brake
    gravity_gain: float = 1.0      # 1.0 = physical g·sin(grade)
    drag_per_mps: float = 0.0
    actuator_lag_frames: int = 1

    def __post_init__(self) -> None:
        n = max(1, self.actuator_lag_frames + 1)
        self._buf: collections.deque = collections.deque([(0.0, 0.0)] * n, maxlen=n)

    def step(self, v: float, throttle: float, brake: float, grade_rad: float, dt: float) -> float:
        self._buf.append((float(throttle), float(brake)))
        thr, brk = self._buf[0]
        a = (self.k_throttle * thr
             - self.k_brake * brk
             - G_MPS2 * math.sin(grade_rad) * self.gravity_gain
             - self.drag_per_mps * v)
        v_new = v + a * dt
        # A stopped car on an incline does not roll backwards in Unity (brake hold).
        return max(0.0, v_new)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClosedLoopResult:
    trace: Dict[str, List] = field(default_factory=lambda: collections.defaultdict(list))
    collided: bool = False
    contact_frame: int = -1

    # -- derived metrics -------------------------------------------------------
    def col(self, name: str) -> List:
        return self.trace[name]

    def states(self) -> List[str]:
        return list(dict.fromkeys(self.col("acc_state_code")))

    def frames_in(self, *states: str) -> int:
        return sum(1 for s in self.col("acc_state_code") if s in states)

    def min_gap(self) -> float:
        """Minimum REPORTED gap (radar frame) — comparable to the sweep's numbers."""
        return min(self.col("radar_fwd_distance_m"))

    def min_true_gap(self) -> float:
        """Minimum TRUE bumper-to-bumper gap; <= 0 means physical contact."""
        return min(self.col("true_bumper_gap_m"))

    def ttc_min(self, min_ego_speed: float = 0.5, min_gap: float = 0.5) -> float:
        """TTC minimum over frames where the number is meaningful.

        Excludes near-standstill and collapsed-gap frames, where gap/range-rate
        degenerates (the raw recording hits 0.80 s at v=0.04 m/s, gap 0.36 m).
        """
        best = math.inf
        for ttc, v, gap, det in zip(self.col("acc_ttc_s"), self.col("speed"),
                                    self.col("radar_fwd_distance_m"), self.col("radar_fwd_detected")):
            if det and v > min_ego_speed and gap > min_gap:
                best = min(best, ttc)
        return best

    def ttc_min_acc_active(self) -> float:
        """TTC minimum the way acc_pipeline_analysis._card3 computes it:
        min over frames with acc_active > 0.5 (finite values)."""
        vals = [ttc for ttc, a in zip(self.col("acc_ttc_s"), self.col("acc_active"))
                if a > 0.5 and math.isfinite(ttc)]
        return min(vals) if vals else math.inf

    def cutout_throttle_frames(self, gap_below_m: float = 15.0) -> int:
        """Frames in CUTOUT where the stack is THROTTLING toward a stopped lead.

        This is the G2 mechanism signature: ACC cuts out below cutout_speed_mps,
        hands ownership to the governor's free-flow target, and the longitudinal
        controller accelerates into a stationary obstacle.
        """
        n = 0
        for st, thr, gap, v_lead in zip(self.col("acc_state_code"), self.col("throttle"),
                                        self.col("radar_fwd_distance_m"), self.col("lead_speed")):
            if st == "CUTOUT" and thr > 0.0 and gap < gap_below_m and v_lead < 0.1:
                n += 1
        return n

    def summary(self) -> str:
        return (f"frames={len(self.col('speed'))} states={self.states()} "
                f"min_gap={self.min_gap():.2f}m true_min_gap={self.min_true_gap():.2f}m ttc_min={self.ttc_min():.2f}s "
                f"estop_frames={self.frames_in(*_ESTOP_STATES)} "
                f"eb_frames={self.frames_in('EMERGENCY_BRAKE')} "
                f"cutout_throttle_frames={self.cutout_throttle_frames()} "
                f"collided={self.collided}")


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_closedloop(
    *,
    cfg: dict,
    lead: LeadProfile,
    grade_rad: float = 0.0,
    ego_speed0: float = 0.0,
    gap0: float = 15.0,
    free_flow_target: float = 12.0,
    n_frames: int = 400,
    dt: float = FRAME_DT_MEASURED_S,
    acc_dt: Optional[float] = None,        # None → 1/30 unless cfg acc.use_measured_dt (T-ACC-DT-HARDCODE)
    plant: Optional[PointMassPlant] = None,
    acc: Optional[ACCController] = None,
    longitudinal: Optional[LongitudinalController] = None,
    planned_accel: float = 0.0,
    radar_detect_fn: Optional[Callable[[int, float], bool]] = None,
    stop_on_collision: bool = True,
    radar_range_offset_m: float = RADAR_RANGE_OFFSET_MEASURED_M,
    contact_frames_after: int = 40,
    radar_noise_sigma_m: float = 0.0,        # AVBridge.cs radarDistanceNoiseSigma = 0.15
    radar_rate_noise_sigma_mps: float = 0.0, # AVBridge.cs radarRateNoiseSigma   = 0.05
    rng_seed: int = 7,
) -> ClosedLoopResult:
    """Run the ACC longitudinal stack against a scripted lead.

    ``radar_detect_fn(frame, gap) -> bool`` can inject dropouts; default is
    perfect detection inside ``acc.detection_range_m``.

    ``gap`` is always the TRUE bumper-to-bumper gap. ``radar_range_offset_m`` is
    added to what the radar reports (Unity's centre-to-centre frame, measured
    4.43 m); on contact (true gap <= 0) the radar reports
    ``COLLISION_OVERRIDE_RANGE_M`` like AVBridge.cs does, the lead stops, and the
    run continues for ``contact_frames_after`` frames so post-contact behaviour
    is visible. The production ``ForwardRadarSensor`` subtracts
    ``cfg["acc"]["radar_range_offset_m"]`` — pass a cfg with 0.0 to model the
    pre-2026-09-22 stack, or ``radar_range_offset_m=0.0`` for an idealised radar.
    """
    acc_cfg = cfg.get("acc", {})
    lon_cfg = cfg.get("control", {}).get("longitudinal", {})
    safety_cfg = cfg.get("safety", {})
    if acc_dt is None:
        acc_dt = dt if bool(acc_cfg.get("use_measured_dt", False)) else ACC_DT_PRODUCTION_S

    sensor = ForwardRadarSensor(gap_alpha=float(acc_cfg.get("gap_alpha", 0.30)),
                                rate_alpha=float(acc_cfg.get("rate_alpha", 0.20)),
                                range_offset_m=float(acc_cfg.get("radar_range_offset_m", 0.0)))
    acc = acc or build_acc_controller(cfg)
    lon = longitudinal or build_longitudinal_controller(cfg)
    plant = plant or PointMassPlant()

    detection_range = float(acc_cfg.get("detection_range_m", 60.0))
    release_speed = float(safety_cfg.get("emergency_stop_release_speed", 0.2))

    # orchestrator.py:8560 routing flags + VehicleController.compute_control kwargs
    routing_enabled = bool(lon_cfg.get("acc_idm_accel_routing_enabled", False))
    routing_shadow = bool(lon_cfg.get("acc_idm_accel_routing_shadow_mode", True))
    relax_enabled = bool(lon_cfg.get("acc_idm_accel_routing_emergency_relax_in_estate", False))
    relax_decel_floor = float(lon_cfg.get("acc_idm_accel_routing_emergency_decel_floor", -3.0))
    relax_jerk_max = float(lon_cfg.get("acc_idm_accel_routing_emergency_jerk_max", 4.0))
    b1_bypass = bool(lon_cfg.get("acc_idm_accel_routing_b1_hard_brake_bypass", False))
    floor_mode = str(lon_cfg.get("acc_idm_accel_floor_mode", "off"))
    floor_states = tuple(lon_cfg.get("acc_idm_accel_floor_states", ()) or ())
    floor_min_neg = float(lon_cfg.get("acc_idm_accel_floor_min_negative", 0.0))

    res = ClosedLoopResult()
    tr = res.trace
    _rng = __import__('numpy').random.default_rng(rng_seed)

    v = float(ego_speed0)
    gap = float(gap0)
    estop_latched = False
    t = 0.0

    for i in range(n_frames):
        v_lead = lead.speed_at(t)

        # ── radar (centre-to-centre frame + collision override) ─────────────
        if res.collided:
            v_lead = 0.0                                    # LeadVehicle.StopMovement()
            reported = COLLISION_OVERRIDE_RANGE_M
            detected = True
        else:
            reported = gap + radar_range_offset_m
            detected = 0.0 < reported <= detection_range
            if radar_detect_fn is not None:
                detected = detected and bool(radar_detect_fn(i, gap))
        rr_raw = 0.0 if res.collided else v - v_lead                       # + = closing
        if radar_noise_sigma_m > 0.0 and detected and not res.collided:
            reported = max(0.1, reported + float(_rng.normal(0.0, radar_noise_sigma_m)))
        if radar_rate_noise_sigma_mps > 0.0 and detected and not res.collided:
            rr_raw = rr_raw + float(_rng.normal(0.0, radar_rate_noise_sigma_mps))
        raw = {
            "radar_fwd_detected": detected,
            "radar_fwd_distance_m": reported,
            "radar_fwd_range_rate_mps": rr_raw,
            "radar_fwd_snr": 1.0,
        }
        reading = sensor.read_frame(raw)

        # ── ACC (note: production steps this with a fixed 1/30 s) ───────────
        out = acc.compute_target_speed(ego_speed=v, free_flow_target=free_flow_target,
                                       reading=reading, dt=acc_dt)
        state = out.state.value

        # ── single-owner resolver  (orchestrator.py:9927) ───────────────────
        final_target = free_flow_target
        owner = "speed_governor"
        acc_state_active = state not in _ACC_OWNER_INACTIVE_STATES
        if acc_state_active or out.request_estop:
            final_target = min(free_flow_target, max(0.0, out.target_speed))
            owner = "acc"
        if out.request_estop:
            final_target = 0.0
            owner = "acc_collapsed_gap_stop" if state == "COLLAPSED_GAP_STOP" else "acc_ttc_estop"
            estop_latched = True

        # ── IDM → reference_accel routing  (orchestrator.py:8560) ───────────
        route_idm = (routing_enabled and not routing_shadow
                     and bool(out.acc_active) and out.idm_accel_mps2 < 0.0)
        reference_accel = out.idm_accel_mps2 if route_idm else planned_accel

        # ── longitudinal controller  (VehicleController.compute_control) ────
        throttle, brake = lon.compute_control(
            current_speed=v,
            reference_velocity=final_target,
            dt=dt,
            reference_accel=reference_accel,
            current_curvature=0.0,
            min_speed_floor=None,
            grade_rad=grade_rad,
            acc_state_code=state,
            emergency_relax_enabled=relax_enabled,
            emergency_decel_floor=relax_decel_floor,
            emergency_jerk_max=relax_jerk_max,
            idm_accel_mps2=out.idm_accel_mps2,
            acc_idm_accel_floor_mode=floor_mode,
            acc_idm_accel_floor_states=floor_states,
            acc_idm_accel_floor_min_negative=floor_min_neg,
        )
        accel_cmd_raw = float(getattr(lon, "last_accel_cmd_raw", float("nan")))

        # ── safety clip  (orchestrator.py:9445-9530) ────────────────────────
        if estop_latched and v <= release_speed:
            estop_latched = False
        forced = None
        if estop_latched and v > release_speed:
            forced = "estop_latch"
        elif b1_bypass and state == "EMERGENCY_BRAKE":
            forced = "b1_bypass"
        if forced:
            throttle, brake = 0.0, 1.0

        # ── record (HDF5 names) ─────────────────────────────────────────────
        tr["t"].append(t)
        tr["speed"].append(v)
        tr["lead_speed"].append(v_lead)
        tr["road_grade"].append(grade_rad)
        tr["radar_fwd_detected"].append(detected)
        tr["radar_fwd_distance_m"].append(reported)
        tr["true_bumper_gap_m"].append(gap)
        tr["radar_fwd_range_rate_mps"].append(v - v_lead)
        tr["acc_state_code"].append(state)
        tr["acc_active"].append(out.acc_active)
        tr["acc_target_speed_mps"].append(out.target_speed)
        tr["acc_idm_accel_mps2"].append(out.idm_accel_mps2)
        tr["acc_ttc_s"].append(out.ttc_s)
        tr["acc_request_estop"].append(out.request_estop)
        tr["target_speed_final"].append(final_target)
        tr["final_longitudinal_owner_code"].append(owner)
        tr["reference_accel_source"].append("acc_idm" if route_idm else "planner")
        tr["longitudinal_accel_cmd_raw"].append(accel_cmd_raw)
        tr["throttle"].append(throttle)
        tr["brake"].append(brake)
        tr["safety_override"].append(forced or "")

        # ── plant ───────────────────────────────────────────────────────────
        v_next = plant.step(v, throttle, brake, grade_rad, dt)
        gap_next = gap + (v_lead - 0.5 * (v + v_next)) * dt
        if gap_next <= 0.0 and not res.collided:
            res.collided = True
            res.contact_frame = i
        if res.collided:
            gap_next = max(0.0, gap_next)
            if stop_on_collision and i >= res.contact_frame + contact_frames_after:
                v, gap, t = v_next, gap_next, t + dt
                break
        v, gap, t = v_next, gap_next, t + dt

    return res


# ---------------------------------------------------------------------------
# Scenario shortcuts
# ---------------------------------------------------------------------------

def run_g2_from_brake_onset(cfg: Optional[dict] = None, **kw) -> ClosedLoopResult:
    """hill_g2_stop_on_grade from the instant the lead begins braking.

    Initial state is the recorded onset in recording_20260908_041941.h5 frame
    210: ego 10.2 m/s, gap 58.6 m, lead 10 m/s, grade 0.050.  Starting here
    skips ~23 s of startup and keeps the test under a second.
    """
    cfg = cfg or load_scenario_config("config/acc_hill_highway.yaml")
    params = dict(
        cfg=cfg,
        lead=LeadProfile(speed=10.0, brake_at_s=0.0, decel=1.5, brake_to=0.0),
        grade_rad=0.050,
        ego_speed0=10.2,
        gap0=58.6,
        free_flow_target=12.0,
        n_frames=int(40.0 / FRAME_DT_MEASURED_S),
    )
    params.update(kw)
    return run_closedloop(**params)


def run_g2_from_rest(cfg: Optional[dict] = None, **kw) -> ClosedLoopResult:
    """Full scenario: ego from rest 15 m behind a 10 m/s lead that brakes later.

    Used as a plant self-consistency check — the real run reached ~58 m gap by
    the time the lead braked.
    """
    cfg = cfg or load_scenario_config("config/acc_hill_highway.yaml")
    params = dict(
        cfg=cfg,
        lead=LeadProfile(speed=10.0, brake_at_s=23.0, decel=1.5, brake_to=0.0),
        grade_rad=0.050,
        ego_speed0=0.0,
        gap0=15.0,
        free_flow_target=12.0,
        n_frames=int(65.0 / FRAME_DT_MEASURED_S),
    )
    params.update(kw)
    return run_closedloop(**params)


def run_h5_stop_go(cfg: Optional[dict] = None, **kw) -> ClosedLoopResult:
    """highway_h5_stop_go: flat highway_65, lead sinusoid 0↔8 m/s over 20 s,
    17 m initial gap, ego target 15 m/s.  Sweep verdict on 2026-09-08: PASS 100.0
    — this scenario is the harness's known-GOOD calibration point.
    """
    cfg = cfg or load_scenario_config("config/acc_highway.yaml")
    params = dict(
        cfg=cfg,
        lead=LeadProfile(kind="stop_go", stop_go_top_speed=8.0, stop_go_period_s=20.0),
        grade_rad=0.0,
        ego_speed0=0.0,
        gap0=17.0,
        free_flow_target=15.0,
        n_frames=int(60.0 / FRAME_DT_MEASURED_S),
    )
    params.update(kw)
    return run_closedloop(**params)


def print_trace(res: ClosedLoopResult, every: int = 3, start: int = 0, stop: Optional[int] = None) -> None:
    """Debug printer with the same columns as the recording dump."""
    tr = res.trace
    n = len(tr["speed"]); stop = n if stop is None else min(stop, n)
    print(f"{'fr':>4} {'t':>6} {'v':>5} {'vl':>5} {'gap':>6} {'rr':>5} {'ttc':>6} {'state':<18} "
          f"{'acc_ts':>6} {'idm':>6} {'final':>6} {'owner':<16} {'a_raw':>6} {'thr':>5} {'brk':>5} ovr")
    for i in range(start, stop, every):
        print(f"{i:4d} {tr['t'][i]:6.2f} {tr['speed'][i]:5.2f} {tr['lead_speed'][i]:5.2f} "
              f"{tr['radar_fwd_distance_m'][i]:6.2f} {tr['radar_fwd_range_rate_mps'][i]:5.2f} "
              f"{min(tr['acc_ttc_s'][i], 99):6.2f} {tr['acc_state_code'][i]:<18} "
              f"{tr['acc_target_speed_mps'][i]:6.2f} {tr['acc_idm_accel_mps2'][i]:6.2f} "
              f"{tr['target_speed_final'][i]:6.2f} {tr['final_longitudinal_owner_code'][i]:<16} "
              f"{tr['longitudinal_accel_cmd_raw'][i]:6.2f} {tr['throttle'][i]:5.2f} {tr['brake'][i]:5.2f} "
              f"{tr['safety_override'][i]}")


if __name__ == "__main__":  # exploratory: python3 tests/acc_closedloop_harness.py
    import logging
    logging.disable(logging.WARNING)
    r = run_g2_from_brake_onset()
    print(r.summary())
    print_trace(r, every=4)
