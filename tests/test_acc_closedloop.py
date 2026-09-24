"""Closed-loop tests for the ACC longitudinal stack — no Unity required.

Longitudinal counterpart of ``test_closedloop_stability.py``.  Every controller
is the production object built from the real merged YAML; only the plant and the
radar are modelled.  See ``acc_closedloop_harness.py`` for the calibration.

Four groups:

  1. Calibration — the harness agrees with the sweep on a known-GOOD scenario
     (H5 stop-and-go: PASS 100.0 on 2026-09-08) and a known-BAD one (G2
     stop-on-grade: TTC 1.54–1.67 s on 7 consecutive nights).
  2. G2 with production config — passes since the 2026-09-22 fix
     (``acc.cutout_requires_no_lead`` + ``control.longitudinal.
     acc_jerk_cooldown_bypass_states``).  Parametrised over the plant calibration
     so a fix that only works for one plant gain is visible immediately.  These
     were strict-xfail reproducers before the fix.
  3. Mechanism — run with BOTH flags in their legacy (off) position and pin the
     diagnosis, so the record of *why* survives the fix.  Two stacked causes,
     neither of which is the grade:
       (a) CUTOUT hand-off: below ``acc.cutout_speed_mps`` the ACC state machine
           ramps its target UP toward free-flow and the owner-resolver gives the
           governor's 12 m/s target to the longitudinal controller — with a
           stopped lead 7 m ahead.  Dominant on grade (G2).
       (b) Jerk-cooldown pin: ``LongitudinalController`` multiplies accel_cmd by
           ``jerk_cooldown_scale`` (0.4) every frame the cooldown is armed, which
           pins IDM's −4…−10 m/s² demand at ≈ −0.43 m/s² (brake 0.18).  The
           bypass at pid_controller.py:5114 only fires when |gravity| ≥ 0.1 or in
           an emergency state, so this is MASKED on grades and dominant on flat.
     The counterfactuals refute the standing explanations (grade feed-forward,
     actuator latency, ``idm_comfortable_decel``) — see
     ``project_acc_brake_authority_findings`` for the history they correct.
  4. Latent — EMERGENCY_BRAKE persists at standstill because the state machine
     tests ``range_rate_mps > 0.0`` on an EMA that asymptotes but never reaches
     zero.
  5. Radar frame (found by the 2026-09-22 Unity A/B) — AVBridge.cs reports the
     centre-to-centre range; physical contact occurs at a reported ~4.43 m, so
     before ``acc.radar_range_offset_m`` every gap threshold in the stack sat
     inside the lead vehicle's body. The harness always models the real sensor;
     the production config compensates it, ``_legacy_cfg`` does not.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acc_closedloop_harness import (  # noqa: E402
    ACC_DT_PRODUCTION_S,
    FRAME_DT_MEASURED_S,
    RADAR_RANGE_OFFSET_MEASURED_M,
    LeadProfile,
    PointMassPlant,
    build_acc_controller,
    build_longitudinal_controller,
    load_scenario_config,
    run_g2_from_brake_onset,
    run_g2_from_rest,
    run_h5_stop_go,
)
from control.acc_controller import ACCController, ACCParams  # noqa: E402
from control.radar_sensor import RadarReading  # noqa: E402

pytestmark = pytest.mark.control

TTC_GATE_S = 2.0          # scoring_registry.ACC_TTC_MIN_GATE_S — the G2/H5 Expected: line
ESTOP_STATES = ("TTC_ESTOP", "COLLAPSED_GAP_STOP")

# Plant calibration grid: ±30 % around the values fitted to the 2026-09-08 recording.
PLANT_GRID = [
    pytest.param(PointMassPlant(k_throttle=kt, k_brake=kb), id=f"kt{kt:g}-kb{kb:g}")
    for kt in (5.0, 7.0, 9.0)
    for kb in (3.5, 4.5, 6.0)
]

LEGACY_ACC = dict(cutout_requires_no_lead=False)
LEGACY_LON = dict(acc_jerk_cooldown_bypass_states=())


@pytest.fixture(scope="module")
def hill_cfg() -> dict:
    return load_scenario_config("config/acc_hill_highway.yaml")


@pytest.fixture(scope="module")
def highway_cfg() -> dict:
    return load_scenario_config("config/acc_highway.yaml")


def _legacy_cfg(cfg: dict) -> dict:
    """Config as it stood before 2026-09-22 for the SENSOR: no radar range offset."""
    import copy
    c = copy.deepcopy(cfg)
    c["acc"]["radar_range_offset_m"] = 0.0
    return c


def _legacy_acc(cfg: dict, **kw) -> ACCController:
    """ACCController as it behaved before 2026-09-22 (cutout_requires_no_lead off)."""
    return build_acc_controller(cfg, **{**LEGACY_ACC, **kw})


def _legacy_lon(cfg: dict, **kw):
    """LongitudinalController with the jerk-cooldown bypass in its legacy (off) position."""
    return build_longitudinal_controller(cfg, **{**LEGACY_LON, **kw})


def _acc_no_cutout(cfg: dict) -> ACCController:
    """Legacy controller with CUTOUT disabled by speed threshold — the counterfactual
    that first isolated mechanism (a)."""
    return _legacy_acc(cfg, cutout_speed_mps=0.0)


# ===========================================================================
# 1.  Calibration — harness agrees with the sweep on known-good and known-bad
# ===========================================================================

class TestHarnessCalibration:

    def test_production_config_builds_production_controllers(self, hill_cfg):
        """Guard the config keys the orchestrator glue in the harness depends on."""
        assert hill_cfg["acc"]["enabled"] is True
        lon = hill_cfg["control"]["longitudinal"]
        for key in ("acc_idm_accel_routing_enabled", "acc_idm_accel_routing_shadow_mode",
                    "acc_idm_accel_floor_mode", "jerk_cooldown_frames", "grade_ff_gain"):
            assert key in lon, f"harness assumes control.longitudinal.{key} exists"
        build_acc_controller(hill_cfg)
        build_longitudinal_controller(hill_cfg)

    def test_h5_stop_go_matches_sweep_pass(self, highway_cfg):
        """Known-GOOD: H5 scored PASS 100.0 on night 27.  The harness must agree."""
        r = run_h5_stop_go(_legacy_cfg(highway_cfg))
        assert r.frames_in(*ESTOP_STATES) == 0, r.summary()
        assert not r.collided, r.summary()
        assert r.ttc_min_acc_active() >= TTC_GATE_S, r.summary()

    def test_h5_still_passes_with_production_config(self, highway_cfg):
        """Regression guard for the 2026-09-22 fixes on the known-good scenario."""
        r = run_h5_stop_go(highway_cfg)
        assert r.frames_in(*ESTOP_STATES) == 0, r.summary()
        assert not r.collided, r.summary()
        assert r.ttc_min_acc_active() >= TTC_GATE_S, r.summary()
        assert r.min_true_gap() > 2.0, r.summary()

    def test_g2_ttc_matches_sweep_measurement(self, hill_cfg):
        """Known-BAD: the sweep measured TTC_min 1.54–1.67 s on 7 nights with the
        legacy flags.  The harness (sweep-style metric: min over acc_active
        frames) lands at ~1.58 in the same configuration."""
        lc = _legacy_cfg(hill_cfg)
        r = run_g2_from_brake_onset(lc, acc=_legacy_acc(lc), longitudinal=_legacy_lon(lc))
        assert 1.3 <= r.ttc_min_acc_active() <= 1.9, r.summary()
        assert r.frames_in(*ESTOP_STATES) > 0, r.summary()

    def test_from_rest_reaches_recorded_onset_state(self, hill_cfg):
        """The real run had ego 10.2 m/s at 58.6 m gap when the lead began braking
        (t≈23 s).  A point-mass plant with no curve governor should land within
        ~20 % — this bounds how far the plant can be trusted."""
        r = run_g2_from_rest(hill_cfg)
        i = min(range(len(r.col("t"))), key=lambda k: abs(r.col("t")[k] - 23.0))
        v, gap = r.col("speed")[i], r.col("radar_fwd_distance_m")[i]
        assert 9.0 <= v <= 13.0, f"ego {v:.2f} m/s at t=23 s (recording 10.2)"
        assert 40.0 <= gap <= 70.0, f"gap {gap:.1f} m at t=23 s (recording 58.6)"


# ===========================================================================
# 2.  G2 reproducer — desired behaviour, expected to FAIL until fixed
# ===========================================================================

class TestG2StopOnGrade:
    """Production config (both fixes on), idealised bumper-gap radar.  Were
    strict-xfail until 2026-09-22.  In the REAL radar frame the same config still
    contacts the lead — see TestRadarFrame."""

    @pytest.mark.parametrize("plant", PLANT_GRID)
    def test_g2_stops_without_estop(self, hill_cfg, plant):
        """hill_g2 Expected: 'ego stops cleanly on grade; no collision; no e-stop'."""
        r = run_g2_from_brake_onset(hill_cfg, plant=plant)
        assert not r.collided, r.summary()
        assert r.frames_in(*ESTOP_STATES) == 0, r.summary()
        assert r.min_gap() > 2.0, r.summary()

    def test_g2_ttc_min_meets_gate(self, hill_cfg):
        r = run_g2_from_brake_onset(hill_cfg)
        assert r.ttc_min_acc_active() >= TTC_GATE_S, r.summary()

    @pytest.mark.parametrize("plant", PLANT_GRID)
    def test_g2_flat_variant_also_passes(self, hill_cfg, plant):
        """The flat variant needed BOTH fixes — the cooldown bypass is what makes
        this one pass; on the 5 % grade it was already bypassed by gravity."""
        r = run_g2_from_brake_onset(hill_cfg, grade_rad=0.0, plant=plant)
        assert not r.collided, r.summary()
        assert r.frames_in(*ESTOP_STATES) == 0, r.summary()
        assert r.ttc_min_acc_active() >= TTC_GATE_S, r.summary()

    @pytest.mark.parametrize("flag", ["cutout_only", "cooldown_only"])
    def test_each_fix_alone_is_insufficient_on_flat(self, hill_cfg, flag):
        """Pins that both fixes are required: on flat ground, either one alone still
        e-stops (cooldown alone: CUTOUT hand-off; cutout alone: −0.43 pin)."""
        if flag == "cutout_only":
            r = run_g2_from_brake_onset(hill_cfg, grade_rad=0.0, longitudinal=_legacy_lon(hill_cfg))
        else:
            r = run_g2_from_brake_onset(hill_cfg, grade_rad=0.0, acc=_legacy_acc(hill_cfg))
        assert r.frames_in(*ESTOP_STATES) > 0, r.summary()


class TestFixFlags:
    """The two config kill-switches — defaults, production values, and effect."""

    def test_code_defaults_are_legacy(self):
        assert ACCParams().cutout_requires_no_lead is False
        assert ACCParams.from_config({}).cutout_requires_no_lead is False
        assert build_longitudinal_controller({"control": {"longitudinal": {}}}).acc_jerk_cooldown_bypass_states == ()

    def test_production_config_turns_both_on(self, hill_cfg, highway_cfg):
        for cfg in (hill_cfg, highway_cfg):
            assert cfg["acc"]["cutout_requires_no_lead"] is True
            states = set(cfg["control"]["longitudinal"]["acc_jerk_cooldown_bypass_states"])
            assert {"ACC_ACTIVE", "CUTOUT"} <= states
            assert cfg["acc"]["emergency_brake_min_closing_mps"] > 0.0
            assert cfg["acc"]["emergency_brake_abs_gap_m"] < cfg["acc"]["min_gap_s0_m"]

    def test_code_defaults_for_eb_are_legacy(self):
        assert ACCParams().emergency_brake_min_closing_mps == 0.0
        assert ACCParams().emergency_brake_abs_gap_m == 3.0

    def test_legacy_flags_reproduce_the_failure(self, hill_cfg):
        """Rollback path: flipping both switches back restores the original e-stop."""
        r = run_g2_from_brake_onset(hill_cfg, acc=_legacy_acc(hill_cfg), longitudinal=_legacy_lon(hill_cfg))
        assert r.frames_in(*ESTOP_STATES) > 0, r.summary()


# ===========================================================================
# 3a. Mechanism — CUTOUT hand-off (dominant on grade)
# ===========================================================================

class TestG2Mechanism:

    def test_cutout_hands_target_to_governor_with_stopped_lead_ahead(self, hill_cfg):
        """The signature of the failure: a CUTOUT frame where the final target is
        the governor's free-flow speed, well above ego, while a STOPPED lead sits
        inside 12 m — and the stack is throttling toward it."""
        r = run_g2_from_brake_onset(hill_cfg, acc=_legacy_acc(hill_cfg), longitudinal=_legacy_lon(hill_cfg))
        assert r.cutout_throttle_frames() > 0, r.summary()
        i = r.col("acc_state_code").index("CUTOUT")
        assert r.col("final_longitudinal_owner_code")[i] == "speed_governor"
        assert r.col("target_speed_final")[i] > r.col("speed")[i] + 1.0
        assert r.col("lead_speed")[i] < 0.1
        assert r.col("radar_fwd_distance_m")[i] < 12.0
        assert r.col("speed")[i] < hill_cfg["acc"]["cutout_speed_mps"]

    @pytest.mark.parametrize("plant", PLANT_GRID)
    def test_disabling_cutout_removes_estop_on_grade(self, hill_cfg, plant):
        """Counterfactual: with cutout_speed_mps=0 the same run stops ~2.7 m behind
        the lead with TTC ≥ 2 s, on every plant calibration.  This is the one knob
        that moves the outcome — see the negative controls below."""
        r = run_g2_from_brake_onset(hill_cfg, plant=plant, acc=_acc_no_cutout(hill_cfg),
                                    longitudinal=_legacy_lon(hill_cfg))
        assert r.frames_in(*ESTOP_STATES) == 0, r.summary()
        assert not r.collided, r.summary()
        assert r.ttc_min_acc_active() >= TTC_GATE_S, r.summary()
        assert r.min_gap() > 2.0, r.summary()

    @pytest.mark.parametrize("gain", [1.0, 0.0])
    def test_grade_feedforward_is_not_the_cause(self, hill_cfg, gain):
        """Night-27 sweep blamed 'grade FF propulsive bias' and proposed a
        range-rate guard on it.  Removing grade FF entirely does not remove the
        e-stop."""
        lon = _legacy_lon(hill_cfg, grade_ff_gain=gain)
        r = run_g2_from_brake_onset(hill_cfg, acc=_legacy_acc(hill_cfg), longitudinal=lon)
        assert r.frames_in(*ESTOP_STATES) > 0, r.summary()

    def test_idm_comfortable_decel_is_not_the_cause(self, hill_cfg):
        """project_acc_brake_authority_findings: do NOT tune idm_comfortable_decel.
        Here is why — 2.5→4.0 changes nothing."""
        acc = _legacy_acc(hill_cfg, comfortable_decel_mps2=4.0)
        r = run_g2_from_brake_onset(hill_cfg, acc=acc, longitudinal=_legacy_lon(hill_cfg))
        assert r.frames_in(*ESTOP_STATES) > 0, r.summary()

    def test_failure_persists_on_flat_ground(self, hill_cfg):
        """The scenario is named stop-on-GRADE and every write-up has blamed the
        grade.  Set it to zero and the e-stop still fires."""
        r = run_g2_from_brake_onset(hill_cfg, grade_rad=0.0, acc=_legacy_acc(hill_cfg), longitudinal=_legacy_lon(hill_cfg))
        assert r.frames_in(*ESTOP_STATES) > 0, r.summary()

    def test_acc_dt_hardcode_is_not_the_cause(self, hill_cfg):
        """orchestrator.py:9864 steps ACC with dt=1/30 while frames arrive at 1/13.
        Correcting it does not rescue G2 (the CUTOUT hand-off is unaffected by
        integration rate) — but it IS a real bug, quantified in the next test."""
        r = run_g2_from_brake_onset(hill_cfg, acc_dt=FRAME_DT_MEASURED_S,
                                    acc=_legacy_acc(hill_cfg), longitudinal=_legacy_lon(hill_cfg))
        assert r.frames_in(*ESTOP_STATES) > 0, r.summary()

    def test_acc_dt_hardcode_slows_idm_integration_2p3x(self, hill_cfg):
        """ACC target speed integrates idm_accel × dt per frame.  With the
        production 1/30 s hardcode and real 76.9 ms frames the target moves at
        43 % of the designed rate.  Isolated: same reading, same ego speed, 10
        frames, two dt values → Δtarget ratio must equal the dt ratio."""
        reading = RadarReading(detected=True, gap_m=40.0, range_rate_mps=0.0, snr=1.0,
                               gap_raw=40.0, range_rate_raw=0.0)

        def delta_target(dt: float) -> float:
            acc = ACCController(ACCParams.from_config(hill_cfg["acc"]))
            first = last = None
            for _ in range(10):
                out = acc.compute_target_speed(ego_speed=8.0, free_flow_target=12.0,
                                               reading=reading, dt=dt)
                assert out.state.value == "ACC_ACTIVE"
                first = out.target_speed if first is None else first
                last = out.target_speed
            return last - first

        ratio = delta_target(FRAME_DT_MEASURED_S) / delta_target(ACC_DT_PRODUCTION_S)
        expected = FRAME_DT_MEASURED_S / ACC_DT_PRODUCTION_S      # ≈ 2.31
        assert math.isclose(ratio, expected, rel_tol=0.05), (ratio, expected)


# ===========================================================================
# 3b. Mechanism — jerk-cooldown pin (dominant on flat, masked on grade)
# ===========================================================================

class TestFlatGroundJerkCooldownPin:

    def test_idm_decel_is_pinned_by_jerk_cooldown_on_flat(self, hill_cfg):
        """With CUTOUT already disabled, flat ground STILL e-stops: IDM demands
        −4…−10 m/s² and accel_cmd_raw sits at ≈ −0.43 (brake 0.18).  This is the
        same −0.43 the 2026-04-20 H5 probe recorded and attributed to a
        'routing gap'."""
        r = run_g2_from_brake_onset(hill_cfg, grade_rad=0.0, acc=_acc_no_cutout(hill_cfg),
                                    longitudinal=_legacy_lon(hill_cfg))
        assert r.frames_in(*ESTOP_STATES) > 0, r.summary()
        pairs = [(a, idm) for a, idm in zip(r.col("longitudinal_accel_cmd_raw"), r.col("acc_idm_accel_mps2"))
                 if idm < -3.0 and math.isfinite(a)]
        assert len(pairs) >= 5, "expected a sustained window of IDM demanding < -3 m/s²"
        cmds = sorted(a for a, _ in pairs)
        median_cmd = cmds[len(cmds) // 2]
        median_idm = sorted(i for _, i in pairs)[len(pairs) // 2]
        delivered = median_cmd / median_idm            # fraction of IDM demand reaching the actuator
        assert -0.6 < median_cmd < -0.3, f"median accel_cmd_raw {median_cmd:.3f} (expected the -0.43 pin)"
        assert delivered < 0.15, f"controller delivered {delivered:.0%} of IDM demand (median idm {median_idm:.2f})"
        assert sum(1 for a in cmds if a > -0.7) / len(cmds) > 0.7, "pin should hold on most frames"

    @pytest.mark.parametrize("knob", [
        pytest.param({"jerk_cooldown_frames": 0}, id="cooldown_frames=0"),
        pytest.param({"jerk_cooldown_scale": 1.0}, id="cooldown_scale=1"),
        pytest.param({"max_jerk": 0.0}, id="measured_jerk_cap_off"),
    ])
    @pytest.mark.parametrize("plant", PLANT_GRID)
    def test_removing_cooldown_unpins_idm_and_passes_flat(self, hill_cfg, plant, knob):
        """Any of the three ways to disarm the cooldown lets the IDM demand through
        and the flat-ground stop passes on every plant calibration.  (The cooldown
        is armed by the *measured*-jerk cap at pid_controller.py:5330 — braking
        harder than max_jerk=0.7 m/s³ re-arms it every frame.)"""
        lon = _legacy_lon(hill_cfg, **knob)
        r = run_g2_from_brake_onset(hill_cfg, grade_rad=0.0, plant=plant,
                                    acc=_acc_no_cutout(hill_cfg), longitudinal=lon)
        assert r.frames_in(*ESTOP_STATES) == 0, r.summary()
        assert not r.collided, r.summary()
        assert r.ttc_min_acc_active() >= TTC_GATE_S, r.summary()

    def test_pin_is_bypassed_on_grade(self, hill_cfg):
        """Same controller, cooldown armed: grade passes, flat fails.  The only
        difference is |gravity_accel| ≥ 0.1 taking the bypass branch at
        pid_controller.py:5114 — which is why G2 never showed this and H5 did."""
        on_grade = run_g2_from_brake_onset(hill_cfg, grade_rad=0.05, acc=_acc_no_cutout(hill_cfg),
                                           longitudinal=_legacy_lon(hill_cfg))
        on_flat = run_g2_from_brake_onset(hill_cfg, grade_rad=0.0, acc=_acc_no_cutout(hill_cfg),
                                          longitudinal=_legacy_lon(hill_cfg))
        assert on_grade.frames_in(*ESTOP_STATES) == 0, on_grade.summary()
        assert on_flat.frames_in(*ESTOP_STATES) > 0, on_flat.summary()


# ===========================================================================
# 4.  Latent — EMERGENCY_BRAKE never releases at standstill
# ===========================================================================

class TestStandstillEmergencyBrakeLatch:
    """T-ACC-EB-STANDSTILL. Legacy: after a clean stop the state machine sat in
    EMERGENCY_BRAKE for the rest of the run (`range_rate > 0.0` on an EMA that
    never reaches zero; EB floor 3.0 m > s0 2.0 m). In Unity the Doppler noise
    toggled it 124 times in one 90 s stop and every entry was scored as an
    e-stop.  Fixed 2026-09-22 by ``acc.emergency_brake_min_closing_mps: 0.1``
    and ``acc.emergency_brake_abs_gap_m: 1.5`` (< s0)."""

    @staticmethod
    def _standstill(r):
        st, v = r.col("acc_state_code"), r.col("speed")
        i0 = next(i for i, x in enumerate(v) if x == 0.0)
        entries = sum(1 for i in range(1, len(st)) if st[i] == "EMERGENCY_BRAKE" and st[i - 1] != "EMERGENCY_BRAKE")
        tail = st[i0:]
        return entries, tail.count("EMERGENCY_BRAKE") / len(tail), st[-1]

    def test_production_releases_emergency_brake_at_standstill(self, hill_cfg):
        r = run_g2_from_brake_onset(hill_cfg)
        entries, eb_frac, final = self._standstill(r)
        assert entries <= 1, r.summary()
        assert eb_frac < 0.05, f"EMERGENCY_BRAKE held for {eb_frac:.0%} of standstill"
        assert final == "ACC_ACTIVE"
        assert 1.2 < r.min_true_gap() < 3.0, r.summary()     # parked short of s0, not touching

    def test_legacy_thresholds_latch_emergency_brake(self, hill_cfg):
        acc = build_acc_controller(hill_cfg, emergency_brake_min_closing_mps=0.0, emergency_brake_abs_gap_m=3.0)
        r = run_g2_from_brake_onset(hill_cfg, acc=acc)
        entries, eb_frac, final = self._standstill(r)
        assert eb_frac > 0.95 and final == "EMERGENCY_BRAKE"

    def test_floor_above_s0_causes_creep_brake_cycles(self, hill_cfg):
        """The second half of the defect: with the floor at 3.0 m but the closing
        threshold fixed, IDM's approach to s0 = 2.0 m keeps re-entering EB."""
        acc = build_acc_controller(hill_cfg, emergency_brake_abs_gap_m=3.0)
        r = run_g2_from_brake_onset(hill_cfg, acc=acc)
        entries, _, _ = self._standstill(r)
        assert entries >= 5, r.summary()


# ===========================================================================
# 5.  Radar frame — the sensor reports centre-to-centre, not bumper-to-bumper
# ===========================================================================

class TestRadarFrame:
    """Unity A/B 2026-09-22: with the CUTOUT and cooldown fixes on, 5/5 runs still
    ended in COLLAPSED_GAP_STOP at a reported 0.10 m because
    LeadVehicle.OnTriggerEnter fired while the radar read 4.34–4.64 m.  Fixed the
    same night by ``acc.radar_range_offset_m: 4.43`` in ``ForwardRadarSensor``."""

    def test_production_config_compensates_the_measured_offset(self, hill_cfg):
        assert hill_cfg["acc"]["radar_range_offset_m"] == pytest.approx(RADAR_RANGE_OFFSET_MEASURED_M)
        r = run_g2_from_brake_onset(hill_cfg)
        rep, true = r.col("radar_fwd_distance_m"), r.col("true_bumper_gap_m")
        assert all(abs((a - b) - RADAR_RANGE_OFFSET_MEASURED_M) < 1e-9 for a, b in zip(rep[:50], true[:50]))

    def test_legacy_sensor_reproduces_unity_contact(self, hill_cfg):
        """Pins the A/B result: fixed controllers + legacy sensor frame → contact
        and a collapsed-gap stop, exactly as all five fix-arm recordings."""
        r = run_g2_from_brake_onset(_legacy_cfg(hill_cfg))
        assert r.collided, r.summary()
        assert "COLLAPSED_GAP_STOP" in r.states(), r.summary()
        assert r.min_true_gap() <= 0.0

    def test_g2_does_not_contact_lead_with_production_config(self, hill_cfg):
        """Was strict-xfail (T-ACC-RADAR-FRAME) until the offset landed."""
        r = run_g2_from_brake_onset(hill_cfg)
        assert not r.collided, r.summary()
        assert r.frames_in(*ESTOP_STATES) == 0, r.summary()
        # Parks ~1.7 m short: IDM's s0 is 2.0 m but the EB floor (1.5 m) + brake
        # dynamics overshoot the equilibrium slightly. Unity A/B: 1.55–1.63 m.
        assert 1.2 < r.min_true_gap() < 3.0, r.summary()
        assert r.ttc_min_acc_active() >= TTC_GATE_S, r.summary()

    @pytest.mark.parametrize("offset_m", [RADAR_RANGE_OFFSET_MEASURED_M, 4.0])
    def test_offset_within_half_metre_still_prevents_contact(self, hill_cfg, offset_m):
        """The measured value has ±0.15 m spread across recordings; 0.43 m under it
        still leaves > 1 m of true margin (the under-compensation shows up 1:1 in
        the parked distance: 1.7 m → 1.27 m)."""
        import copy
        c = copy.deepcopy(hill_cfg); c["acc"]["radar_range_offset_m"] = offset_m
        r = run_g2_from_brake_onset(c)
        assert not r.collided, r.summary()
        assert r.frames_in(*ESTOP_STATES) == 0, r.summary()
        assert r.min_true_gap() > 1.0, r.summary()

    def test_raw_gap_thresholds_sit_inside_the_lead_body(self, hill_cfg):
        """Why the offset is necessary: s0, the EB absolute floor, the collapsed-gap
        stop and the near-miss gate are all smaller than the reported range at
        which the bumpers touch. In the raw frame they were unreachable."""
        from control import acc_controller as ac
        from tools import scoring_registry as reg
        thresholds = {
            "acc.min_gap_s0_m": float(hill_cfg["acc"]["min_gap_s0_m"]),
            "_EMERGENCY_BRAKE_ABS_GAP_M": ac._EMERGENCY_BRAKE_ABS_GAP_M,
            "_COLLAPSED_GAP_STOP_M": ac._COLLAPSED_GAP_STOP_M,
            "ACC_NEAR_MISS_GAP_M": reg.ACC_NEAR_MISS_GAP_M,
        }
        assert all(v < RADAR_RANGE_OFFSET_MEASURED_M for v in thresholds.values()), thresholds
        assert reg.ACC_RADAR_RANGE_OFFSET_M == pytest.approx(hill_cfg["acc"]["radar_range_offset_m"])

    def test_h5_legacy_pass_hid_a_half_metre_true_margin(self, highway_cfg):
        """The sweep scored H5 PASS 100.0 with a reported minimum gap of ~5.8 m.
        In the legacy frame that is ~0.5 m bumper-to-bumper — which is why the
        2026-09-09 H5 run physically contacted the lead (6615 override frames) and
        was still reported as 0 collisions."""
        r = run_h5_stop_go(_legacy_cfg(highway_cfg))
        assert not r.collided, r.summary()
        assert r.min_gap() > 4.0, r.summary()          # what the sweep saw
        assert r.min_true_gap() < 1.0, r.summary()     # what was physically there
