"""Lateral sign-convention contract — one plant in Unity's measured frame, every lateral controller must drive it.

A minus sign has been blamed four times (NMPC e_lat b6a8c5b, NMPC κ 2026-04, LMPC e_lat Frenet alignment
2026-04-18, LMPC κ `mpc_curvature_sign` 2026-10-11) and each time it was found in Unity after the fact because
the only closed-loop test (`test_closedloop_stability.py`) writes its plant in the SOLVER's own convention —
self-consistent by construction, blind to a disagreement between the orchestrator and the solver, and driven
in one direction only. This file pins the conventions measured on two production recordings that are exact
mirrors of each other and closes the loop through the production `VehicleController.compute_control`.

CONVENTIONS (recording_20261011_140140.h5 hill_highway, all LEFT arcs; recording_20261011_140830.h5
mixed_radius, all RIGHT arcs; medians over arc frames |κ| > 0.003):

    orchestrator / planner curvature (`reference_point['curvature']`, `mpc_kappa_ref`)   + = LEFT turn
    Unity steering command (`control/steering` == `vehicle/steering_input`)              + = RIGHT (−0.081 on left arcs)
    Unity yaw rate (`vehicle/angular_velocity[1]`)                                        − on left arcs; |ω| = κ·v
    `groundTruthLaneCenterX{AtCar,Lookahead}` → `gt_cross_track_{at_car,lookahead}_m`    lane-CENTRE offset from the car,
                                                                                          + = centre to the RIGHT of the car
    `gt_cross_track_road_frame_at_car_m` (= scorer's at-car error)                        CAR offset from the centre, + = RIGHT
    `headingDeltaDeg` → `gt_heading_error_rad` → `mpc_e_heading`                          + = pointing RIGHT of the path
                                                                                          (corr with d/dt at-car cross-track +0.85)
    `mpc_e_lat` = −gt_cross_track_lookahead_m                                             + = car RIGHT of the lookahead centre
    `reference_point['x']`                                                                centre offset at the lookahead, + RIGHT
    `reference_point['heading']`                                                          path heading in vehicle frame, + RIGHT
    Unity ground-truth lookahead distance                                                 6.0 m; lookahead − at-car offset = −½Ld²κ

PROMOTED 2026-10-11 23:40: mpc_curvature_sign -1.0 after the valid A/B (highway 99.4→99.8, sweeping Traj 92.8→96.0, hill
neutral, mixed@9.8 within bimodal noise with more steering activity, mixed@11.6 margin unchanged). 1.0 is the kill-switch.

TWO WRITERS: the orchestrator sends steering from compute_control once per camera frame AND from
compute_interframe_steering up to three times in between (57 % of recorded frames carry an inter-frame update,
and the inter-frame command differs from the camera-frame one by > 0.03 on 64–74 % of arc frames). Any sign
adaptation must therefore live where both paths meet (MPCParams.curvature_sign), never at one call site: the
2026-10-11 hill A/B that flipped only the camera-frame site had Unity's steering input alternating sign every frame.

Actuator (closed-loop multisine identification 2026-10-11 and the two recordings above): wheel angle settles at
≈ 0.8 × max_steer_map(v) per unit of the raw command (map 0.5236 rad @ 0 m/s → 0.2793 @ 12), τ ≈ 0.25 s,
≈ 2 frames of pipeline delay; vehicle ≈ kinematic bicycle with understeer, L_eff = 2.5·(1 + 0.0046·v²).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from av_stack.config import load_config
from control.mpc_controller import MPCController
from control.pid_controller import VehicleController
from control.regime_selector import ControlRegime

FRAME_DT = 0.05           # 20 Hz control loop (the machine runs ~21.6 Hz)
GT_LOOKAHEAD_M = 6.0      # Unity's ground-truth lookahead distance (vehicle/ground_truth_lookahead_distance)
ACTUATOR_GAIN = 0.80      # fraction of the modelled max-steer map actually delivered
ACTUATOR_TAU_S = 0.25
PIPELINE_DELAY_FRAMES = 2
TRACK_SPEED_LIMIT_MPS = 15.7   # vehicle/speed_limit on the mixed_radius arcs; hill is 15.1


def max_steer_map(v: float) -> float:
    return float(np.interp(v, [0.0, 12.0], [0.5236, 0.2793]))


def l_eff(v: float) -> float:
    return 2.5 * (1.0 + 0.0046 * v * v)


# ---------------------------------------------------------------------------
# Path: a straight run-in followed by a constant-radius arc (signed, + = left)
# ---------------------------------------------------------------------------

@dataclass
class Path:
    """Straight along +x for `straight_m`, then an arc of signed curvature `kappa` (+ = left, the
    orchestrator's convention). World frame is right-handed with heading counter-clockwise from +x."""
    kappa: float
    straight_m: float = 40.0

    def heading(self, s: float) -> float:
        return 0.0 if s <= self.straight_m else self.kappa * (s - self.straight_m)

    def point(self, s: float) -> np.ndarray:
        if s <= self.straight_m or abs(self.kappa) < 1e-9:
            return np.array([s, 0.0])
        R = 1.0 / abs(self.kappa)
        sgn = 1.0 if self.kappa > 0 else -1.0
        centre = np.array([self.straight_m, sgn * R])
        ang = -sgn * math.pi / 2.0 + self.kappa * (s - self.straight_m)
        return centre + R * np.array([math.cos(ang), math.sin(ang)])

    def nearest(self, q: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """Return (s, centre point, path heading) of the nearest path point to q."""
        s_line = float(np.clip(q[0], 0.0, self.straight_m))
        c_line = np.array([s_line, 0.0])
        if abs(self.kappa) < 1e-9:
            c_line = np.array([q[0], 0.0])
            return float(q[0]), c_line, 0.0
        R = 1.0 / abs(self.kappa)
        sgn = 1.0 if self.kappa > 0 else -1.0
        centre = np.array([self.straight_m, sgn * R])
        rel = q - centre
        ang = math.atan2(rel[1], rel[0])
        ang0 = -sgn * math.pi / 2.0
        dang = (ang - ang0) * sgn
        dang = dang % (2.0 * math.pi)
        s_arc = self.straight_m + dang * R
        c_arc = centre + R * np.array([math.cos(ang), math.sin(ang)])
        if np.linalg.norm(q - c_arc) <= np.linalg.norm(q - c_line) and q[0] >= self.straight_m - 1e-9:
            return s_arc, c_arc, self.heading(s_arc)
        return s_line, c_line, 0.0

    def curvature_at(self, s: float) -> float:
        return 0.0 if s <= self.straight_m else self.kappa


# ---------------------------------------------------------------------------
# Plant in Unity's conventions
# ---------------------------------------------------------------------------

@dataclass
class UnityFramePlant:
    path: Path
    speed: float
    d0_right: float = 0.0                      # initial car offset, + right (road frame)
    gain: float = ACTUATOR_GAIN
    tau_s: float = ACTUATOR_TAU_S
    delay_frames: int = PIPELINE_DELAY_FRAMES
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    theta: float = 0.0                         # CCW heading, world frame
    wheel_rad: float = 0.0                     # signed in Unity's command sense (+ right)
    yaw_rate_ccw: float = 0.0
    _queue: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.pos = np.array([5.0, -self.d0_right])   # +right (road frame along +x) is −y
        self._queue = [0.0] * self.delay_frames

    # -- observation in the orchestrator's reference_point / current_state shape --------------------
    def observe(self) -> Tuple[Dict, Dict]:
        fwd = np.array([math.cos(self.theta), math.sin(self.theta)])
        right = np.array([math.sin(self.theta), -math.cos(self.theta)])
        s_car, c_car, th_path = self.path.nearest(self.pos)
        right_path = np.array([math.sin(th_path), -math.cos(th_path)])
        d_car_right = float(np.dot(self.pos - c_car, right_path))          # car offset, + right
        centre_at_car_right = float(np.dot(c_car - self.pos, right))       # centre offset, + right
        heading_err_right = float(-(self.theta - th_path))                  # + pointing right
        heading_err_right = (heading_err_right + math.pi) % (2 * math.pi) - math.pi
        p_look = self.pos + GT_LOOKAHEAD_M * fwd
        _, c_look, th_look = self.path.nearest(p_look)
        centre_at_look_right = float(np.dot(c_look - p_look, right))
        kappa_here = self.path.curvature_at(s_car)
        dists = np.arange(0.0, 62.0, 2.0)
        kappas = np.array([self.path.curvature_at(s_car + d) for d in dists])
        reference_point = {
            'x': centre_at_look_right,
            'y': GT_LOOKAHEAD_M,
            'heading': float(-(th_look - self.theta)),                      # path heading in vehicle frame, + right
            'velocity': self.speed,
            'curvature': kappa_here,
            'curvature_map_abs': abs(kappa_here),
            'curvature_horizon_signed': kappas,
            'curvature_horizon_distances': dists,
            'gt_cross_track_m': d_car_right,
            'gt_cross_track_road_frame_at_car_m': d_car_right,
            'gt_cross_track_at_car_m': centre_at_car_right,
            'gt_cross_track_vehicle_frame_at_car_m': centre_at_car_right,
            'gt_cross_track_lookahead_m': centre_at_look_right,
            'gt_cross_track_control_source_code': 'road_frame_at_car',
            'gt_cross_track_source_code': 'road_frame_at_car',
            'gt_heading_error_rad': heading_err_right,
        }
        current_state = {
            'heading': self.theta,
            'speed': self.speed,
            'position': (float(self.pos[0]), float(self.pos[1])),
            'angular_velocity_y': -self.yaw_rate_ccw,                        # Unity: negative = left turn
        }
        return current_state, reference_point

    def d_car_right(self) -> float:
        return self.observe()[1]['gt_cross_track_road_frame_at_car_m']

    # -- one frame: apply the raw command Unity would receive -----------------------------------------
    def step(self, command: float, dt: float = FRAME_DT) -> None:
        self._queue.append(float(command))
        cmd = self._queue.pop(0)
        target = self.gain * max_steer_map(self.speed) * cmd
        a = dt / max(self.tau_s, 1e-6)
        self.wheel_rad += (target - self.wheel_rad) * min(1.0, a)
        self.yaw_rate_ccw = -self.speed * math.tan(self.wheel_rad) / l_eff(self.speed)   # + command = right = CW
        self.theta += self.yaw_rate_ccw * dt
        self.pos = self.pos + self.speed * dt * np.array([math.cos(self.theta), math.sin(self.theta)])


# ---------------------------------------------------------------------------
# Production controller driver
# ---------------------------------------------------------------------------

def _production_cfg(**mpc_overrides) -> dict:
    cfg = copy.deepcopy(load_config(None))
    cfg['trajectory']['mpc'].update(mpc_overrides)
    return cfg


def run_closed_loop(kappa: float, speed: float, n_frames: int = 400, d0_right: float = 0.0,
                    cfg: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Drive the production VehicleController around `Path(kappa)` at constant speed. Returns the
    at-car offset (+right), the raw command, the regime and the controller's e_lat per frame."""
    cfg = cfg or _production_cfg()
    vc = VehicleController(full_config=cfg)
    # The orchestrator sets the longitudinal max_speed to the track limit at runtime (hill 15.1, mixed 15.7 m/s);
    # built from config alone it is 10.0, and the QP's speed bound then makes every solve above 10 m/s
    # "infeasible" → fallback (harness artifact found 2026-10-11; Unity solves 100 % of arc frames at 11.6).
    vc.longitudinal_controller.max_speed = TRACK_SPEED_LIMIT_MPS
    plant = UnityFramePlant(Path(kappa), speed, d0_right=d0_right)
    d, u, regime, e_lat = [], [], [], []
    for _ in range(n_frames):
        state, ref = plant.observe()
        out = vc.compute_control(state, ref, return_metadata=True)
        cmd = float(out['steering'])
        d.append(ref['gt_cross_track_road_frame_at_car_m']); u.append(cmd)
        regime.append(int(out.get('regime', -9))); e_lat.append(float(out.get('mpc_e_lat', np.nan)))
        plant.step(cmd)
    return {'d': np.array(d), 'u': np.array(u), 'regime': np.array(regime), 'e_lat': np.array(e_lat)}


def _arc_frames(speed: float, straight_m: float = 40.0, settle_s: float = 6.0) -> slice:
    """Frames after the car has been on the arc for `settle_s` seconds (steady state)."""
    start = int((straight_m / speed + settle_s) / FRAME_DT)
    return slice(start, None)


# ===========================================================================
# 1. The plant obeys the measured contract
# ===========================================================================

class TestPlantMatchesRecordedConventions:

    def test_left_arc_on_centre_has_negative_lookahead_centre_offset(self):
        """hill: lookahead − at-car = −0.184 m for κ=+0.010, Ld=6 → −½Ld²κ = −0.180."""
        p = UnityFramePlant(Path(0.010, straight_m=0.0), speed=6.6)
        p.pos = p.path.point(20.0); p.theta = p.path.heading(20.0)
        _, ref = p.observe()
        assert ref['gt_cross_track_road_frame_at_car_m'] == pytest.approx(0.0, abs=1e-6)
        assert ref['gt_cross_track_lookahead_m'] == pytest.approx(-0.5 * GT_LOOKAHEAD_M ** 2 * 0.010, rel=0.05)
        assert ref['heading'] < 0.0, "path bends left → path heading in vehicle frame is negative (+ = right)"

    def test_right_arc_is_the_exact_mirror(self):
        pl = UnityFramePlant(Path(+0.005, straight_m=0.0), speed=11.6); pr = UnityFramePlant(Path(-0.005, straight_m=0.0), speed=11.6)
        for p in (pl, pr):
            p.pos = p.path.point(30.0); p.theta = p.path.heading(30.0)
        _, a = pl.observe(); _, b = pr.observe()
        for k in ('x', 'heading', 'curvature', 'gt_cross_track_lookahead_m', 'gt_cross_track_at_car_m',
                  'gt_cross_track_road_frame_at_car_m', 'gt_heading_error_rad'):
            assert float(a[k]) == pytest.approx(-float(b[k]), abs=1e-9), k

    def test_car_right_of_centre_reads_positive_road_frame_and_negative_centre_offset(self):
        p = UnityFramePlant(Path(0.0), speed=10.0, d0_right=+0.3)
        _, ref = p.observe()
        assert ref['gt_cross_track_road_frame_at_car_m'] == pytest.approx(+0.3)
        assert ref['gt_cross_track_at_car_m'] == pytest.approx(-0.3)
        assert ref['gt_cross_track_lookahead_m'] == pytest.approx(-0.3)

    def test_positive_command_turns_right_and_unity_yaw_rate_is_negative_on_left_turns(self):
        p = UnityFramePlant(Path(0.0), speed=10.0)
        for _ in range(40):
            p.step(-0.1)                                   # negative command = left (hill: −0.081 on left arcs)
        state, ref = p.observe()
        assert p.yaw_rate_ccw > 0 and state['angular_velocity_y'] < 0
        assert ref['gt_cross_track_road_frame_at_car_m'] < 0, "turning left moves the car LEFT of a straight centre"

    def test_steady_yaw_rate_matches_kappa_v_for_the_recorded_command(self):
        """hill: command −0.081 at 6.6 m/s on κ=0.010 → |ω| ≈ κ·v = 0.066 rad/s (recorded 0.061)."""
        p = UnityFramePlant(Path(0.0), speed=6.6)
        for _ in range(100):
            p.step(-0.081)
        assert abs(p.yaw_rate_ccw) == pytest.approx(0.066, rel=0.25)


# ===========================================================================
# 2. Mirror symmetry of the LMPC wrapper: u(−x) must equal −u(x)
# ===========================================================================

def _wrapper_sequence(sign: float, speed: float = 6.6, kappa: float = 0.010, n: int = 60, entry: int = 10) -> np.ndarray:
    c = MPCController(_production_cfg())
    out, last, dact = [], 0.0, 0.0
    e, psi = 0.28 * sign, 0.01 * sign
    for k in range(n):
        kap = kappa * sign if k >= entry else 0.0
        r = c.compute_steering(e_lat=e, e_heading=psi, current_speed=speed, last_delta_norm=last, kappa_ref=kap,
                               v_target=12.0, v_max=15.0, dt=FRAME_DT, delta_actual_norm=dact)
        u = float(r['steering_normalized']); out.append(u); last = u; dact = 0.8 * dact + 0.2 * u
    return np.array(out)


class TestLMPCMirrorSymmetry:

    def test_steady_state_is_symmetric(self):
        a, b = _wrapper_sequence(+1.0), _wrapper_sequence(-1.0)
        # Tolerance 1e-5, not 1e-6: OSQP is an iterative solver and its last-digit result shifts with the
        # BLAS backend and under xdist worker subprocesses (feedback_mpc_tests_xdist_flaky, feedback_solver_
        # cold_start_determinism). Serial runs sit ~1e-9; a 2026-10-12 full-suite xdist run measured 3.1e-6.
        # 1e-5 still catches a real sign/gain asymmetry (those are O(1e-1)) without flagging solver jitter.
        assert float(np.max(np.abs(a[30:] + b[30:]))) < 1e-5

    @pytest.mark.xfail(strict=True, reason=(
        "kappa_transition_warmup fires on `kappa_ref - last > threshold` (mpc_controller.py:1768), a SIGNED increase, "
        "so it only triggers when entering a LEFT arc: the left-arc command steps 0.14 toward zero for ~5 frames at "
        "entry, the right-arc command does not (asymmetry 0.076, measured 2026-10-11)."))
    def test_arc_entry_is_symmetric(self):
        a, b = _wrapper_sequence(+1.0), _wrapper_sequence(-1.0)
        assert float(np.max(np.abs(a + b))) < 1e-6


# ===========================================================================
# 3. Feedforward direction: zero error on a LEFT arc must command LEFT (negative)
# ===========================================================================

class TestLMPCFeedforwardDirection:

    @staticmethod
    def _zero_error_command(kappa, speed, sign=None):
        cfg = _production_cfg() if sign is None else _production_cfg(mpc_curvature_sign=sign)
        c = MPCController(cfg); u = 0.0
        for _ in range(8):
            r = c.compute_steering(e_lat=0.0, e_heading=0.0, current_speed=speed, last_delta_norm=u,
                                   kappa_ref=kappa, v_target=12.0, v_max=15.0, dt=FRAME_DT, delta_actual_norm=u)
            u = float(r['steering_normalized'])
        return u

    @pytest.mark.parametrize("kappa,speed", [(+0.010, 6.6), (+0.005, 11.6)])
    def test_zero_error_feedforward_points_into_the_left_arc(self, kappa, speed):
        """Production config (mpc_curvature_sign promoted to -1.0 on 2026-10-11): on an orchestrator LEFT arc
        the zero-error command must be negative, like the recorded -0.081 on hill's left arcs."""
        u = self._zero_error_command(kappa, speed)
        assert u < 0.0, f"zero-error command {u:+.4f} on a left arc must be negative (left)"

    @pytest.mark.parametrize("kappa,speed", [(+0.010, 6.6), (+0.005, 11.6)])
    def test_legacy_sign_pointed_the_feedforward_out_of_the_arc(self, kappa, speed):
        """The defect on record: with curvature_sign 1.0 the LMPC heading row e_heading += steer_gain*delta - kappa*v*dt
        makes +kappa demand +delta (RIGHT) while the orchestrator's +kappa is a LEFT turn — the kinematic feedforward
        opposed every arc (+0.03 to +0.045 for kappa=+0.005 at every speed) and the feedback carried the curve. The NMPC
        path had negated kappa at dispatch since April; the LMPC path never did. Kill-switch: mpc_curvature_sign 1.0."""
        u = self._zero_error_command(kappa, speed, sign=1.0)
        assert u > 0.0, f"legacy zero-error command {u:+.4f} should point out of the left arc (documented defect)"

    def test_feedforward_is_mirror_symmetric(self):
        def ff(k):
            c = MPCController(_production_cfg())
            return float(c.compute_steering(e_lat=0.0, e_heading=0.0, current_speed=6.6, last_delta_norm=0.0, kappa_ref=k,
                                            v_target=12.0, v_max=15.0, dt=FRAME_DT, delta_actual_norm=0.0)['steering_normalized'])
        assert ff(+0.010) == pytest.approx(-ff(-0.010), abs=1e-9)


# ===========================================================================
# 4. Closed loop through the production dispatch, both directions, in Unity's frame
# ===========================================================================

ARCS = [
    pytest.param(+0.010, 6.6, id="hill_R100_left_6.6"),
    pytest.param(-0.010, 6.6, id="R100_right_6.6"),
    pytest.param(-0.005, 11.6, id="mixed_R200_right_11.6"),
    pytest.param(+0.005, 11.6, id="R200_left_11.6"),
    pytest.param(+0.002, 12.0, id="highway_R500_left_12"),
    pytest.param(-0.005, 13.5, id="R200_right_13.5"),
]


class TestClosedLoopUnityFrame:

    @pytest.mark.parametrize("kappa,speed", ARCS)
    def test_production_lmpc_stays_in_lane_on_both_arc_directions(self, kappa, speed):
        r = run_closed_loop(kappa, speed)
        steady = _arc_frames(speed)
        assert (r['regime'][steady] == int(ControlRegime.LINEAR_MPC)).all(), f"regimes {np.unique(r['regime'][steady])}"
        assert float(np.max(np.abs(r['d']))) < 0.60, f"left the lane: max |d| {np.max(np.abs(r['d'])):.2f} m"
        assert abs(float(np.mean(r['d'][steady]))) < 0.30, f"steady offset {np.mean(r['d'][steady]):+.3f} m"

    def test_legacy_sign_parked_0p3m_outside_at_r200_13p5(self):
        """Speed-margin reproducer on record (T-LATERAL-LMPC-SPEED-MARGIN): with the legacy sign the production LMPC
        parks 0.31 m outside at R200/13.5 m/s (0.19 at 11.6, 0.11 at R500/12 — Unity 0.102). The promoted sign
        halves these (0.16 / 0.07 / 0.05) but does NOT restore the dynamic margin at 11.6 on the real stack
        (Unity 2026-10-11: 3 OOL in 4 flipped runs vs 7 in 5 legacy; the 0.42 Hz mode grows 1.6-3x in both)."""
        r = run_closed_loop(-0.005, 13.5, cfg=_production_cfg(mpc_curvature_sign=1.0))
        assert abs(float(np.mean(r['d'][_arc_frames(13.5)]))) > 0.28

    def test_straight_from_offset_converges_without_sign_reversal(self):
        r = run_closed_loop(0.0, 10.0, n_frames=200, d0_right=+0.3)
        assert r['u'][0] < 0.0, "car right of centre must steer LEFT (negative) first"
        assert abs(float(r['d'][-1])) < 0.05 and float(np.max(np.abs(r['d']))) < 0.35

    @pytest.mark.parametrize("kappa,speed", [(0.010, 6.6), (0.005, 11.6)])
    def test_left_and_right_arcs_reach_mirrored_steady_offsets(self, kappa, speed):
        a, b = run_closed_loop(+kappa, speed), run_closed_loop(-kappa, speed)
        steady = _arc_frames(speed)
        assert float(np.mean(a['d'][steady])) == pytest.approx(-float(np.mean(b['d'][steady])), abs=0.02)

    @pytest.mark.parametrize("direction", [+1.0, -1.0], ids=["left", "right"])
    def test_regime_zero_path_steers_toward_the_reference_on_a_tight_arc(self, direction):
        """κ=±0.067 (R15) > mpc_max_map_curvature 0.015 → regime 0 after the 30-frame hold. Sign only: the
        actuator gain here is identified for 6–13 m/s and under-delivers at 4 m/s (the plant cannot hold R15 at
        the 0.7·max_steer command cap), so convergence is not asserted — the direction of the reaction is."""
        r = run_closed_loop(direction / 15.0, 4.0, n_frames=260)
        on_arc = r['regime'] == int(ControlRegime.PURE_PURSUIT)
        assert on_arc.any(), f"regime never left LINEAR_MPC: {np.unique(r['regime'])}"
        u_pp = r['u'][on_arc]
        # Assert the net reaction, not every frame. A per-frame `np.all(sign == ...)` over a >1e-3 deadband is
        # brittle: one near-zero command whose last digit flips under an xdist worker (the OSQP determinism
        # class) failed this as a REAL_BREAK on 2026-10-12 though it passed every serial re-run. The physical
        # claim is "the regime-0 path steers INTO the arc" — so check the mean command is clearly signed the
        # right way and that no CLEARLY-large command (>5e-3) points the wrong way.
        assert np.sign(np.mean(u_pp)) == -direction and abs(np.mean(u_pp)) > 5e-3, \
            f"regime-0 path must steer into the arc: mean command {np.mean(u_pp):+.4f}, expected sign {-direction:+.0f}"
        wrong = u_pp[np.sign(u_pp) == direction]
        assert not np.any(np.abs(wrong) > 5e-3), \
            f"a clearly-large command pointed OUT of the arc: {wrong[np.abs(wrong) > 5e-3]}"


# ===========================================================================
# 5. Two writers, one sign: the camera-frame dispatch and the inter-frame extrapolation
# ===========================================================================

class TestInterframePathSharesTheCurvatureSign:
    """The orchestrator runs the LMPC twice per camera cycle: once through compute_control (camera frame) and
    up to three times through compute_interframe_steering (fresh GT state, orchestrator curvature). On
    2026-10-11 `mpc_curvature_sign: -1` was applied at the camera-frame call site only: the two writers alternated
    opposite-sign commands into Unity (recorded `vehicle/steering_input` flipped sign every frame on hill while
    the camera-frame command sat at −0.105), and the hill arm parked 0.4–0.9 m outside. That Unity A/B was
    invalid, not a refutation. The sign now lives in MPCParams.curvature_sign; this pins both paths to it."""

    @staticmethod
    def _kappa_used_by_both_paths(sign: float, kappa: float = 0.010, speed: float = 6.6):
        from control import mpc_controller as mc
        seen = []
        orig = mc.MPCController.compute_steering
        def spy(self, *a, **k):
            r = orig(self, *a, **k); seen.append(float(r.get('kappa_ref_used'))); return r
        mc.MPCController.compute_steering = spy
        try:
            vc = VehicleController(full_config=_production_cfg(mpc_curvature_sign=sign))
            vc.longitudinal_controller.max_speed = TRACK_SPEED_LIMIT_MPS
            plant = UnityFramePlant(Path(kappa, straight_m=0.0), speed)
            plant.pos = plant.path.point(30.0); plant.theta = plant.path.heading(30.0)
            state, ref = plant.observe()
            for _ in range(3):
                vc.compute_control(state, ref, return_metadata=True)
            camera = seen[-1]
            seen.clear()
            r = vc.compute_interframe_steering(e_lat=0.0, e_heading=0.0, speed=speed, curvature=kappa, dt=0.033)
            assert r is not None, "inter-frame path must run while the regime is LINEAR_MPC"
            interframe = seen[-1]
        finally:
            mc.MPCController.compute_steering = orig
        return camera, interframe

    @pytest.mark.parametrize("sign", [1.0, -1.0], ids=["legacy", "flipped"])
    def test_both_paths_hand_the_solver_the_same_signed_curvature(self, sign):
        camera, interframe = self._kappa_used_by_both_paths(sign)
        assert camera == pytest.approx(sign * 0.010, abs=1e-9)
        assert interframe == pytest.approx(sign * 0.010, abs=1e-9)

    def test_interframe_command_agrees_in_sign_with_the_camera_frame_command(self):
        """Same state, same sign: the two writers must not fight (left arc → both negative)."""
        vc = VehicleController(full_config=_production_cfg())
        vc.longitudinal_controller.max_speed = TRACK_SPEED_LIMIT_MPS
        plant = UnityFramePlant(Path(0.010, straight_m=0.0), 6.6, d0_right=0.05)
        plant.pos = plant.path.point(30.0) + 0.05 * np.array([0.0, -1.0]); plant.theta = plant.path.heading(30.0)
        out = None
        for _ in range(3):
            state, ref = plant.observe(); out = vc.compute_control(state, ref, return_metadata=True)
        inter = vc.compute_interframe_steering(e_lat=float(out['mpc_e_lat']), e_heading=float(out['mpc_e_heading']),
                                               speed=6.6, curvature=0.010, dt=0.033)
        assert out['steering'] < 0 and inter['steering'] < 0, (out['steering'], inter['steering'])


# ===========================================================================
# 6. Harness calibration against Unity (known-good / known-bad rule)
# ===========================================================================

class TestHarnessCalibration:
    """A harness that cannot reproduce Unity's ranking is missing a mechanism — diff the plant, do not trust it.
    Legacy sign, production config, steady at-car offset: highway R500 @12 Unity +0.094 m / harness +0.107;
    hill R100 @6.6 Unity +0.05–0.08 (command −0.081) / harness +0.033 (−0.096); mixed R200 @11.6 Unity 0.33–0.49 RMSE
    and 2/2 out-of-lane with the lateral budget on / harness 0.19 m steady (the kinematic plant has no inter-frame
    writer and no perception noise, so it under-states the margin loss). The flipped-sign Unity arms of 2026-10-11
    are NOT calibration points: they flipped one writer of two (TestInterframePathSharesTheCurvatureSign)."""

    @staticmethod
    def _steady(kappa, speed, sign):
        r = run_closed_loop(kappa, speed, cfg=_production_cfg(mpc_curvature_sign=sign))
        return float(np.mean(np.abs(r['d'][_arc_frames(speed)]))), float(np.max(np.abs(r['d'])))

    @pytest.mark.parametrize("kappa,speed,sign,unity_signed", [
        pytest.param(+0.002, 12.0, 1.0, +0.102, id="highway_R500_left_12-legacy"),
        pytest.param(+0.002, 12.0, -1.0, +0.048, id="highway_R500_left_12-flipped"),
        pytest.param(+0.010, 6.6, 1.0, +0.050, id="hill_R100_left_6.6-legacy"),
        pytest.param(+0.010, 6.6, -1.0, -0.070, id="hill_R100_left_6.6-flipped"),
    ])
    def test_steady_offset_matches_the_valid_unity_ab_within_0p04m(self, kappa, speed, sign, unity_signed):
        """Valid A/B 2026-10-11 20:25–21:04 (sign in MPCParams, 5 pairs/track, steady arc frames): highway
        +0.102 → +0.048 m, hill +0.050 → −0.070 m. Harness: +0.107 → +0.048, +0.033 → −0.082."""
        harness = self._steady_signed(kappa, speed, sign)
        assert harness == pytest.approx(unity_signed, abs=0.04), f"harness {harness:+.3f} vs Unity {unity_signed:+.3f}"

    @staticmethod
    def _steady_signed(kappa, speed, sign):
        r = run_closed_loop(kappa, speed, cfg=_production_cfg(mpc_curvature_sign=sign))
        return float(np.mean(r['d'][_arc_frames(speed)]))

    @pytest.mark.parametrize("kappa,speed,legacy_m,flipped_m", [
        pytest.param(+0.010, 6.6, +0.033, -0.082, id="hill_R100_left_6.6"),
        pytest.param(+0.010, 9.8, +0.253, +0.052, id="R100_left_9.8"),
        pytest.param(-0.005, 11.6, -0.192, -0.073, id="mixed_R200_right_11.6"),
        pytest.param(+0.002, 12.0, +0.107, +0.048, id="highway_R500_left_12"),
    ])
    def test_flipping_the_sign_for_both_writers_moves_the_car_inward_on_every_arc(self, kappa, speed, legacy_m, flipped_m):
        """What a VALID Unity A/B (both writers flipped) must show. The flip always moves the car toward the
        inside of the arc; it shrinks |offset| from 9.8 m/s up but OVERSHOOTS inside at 6.6 m/s (0.033 out →
        0.082 in): the lookahead reference `−gt_cross_track_lookahead` carries a −½Ld²κ bias (0.18 m at R100)
        that the wrong-way feedforward happened to cancel. The sign is half of the curve-offset story; the
        reference bias is the other half. If Unity disagrees with these numbers, diff the plant
        (feedback_harness_sensor_frame_and_ab_loop) — do not tune."""
        inward = -np.sign(kappa)                      # left arc: inward = negative (left) in +right coordinates
        legacy, flipped = self._steady_signed(kappa, speed, 1.0), self._steady_signed(kappa, speed, -1.0)
        assert legacy == pytest.approx(legacy_m, abs=0.02) and flipped == pytest.approx(flipped_m, abs=0.02), (legacy, flipped)
        assert (flipped - legacy) * inward > 0.04, "flip must move the car inward"
        if speed >= 9.0:
            assert abs(flipped) < abs(legacy)
