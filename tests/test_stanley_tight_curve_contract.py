"""Stanley tight-curve contract (tight-curve controller research, Phase 1, 2026-10-15).

Two implementation defects were found while preparing Stanley for the 3–9 m/s tight-curve band, both behind
kill-switches with legacy defaults:

1. **Cross-track sign (source-aware).** Stanley's call site assumes the LANE-CENTRE convention
   (`gt_cross_track > 0 → lane centre right of car → steer right`). Since 2026-09-28 the first-priority source is
   `groundTruthSelectedLaneCrossTrackRoadFrameAtCar`, the CAR's offset (+right) — the exact negative: on the
   2026-10-15 s_loop run corr(lane_center_x_at_car, road_frame_at_car) = −1.000 over 993 frames. With that
   source Stanley's cross-track feedback is inverted (car right → steer right). `stanley_cross_track_source_aware`
   negates the road-frame source only; the lane-centre sources are untouched.
2. **Output units.** The law's radian sum (heading + arctan cross-track + feedforward) was clipped to ±1 and
   handed on as a NORMALIZED command, then scaled by max_steering (0.7) — ≈0.37 of the intended wheel angle,
   and the "feedforward" −atan(L·κ) was therefore not physical. Pure Pursuit normalizes by radians(30°);
   `stanley_output_units: radians` does the same.

The kinematic harness is NOT a verdict on tight-curve numbers (it mis-models PP 3× at R40 and diverges at R15);
it is used here only in its calibrated gentle regime (R200) to show the sign fix turns a runaway into a hold.
"""
from __future__ import annotations

import numpy as np
import pytest

from control.pid_controller import LateralController, STANLEY_MAX_STEER_RAD
from control.regime_selector import ControlRegime

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from test_lateral_sign_contract import run_closed_loop, _production_cfg, _arc_frames  # noqa: E402


def _lc(**kw) -> LateralController:
    lc = LateralController.__new__(LateralController)
    lc.stanley_k = kw.get('k', 1.5)
    lc.stanley_soft_speed = kw.get('soft', 2.0)
    lc.stanley_heading_weight = kw.get('w', 1.0)
    lc.stanley_curvature_ff_enabled = kw.get('ff', False)
    lc.stanley_wheelbase_m = kw.get('L', 2.7)
    lc.stanley_ff_slew_rate_rad = kw.get('slew', 10.0)   # effectively no slew unless a test sets it
    lc.stanley_output_units = kw.get('units', 'normalized_legacy')
    lc.stanley_cross_track_source_aware = kw.get('aware', False)
    lc._prev_stanley_ff_term = 0.0
    return lc


class TestOutputUnits:
    def test_legacy_passes_the_radian_sum_through_as_normalized(self):
        r = _lc(units='normalized_legacy').compute_stanley_from_errors(0.0, 0.10, 5.0)
        assert r['steering_normalized'] == pytest.approx(0.10)                 # the defect: 0.10 rad → "0.10 normalized"

    def test_radians_mode_normalizes_by_the_same_30deg_pure_pursuit_uses(self):
        r = _lc(units='radians').compute_stanley_from_errors(0.0, 0.10, 5.0)
        assert r['steering_normalized'] == pytest.approx(0.10 / np.radians(30.0))
        assert STANLEY_MAX_STEER_RAD == pytest.approx(np.radians(30.0))

    def test_radians_mode_still_clips_to_unit(self):
        r = _lc(units='radians').compute_stanley_from_errors(5.0, 1.0, 2.0)
        assert r['steering_normalized'] == 1.0


class TestFeedforwardIsPhysicalAndSigned:
    def test_left_curve_feedforward_steers_left_with_atan_L_kappa(self):
        r = _lc(ff=True, units='radians').compute_stanley_from_errors(0.0, 0.0, 3.9, curvature=+1 / 15)
        assert r['stanley_curvature_ff_term'] == pytest.approx(-np.arctan(2.7 / 15))   # κ>0 = left → negative = left
        assert r['steering_normalized'] == pytest.approx(-np.arctan(2.7 / 15) / np.radians(30.0))

    def test_feedforward_is_mirror_symmetric(self):
        a = _lc(ff=True, units='radians').compute_stanley_from_errors(0.0, 0.0, 3.9, curvature=+1 / 15)
        b = _lc(ff=True, units='radians').compute_stanley_from_errors(0.0, 0.0, 3.9, curvature=-1 / 15)
        assert a['steering_normalized'] == pytest.approx(-b['steering_normalized'], abs=1e-12)

    def test_feedforward_off_contributes_nothing(self):
        r = _lc(ff=False, units='radians').compute_stanley_from_errors(0.0, 0.0, 3.9, curvature=+1 / 15)
        assert r['stanley_curvature_ff_term'] == 0.0 and r['steering_normalized'] == 0.0


def _stanley_cfg(aware: bool, units: str, ff: bool, cap: float = 9.0) -> dict:
    cfg = _production_cfg()
    reg = cfg['control']['regime']; reg['stanley_max_speed_mps'] = cap; reg['stanley_inhibit_curvature_abs'] = 0.0
    lat = cfg['control']['lateral']
    lat.update({'stanley_cross_track_source_aware': aware, 'stanley_output_units': units,
                'stanley_curvature_ff_enabled': ff})
    return cfg


class TestCrossTrackSignInTheClosedLoop:
    """The harness plant publishes the car offset under source code 'road_frame_at_car' — the production
    semantics. Straight-line regulation from a 0.5 m right offset at 8 m/s isolates the sign from curvature and
    feedforward; R200 at 8 m/s is inside the harness's calibrated regime (±0.02 m on steady offset)."""

    def test_legacy_sign_alone_is_positive_feedback_documenting_the_defect(self):
        # heading weight 0 → pure cross-track feedback: inverted sign must drive the car away, not back.
        c = _stanley_cfg(aware=False, units='radians', ff=False); c['control']['lateral']['stanley_heading_weight'] = 0.0
        r = run_closed_loop(0.0, 8.0, n_frames=400, d0_right=0.5, cfg=c)
        assert np.abs(r['d']).max() > 2.0

    def test_legacy_sign_with_heading_term_still_drifts_away(self):
        r = run_closed_loop(0.0, 8.0, n_frames=400, d0_right=0.5, cfg=_stanley_cfg(aware=False, units='radians', ff=False))
        assert r['d'][-1] > 1.0                                  # ends further right than it started

    def test_source_aware_sign_returns_the_car_to_centre(self):
        r = run_closed_loop(0.0, 8.0, n_frames=400, d0_right=0.5, cfg=_stanley_cfg(aware=True, units='radians', ff=False))
        sl = _arc_frames(8.0)
        assert (r['regime'][sl] == ControlRegime.STANLEY.value).all()
        assert np.abs(r['d']).max() <= 0.5 + 1e-9                # never worse than the initial offset
        assert abs(float(r['d'][-1])) < 0.3

    def test_source_aware_stanley_holds_r200_in_the_calibrated_regime(self):
        r = run_closed_loop(0.005, 8.0, n_frames=500, cfg=_stanley_cfg(aware=True, units='radians', ff=False))
        assert np.abs(r['d']).max() < 0.25                       # observed 0.08 m (2026-10-15)

    def test_source_aware_sign_is_mirror_symmetric(self):
        a = run_closed_loop(+0.005, 8.0, n_frames=500, cfg=_stanley_cfg(True, 'radians', True))['d']
        b = run_closed_loop(-0.005, 8.0, n_frames=500, cfg=_stanley_cfg(True, 'radians', True))['d']
        assert np.abs(a + b).max() < 1e-6


class TestKillSwitchDefaults:
    def test_production_config_runs_the_promoted_law(self):
        cfg = _production_cfg()
        lat = cfg['control']['lateral']
        # PROMOTED 2026-10-16 (option B) — production runs the corrected law; legacy values remain kill-switch positions.
        assert lat.get('stanley_cross_track_source_aware') is True
        assert lat.get('stanley_output_units') == 'radians_speed_map'   # PROMOTED 2026-10-17 (steer-command interface)


def _stanley_cfg_h(source: str) -> dict:
    cfg = _stanley_cfg(aware=True, units='radians', ff=False)
    cfg['control']['lateral']['stanley_heading_source'] = source
    return cfg


class TestHeadingSource:
    """PP's heading_error is the pursuit angle to the lookahead point and contains cross-track (s_loop 2026-10-15:
    corr +0.59 with the car offset, −0.68 with the true heading error) — Stanley's two terms fight (0.4 Hz weave,
    Control 87–98). 'gt_path_tangent' feeds the orchestrator's headingDeltaDeg (+ = car pointing right), negated
    into Stanley's steer-right-positive convention."""

    def test_production_uses_the_path_tangent_heading(self):
        assert _production_cfg()['control']['lateral'].get('stanley_heading_source') == 'gt_path_tangent'   # PROMOTED 2026-10-16

    def test_gt_heading_source_regulates_with_less_overshoot_than_the_pursuit_angle(self):
        pp = run_closed_loop(0.0, 8.0, n_frames=400, d0_right=0.5, cfg=_stanley_cfg_h('pp_heading_to_ref'))['d']
        gt = run_closed_loop(0.0, 8.0, n_frames=400, d0_right=0.5, cfg=_stanley_cfg_h('gt_path_tangent'))['d']
        assert np.abs(gt).max() <= 0.5 + 1e-9 and abs(float(gt[-1])) < 0.2          # converges
        assert np.abs(gt[60:]).max() < np.abs(pp[60:]).max()                          # less overshoot after the first swing

    def test_gt_heading_source_holds_r200(self):
        r = run_closed_loop(0.005, 8.0, n_frames=500, cfg=_stanley_cfg_h('gt_path_tangent'))
        assert np.abs(r['d']).max() < 0.25

    def test_gt_heading_source_is_mirror_symmetric(self):
        a = run_closed_loop(+0.005, 8.0, n_frames=500, cfg=_stanley_cfg_h('gt_path_tangent'))['d']
        b = run_closed_loop(-0.005, 8.0, n_frames=500, cfg=_stanley_cfg_h('gt_path_tangent'))['d']
        assert np.abs(a + b).max() < 1e-6


class TestFeedforwardPlantGainCompensation:
    """The plant delivers ≈0.78 of a commanded steer (closed-loop sysid 2026-10-11). Uncompensated, the arc is held
    by a persistent cross-track error (Phase 1b s_loop −0.124 m, hairpin ≈0.2 m); dividing the FF by the measured
    ratio lets feedforward hold the arc."""

    def test_production_gain_constant_is_kept_only_for_the_legacy_radians_mode(self):
        lat = _production_cfg()['control']['lateral']
        assert lat.get('stanley_ff_plant_gain') == 0.78 and lat.get('stanley_output_units') == 'radians_speed_map'   # ignored in production (the map replaces it)

    def test_ff_is_divided_by_the_plant_gain(self):
        lc = _lc(ff=True, units='radians'); lc.stanley_ff_plant_gain = 0.78
        r = lc.compute_stanley_from_errors(0.0, 0.0, 3.9, curvature=+1 / 15)
        assert r['stanley_curvature_ff_term'] == pytest.approx(-np.arctan(2.7 / 15) / 0.78)

    def test_compensated_ff_keeps_mirror_symmetry(self):
        a = _lc(ff=True, units='radians'); a.stanley_ff_plant_gain = 0.78
        b = _lc(ff=True, units='radians'); b.stanley_ff_plant_gain = 0.78
        assert a.compute_stanley_from_errors(0, 0, 3.9, +1 / 15)['steering_normalized'] == pytest.approx(
            -b.compute_stanley_from_errors(0, 0, 3.9, -1 / 15)['steering_normalized'], abs=1e-12)


class TestFeedforwardLagPreview:
    """The at-car FF cannot anticipate an S-reversal; the plant answers ≈0.35 s late (τ 0.25 s + 2 frames). With
    `stanley_ff_preview_tau_s` the FF curvature is read v·τ ahead on the map horizon. On a constant-κ arc the horizon
    is flat, so the hold and mirror results must be unchanged; the reversal benefit is a Unity verdict."""

    def test_production_preview_is_the_measured_plant_lag(self):
        assert _production_cfg()['control']['lateral'].get('stanley_ff_preview_tau_s') == 0.35   # PROMOTED 2026-10-16

    def test_constant_arc_hold_unchanged_with_preview(self):
        cfg = _stanley_cfg_h('gt_path_tangent'); cfg['control']['lateral'].update({'stanley_ff_preview_tau_s': 0.35, 'stanley_curvature_ff_enabled': True, 'stanley_ff_curvature_source': 'map_horizon'})
        r = run_closed_loop(0.005, 8.0, n_frames=500, cfg=cfg)
        assert np.abs(r['d']).max() < 0.25

    def test_preview_is_mirror_symmetric(self):
        cfg = _stanley_cfg_h('gt_path_tangent'); cfg['control']['lateral'].update({'stanley_ff_preview_tau_s': 0.35, 'stanley_curvature_ff_enabled': True, 'stanley_ff_curvature_source': 'map_horizon'})
        a = run_closed_loop(+0.005, 8.0, n_frames=500, cfg=cfg)['d']; b = run_closed_loop(-0.005, 8.0, n_frames=500, cfg=cfg)['d']
        assert np.abs(a + b).max() < 1e-6


class TestFeedforwardCurvatureSource:
    """Two signed curvatures live in one reference_point with OPPOSITE conventions (2026-10-15, s_loop recordings):
    reference_point['curvature'] (+right: sign from the reference heading; 1.00 agreement with groundTruthPathCurvature)
    and curvature_horizon_signed (+left: track-profile direction_sign = −1 for 'right'). The FF formula −atan(L·κ) is
    written for +left. Legacy source 'reference_point' therefore steers OUT of the curve — Phase 1c arm B got worse
    with more FF gain; arm C ('map_horizon') scored s_loop Traj 95.4 / Control 100. The harness plant emits both
    fields from ONE path and cannot see this; the verdict is the recordings."""

    def test_production_ff_reads_the_map_horizon(self):
        assert _production_cfg()['control']['lateral'].get('stanley_ff_curvature_source') == 'map_horizon'   # PROMOTED 2026-10-16

    def test_map_horizon_source_at_zero_tau_matches_at_car_on_a_constant_arc(self):
        cfg = _stanley_cfg_h('gt_path_tangent'); cfg['control']['lateral'].update({'stanley_curvature_ff_enabled': True, 'stanley_ff_curvature_source': 'map_horizon', 'stanley_ff_preview_tau_s': 0.0})
        r = run_closed_loop(0.005, 8.0, n_frames=500, cfg=cfg)
        assert np.abs(r['d']).max() < 0.25


from control.steer_interface import unity_max_steer_rad  # noqa: E402


class TestSpeedMapNormalization:
    """'radians_speed_map' (T-STANLEY-FF-SPEED-MAP): the law's radians are normalized by the plant's max steer AT THIS
    SPEED (Unity Lerp 30°→16° over 0→12 m/s) so the delivered wheel angle equals the law at every speed; the fixed-30°
    modes under-deliver by r(v) and needed stanley_ff_plant_gain 0.78 — right at s_loop's 5.3 m/s (ideal 1.26),
    over-steering hairpin at 3.4 m/s (ideal 1.14). The constant is ignored in this mode."""

    def test_scales_the_radians_mode_by_30deg_over_max_at_speed(self):
        a = _lc(units='radians').compute_stanley_from_errors(0.0, 0.10, 5.3)
        b = _lc(units='radians_speed_map').compute_stanley_from_errors(0.0, 0.10, 5.3)
        assert b['steering_normalized'] == pytest.approx(a['steering_normalized'] * np.radians(30.0) / unity_max_steer_rad(5.3))
        assert b['stanley_max_steer_rad_used'] == pytest.approx(unity_max_steer_rad(5.3))

    def test_reproduces_the_promoted_gain_at_s_loop_speed_and_a_smaller_one_at_hairpin_speed(self):
        assert np.radians(30.0) / unity_max_steer_rad(5.2) == pytest.approx(1.26, abs=0.02)   # ≈ today's 1/0.78 = 1.28
        assert np.radians(30.0) / unity_max_steer_rad(3.4) == pytest.approx(1.15, abs=0.02)   # < 1.28: hairpin over-steer gone

    def test_plant_reported_max_steer_is_preferred(self):
        r = _lc(units='radians_speed_map').compute_stanley_from_errors(0.0, 0.10, 5.3, max_steer_rad_at_speed=np.radians(20.0))
        assert r['stanley_max_steer_rad_used'] == pytest.approx(np.radians(20.0))
        assert r['steering_normalized'] == pytest.approx(0.10 / np.radians(20.0))

    def test_ff_plant_gain_is_ignored_in_speed_map_mode(self):
        a = _lc(ff=True, units='radians_speed_map'); a.stanley_ff_plant_gain = 0.78
        b = _lc(ff=True, units='radians_speed_map'); b.stanley_ff_plant_gain = 1.0
        ra = a.compute_stanley_from_errors(0, 0, 3.4, curvature=+1 / 15); rb = b.compute_stanley_from_errors(0, 0, 3.4, curvature=+1 / 15)
        assert ra['stanley_curvature_ff_term'] == pytest.approx(rb['stanley_curvature_ff_term']) == pytest.approx(-np.arctan(2.7 / 15))

    def test_legacy_modes_are_bit_exact(self):
        for units in ('normalized_legacy', 'radians'):
            lc = _lc(ff=True, units=units); lc.stanley_ff_plant_gain = 0.78
            r = lc.compute_stanley_from_errors(0.1, 0.05, 4.0, curvature=+1 / 15)
            assert r['stanley_curvature_ff_term'] == pytest.approx(-np.arctan(2.7 / 15) / 0.78)
            assert np.isnan(r['stanley_max_steer_rad_used'])

    def test_r200_hold_in_the_calibrated_regime(self):
        cfg = _stanley_cfg_h('gt_path_tangent'); cfg['control']['lateral'].update({'stanley_output_units': 'radians_speed_map', 'stanley_curvature_ff_enabled': True, 'stanley_ff_curvature_source': 'map_horizon', 'stanley_max_steer_source': 'mirror'})
        r = run_closed_loop(0.005, 8.0, n_frames=500, cfg=cfg)
        assert np.abs(r['d']).max() < 0.25
