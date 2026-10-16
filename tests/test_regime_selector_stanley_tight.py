"""Tight-curve Stanley routing (2026-10-15 tight-curve controller research).

Stanley takes exactly the slot Pure Pursuit holds today — the LMPC curvature ceiling (map κ ≥ mpc_max_map_curvature,
preview-fed) — while below `stanley_tight_curve_max_speed_mps`. Everything else (low-speed Stanley with its curve
inhibit, LMPC default, NMPC guards) is untouched. Kill-switch `stanley_tight_curve_routing`, default False = legacy PP.
"""
from __future__ import annotations

from control.regime_selector import ControlRegime, RegimeConfig, RegimeSelector


def _sel(routing: bool, **over) -> RegimeSelector:
    kw = dict(enabled=True, stanley_enabled=True, stanley_max_speed_mps=2.0, stanley_inhibit_curvature_abs=0.004,
              mpc_min_speed_absolute_mps=2.0, mpc_max_map_curvature=0.015, nmpc_curvature_threshold=1.0,
              lmpc_max_speed_mps=20.0, min_hold_frames=5, blend_frames=5, stanley_min_hold_frames=5, stanley_blend_frames=5,
              stanley_tight_curve_routing=routing, stanley_tight_curve_max_speed_mps=9.0)
    kw.update(over)
    return RegimeSelector(RegimeConfig(**kw))


def _run(sel, speed, kappa, n=30):
    r = None
    for _ in range(n):
        r, _ = sel.update(speed=speed, curvature_abs=kappa, map_curvature_abs=kappa)
    return r


class TestLegacyUnchanged:
    def test_flag_off_keeps_pp_on_tight_curves(self):
        assert _run(_sel(False), 5.3, 0.025) == ControlRegime.PURE_PURSUIT      # s_loop R40
        assert _run(_sel(False), 4.0, 1 / 15) == ControlRegime.PURE_PURSUIT     # hairpin R15

    def test_gentle_curves_stay_lmpc_either_way(self):
        for flag in (False, True):
            assert _run(_sel(flag), 11.6, 0.005) == ControlRegime.LINEAR_MPC    # mixed_radius R200
            assert _run(_sel(flag), 15.0, 0.0) == ControlRegime.LINEAR_MPC      # highway straight

    def test_low_speed_stanley_path_unchanged(self):
        # launch straight below 2 m/s → low-speed Stanley (legacy path), independent of the new flag
        for flag in (False, True):
            assert _run(_sel(flag), 1.0, 0.0) == ControlRegime.STANLEY
            # the curve inhibit on the low-speed path still blocks it: slow on a highway curve → not Stanley via that path
            assert _run(_sel(flag, stanley_tight_curve_routing=False), 1.5, 0.01) != ControlRegime.STANLEY


class TestTightCurveRouting:
    def test_flag_on_routes_tight_curves_to_stanley(self):
        assert _run(_sel(True), 5.3, 0.025) == ControlRegime.STANLEY             # s_loop R40 at its cruise speed
        assert _run(_sel(True), 4.0, 1 / 15) == ControlRegime.STANLEY            # hairpin R15
        assert _run(_sel(True), 8.4, 0.025) == ControlRegime.STANLEY             # s_loop's fastest frame

    def test_above_the_tight_curve_speed_cap_falls_back_to_pp(self):
        assert _run(_sel(True), 12.0, 0.025) == ControlRegime.PURE_PURSUIT

    def test_requires_stanley_enabled(self):
        assert _run(_sel(True, stanley_enabled=False), 5.3, 0.025) == ControlRegime.PURE_PURSUIT

    def test_exits_to_lmpc_when_the_curve_ends(self):
        sel = _sel(True)
        assert _run(sel, 5.3, 0.025) == ControlRegime.STANLEY
        assert _run(sel, 6.0, 0.0) == ControlRegime.LINEAR_MPC

    def test_boundary_is_the_same_as_pp_used(self):
        # just under the ceiling → LMPC; at the ceiling → Stanley (PP in legacy)
        assert _run(_sel(True), 5.0, 0.0149) == ControlRegime.LINEAR_MPC
        assert _run(_sel(True), 5.0, 0.015) == ControlRegime.STANLEY
        assert _run(_sel(False), 5.0, 0.015) == ControlRegime.PURE_PURSUIT
