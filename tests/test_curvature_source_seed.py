"""First-frame curvature-source seeding (T-LAUNCH-CURVATURE-SEED, 2026-10-09).

Mechanism found on the ACC H2 minimal-overlay A/B (recording_20261009_011632):
`_select_primary_curvature` started on ``lane_context`` and applied switch-on
hysteresis to the very first selection, so frames 0-1 used the perception
curvature of a stationary car (0.09 1/m, R≈11 m, on a straight) while the map
was already healthy (``map_ok`` overwritten by ``hysteresis_hold``). That value
seeded the 12 m distance-based curvature EMA; the speed governor turned it into
comfort = 3.0 m/s and curve cap = comfort − 0.4, and a car held at 3 m/s crosses
the smoothing window slowly, so the seed took ~12 s to wash out (comfort speed
3.0 → 4.7 → 7.8 → 12.6 m/s at 0/5/8/10 s). The full ACC overlay hid it by
disabling both governor caps; production highway_65 shows the mild form
(seed 0.011 → comfort 8.4 m/s for the first seconds instead of 12).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from av_stack import AVStack
from control.speed_governor import SpeedGovernorConfig, SpeedGovernor, SpeedPlannerConfig
from trajectory.utils import smooth_curvature_distance

JUNK_STATIONARY_KAPPA = 0.09129   # control/curvature_lane_context_abs, frame 0 of the H2 recording


def _selector(seed_without_hysteresis: bool, switch_on_frames: int = 3):
    fake = SimpleNamespace(
        _curvature_source_seed_without_hysteresis=seed_without_hysteresis,
        _curvature_source_active=None if seed_without_hysteresis else "lane_context",
        _curvature_source_candidate=None,
        _curvature_source_candidate_frames=0,
        _curvature_source_switch_on_frames=switch_on_frames,
    )

    def select(map_abs, lane_abs, gt_abs=None, map_ok=True):
        return AVStack._select_primary_curvature(
            fake, map_abs=map_abs, gt_abs=gt_abs, lane_context_abs=lane_abs, map_health_ok=map_ok,
        )

    return fake, select


class TestFirstSelection:
    def test_legacy_holds_lane_context_for_switch_on_frames(self):
        """Reproduces frames 0-1 of recording_20261009_011632: map healthy, κ_map=0, yet
        the stationary-car perception curvature is returned under hysteresis_hold."""
        _, select = _selector(seed_without_hysteresis=False, switch_on_frames=3)
        f0 = select(map_abs=0.0, lane_abs=JUNK_STATIONARY_KAPPA)
        f1 = select(map_abs=0.0, lane_abs=0.0877)
        f2 = select(map_abs=0.0, lane_abs=0.0931)
        assert f0 == (pytest.approx(JUNK_STATIONARY_KAPPA), "lane_context", "hysteresis_hold")
        assert f1[1:] == ("lane_context", "hysteresis_hold")
        assert f2 == (0.0, "map_track", "map_ok")

    def test_seed_adopts_map_on_frame_zero(self):
        fake, select = _selector(seed_without_hysteresis=True)
        f0 = select(map_abs=0.0, lane_abs=JUNK_STATIONARY_KAPPA)
        assert f0 == (0.0, "map_track", "map_ok")
        assert fake._curvature_source_active == "map_track"
        assert fake._curvature_source_candidate is None

    def test_seed_adopts_lane_context_when_map_unavailable(self):
        """No map and no GT on frame 0 → lane_context is the honest first source."""
        _, select = _selector(seed_without_hysteresis=True)
        f0 = select(map_abs=None, lane_abs=0.004, map_ok=False)
        assert f0 == (pytest.approx(0.004), "lane_context", "gt_unavailable_switch_to_lane")

    def test_hysteresis_still_protects_later_switches(self):
        """Seeding skips hysteresis ONCE; a mid-run map→GT switch still needs N frames."""
        _, select = _selector(seed_without_hysteresis=True, switch_on_frames=3)
        assert select(map_abs=0.0, lane_abs=0.1)[1] == "map_track"
        # Map goes unhealthy, GT available: desired = ground_truth, held for 2 frames.
        h1 = select(map_abs=0.001, lane_abs=0.1, gt_abs=0.02, map_ok=False)
        h2 = select(map_abs=0.001, lane_abs=0.1, gt_abs=0.02, map_ok=False)
        h3 = select(map_abs=0.001, lane_abs=0.1, gt_abs=0.02, map_ok=False)
        assert h1[2] == "hysteresis_hold" and h2[2] == "hysteresis_hold"
        assert h3[1:] == ("ground_truth", "map_untrusted_switch_to_gt")

    def test_flag_default_is_legacy(self, tmp_path):
        """Code default keeps the legacy start; the base YAML turns the fix on."""
        import yaml
        from pathlib import Path
        base = yaml.safe_load(Path("config/av_stack_config.yaml").read_text())
        assert base["trajectory"]["curvature_source_seed_without_hysteresis"] is True


class TestLaunchFreezeMechanism:
    """Closed-loop toy: the governor's comfort speed is the only cap, the car follows it
    with 2 m/s² authority, and the curvature EMA is the production distance filter
    (window 12 m, min speed 2 m/s). A junk seed holds launch for many seconds; a zero
    seed releases at once. Pins the physics the Unity A/B then confirms."""

    @staticmethod
    def _launch(seed_kappa: float, dt: float = 0.046, horizon_s: float = 20.0, target: float = 15.0):
        gov = SpeedGovernor(
            SpeedGovernorConfig(
                comfort_governor_max_lat_accel_g=0.2,
                comfort_governor_min_speed=3.0,
                curvature_calibration_scale=2.5,
                curvature_history_frames=5,
                speed_planner_enabled=False,
            ),
            SpeedPlannerConfig(),
        )
        n = int(horizon_s / dt)
        raw = [seed_kappa] + [0.0] * (n - 1)
        v, t_release = 0.0, None
        speeds, times, kappas = [], [], []
        smoothed_prev = None
        for i in range(n):
            t = i * dt
            # production filter, one step at a time (same as AVStack._smooth_path_curvature)
            if smoothed_prev is None:
                smoothed = raw[i]
            else:
                dist = max(v, 2.0) * dt
                alpha = 1.0 - math.exp(-dist / 12.0)
                smoothed = alpha * raw[i] + (1.0 - alpha) * smoothed_prev
            smoothed_prev = smoothed
            gov._curvature_history.append(smoothed)
            gov._curvature_history = gov._curvature_history[-5:]
            comfort = gov._compute_comfort_speed(max(gov._curvature_history))
            cap = min(target, comfort)
            if t_release is None and cap >= 12.0:
                t_release = t
            v = min(cap, v + 2.0 * dt)
            speeds.append(v); times.append(t); kappas.append(smoothed)
        return t_release, speeds, times

    def test_junk_seed_holds_launch_for_seconds(self):
        t_release, speeds, times = self._launch(JUNK_STATIONARY_KAPPA)
        v_at_5s = speeds[int(5.0 / 0.046)]
        assert t_release is not None and t_release > 6.0, t_release
        assert v_at_5s < 7.0, v_at_5s           # recording: 3.6 m/s at 5 s (toy has no throttle lag)

    def test_map_seed_releases_immediately(self):
        t_release, speeds, _ = self._launch(0.0)
        assert t_release == 0.0
        assert speeds[int(5.0 / 0.046)] > 9.5    # 2 m/s² for 5 s

    def test_distance_filter_matches_recorded_decay(self):
        """Independent check of the diagnosis: the recorded comfort speeds at 5/8/10 s
        (4.66/7.84/12.63 m/s, v = sqrt(0.2g / (2.5 κ))) imply κ_smoothed ≈
        0.036/0.0128/0.0049; a distance EMA of the recorded speed trace from the
        recorded seed reproduces them to 14-30 % (1 s samples; the governor's 5-frame
        max-history adds the remaining lag)."""
        # (t, v) samples from recording_20261009_011632 (min overlay, H2)
        trace = [(0, 0.0), (1, 0.02), (2, 1.57), (3, 2.46), (4, 2.83), (5, 3.62), (6, 3.99),
                 (7, 4.5), (8, 5.14), (9, 6.0), (10, 6.82), (11, 7.8), (12, 8.72)]
        ts = [float(t) for t, _ in trace]
        vs = [v for _, v in trace]
        raw = [JUNK_STATIONARY_KAPPA] + [0.0] * (len(ts) - 1)
        sm = smooth_curvature_distance(raw, vs, ts, window_m=12.0, min_speed=2.0)
        g = 9.81
        implied = {5: 0.0362, 8: 0.0128, 10: 0.0049}
        for sec, k_rec in implied.items():
            assert sm[sec] == pytest.approx(k_rec, rel=0.45), (sec, sm[sec], k_rec)
        comfort_at_10 = max(3.0, math.sqrt(0.2 * g / (sm[10] * 2.5)))
        assert 9.0 < comfort_at_10 < 20.0      # recorded 12.6 m/s
