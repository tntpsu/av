"""Config YAML schema validation tests.

These tests catch silent mis-configurations that cause wrong behaviour at
runtime without raising an error.  All discovered by past bugs:

  - Non-monotone speed tables         → wrong lookahead interpolation
  - camera_height wrong value         → systematic lateral error on curves
  - target_lane='right' on sloop     → 0.22 m persistent lateral bias
  - pp_curve_local_floor_state_min   → wrong enum caused no-floor on curves
  - tight_scale out of (0, 1]        → negative or >1 lookahead scaling
  - mpc cost weights ≤ 0             → degenerate QP (slow or infeasible)

All config files are validated: the base config plus every track overlay.
Track overlays are also validated merged with the base so that overrides
do not produce an inconsistent combined configuration.
"""

import pytest
import yaml
from pathlib import Path


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_BASE_CONFIG = _CONFIG_DIR / "av_stack_config.yaml"

# All overlay files that are layered on top of the base config
_OVERLAY_FILES = [
    p for p in _CONFIG_DIR.glob("*.yaml")
    if p != _BASE_CONFIG
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _get(cfg: dict, *keys, default=None):
    """Nested key lookup with default."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _assert_speed_table_monotone(table, name: str) -> None:
    """Speed table entries must have strictly increasing speed_mps values."""
    if not table:
        return
    speeds = [float(e["speed_mps"]) for e in table]
    for i in range(1, len(speeds)):
        assert speeds[i] > speeds[i - 1], (
            f"{name}: speed_mps values not strictly increasing at index {i}: "
            f"{speeds[i - 1]:.3f} → {speeds[i]:.3f}"
        )


def _assert_lookahead_table_positive(table, name: str) -> None:
    """All lookahead_m values must be positive."""
    for e in (table or []):
        v = float(e.get("lookahead_m", 0))
        assert v > 0, f"{name}: lookahead_m={v} must be > 0"


def _configs_to_validate():
    """Return (label, merged_cfg) pairs for base + each overlay."""
    base = _load(_BASE_CONFIG)
    pairs = [("base", base)]
    for overlay_path in sorted(_OVERLAY_FILES):
        overlay = _load(overlay_path)
        merged  = _deep_merge(base, overlay)
        pairs.append((overlay_path.name, merged))
    return pairs


# ---------------------------------------------------------------------------
# Speed table monotonicity
# ---------------------------------------------------------------------------

class TestSpeedTableMonotonicity:

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_reference_lookahead_speed_table_monotone(self, label, cfg):
        table = _get(cfg, "trajectory", "reference_lookahead_speed_table")
        _assert_speed_table_monotone(
            table, f"[{label}] trajectory.reference_lookahead_speed_table"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_reference_lookahead_speed_table_entry_monotone(self, label, cfg):
        table = _get(cfg, "trajectory", "reference_lookahead_speed_table_entry")
        _assert_speed_table_monotone(
            table, f"[{label}] trajectory.reference_lookahead_speed_table_entry"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_reference_lookahead_speed_table_commit_monotone(self, label, cfg):
        table = _get(cfg, "trajectory", "reference_lookahead_speed_table_commit")
        _assert_speed_table_monotone(
            table, f"[{label}] trajectory.reference_lookahead_speed_table_commit"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_pp_floor_speed_table_monotone(self, label, cfg):
        table = _get(cfg, "control", "lateral", "pp_curve_local_lookahead_floor_speed_table")
        _assert_speed_table_monotone(
            table, f"[{label}] control.lateral.pp_curve_local_lookahead_floor_speed_table"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_all_lookahead_values_positive(self, label, cfg):
        for key in ("reference_lookahead_speed_table",
                    "reference_lookahead_speed_table_entry",
                    "reference_lookahead_speed_table_commit"):
            table = _get(cfg, "trajectory", key)
            _assert_lookahead_table_positive(
                table, f"[{label}] trajectory.{key}"
            )


# ---------------------------------------------------------------------------
# Enum / string parameters
# ---------------------------------------------------------------------------

_VALID_FLOOR_STATES = {"STRAIGHT", "ENTRY", "COMMIT"}
_VALID_TARGET_LANES = {"center", "left", "right"}
_VALID_CONTROL_MODES = {"pure_pursuit", "pid", "stanley", "mpc"}
_VALID_CURVE_SCHEDULERS = {"phase_active", "phase_shadow", "binary", "legacy_scale"}


class TestEnumParameters:

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_pp_curve_floor_state_min_is_valid(self, label, cfg):
        val = _get(cfg, "control", "lateral", "pp_curve_local_floor_state_min")
        if val is None:
            return
        assert str(val).upper() in _VALID_FLOOR_STATES, (
            f"[{label}] pp_curve_local_floor_state_min='{val}' "
            f"must be one of {_VALID_FLOOR_STATES}"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_target_lane_is_valid(self, label, cfg):
        val = _get(cfg, "trajectory", "target_lane")
        if val is None:
            return
        assert str(val).lower() in _VALID_TARGET_LANES, (
            f"[{label}] trajectory.target_lane='{val}' "
            f"must be one of {_VALID_TARGET_LANES}"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_control_mode_is_valid(self, label, cfg):
        val = _get(cfg, "control", "lateral", "control_mode")
        if val is None:
            return
        assert str(val).lower() in _VALID_CONTROL_MODES, (
            f"[{label}] control.lateral.control_mode='{val}' "
            f"must be one of {_VALID_CONTROL_MODES}"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_curve_scheduler_mode_is_valid(self, label, cfg):
        val = _get(cfg, "trajectory", "curve_scheduler_mode")
        if val is None:
            return
        assert str(val).lower() in _VALID_CURVE_SCHEDULERS, (
            f"[{label}] trajectory.curve_scheduler_mode='{val}' "
            f"must be one of {_VALID_CURVE_SCHEDULERS}"
        )


# ---------------------------------------------------------------------------
# Numeric bounds
# ---------------------------------------------------------------------------

class TestNumericBounds:

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_tight_scale_in_bounds(self, label, cfg):
        val = _get(cfg, "trajectory", "reference_lookahead_tight_scale")
        if val is None:
            return
        assert 0.0 < float(val) <= 1.0, (
            f"[{label}] reference_lookahead_tight_scale={val} must be in (0, 1]"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_camera_height_positive(self, label, cfg):
        val = _get(cfg, "trajectory", "camera_height")
        if val is None:
            return
        assert float(val) > 0, (
            f"[{label}] trajectory.camera_height={val} must be positive"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_camera_height_plausible(self, label, cfg):
        """Camera height outside 0.5–2.5 m is almost certainly misconfigured."""
        val = _get(cfg, "trajectory", "camera_height")
        if val is None:
            return
        assert 0.5 <= float(val) <= 2.5, (
            f"[{label}] camera_height={val} m is outside plausible range [0.5, 2.5]. "
            f"The physical camera sits at 1.2 m."
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_mpc_cost_weights_positive(self, label, cfg):
        for key in ("mpc_q_lat", "mpc_q_heading", "mpc_r_steer", "mpc_r_steer_rate"):
            val = _get(cfg, "trajectory", "mpc", key)
            if val is None:
                continue
            assert float(val) > 0, (
                f"[{label}] trajectory.mpc.{key}={val} must be > 0 "
                f"(zero or negative breaks the QP)"
            )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_mpc_horizon_positive_integer(self, label, cfg):
        val = _get(cfg, "trajectory", "mpc", "mpc_horizon")
        if val is None:
            return
        assert int(val) >= 5, (
            f"[{label}] mpc_horizon={val} must be ≥ 5 (shorter horizon is unstable)"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_pp_max_steering_jerk_positive(self, label, cfg):
        val = _get(cfg, "control", "lateral", "pp_max_steering_jerk")
        if val is None:
            return
        assert float(val) > 0, (
            f"[{label}] pp_max_steering_jerk={val} must be > 0"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_reference_lookahead_min_positive(self, label, cfg):
        val = _get(cfg, "trajectory", "reference_lookahead_min")
        if val is None:
            return
        assert float(val) > 0, (
            f"[{label}] reference_lookahead_min={val} must be > 0"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_heading_gate_on_less_than_off(self, label, cfg):
        """The heading gate ON threshold must be less than the OFF threshold
        for the hysteresis to work correctly.  If ON >= OFF the gate can never
        latch or can never release — both are bugs.
        """
        on_val  = _get(cfg, "trajectory", "traj_heading_zero_gate_heading_on_abs_rad")
        off_val = _get(cfg, "trajectory", "traj_heading_zero_gate_heading_off_abs_rad")
        if on_val is None or off_val is None:
            return
        assert float(on_val) < float(off_val), (
            f"[{label}] heading gate ON={on_val} must be < OFF={off_val} "
            f"for hysteresis to work"
        )


# ---------------------------------------------------------------------------
# Cross-parameter consistency
# ---------------------------------------------------------------------------

class TestCrossParameterConsistency:

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_lookahead_min_less_than_base(self, label, cfg):
        base_lh = _get(cfg, "trajectory", "reference_lookahead")
        min_lh  = _get(cfg, "trajectory", "reference_lookahead_min")
        if base_lh is None or min_lh is None:
            return
        assert float(min_lh) < float(base_lh), (
            f"[{label}] reference_lookahead_min={min_lh} must be < "
            f"reference_lookahead={base_lh}"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_curvature_threshold_ordering(self, label, cfg):
        """reference_lookahead_curvature_min < curvature_max."""
        c_min = _get(cfg, "trajectory", "reference_lookahead_curvature_min")
        c_max = _get(cfg, "trajectory", "reference_lookahead_curvature_max")
        if c_min is None or c_max is None:
            return
        assert float(c_min) < float(c_max), (
            f"[{label}] curvature_min={c_min} must be < curvature_max={c_max}"
        )

    @pytest.mark.parametrize("label,cfg", _configs_to_validate())
    def test_mpc_dt_times_horizon_reasonable(self, label, cfg):
        """MPC prediction window (horizon × dt) must be between 0.5 s and 5 s."""
        horizon = _get(cfg, "trajectory", "mpc", "mpc_horizon")
        dt      = _get(cfg, "trajectory", "mpc", "mpc_dt")
        if horizon is None or dt is None:
            return
        window = int(horizon) * float(dt)
        assert 0.5 <= window <= 5.0, (
            f"[{label}] MPC prediction window={window:.2f} s "
            f"(horizon={horizon} × dt={dt}) is outside [0.5, 5.0] s"
        )

    def test_base_config_target_lane_is_center(self):
        """Production regression guard: base config must use target_lane='center'.
        Reverted to 'right' once — caused 0.22 m lateral bias on sloop.
        """
        cfg = _load(_BASE_CONFIG)
        val = _get(cfg, "trajectory", "target_lane")
        assert val == "center", (
            f"av_stack_config.yaml target_lane='{val}' must be 'center'. "
            f"Changing to 'right' caused a 0.22 m lateral bias on the sloop track."
        )

    def test_floor_state_min_is_entry_in_base(self):
        """Production regression guard: pp_curve_local_floor_state_min='ENTRY'.
        Was 'COMMIT' — caused steering jerk discontinuity at curve entry.
        """
        cfg = _load(_BASE_CONFIG)
        val = _get(cfg, "control", "lateral", "pp_curve_local_floor_state_min")
        assert str(val).upper() == "ENTRY", (
            f"pp_curve_local_floor_state_min='{val}' must be 'ENTRY'. "
            f"Using 'COMMIT' caused a steering jerk spike at curve entry."
        )
