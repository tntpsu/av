"""Scenario overlays must not re-enable what the base config disables for stability.

2026-10-07 (T-ACC-G1-LEFF-OSCILLATION): `config/mpc_hill_highway.yaml` kept
`mpc_leff_estimation_enabled: true` for six months after the base turned the RLS wheelbase
estimator off as unstable; `acc_hill_highway.yaml` inherited it, so every hill ACC
scenario ran an estimator that pegged L_eff at 8 m in the arcs and drove the LMPC into a
±0.70 steering limit cycle. The lateral sweep never saw it (no overlay). This test loads
every overlay through the real loader (with `_inherits`) and pins the keys the base marks
as stability-gated.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from av_stack.config import load_config  # noqa: E402

# key path → value the base pins for stability (and why)
STABILITY_GATED = {
    ("trajectory", "mpc", "mpc_leff_estimation_enabled"): (False, "RLS L_eff estimator unstable on curve transitions (base 8a5c0c5); pegged at 8 m on G1"),
    ("safety", "emergency_stop_use_gt_lane_boundaries"): (True, "a scenario overlay must not disable the off-road e-stop (acc_highway.yaml had it off since March 2026; re-enabled 2026-10-09)"),
}

OVERLAYS = sorted(p for p in glob.glob(str(REPO_ROOT / "config" / "*.yaml")) if Path(p).name != "av_stack_config.yaml")


def _get(cfg: dict, path: tuple):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda p: Path(p).name)
@pytest.mark.parametrize("path,expected", [(k, v[0]) for k, v in STABILITY_GATED.items()], ids=lambda x: ".".join(x) if isinstance(x, tuple) else str(x))
def test_overlay_keeps_base_stability_setting(overlay, path, expected):
    merged = load_config(overlay)
    actual = _get(merged, path)
    reason = STABILITY_GATED[path][1]
    assert actual == expected, f"{Path(overlay).name}: {'.'.join(path)} = {actual!r}, base pins {expected!r} — {reason}"


def test_base_itself_pins_the_values():
    base = load_config(None)
    for path, (expected, _) in STABILITY_GATED.items():
        assert _get(base, path) == expected
