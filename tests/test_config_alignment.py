"""
S1-M39: Config alignment tests. No av_stack import (avoids cv2); uses yaml directly.
Ensures speed planner limits do not exceed longitudinal controller limits for smooth chase.
"""

import pytest
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
CONFIG_PATH = project_root / 'config' / 'av_stack_config.yaml'


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f) or {}


class TestLongitudinalPlannerAlignment:
    """Ensure speed planner limits do not exceed longitudinal controller limits."""

    def test_planner_accel_not_exceeds_longitudinal(self):
        """Planner max_accel should be <= longitudinal max_accel for smooth chase."""
        config = _load_config()
        if not config:
            pytest.skip('av_stack_config.yaml not found')
        long_cfg = config.get('control', {}).get('longitudinal', {})
        speed_planner_cfg = config.get('trajectory', {}).get('speed_planner', {})
        if not speed_planner_cfg.get('enabled', False):
            pytest.skip('speed_planner not enabled')
        planner_max_accel = float(speed_planner_cfg.get('max_accel', 2.0))
        long_max_accel = float(long_cfg.get('max_accel', 1.2))
        assert planner_max_accel <= long_max_accel + 0.01, (
            f'Planner max_accel {planner_max_accel} > longitudinal {long_max_accel}'
        )

    def test_planner_jerk_not_exceeds_longitudinal_dynamic_max(self):
        """Planner max_jerk should be <= longitudinal max_jerk_max for smooth chase."""
        config = _load_config()
        if not config:
            pytest.skip('av_stack_config.yaml not found')
        long_cfg = config.get('control', {}).get('longitudinal', {})
        speed_planner_cfg = config.get('trajectory', {}).get('speed_planner', {})
        if not speed_planner_cfg.get('enabled', False):
            pytest.skip('speed_planner not enabled')
        planner_max_jerk = float(speed_planner_cfg.get('max_jerk', 1.2))
        long_max_jerk = float(long_cfg.get('max_jerk_max', long_cfg.get('max_jerk', 6.0)))
        assert planner_max_jerk <= long_max_jerk + 0.01, (
            f'Planner max_jerk {planner_max_jerk} > longitudinal max_jerk_max {long_max_jerk}'
        )
