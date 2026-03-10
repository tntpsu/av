import logging
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ControlConfig:
    """Control configuration parameters."""
    lateral_kp: float = 0.3
    lateral_ki: float = 0.0
    lateral_kd: float = 0.1
    lateral_max_steering: float = 0.5
    lateral_deadband: float = 0.02
    lateral_heading_weight: float = 0.5
    lateral_lateral_weight: float = 0.5
    lateral_error_clip: float = np.pi / 4
    lateral_integral_limit: float = 0.3

    longitudinal_kp: float = 0.3
    longitudinal_ki: float = 0.05
    longitudinal_kd: float = 0.02
    longitudinal_target_speed: float = 8.0
    longitudinal_max_speed: float = 10.0
    longitudinal_speed_smoothing: float = 0.7
    longitudinal_speed_deadband: float = 0.1
    longitudinal_throttle_limit_threshold: float = 0.8
    longitudinal_throttle_reduction_factor: float = 0.3
    longitudinal_brake_aggression: float = 3.0


@dataclass
class TrajectoryConfig:
    """Trajectory planning configuration parameters."""
    lookahead_distance: float = 20.0
    point_spacing: float = 1.0
    target_speed: float = 8.0
    reference_lookahead: float = 8.0
    image_width: float = 640.0
    image_height: float = 480.0
    camera_fov: float = 75.0
    camera_height: float = 1.2
    bias_correction_threshold: float = 10.0


@dataclass
class SafetyConfig:
    """Safety configuration parameters."""
    max_speed: float = 10.0
    emergency_brake_threshold: float = 2.0
    speed_prevention_threshold: float = 0.85
    speed_prevention_brake_threshold: float = 0.9
    speed_prevention_brake_amount: float = 0.2
    lane_width: float = 7.0  # Lane width in meters
    car_width: float = 1.85  # Car width in meters
    allowed_outside_lane: float = 1.0  # Allowed distance outside lane before emergency stop (meters)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base.

    Dicts are merged recursively so that nested keys in base that are absent
    from overlay are preserved.  Lists and all other types are replaced in
    full by the overlay value — speed tables and similar sequences must
    replace the base list, not extend it.
    """
    result = dict(base)
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration, deep-merging an optional overlay over the base config.

    Always loads av_stack_config.yaml as the base so that every parameter has
    a canonical default.  If *config_path* is provided and differs from the
    base, its values are merged on top (overlay wins on any key it specifies,
    base fills in everything else).
    """
    base_path = Path(__file__).parent.parent / "config" / "av_stack_config.yaml"

    config: dict = {}
    if base_path.exists():
        with open(base_path, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Loaded base config from %s", base_path)
    else:
        logger.warning("Base config not found at %s", base_path)

    if config_path is None:
        return config

    overlay_path = Path(config_path)
    if overlay_path.resolve() == base_path.resolve():
        return config  # overlay IS the base — nothing to merge

    if not overlay_path.exists():
        logger.warning("Config overlay not found at %s — using base only", overlay_path)
        return config

    with open(overlay_path, "r") as f:
        overlay = yaml.safe_load(f) or {}
    logger.info("Merged config overlay from %s", overlay_path)
    return _deep_merge(config, overlay)
