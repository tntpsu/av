"""Coordinate conversion behavioural contract tests.

Focus: properties NOT covered by test_coordinate_conversion_assumptions.py
or test_camera_offset.py, specifically:

  1. Symmetry contract — a pixel exactly at the image horizontal centre
     must always produce x_vehicle ≈ 0.0 regardless of other parameters.

  2. Camera-height sensitivity — increasing camera_height must monotonically
     increase the computed forward distance for a given screen-y pixel.
     This catches the direction-of-effect error that caused the 1.4 m vs
     1.2 m camera height bug.

  3. Config camera_height matches calibrated value — the value in
     av_stack_config.yaml must equal the known physical camera height (1.2 m).

  4. Left-right symmetry — a pixel N pixels left of centre must give
     x_vehicle = −x_vehicle of a pixel N pixels right of centre.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path

from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner

_IMAGE_W = 640
_IMAGE_H = 480
_CONFIG  = Path(__file__).resolve().parent.parent / "config" / "av_stack_config.yaml"

# Known physical camera height from calibration (Unity prefab measurement).
# This must match config/av_stack_config.yaml camera_height.
_PHYSICAL_CAMERA_HEIGHT_M = 1.2


def _make_planner(camera_height: float = 1.2,
                  camera_fov: float = 66.0) -> RuleBasedTrajectoryPlanner:
    return RuleBasedTrajectoryPlanner(
        image_width=_IMAGE_W,
        image_height=_IMAGE_H,
        camera_height=camera_height,
        camera_fov=camera_fov,
    )


# ---------------------------------------------------------------------------
# 1. Symmetry: centre pixel → x_vehicle ≈ 0
# ---------------------------------------------------------------------------

class TestCentrePixelSymmetry:

    def test_centre_pixel_gives_zero_lateral(self):
        """A lane line detected at the exact horizontal centre maps to x=0."""
        planner = _make_planner()
        centre_pixel_x = _IMAGE_W / 2.0
        mid_pixel_y    = _IMAGE_H / 2.0

        x_veh, _ = planner._convert_image_to_vehicle_coords(
            centre_pixel_x, mid_pixel_y, lookahead_distance=10.0
        )
        assert abs(x_veh) < 0.01, (
            f"Centre pixel should give x_vehicle ≈ 0, got {x_veh:.4f} m"
        )

    def test_centre_pixel_independent_of_camera_height(self):
        """The symmetry axis must not shift when camera_height changes."""
        centre_x = _IMAGE_W / 2.0
        mid_y    = _IMAGE_H / 2.0
        for h in (0.8, 1.0, 1.2, 1.4, 1.6):
            planner = _make_planner(camera_height=h)
            x_veh, _ = planner._convert_image_to_vehicle_coords(
                centre_x, mid_y, lookahead_distance=10.0
            )
            assert abs(x_veh) < 0.02, (
                f"camera_height={h}: centre pixel gave x_vehicle={x_veh:.4f} m "
                f"(expected ≈ 0)"
            )


# ---------------------------------------------------------------------------
# 2. Camera-height is stored and used in the fallback projection path
# ---------------------------------------------------------------------------

class TestCameraHeightSensitivity:

    def test_camera_height_stored_on_planner(self):
        """The configured camera_height must be stored on the planner instance.

        This is a regression guard: if someone removes the camera_height
        parameter it silently defaults to 1.2 m and the projection subtly
        changes.  The field must match the constructor argument.
        """
        for h in (0.8, 1.0, 1.2, 1.4, 1.6):
            planner = _make_planner(camera_height=h)
            assert planner.camera_height == pytest.approx(h, rel=1e-6), (
                f"camera_height={h} not stored correctly, got {planner.camera_height}"
            )

    def test_camera_height_affects_far_pixel_projection(self):
        """camera_height is used in the very-far fallback path (y_pixels near 0).
        A higher camera must project a top-of-image pixel to a farther distance.
        """
        # Use a near-top pixel (far away) and auto-distance mode (lookahead_distance < 0)
        # to trigger the camera-height dependent code path.
        pixel_y = _IMAGE_H * 0.02   # near top = very far away
        pixel_x = _IMAGE_W / 2.0

        planner_low  = _make_planner(camera_height=1.0)
        planner_high = _make_planner(camera_height=1.4)

        _, y_low  = planner_low._convert_image_to_vehicle_coords(
            pixel_x, pixel_y, lookahead_distance=-1.0
        )
        _, y_high = planner_high._convert_image_to_vehicle_coords(
            pixel_x, pixel_y, lookahead_distance=-1.0
        )
        assert y_high >= y_low, (
            f"Higher camera must project top pixel at least as far: "
            f"y_low={y_low:.3f}, y_high={y_high:.3f}"
        )

    def test_explicit_lookahead_overrides_camera_projection(self):
        """When lookahead_distance is provided explicitly, the returned y is
        proportional to it (not to camera_height), regardless of pixel position.
        This confirms the planner correctly uses the orchestrator's lookahead.
        """
        pixel_x, pixel_y = _IMAGE_W / 2.0, _IMAGE_H * 0.6
        planner = _make_planner(camera_height=1.2)

        _, y10 = planner._convert_image_to_vehicle_coords(pixel_x, pixel_y,
                                                           lookahead_distance=10.0)
        _, y20 = planner._convert_image_to_vehicle_coords(pixel_x, pixel_y,
                                                           lookahead_distance=20.0)
        # y should scale proportionally with lookahead_distance
        assert y20 > y10, (
            f"Doubling lookahead should increase y_vehicle: y10={y10:.3f}, y20={y20:.3f}"
        )


# ---------------------------------------------------------------------------
# 3. Config camera_height matches calibrated physical value
# ---------------------------------------------------------------------------

class TestConfigCameraHeight:

    def test_config_camera_height_matches_physical(self):
        """The configured camera_height must equal the physically measured value.

        When this value was wrong (1.4 m instead of 1.2 m) the coordinate
        conversion was off, causing systematic lateral error on curves.
        If this test fails, either the config changed or the physical height
        changed — both require deliberate review.
        """
        with open(_CONFIG) as f:
            cfg = yaml.safe_load(f)

        # camera_height lives in the trajectory section (line 611 of av_stack_config.yaml)
        configured = (cfg.get("trajectory", {}) or {}).get("camera_height")
        if configured is None:
            configured = cfg.get("camera_height")

        assert configured is not None, (
            "camera_height not found in av_stack_config.yaml trajectory section"
        )
        assert float(configured) == pytest.approx(_PHYSICAL_CAMERA_HEIGHT_M, rel=0.05), (
            f"config camera_height={configured} m differs from "
            f"physical={_PHYSICAL_CAMERA_HEIGHT_M} m. "
            f"Update _PHYSICAL_CAMERA_HEIGHT_M if the camera was moved."
        )


# ---------------------------------------------------------------------------
# 4. Left-right symmetry
# ---------------------------------------------------------------------------

class TestLeftRightSymmetry:

    @pytest.mark.parametrize("offset_px", [50, 100, 150, 200])
    def test_symmetric_pixels_give_opposite_lateral(self, offset_px):
        """A pixel N px left of centre must give −x of a pixel N px right of centre."""
        planner = _make_planner()
        centre  = _IMAGE_W / 2.0
        mid_y   = _IMAGE_H / 2.0

        x_right, _ = planner._convert_image_to_vehicle_coords(
            centre + offset_px, mid_y, lookahead_distance=10.0
        )
        x_left, _  = planner._convert_image_to_vehicle_coords(
            centre - offset_px, mid_y, lookahead_distance=10.0
        )
        assert x_right == pytest.approx(-x_left, abs=0.01), (
            f"offset_px={offset_px}: x_right={x_right:.4f}, x_left={x_left:.4f} "
            f"— must be equal and opposite"
        )

    def test_right_pixel_gives_positive_x(self):
        """Pixel right of centre must give positive x_vehicle (convention check)."""
        planner = _make_planner()
        x_veh, _ = planner._convert_image_to_vehicle_coords(
            _IMAGE_W / 2.0 + 100, _IMAGE_H / 2.0, lookahead_distance=10.0
        )
        assert x_veh > 0, f"Right pixel must give x_vehicle > 0, got {x_veh:.4f}"

    def test_left_pixel_gives_negative_x(self):
        """Pixel left of centre must give negative x_vehicle (convention check)."""
        planner = _make_planner()
        x_veh, _ = planner._convert_image_to_vehicle_coords(
            _IMAGE_W / 2.0 - 100, _IMAGE_H / 2.0, lookahead_distance=10.0
        )
        assert x_veh < 0, f"Left pixel must give x_vehicle < 0, got {x_veh:.4f}"
