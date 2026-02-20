"""
Trajectory planning module.
Takes lane lines from perception and generates desired trajectory.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Single point in trajectory."""
    x: float
    y: float
    heading: float  # radians
    velocity: float  # m/s
    curvature: float  # 1/m


@dataclass
class Trajectory:
    """Complete trajectory."""
    points: List[TrajectoryPoint]
    length: float  # total length in meters


class RuleBasedTrajectoryPlanner:
    """
    Rule-based trajectory planner.
    Follows center of detected lanes.
    """
    
    def __init__(self, lookahead_distance: float = 50.0, 
                 point_spacing: float = 1.0, target_speed: float = 10.0,
                 image_width: float = 640.0, image_height: float = 480.0,
                 camera_fov: float = 75.0, camera_height: float = 1.2,
                 lookahead_pixels: float = 200.0, bias_correction_threshold: float = 10.0,
                 reference_smoothing: float = 0.7, lane_smoothing_alpha: float = 0.7,
                 camera_offset_x: float = 0.0, distance_scaling_factor: float = 0.875,
                 center_spline_enabled: bool = False, center_spline_degree: int = 2,
                 center_spline_samples: int = 20, center_spline_alpha: float = 0.7,
                 x_clip_enabled: bool = True, x_clip_limit_m: float = 10.0,
                 trajectory_eval_y_min_ratio: float = 0.0,
                 centerline_midpoint_mode: str = "pointwise",
                 projection_model: str = "legacy",
                 calibrated_fx_px: float = 0.0,
                 calibrated_fy_px: float = 0.0,
                 calibrated_cx_px: float = 0.0,
                 calibrated_cy_px: float = 0.0,
                 dynamic_effective_horizon_farfield_scale_min: float = 0.85,
                 far_band_contribution_cap_enabled: bool = True,
                 far_band_contribution_cap_start_m: float = 12.0,
                 far_band_contribution_cap_gain: float = 0.35,
                 heading_zero_quad_threshold: float = 0.005,
                 heading_zero_curvature_guard: float = 0.002):
        """
        Initialize trajectory planner.
        
        Args:
            lookahead_distance: How far ahead to plan (meters)
            point_spacing: Spacing between trajectory points (meters)
            target_speed: Target speed for trajectory (m/s)
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
            camera_fov: Camera field of view in degrees
            camera_height: Camera height above ground in meters
            lookahead_pixels: Lookahead distance in pixels (for coordinate conversion)
            reference_smoothing: Exponential smoothing for reference points (0.0-1.0)
            lane_smoothing_alpha: Exponential smoothing for lane coefficients (0.0-1.0, higher = more smoothing)
        """
        self.lookahead_distance = lookahead_distance
        self.point_spacing = point_spacing
        self.target_speed = target_speed
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fov = camera_fov
        self.camera_height = camera_height
        self.lookahead_pixels = lookahead_pixels
        self.bias_correction_threshold = bias_correction_threshold
        self.reference_smoothing = reference_smoothing
        
        # NEW: Camera offset correction (for calibration)
        # camera_offset_x: Lateral offset of camera from vehicle center (meters, positive = right)
        # This corrects for camera misalignment
        self.camera_offset_x = camera_offset_x
        
        # NEW: Distance scaling factor to account for camera pitch
        # When camera looks down, point 8m straight ahead in camera space is actually
        # closer in world space when projected to ground. This scaling factor corrects for that.
        # Default: 7/8 = 0.875 (based on visualizer tuning where 7m aligns correctly)
        # This makes lanes appear correct width (7m instead of 8.5m)
        self.distance_scaling_factor = distance_scaling_factor
        
        # NEW: Optional center spline smoothing to reduce trajectory jitter.
        self.center_spline_enabled = center_spline_enabled
        self.center_spline_degree = int(center_spline_degree)
        self.center_spline_samples = int(center_spline_samples)
        self.center_spline_alpha = float(center_spline_alpha)
        self.x_clip_enabled = bool(x_clip_enabled)
        self.x_clip_limit_m = max(1.0, float(x_clip_limit_m))
        self.trajectory_eval_y_min_ratio = float(np.clip(trajectory_eval_y_min_ratio, 0.0, 0.95))
        mode = str(centerline_midpoint_mode).strip().lower()
        self.centerline_midpoint_mode = mode if mode in {"pointwise", "coeff_average"} else "pointwise"
        proj_model = str(projection_model).strip().lower()
        self.projection_model = proj_model if proj_model in {"legacy", "calibrated_pinhole"} else "legacy"
        self.calibrated_fx_px = float(calibrated_fx_px)
        self.calibrated_fy_px = float(calibrated_fy_px)
        self.calibrated_cx_px = float(calibrated_cx_px) if calibrated_cx_px > 0.0 else (float(image_width) * 0.5)
        self.calibrated_cy_px = float(calibrated_cy_px) if calibrated_cy_px > 0.0 else (float(image_height) * 0.5)
        self.dynamic_effective_horizon_farfield_scale_min = float(
            np.clip(dynamic_effective_horizon_farfield_scale_min, 0.5, 1.0)
        )
        self.far_band_contribution_cap_enabled = bool(far_band_contribution_cap_enabled)
        self.far_band_contribution_cap_start_m = float(np.clip(far_band_contribution_cap_start_m, 0.0, 50.0))
        self.far_band_contribution_cap_gain = float(np.clip(far_band_contribution_cap_gain, 0.0, 1.0))
        self.heading_zero_quad_threshold = float(max(0.0, heading_zero_quad_threshold))
        self.heading_zero_curvature_guard = float(max(0.0, heading_zero_curvature_guard))
        
        # For reference point smoothing
        self.last_reference_x = None
        self.last_reference_y = None
        self.last_reference_heading = None
        
        # FIXED: Initialize temporal filtering for lane coefficients
        self.lane_coeffs_history = []  # History of lane coefficients for smoothing
        self.lane_smoothing_alpha = lane_smoothing_alpha  # Exponential smoothing factor
        
        # CRITICAL FIX: Track frames since last_center_coeffs was set
        # This prevents locking in a biased first-frame trajectory
        self._frames_since_last_center = 0
        
        # NEW: Adaptive bias correction - track bias history for persistent bias detection
        self.bias_history = []  # History of reference point biases (in pixels)
        self.bias_history_size = 30  # Track last 30 frames (~1 second at 30 FPS)
        # FIXED: Relaxed threshold from 3.0 to 4.0 pixels to match current variance (3.78 pixels = 0.0545m)
        # This allows adaptive corrections to trigger even with higher variance
        self.persistent_bias_threshold_std = 4.0  # If std < this, bias is persistent (was 3.0, current variance = 3.78 pixels)
        self.persistent_bias_threshold_mean = 1.0  # If mean > this, apply correction
        
        # Compute conversion factors from image to vehicle coordinates
        # Using camera model: at lookahead distance, estimate pixel-to-meter conversion
        # For typical AV camera: 75° FOV, 640x480 image
        # At 10m lookahead: horizontal FOV ≈ 2 * 10 * tan(30°) ≈ 11.5m
        # So 640 pixels ≈ 11.5m, or 1 pixel ≈ 0.018m horizontally
        # For vertical: similar calculation
        # More accurate would use camera calibration, but this is a reasonable approximation
        fov_rad = np.radians(camera_fov)
        # At typical lookahead (10-20m), estimate conversion
        typical_lookahead = 15.0  # meters
        width_at_lookahead = 2.0 * typical_lookahead * np.tan(fov_rad / 2.0)
        self.pixel_to_meter_x = width_at_lookahead / image_width  # meters per pixel horizontally
        # For vertical, assume similar FOV
        height_at_lookahead = width_at_lookahead * (image_height / image_width)
        self.pixel_to_meter_y = height_at_lookahead / image_height  # meters per pixel vertically
        self.last_generation_diagnostics: Dict[str, float] = self._empty_generation_diagnostics()

    def _empty_generation_diagnostics(self) -> Dict[str, float]:
        """Default per-frame trajectory generation diagnostics (instrumentation-only)."""
        return {
            "diag_available": 0.0,
            "diag_generated_by_fallback": 0.0,
            "diag_points_generated": 0.0,
            "diag_x_clip_count": 0.0,
            "diag_pre_y0": np.nan,
            "diag_pre_y1": np.nan,
            "diag_pre_y2": np.nan,
            "diag_post_y0": np.nan,
            "diag_post_y1": np.nan,
            "diag_post_y2": np.nan,
            "diag_used_provided_distance0": np.nan,
            "diag_used_provided_distance1": np.nan,
            "diag_used_provided_distance2": np.nan,
            "diag_post_minus_pre_y0": np.nan,
            "diag_post_minus_pre_y1": np.nan,
            "diag_post_minus_pre_y2": np.nan,
            "diag_preclip_x0": np.nan,
            "diag_preclip_x1": np.nan,
            "diag_preclip_x2": np.nan,
            "diag_preclip_x_abs_max": np.nan,
            "diag_preclip_x_abs_p95": np.nan,
            "diag_preclip_mean_12_20m_lane_source_x": np.nan,
            "diag_preclip_mean_12_20m_distance_scale_delta_x": np.nan,
            "diag_preclip_mean_12_20m_camera_offset_delta_x": np.nan,
            "diag_preclip_abs_mean_12_20m_lane_source_x": np.nan,
            "diag_preclip_abs_mean_12_20m_distance_scale_delta_x": np.nan,
            "diag_preclip_abs_mean_12_20m_camera_offset_delta_x": np.nan,
            "diag_postclip_x0": np.nan,
            "diag_postclip_x1": np.nan,
            "diag_postclip_x2": np.nan,
            "diag_first_segment_y0_gt_y1_pre": np.nan,
            "diag_first_segment_y0_gt_y1_post": np.nan,
            "diag_inversion_introduced_after_conversion": np.nan,
            "diag_far_band_contribution_limited_active": 0.0,
            "diag_far_band_contribution_limit_start_m": np.nan,
            "diag_far_band_contribution_limit_gain": np.nan,
            "diag_far_band_contribution_scale_mean_12_20m": np.nan,
            "diag_far_band_contribution_limited_frac_12_20m": np.nan,
        }

    @staticmethod
    def _diag_value_at(values: List[float], idx: int) -> float:
        if idx < len(values):
            return float(values[idx])
        return np.nan

    def get_last_generation_diagnostics(self) -> Dict[str, float]:
        """Get most recent planner generation diagnostics."""
        return dict(self.last_generation_diagnostics)
    
    def plan(self, lane_coeffs: List[Optional[np.ndarray]], 
             vehicle_state: Optional[dict] = None) -> Trajectory:
        """
        Plan trajectory from lane lines.
        
        Args:
            lane_coeffs: List of lane polynomial coefficients
            vehicle_state: Current vehicle state (position, heading, etc.)
        
        Returns:
            Planned trajectory
        """
        # Reset diagnostics per planning step (avoid stale values on fallback paths).
        self.last_generation_diagnostics = self._empty_generation_diagnostics()

        # FIXED: Apply temporal filtering to lane coefficients to reduce noise
        lane_coeffs = self._smooth_lane_coeffs(lane_coeffs)
        
        # Filter out None lanes and ensure they're 1D arrays
        valid_lanes = []
        for coeffs in lane_coeffs:
            if coeffs is not None:
                # Convert to numpy array and ensure it's 1D
                coeffs_array = np.asarray(coeffs)
                if coeffs_array.ndim == 0:
                    # Scalar - skip it (invalid)
                    continue
                elif coeffs_array.ndim == 1 and len(coeffs_array) >= 2:
                    # Valid 1D array with at least 2 coefficients
                    valid_lanes.append(coeffs_array)
                elif coeffs_array.ndim > 1:
                    # Multi-dimensional - flatten it
                    coeffs_array = coeffs_array.flatten()
                    if len(coeffs_array) >= 2:
                        valid_lanes.append(coeffs_array)
        
        if len(valid_lanes) == 0:
            # No lanes detected - return straight trajectory
            # IMPORTANT: Do NOT use last_center_coeffs here - that preserves bias!
            # Always return a fresh, centered straight trajectory when lanes fail
            return self._create_straight_trajectory(vehicle_state)
        
        # Find center lane or use single lane
        # Note: lane_coeffs[0] should be left lane, lane_coeffs[1] should be right lane
        if len(valid_lanes) >= 2:
            # NEW: Solution 4 - Lane Detection Validation
            # Validate lane detection before using
            lane1 = np.asarray(valid_lanes[0])
            lane2 = np.asarray(valid_lanes[1])
            
            # Ensure they're 1D arrays
            if lane1.ndim == 0:
                return self._create_straight_trajectory(vehicle_state)
            if lane2.ndim == 0:
                return self._create_straight_trajectory(vehicle_state)
            lane1 = lane1.flatten() if lane1.ndim > 1 else lane1
            lane2 = lane2.flatten() if lane2.ndim > 1 else lane2
            
            # CRITICAL FIX: Check lane order at BOTTOM of image (y=image_height), not top!
            # During turns, lanes can cross at the top (far away), but at bottom (close to vehicle)
            # the left lane should still be on the left, right lane on the right
            # y=0 is TOP of image (far away), y=image_height is BOTTOM (close to vehicle)
            x1_at_bottom = np.polyval(lane1, self.image_height)
            x2_at_bottom = np.polyval(lane2, self.image_height)
            
            # Also check at a middle position for robustness (in case bottom evaluation is unreliable)
            y_middle = self.image_height // 2
            x1_at_middle = np.polyval(lane1, y_middle)
            x2_at_middle = np.polyval(lane2, y_middle)
            
            # Determine which is left and which is right
            # Use bottom position as primary (closest to vehicle, most reliable)
            # Fall back to middle if bottom is ambiguous (both near center)
            image_center_x = self.image_width / 2.0
            margin = self.image_width * 0.1  # 10% margin
            
            # Check if bottom positions are clearly separated
            if abs(x1_at_bottom - x2_at_bottom) > margin:
                # Bottom positions are clearly different - use them
                if x1_at_bottom < x2_at_bottom:
                    left_lane = lane1
                    right_lane = lane2
                else:
                    left_lane = lane2
                    right_lane = lane1
            else:
                # Bottom positions are too close (ambiguous) - use middle position
                if x1_at_middle < x2_at_middle:
                    left_lane = lane1
                    right_lane = lane2
                else:
                    left_lane = lane2
                    right_lane = lane1
            
            # NEW: Validate lane detection
            # Check lane width (should be ~3.5m ± 0.5m at bottom of image)
            # Convert pixel width to meters at bottom of image (closest to vehicle)
            left_x_at_bottom = np.polyval(left_lane, self.image_height)
            right_x_at_bottom = np.polyval(right_lane, self.image_height)
            lane_width_pixels = abs(right_x_at_bottom - left_x_at_bottom)
            
            # Estimate distance at bottom of image (closest, ~0-2m ahead)
            distance_at_bottom = 2.0  # meters (rough estimate)
            fov_rad = np.radians(self.camera_fov)
            width_at_bottom = 2.0 * distance_at_bottom * np.tan(fov_rad / 2.0)
            pixel_to_meter_at_bottom = width_at_bottom / self.image_width
            lane_width_meters = lane_width_pixels * pixel_to_meter_at_bottom
            
            # Validate lane width (typical: 3.5m ± 0.5m)
            valid_lane_width = 3.0 <= lane_width_meters <= 4.0
            
            # Validate lane order (left should be left of center, right should be right of center)
            image_center_x = self.image_width / 2.0
            valid_lane_order = left_x_at_bottom < image_center_x < right_x_at_bottom
            
            # If validation fails, be more careful about using previous center lane
            # CRITICAL FIX: Don't use last_center_coeffs if it's the first frame or if it's been too long
            # This prevents locking in an initial bias
            # CRITICAL FIX: Apply bias correction to last_center_coeffs when reusing it
            # This ensures corrections happen even when validation fails
            if not (valid_lane_width and valid_lane_order):
                # Only use previous center lane if we have enough history (at least 5 frames)
                # This prevents using a biased first-frame trajectory
                if (hasattr(self, 'last_center_coeffs') and self.last_center_coeffs is not None and
                    hasattr(self, '_frames_since_last_center') and self._frames_since_last_center < 10):
                    # Use previous center lane to prevent bad detections from causing bias
                    # CRITICAL: Apply bias correction to last_center_coeffs before using
                    # This ensures corrections happen even when validation fails
                    center_coeffs = self._apply_bias_correction_to_coeffs(
                        self.last_center_coeffs, left_lane, right_lane
                    )
                    if not hasattr(self, '_frames_since_last_center'):
                        self._frames_since_last_center = 0
                    self._frames_since_last_center += 1
                else:
                    # No previous center or too old - use current (even if validation failed)
                    # This prevents locking in bias from first frame
                    center_coeffs = self._compute_center_lane(left_lane, right_lane)
                    if hasattr(self, '_frames_since_last_center'):
                        self._frames_since_last_center = 0
            else:
                # Validation passed - compute center lane
                center_coeffs = self._compute_center_lane(left_lane, right_lane)
                if hasattr(self, '_frames_since_last_center'):
                    self._frames_since_last_center = 0
            
            trajectory_points = self._generate_trajectory_points(center_coeffs, vehicle_state)
        else:
            # Use single lane, offset to center
            # If only one lane detected, offset by half lane width (1.75m) toward center
            # CRITICAL FIX: Determine which lane it is and offset in correct direction
            single_lane = np.asarray(valid_lanes[0])
            # Ensure it's a 1D array
            if single_lane.ndim == 0:
                return self._create_straight_trajectory(vehicle_state)
            single_lane = single_lane.flatten() if single_lane.ndim > 1 else single_lane
            
            # Determine if it's left or right lane by checking position at bottom of image
            lane_x_at_bottom = np.polyval(single_lane, self.image_height)
            image_center_x = self.image_width / 2.0
            
            if lane_x_at_bottom < image_center_x:
                # Left lane detected - offset RIGHT (positive) to get to center
                offset_direction = 1.0
            else:
                # Right lane detected - offset LEFT (negative) to get to center
                offset_direction = -1.0
            
            # Convert offset from meters to pixels at lookahead distance
            # At lookahead distance, calculate pixel-to-meter conversion
            fov_rad = np.radians(self.camera_fov)
            width_at_lookahead = 2.0 * self.lookahead_distance * np.tan(fov_rad / 2.0)
            pixel_to_meter = width_at_lookahead / self.image_width
            
            # Offset in meters (half lane width = 1.75m)
            offset_meters = 1.75 * offset_direction
            
            # Convert to pixels
            offset_pixels = offset_meters / pixel_to_meter
            
            # Apply offset to polynomial (in pixels)
            lane_coeffs_offset = single_lane.copy()
            lane_coeffs_offset[-1] += offset_pixels  # Adjust constant term (in pixels)
            
            trajectory_points = self._generate_trajectory_points(lane_coeffs_offset, vehicle_state)
        
        return Trajectory(points=trajectory_points, length=self.lookahead_distance)
    
    def _compute_center_lane(self, left_coeffs: np.ndarray, 
                            right_coeffs: np.ndarray) -> np.ndarray:
        """
        Compute center lane between left and right lanes.
        Assumes lanes are in image coordinates where:
        - Left lane has negative x values (left side of image)
        - Right lane has positive x values (right side of image)
        Center is the average of the two lane polynomials.
        
        Args:
            left_coeffs: Left lane polynomial coefficients
            right_coeffs: Right lane polynomial coefficients
        
        Returns:
            Center lane polynomial coefficients
        """
        # Compute center lane by averaging coefficients
        # NOTE: This works correctly even when lanes converge in perspective
        # Perspective convergence (lanes crossing) is NORMAL - parallel lines converge
        # Averaging coefficients gives correct center because:
        #   - At each y position: center_x = (left_x + right_x) / 2
        #   - This is mathematically equivalent to averaging coefficients for polynomials
        #   - When converted to vehicle coordinates, perspective is accounted for
        # For curved roads, there may be small errors (<1px), but this is acceptable
        # More sophisticated approach: sample points, compute midpoints, fit new polynomial
        # But current approach is simpler and works well for most cases
        
        # CRITICAL FIX: When lanes have very different curvatures (asymmetric), averaging coefficients
        # can give incorrect center. Use point-sampling method for better accuracy.
        # Check if lanes have very different curvatures
        left_curvature = abs(left_coeffs[0]) if len(left_coeffs) >= 3 else 0.0
        right_curvature = abs(right_coeffs[0]) if len(right_coeffs) >= 3 else 0.0
        curvature_diff = abs(left_curvature - right_curvature)
        
        use_pointwise_midpoint = (
            self.centerline_midpoint_mode == "pointwise" or curvature_diff > 0.02
        )
        # If configured for pointwise midpoint or curvature difference is large, use point-sampling.
        if use_pointwise_midpoint:
            # Sample points, compute midpoints, fit polynomial
            y_samples = np.linspace(int(self.image_height * 0.3), int(self.image_height * 0.93), 20)
            left_x_samples = np.polyval(left_coeffs, y_samples)
            right_x_samples = np.polyval(right_coeffs, y_samples)
            center_x_samples = (left_x_samples + right_x_samples) / 2.0
            
            # Fit polynomial to center points
            center_coeffs = np.polyfit(y_samples, center_x_samples, deg=min(2, len(left_coeffs)-1))
        else:
            # Use simple averaging (faster, works well when curvatures are similar)
            center_coeffs = (left_coeffs + right_coeffs) / 2.0

        if self.center_spline_enabled:
            spline_coeffs = self._fit_center_spline(left_coeffs, right_coeffs)
            if spline_coeffs is not None:
                last_center = getattr(self, 'last_center_coeffs', None)
                if (self.center_spline_alpha is not None and last_center is not None and
                        len(spline_coeffs) == len(last_center)):
                    spline_coeffs = (
                        self.center_spline_alpha * last_center +
                        (1.0 - self.center_spline_alpha) * spline_coeffs
                    )
                center_coeffs = spline_coeffs
        
        # CRITICAL FIX: For straight roads, ensure center lane is STRAIGHT (heading = 0°)
        # The 19.7° reference heading bias is caused by non-zero linear coefficient
        # 
        # IMPORTANT: Lanes appear non-parallel in image space when car is offset/angled (perspective)
        # But the ROAD is still straight! We should check road curvature, not lane parallelism.
        #
        # Strategy: Check if road is straight (small quadratic coefficient)
        # If road is straight, force center to be straight regardless of car position/angle
        if len(center_coeffs) >= 3:
            # Check quadratic coefficient (curvature)
            center_quad = center_coeffs[0]  # a in ax^2 + bx + c
            left_quad = left_coeffs[0] if len(left_coeffs) >= 3 else 0.0
            right_quad = right_coeffs[0] if len(right_coeffs) >= 3 else 0.0
            max_quad = max(abs(left_quad), abs(right_quad), abs(center_quad))
            
            # If road curvature is very small, road is straight
            # Force center lane to be straight (zero linear coefficient)
            # CRITICAL FIX: Don't force linear coefficient to 0 based on image-space curvature!
            # The quadratic coefficient is in IMAGE SPACE (1/pixel), not VEHICLE SPACE (1/meter)
            # We can't determine if a road is straight by checking image-space coefficients
            # 
            # REMOVED: This was too aggressive and broke curve detection on the oval track
            # The heading computation will correctly identify straight roads based on vehicle-space coordinates
            # if max_quad <= 0.005:  # This threshold is in wrong units (image space, not vehicle space)
            #     center_coeffs = center_coeffs.copy()
            #     center_coeffs[-2] = 0.0  # Force linear coefficient to zero
        elif len(center_coeffs) >= 2:
            # No quadratic coefficient - road is definitely straight
            center_linear = center_coeffs[-2]
            if abs(center_linear) > 0.01:  # Only fix if there's a significant slope
                center_coeffs = center_coeffs.copy()
                center_coeffs[-2] = 0.0  # Force linear coefficient to zero
        
        # FIXED: Add outlier rejection for stability
        # CRITICAL FIX: Check if center lane is reasonable relative to LANE CENTER, not IMAGE CENTER!
        # The trajectory should be centered between lanes, not at image center
        # Image center ≠ Lane center (if lanes are not symmetric around image center)
        
        # Calculate actual lane center (midpoint between left and right lanes)
        left_x_at_bottom = np.polyval(left_coeffs, self.image_height)
        right_x_at_bottom = np.polyval(right_coeffs, self.image_height)
        lane_center_at_bottom = (left_x_at_bottom + right_x_at_bottom) / 2.0
        lane_width_pixels = abs(right_x_at_bottom - left_x_at_bottom)
        
        # Check if center lane is reasonable relative to lane center (not image center!)
        center_x_at_bottom = np.polyval(center_coeffs, self.image_height)
        center_offset_from_lane_center = abs(center_x_at_bottom - lane_center_at_bottom)
        
        # Reject if center lane is > 50 pixels from lane center OR lane width is unreasonable
        # Typical lane width: 200-400 pixels (3.5-7m lane width)
        # Changed threshold from 100 pixels (from image center) to 50 pixels (from lane center)
        # This prevents rejecting valid trajectories that are correctly centered on lanes
        # CRITICAL FIX: Only use last_center_coeffs if it's recent (within 10 frames)
        # This prevents locking in a biased first-frame trajectory
        # CRITICAL FIX: Apply bias correction to last_center_coeffs when reusing it
        # This ensures corrections happen even when trajectories are rejected
        if center_offset_from_lane_center > 50.0 or lane_width_pixels < 100 or lane_width_pixels > 500:
            if (hasattr(self, 'last_center_coeffs') and self.last_center_coeffs is not None and
                hasattr(self, '_frames_since_last_center') and self._frames_since_last_center < 10):
                # Use previous center lane to prevent sudden jumps (but only if recent)
                # CRITICAL: Apply bias correction to last_center_coeffs before returning
                # This ensures corrections happen even when trajectories are rejected
                corrected_coeffs = self._apply_bias_correction_to_coeffs(
                    self.last_center_coeffs, left_coeffs, right_coeffs
                )
                if not hasattr(self, '_frames_since_last_center'):
                    self._frames_since_last_center = 0
                self._frames_since_last_center += 1
                return corrected_coeffs
        
        # Store for next frame
        self.last_center_coeffs = center_coeffs.copy()
        # Reset counter when we successfully compute a new center lane
        if hasattr(self, '_frames_since_last_center'):
            self._frames_since_last_center = 0
        
        # FIXED: Check bias at multiple distances and apply stronger correction
        # CRITICAL FIX: Check bias relative to LANE CENTER, not IMAGE CENTER!
        # The trajectory should be centered between lanes, not at image center
        # Image center ≠ Lane center (if lanes are not symmetric around image center)
        
        # Calculate actual lane center (midpoint between left and right lanes)
        left_x_at_bottom = np.polyval(left_coeffs, self.image_height)
        right_x_at_bottom = np.polyval(right_coeffs, self.image_height)
        lane_center_at_bottom = (left_x_at_bottom + right_x_at_bottom) / 2.0
        
        # Check bias at bottom of image (y=image_height, closest to vehicle)
        center_x_at_bottom = np.polyval(center_coeffs, self.image_height)
        bias_at_bottom = center_x_at_bottom - lane_center_at_bottom  # FIXED: Use lane center, not image center
        
        # Check bias at lookahead distance (where reference point is computed)
        # Convert lookahead distance to image y coordinate
        typical_lookahead = 8.0  # meters (reference_lookahead)
        y_image_at_lookahead = self.image_height - (typical_lookahead / self.pixel_to_meter_y)
        y_image_at_lookahead = max(0, min(self.image_height - 1, y_image_at_lookahead))
        
        # Calculate lane center at lookahead distance
        left_x_at_lookahead = np.polyval(left_coeffs, y_image_at_lookahead)
        right_x_at_lookahead = np.polyval(right_coeffs, y_image_at_lookahead)
        lane_center_at_lookahead = (left_x_at_lookahead + right_x_at_lookahead) / 2.0
        
        center_x_at_lookahead = np.polyval(center_coeffs, y_image_at_lookahead)
        bias_at_lookahead = center_x_at_lookahead - lane_center_at_lookahead  # FIXED: Use lane center, not image center
        
        # Use the maximum bias (worst case)
        max_bias = max(abs(bias_at_bottom), abs(bias_at_lookahead))
        
        # NEW: Solution 1 - Adaptive Bias Correction Threshold
        # Track bias history to detect persistent small biases
        self.bias_history.append(bias_at_lookahead)
        if len(self.bias_history) > self.bias_history_size:
            self.bias_history.pop(0)
        
        # FIXED: More aggressive bias correction to fix -0.41m trajectory offset
        # Always apply correction when bias is significant, regardless of variance
        should_correct = False
        correction_factor = 0.0
        
        # Convert bias from pixels to approximate meters for threshold comparison
        # At 8m lookahead: pixel_to_meter_x ≈ 0.0144 m/pixel
        bias_meters = abs(bias_at_lookahead) * self.pixel_to_meter_x
        
        if len(self.bias_history) >= 10:  # Need at least 10 frames of history
            bias_array = np.array(self.bias_history)
            bias_mean = np.mean(bias_array)
            bias_std = np.std(bias_array)
            bias_abs_mean = np.mean(np.abs(bias_array))
            bias_abs_mean_meters = bias_abs_mean * self.pixel_to_meter_x
            
            # FIXED: More aggressive thresholds - always correct significant bias
            # Mechanism 1: Large bias (>0.2m) - always correct fully
            if bias_meters > 0.2:  # ~14 pixels = 0.2m
                should_correct = True
                correction_factor = 1.0  # Full correction
            # Mechanism 2: Persistent bias (low variance, >0.05m) - full correction
            elif bias_std < self.persistent_bias_threshold_std and bias_abs_mean_meters > 0.05:
                should_correct = True
                correction_factor = 1.0  # Full correction for persistent biases
            # Mechanism 3: Medium bias (>0.1m) - partial correction
            elif bias_meters > 0.1:  # ~7 pixels = 0.1m
                should_correct = True
                correction_factor = 0.8  # Strong partial correction
            # Mechanism 4: Significant mean bias (>0.05m) even with high variance - partial correction
            elif bias_abs_mean_meters > 0.05:  # ~3.5 pixels = 0.05m
                should_correct = True
                # Reduce correction factor based on variance (higher variance = less correction)
                variance_factor = min(1.0, self.persistent_bias_threshold_std / max(bias_std, 0.1))
                correction_factor = 0.6 * variance_factor  # Partial correction for high variance
            # Mechanism 5: Time-based persistence (if bias persists over time)
            elif len(self.bias_history) >= 20:
                recent_bias = bias_array[-20:]
                consistent_direction = np.all(np.sign(recent_bias) == np.sign(bias_mean))
                if consistent_direction and bias_abs_mean_meters > 0.03:  # ~2 pixels = 0.03m
                    # Bias consistently in same direction - apply correction
                    should_correct = True
                    correction_factor = 0.4  # Conservative correction for high variance
        elif abs(bias_at_lookahead) > self.bias_correction_threshold:
            # Large bias even without history - correct immediately
            should_correct = True
            correction_factor = 1.0
        
        # Apply bias correction
        if should_correct:
            center_coeffs = center_coeffs.copy()
            # Apply correction to constant term (affects all distances)
            # Use bias at lookahead distance (where reference point is computed) for more accurate correction
            center_coeffs[-1] -= bias_at_lookahead * correction_factor
            
            # Also adjust linear term if bias varies significantly with distance
            if abs(bias_at_lookahead - bias_at_bottom) > 5.0:  # Significant variation
                # Estimate linear correction needed
                y_range = self.image_height - y_image_at_lookahead
                if y_range > 0:
                    linear_correction = (bias_at_lookahead - bias_at_bottom) / y_range
                    if len(center_coeffs) >= 2:
                        center_coeffs[-2] -= linear_correction * 0.5  # Partial correction to avoid overcorrection
        
        return center_coeffs

    def _fit_center_spline(self, left_coeffs: np.ndarray,
                           right_coeffs: np.ndarray) -> Optional[np.ndarray]:
        """Fit a smooth centerline polynomial from sampled midpoints."""
        if self.center_spline_samples < 4:
            return None
        y_start = int(self.image_height * 0.3)
        y_end = int(self.image_height * 0.93)
        y_samples = np.linspace(y_start, y_end, self.center_spline_samples)
        left_x_samples = np.polyval(left_coeffs, y_samples)
        right_x_samples = np.polyval(right_coeffs, y_samples)
        center_x_samples = (left_x_samples + right_x_samples) / 2.0
        degree = min(max(self.center_spline_degree, 1), 2)
        if len(y_samples) <= degree:
            return None
        return np.polyfit(y_samples, center_x_samples, deg=degree)
    
    def _apply_bias_correction_to_coeffs(self, center_coeffs: np.ndarray, 
                                         left_coeffs: np.ndarray, 
                                         right_coeffs: np.ndarray) -> np.ndarray:
        """
        Apply bias correction to center coefficients when reusing last_center_coeffs.
        This ensures corrections happen even when trajectories are rejected.
        
        Args:
            center_coeffs: Center lane coefficients to correct
            left_coeffs: Left lane coefficients (for computing lane center)
            right_coeffs: Right lane coefficients (for computing lane center)
        
        Returns:
            Corrected center lane coefficients
        """
        # Calculate lane center at lookahead distance
        typical_lookahead = 8.0  # meters (reference_lookahead)
        y_image_at_lookahead = self.image_height - (typical_lookahead / self.pixel_to_meter_y)
        y_image_at_lookahead = max(0, min(self.image_height - 1, y_image_at_lookahead))
        
        left_x_at_lookahead = np.polyval(left_coeffs, y_image_at_lookahead)
        right_x_at_lookahead = np.polyval(right_coeffs, y_image_at_lookahead)
        lane_center_at_lookahead = (left_x_at_lookahead + right_x_at_lookahead) / 2.0
        
        center_x_at_lookahead = np.polyval(center_coeffs, y_image_at_lookahead)
        bias_at_lookahead = center_x_at_lookahead - lane_center_at_lookahead
        
        # Track bias history
        self.bias_history.append(bias_at_lookahead)
        if len(self.bias_history) > self.bias_history_size:
            self.bias_history.pop(0)
        
        # Apply same bias correction logic as in _compute_center_lane
        should_correct = False
        correction_factor = 0.0
        bias_meters = abs(bias_at_lookahead) * self.pixel_to_meter_x
        
        if len(self.bias_history) >= 10:
            bias_array = np.array(self.bias_history)
            bias_mean = np.mean(bias_array)
            bias_std = np.std(bias_array)
            bias_abs_mean = np.mean(np.abs(bias_array))
            bias_abs_mean_meters = bias_abs_mean * self.pixel_to_meter_x
            
            if bias_meters > 0.2:
                should_correct = True
                correction_factor = 1.0
            elif bias_std < self.persistent_bias_threshold_std and bias_abs_mean_meters > 0.05:
                should_correct = True
                correction_factor = 1.0
            elif bias_meters > 0.1:
                should_correct = True
                correction_factor = 0.8
            elif bias_abs_mean_meters > 0.05:
                should_correct = True
                variance_factor = min(1.0, self.persistent_bias_threshold_std / max(bias_std, 0.1))
                correction_factor = 0.6 * variance_factor
            elif len(self.bias_history) >= 20:
                recent_bias = bias_array[-20:]
                consistent_direction = np.all(np.sign(recent_bias) == np.sign(bias_mean))
                if consistent_direction and bias_abs_mean_meters > 0.03:
                    should_correct = True
                    correction_factor = 0.4
        elif abs(bias_at_lookahead) > self.bias_correction_threshold:
            should_correct = True
            correction_factor = 1.0
        
        # Apply correction
        if should_correct:
            corrected_coeffs = center_coeffs.copy()
            corrected_coeffs[-1] -= bias_at_lookahead * correction_factor
            return corrected_coeffs
        
        return center_coeffs
    
    def _smooth_lane_coeffs(self, lane_coeffs: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """
        Apply temporal smoothing to lane coefficients to reduce noise.
        Includes outlier rejection to handle yellow marker position variability.
        
        Args:
            lane_coeffs: Current frame's lane coefficients
        
        Returns:
            Smoothed lane coefficients
        """
        if len(self.lane_coeffs_history) == 0:
            # First frame - no history, just store and return
            self.lane_coeffs_history.append(lane_coeffs)
            if len(self.lane_coeffs_history) > 5:
                self.lane_coeffs_history.pop(0)
            return lane_coeffs
        
        # Get previous frame's coefficients
        prev_coeffs = self.lane_coeffs_history[-1]
        
        # Apply exponential moving average with outlier rejection
        alpha = self.lane_smoothing_alpha
        smoothed_coeffs = []
        
        for i, coeff in enumerate(lane_coeffs):
            if coeff is not None:
                # Check if we have previous coefficient for this lane
                if i < len(prev_coeffs) and prev_coeffs[i] is not None:
                    prev_coeff = prev_coeffs[i]
                    
                    # ROBUSTNESS FIX: Outlier rejection for sudden jumps
                    # Check if current detection is consistent with previous
                    # Evaluate both at a common y position (middle of image) to compare
                    y_compare = self.image_height * 0.5  # Middle of image
                    prev_x = np.polyval(prev_coeff, y_compare)
                    curr_x = np.polyval(coeff, y_compare)
                    
                    # Calculate change in x position
                    x_change = abs(curr_x - prev_x)
                    max_reasonable_change = self.image_width * 0.15  # 15% of image width (96px for 640px)
                    
                    # If change is too large, it's likely an outlier (wrong detection or marker shift)
                    # Use previous frame instead (temporal persistence)
                    if x_change > max_reasonable_change:
                        logger.warning(f"[ROBUSTNESS] Lane {i}: Large jump detected! "
                                     f"x_change={x_change:.1f}px > {max_reasonable_change:.1f}px. "
                                     f"Using previous frame (outlier rejection).")
                        # Use previous frame (outlier rejection)
                        smoothed_coeffs.append(prev_coeff)
                    else:
                        # Change is reasonable - apply smoothing
                        if len(coeff) == len(prev_coeff):
                            smoothed_coeff = alpha * prev_coeff + (1 - alpha) * coeff
                            smoothed_coeffs.append(smoothed_coeff)
                        else:
                            # Different polynomial order - use current
                            smoothed_coeffs.append(coeff)
                else:
                    # No previous coefficient - use current
                    smoothed_coeffs.append(coeff)
            else:
                # Current is None - use previous if available
                if i < len(prev_coeffs) and prev_coeffs[i] is not None:
                    # Use previous (temporal persistence)
                    smoothed_coeffs.append(prev_coeffs[i])
                else:
                    # Both are None
                    smoothed_coeffs.append(None)
        
        # Store in history
        self.lane_coeffs_history.append(smoothed_coeffs)
        if len(self.lane_coeffs_history) > 5:
            self.lane_coeffs_history.pop(0)
        
        return smoothed_coeffs
    
    def _offset_lane(self, lane_coeffs: np.ndarray, offset: float) -> np.ndarray:
        """
        Offset lane by given distance (perpendicular to lane direction).
        
        Args:
            lane_coeffs: Lane polynomial coefficients
            offset: Offset distance in meters (positive = right)
        
        Returns:
            Offset lane coefficients
        """
        # Simplified: adjust constant term
        # More accurate: compute perpendicular offset at each point
        offset_coeffs = lane_coeffs.copy()
        offset_coeffs[-1] += offset  # Adjust constant term
        return offset_coeffs
    
    def _convert_image_to_vehicle_coords(self, x_pixels: float, y_pixels: float, 
                                         lookahead_distance: float = 15.0,
                                         horizontal_fov_override: Optional[float] = None,
                                         debug_trace: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Convert image pixel coordinates to vehicle coordinates.
        
        Args:
            x_pixels: X coordinate in image (pixels, 0=left, width=right)
            y_pixels: Y coordinate in image (pixels, 0=top, height=bottom)
            lookahead_distance: Reference distance ahead in meters (for initial estimate)
        
        Returns:
            (x_vehicle, y_vehicle) in meters
            x_vehicle: lateral position (-left, +right)
            y_vehicle: forward distance (0=vehicle, +forward)
        """
        # Convert from image coordinates to vehicle coordinates
        # Image: (0,0) at top-left, x increases right, y increases down
        # Vehicle: (0,0) at vehicle, x increases right, y increases forward
        
        # CRITICAL FIX: Use lookahead_distance directly when provided
        # The issue: pixel_to_meter_y is calculated at fixed 15m, but actual distance varies
        # When lookahead_distance is provided (e.g., 8m), it IS the actual distance!
        # Don't try to calculate distance from y_pixels - use the provided distance
        
        fov_rad = np.radians(self.camera_fov)
        
        # DEBUG: Check if lookahead_distance is being used
        # If lookahead_distance is provided and > 0, use it directly
        # Otherwise, fall back to calculating from y_pixels
        using_provided_distance = False
        if lookahead_distance is not None and lookahead_distance >= 0:
            # Use provided lookahead_distance, but apply scaling factor to account for camera pitch
            # When camera looks down, point 8m straight ahead in camera space is actually
            # closer in world space when projected to ground (e.g., 7m instead of 8m)
            # This scaling factor corrects the coordinate conversion to match reality
            actual_distance = lookahead_distance * self.distance_scaling_factor
            using_provided_distance = True
        else:
            # Fallback: estimate distance from y_pixels position
            # Convert y: pixels to normalized position (0=top, 1=bottom)
            y_from_bottom = self.image_height - y_pixels
            y_normalized = y_from_bottom / self.image_height  # 0 (top) to 1 (bottom)
            
            # Use simplified perspective model
            if y_normalized > 0.1:
                # Near bottom: use inverse relationship
                base_distance = 1.5  # meters at bottom
                actual_distance = base_distance / y_normalized
            else:
                # Very far: use camera model
                angle_from_vertical = (1.0 - y_normalized) * fov_rad / 2.0
                if angle_from_vertical > 0.01:
                    actual_distance = self.camera_height / np.tan(angle_from_vertical)
                else:
                    actual_distance = 15.0  # Fallback
        
        # Ensure reasonable bounds
        actual_distance = max(0.5, min(actual_distance, 50.0))
        
        # CRITICAL FIX: Use HORIZONTAL FOV for lateral (x) conversion
        # Unity's Camera.fieldOfView ALWAYS returns VERTICAL FOV, regardless of Inspector "Field of View Axis" setting.
        # However, when Unity Inspector shows "Field of View Axis: Horizontal" and "Field of View: 110°",
        # Unity converts it internally to vertical FOV (84.93° for 640x480), and calculates horizontal = 110°.
        # 
        # Our config (camera_fov = 110.0°) matches Unity's calculated horizontal FOV (110.00°),
        # as verified by logs: "Unity fieldOfView = 84.93° (vertical), calculated horizontal = 110.00°"
        # 
        # Use Unity's reported horizontal FOV if available (most accurate), otherwise use config value
        if horizontal_fov_override is not None and horizontal_fov_override > 0:
            # Use Unity's actual reported horizontal FOV (most accurate)
            horizontal_fov_rad = np.radians(horizontal_fov_override)
        else:
            # Fallback: Use config value directly as horizontal FOV (matches Unity Inspector)
            horizontal_fov_rad = fov_rad
        
        # Diagnostics-only decomposition: isolate distance-scaling contribution.
        # When using provided lookahead, conversion scales distance by distance_scaling_factor.
        # This captures how much that scaling changes lateral x in meters.
        if using_provided_distance and lookahead_distance is not None:
            base_distance_for_x = float(max(0.5, min(float(lookahead_distance), 50.0)))
        else:
            base_distance_for_x = actual_distance

        # Calibrated pinhole model for lateral conversion (A/B with legacy model).
        if self.projection_model == "calibrated_pinhole":
            fx_px = self.calibrated_fx_px if self.calibrated_fx_px > 1.0 else (
                (0.5 * float(self.image_width)) / max(np.tan(horizontal_fov_rad * 0.5), 1e-6)
            )
            x_center = self.calibrated_cx_px
            # y_center reserved for future full ground-plane intersection.
            _ = self.calibrated_cy_px
            x_lane_source = ((x_pixels - x_center) / max(fx_px, 1e-6)) * float(actual_distance)
            x_lane_source_unscaled = ((x_pixels - x_center) / max(fx_px, 1e-6)) * float(base_distance_for_x)
            width_at_actual_distance = 2.0 * actual_distance * np.tan(horizontal_fov_rad / 2.0)
            pixel_to_meter_x_at_distance = width_at_actual_distance / self.image_width
        else:
            width_at_actual_distance = 2.0 * actual_distance * np.tan(horizontal_fov_rad / 2.0)
            width_at_base_distance = 2.0 * base_distance_for_x * np.tan(horizontal_fov_rad / 2.0)
            pixel_to_meter_x_at_distance = width_at_actual_distance / self.image_width
            pixel_to_meter_x_base = width_at_base_distance / self.image_width
            x_center = self.image_width / 2.0
            x_offset_pixels = x_pixels - x_center
            x_lane_source = x_offset_pixels * pixel_to_meter_x_at_distance
            x_lane_source_unscaled = x_offset_pixels * pixel_to_meter_x_base
        x_distance_scale_delta = x_lane_source - x_lane_source_unscaled

        # FIXED: Apply camera offset correction (for calibration)
        x_camera_offset_delta = self.camera_offset_x
        x_vehicle = x_lane_source + x_camera_offset_delta
        
        x_vehicle_pre_clip = x_vehicle

        # Optional guardrail against extreme lateral conversion outliers.
        # Keep configurable for turn-authority A/B testing on curves.
        if self.x_clip_enabled:
            x_vehicle = np.clip(x_vehicle, -self.x_clip_limit_m, self.x_clip_limit_m)
        was_clipped = abs(float(x_vehicle_pre_clip) - float(x_vehicle)) > 1e-6

        if debug_trace is not None:
            debug_trace.setdefault("input_y", []).append(float(lookahead_distance))
            debug_trace.setdefault("post_y", []).append(float(actual_distance))
            debug_trace.setdefault("used_provided_distance", []).append(1.0 if using_provided_distance else 0.0)
            debug_trace.setdefault("pre_clip_x", []).append(float(x_vehicle_pre_clip))
            debug_trace.setdefault("post_clip_x", []).append(float(x_vehicle))
            debug_trace.setdefault("pre_clip_x_lane_source", []).append(float(x_lane_source_unscaled))
            debug_trace.setdefault("pre_clip_x_distance_scale_delta", []).append(float(x_distance_scale_delta))
            debug_trace.setdefault("pre_clip_x_camera_offset_delta", []).append(float(x_camera_offset_delta))
            debug_trace["clip_count"] = int(debug_trace.get("clip_count", 0)) + (1 if was_clipped else 0)
        
        # Use actual_distance for y_vehicle (consistent scaling)
        y_vehicle = actual_distance
        
        # DEBUG: Log coordinate conversion details (only for lane width calculations from av_stack.py)
        # This helps identify if lookahead_distance is being used correctly
        import inspect
        try:
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame else None
            is_lane_width_calc = False
            if caller_frame:
                caller_filename = caller_frame.f_code.co_filename
                if 'av_stack.py' in caller_filename:
                    is_lane_width_calc = True
                    logger.info(f"[COORD DEBUG] Input: x_pixels={x_pixels:.1f}, y_pixels={y_pixels:.1f}, "
                               f"lookahead_distance={lookahead_distance:.2f}m")
                    logger.info(f"[COORD DEBUG] Distance: using_provided={using_provided_distance}, "
                               f"actual_distance={actual_distance:.2f}m")
                    # FIXED: Log correct FOV values
                    if horizontal_fov_override is not None and horizontal_fov_override > 0:
                        logger.info(f"[COORD DEBUG] FOV: using Unity's horizontal={np.degrees(horizontal_fov_rad):.1f}° (from Unity), "
                                   f"config fallback={np.degrees(fov_rad):.1f}°")
                    else:
                        logger.info(f"[COORD DEBUG] FOV: using config horizontal={np.degrees(horizontal_fov_rad):.1f}° "
                                   f"(Unity reports vertical=84.93°, horizontal=110.00°)")
                    logger.info(f"[COORD DEBUG] Scaling: width_at_distance={width_at_actual_distance:.2f}m, "
                               f"pixel_to_meter_x={pixel_to_meter_x_at_distance*1000:.3f}mm/px")
                    logger.info(f"[COORD DEBUG] Output: x_vehicle={x_vehicle:.3f}m, y_vehicle={y_vehicle:.2f}m")
        except:
            pass  # Don't break if inspect fails
        
        # REMOVED: Clamp for lane width calculation - we want to see true detected values
        # Note: Trajectory planning should handle bounds checking itself if needed
        # Previously clamped to ±5m lateral, but this was hiding detection errors
        
        return x_vehicle, y_vehicle
    
    def _generate_trajectory_points(self, lane_coeffs: np.ndarray,
                                    vehicle_state: Optional[dict]) -> List[TrajectoryPoint]:
        """
        Generate trajectory points from lane coefficients.
        
        Args:
            lane_coeffs: Lane polynomial coefficients (in IMAGE coordinates)
            vehicle_state: Current vehicle state
        
        Returns:
            List of trajectory points (in VEHICLE coordinates)
        """
        points = []
        debug_trace: Dict[str, Any] = {"clip_count": 0}
        effective_horizon_m = float(self.lookahead_distance)
        effective_horizon_applied = 0.0
        if isinstance(vehicle_state, dict):
            effective_horizon_applied = float(
                vehicle_state.get(
                    "dynamicEffectiveHorizonApplied",
                    vehicle_state.get("dynamic_effective_horizon_applied", 0.0),
                )
            )
            horizon_candidate = vehicle_state.get(
                "dynamicEffectiveHorizonMeters",
                vehicle_state.get("dynamic_effective_horizon_m", self.lookahead_distance),
            )
            try:
                horizon_candidate = float(horizon_candidate)
            except (TypeError, ValueError):
                horizon_candidate = float(self.lookahead_distance)
            if np.isfinite(horizon_candidate) and horizon_candidate > 0.1:
                effective_horizon_m = min(float(self.lookahead_distance), float(horizon_candidate))
        
        # Generate points along lane in vehicle coordinates
        num_points = int(self.lookahead_distance / self.point_spacing)
        far_band_scale_samples: List[float] = []
        far_band_limited_flags: List[float] = []
        far_band_limited_any_flags: List[float] = []
        far_band_limit_start_m = np.nan
        far_band_limit_gain = np.nan
        
        for i in range(num_points + 1):
            # y_vehicle is forward distance in meters
            y_vehicle = i * self.point_spacing
            
            if y_vehicle > self.lookahead_distance:
                break
            
            # Convert vehicle y to image y for polynomial evaluation
            # y_vehicle = 0 at vehicle, increases forward
            # y_image: bottom of image (y=height) is closest, top (y=0) is farthest
            # Use inverse of conversion: y_image = height - (y_vehicle / pixel_to_meter_y)
            # But need to account for perspective - closer pixels are smaller
            # For now, use simple linear mapping
            y_image_raw = self.image_height - (y_vehicle / self.pixel_to_meter_y)
            y_image = max(0, min(self.image_height - 1, y_image_raw))  # Clamp to image bounds
            if self.trajectory_eval_y_min_ratio > 0.0:
                min_eval_y = float(self.image_height) * self.trajectory_eval_y_min_ratio
                # Reduce far-field extrapolation by avoiding the top-most image rows.
                y_image = max(min_eval_y, y_image)
            
            # Compute x in image coordinates from polynomial
            x_image = np.polyval(lane_coeffs, y_image)
            
            # Convert to vehicle coordinates (pass y_vehicle for distance-based scaling)
            x_vehicle, y_vehicle_corrected = self._convert_image_to_vehicle_coords(
                x_image, y_image, lookahead_distance=y_vehicle, debug_trace=debug_trace
            )
            
            # Compute heading (derivative of polynomial in vehicle coords)
            # FIXED: For straight roads, heading should be ~0°, not constant 9.04°
            # The issue is that coordinate conversion may add constant bias to dx
            # Solution: Use a more robust heading calculation that accounts for coordinate system
            if len(lane_coeffs) >= 2:
                # Sample nearby points for heading calculation
                y1_vehicle = max(0, y_vehicle - 0.5)
                y2_vehicle = y_vehicle + 0.5
                y1_image = self.image_height - (y1_vehicle / self.pixel_to_meter_y)
                y2_image = self.image_height - (y2_vehicle / self.pixel_to_meter_y)
                y1_image = max(0, min(self.image_height - 1, y1_image))
                y2_image = max(0, min(self.image_height - 1, y2_image))
                x1_image = np.polyval(lane_coeffs, y1_image)
                x2_image = np.polyval(lane_coeffs, y2_image)
                
                # FIXED: Use same lookahead_distance for both points to avoid coordinate conversion bias
                # The issue: using different lookahead_distance values causes different pixel_to_meter scaling
                # This can introduce constant bias in dx calculation
                # Solution: Use average distance for both conversions
                avg_distance = (y1_vehicle + y2_vehicle) / 2.0
                x1_vehicle, _ = self._convert_image_to_vehicle_coords(x1_image, y1_image, lookahead_distance=avg_distance)
                x2_vehicle, _ = self._convert_image_to_vehicle_coords(x2_image, y2_image, lookahead_distance=avg_distance)
                
                dx = x2_vehicle - x1_vehicle  # Lateral change (left/right)
                dy = y2_vehicle - y1_vehicle  # Forward change (ahead/behind)
                
                # FIXED: For straight roads, if dx is very small, heading should be ~0°
                # The constant 9.04° suggests coordinate conversion bias
                # Check if the lane is actually straight (small dx relative to dy)
                if abs(dx) < 0.01:  # Very small lateral change (< 1cm over 1m)
                    heading = 0.0
                    curvature = self._compute_curvature(lane_coeffs, y_image)
                else:
                    # Heading: angle from forward direction (y-axis)
                    # For vehicle coords: +y is forward (0°), +x is right (90°), -x is left (-90°)
                    # arctan2(dx, dy) gives angle from +y axis
                    heading = np.arctan2(dx, dy)
                    
                    # Compute curvature early so it can inform the heading zero decision
                    curvature = self._compute_curvature(lane_coeffs, y_image)

                    # For straight roads, heading MUST be near 0° to avoid bias.
                    # But on curve approach, allow heading through even if the
                    # image-space quadratic coefficient appears small.  The
                    # curvature guard prevents zeroing heading when a real curve
                    # is present (R1-validated: 0.002 has zero false positives
                    # on straights).
                    if len(lane_coeffs) >= 3:
                        quadratic_coeff = lane_coeffs[0]  # a in ax^2 + bx + c

                        quad_below = abs(quadratic_coeff) < self.heading_zero_quad_threshold
                        curv_below = abs(curvature) < self.heading_zero_curvature_guard

                        if quad_below and curv_below:
                            heading = 0.0
                        elif abs(heading) < np.radians(1.0):
                            heading = 0.0
                        else:
                            heading = np.clip(heading, -np.radians(30.0), np.radians(30.0))
                    else:
                        heading = 0.0
            else:
                heading = 0.0
                curvature = self._compute_curvature(lane_coeffs, y_image) if len(lane_coeffs) >= 2 else 0.0
            
            # Phase 2: conservatively attenuate far-field lateral contribution beyond effective horizon.
            scale_dynamic_horizon = 1.0
            if (
                effective_horizon_applied > 0.5
                and effective_horizon_m < self.lookahead_distance - 1e-6
                and y_vehicle_corrected > effective_horizon_m
            ):
                far_span = max(1e-3, self.lookahead_distance - effective_horizon_m)
                far_ratio = np.clip((y_vehicle_corrected - effective_horizon_m) / far_span, 0.0, 1.0)
                scale_dynamic_horizon = 1.0 - (
                    1.0 - self.dynamic_effective_horizon_farfield_scale_min
                ) * far_ratio

            scale_far_band_cap = 1.0
            if (
                self.far_band_contribution_cap_enabled
                and effective_horizon_applied > 0.5
                and y_vehicle_corrected >= self.far_band_contribution_cap_start_m
                and self.lookahead_distance > self.far_band_contribution_cap_start_m + 1e-6
            ):
                cap_ratio = np.clip(
                    (y_vehicle_corrected - self.far_band_contribution_cap_start_m)
                    / (self.lookahead_distance - self.far_band_contribution_cap_start_m),
                    0.0,
                    1.0,
                )
                scale_far_band_cap = max(
                    self.dynamic_effective_horizon_farfield_scale_min,
                    1.0 - self.far_band_contribution_cap_gain * cap_ratio,
                )
                far_band_limit_start_m = float(self.far_band_contribution_cap_start_m)
                far_band_limit_gain = float(self.far_band_contribution_cap_gain)

            combined_scale = float(min(scale_dynamic_horizon, scale_far_band_cap))
            x_vehicle = float(x_vehicle) * combined_scale
            far_band_limited_any_flags.append(1.0 if combined_scale < 0.999 else 0.0)
            if 12.0 <= float(y_vehicle_corrected) <= 20.0:
                far_band_scale_samples.append(float(combined_scale))
                far_band_limited_flags.append(1.0 if combined_scale < 0.999 else 0.0)

            point = TrajectoryPoint(
                x=x_vehicle,
                y=y_vehicle_corrected,
                heading=heading,
                velocity=self.target_speed,
                curvature=curvature
            )
            points.append(point)
        
        input_y = [float(v) for v in debug_trace.get("input_y", [])]
        post_y = [float(v) for v in debug_trace.get("post_y", [])]
        used_provided = [float(v) for v in debug_trace.get("used_provided_distance", [])]
        post_minus_pre = [float(py - iy) for iy, py in zip(input_y, post_y)]
        pre_x = [float(v) for v in debug_trace.get("pre_clip_x", [])]
        post_x = [float(v) for v in debug_trace.get("post_clip_x", [])]
        pre_x_lane_source = [float(v) for v in debug_trace.get("pre_clip_x_lane_source", [])]
        pre_x_distance_scale_delta = [float(v) for v in debug_trace.get("pre_clip_x_distance_scale_delta", [])]
        pre_x_camera_offset_delta = [float(v) for v in debug_trace.get("pre_clip_x_camera_offset_delta", [])]
        pre_x_abs = [abs(v) for v in pre_x if np.isfinite(v)]
        post_x_abs = [abs(v) for v in post_x if np.isfinite(v)]
        y_for_bands = [float(v) for v in post_y]
        clip_limit = float(self.x_clip_limit_m)
        near_clip_threshold = max(0.0, clip_limit - 0.05)

        def _band_indices(y0: float, y1: float) -> List[int]:
            idx: List[int] = []
            for i, yv in enumerate(y_for_bands):
                if np.isfinite(yv) and (y0 <= yv <= y1):
                    idx.append(i)
            return idx

        def _mean_abs(vals: List[float], idx: List[int]) -> float:
            if len(idx) == 0:
                return np.nan
            samples = [abs(float(vals[i])) for i in idx if i < len(vals) and np.isfinite(vals[i])]
            if len(samples) == 0:
                return np.nan
            return float(np.mean(np.array(samples, dtype=np.float64)))

        def _near_clip_frac(vals: List[float], idx: List[int]) -> float:
            if len(idx) == 0:
                return np.nan
            samples = [abs(float(vals[i])) for i in idx if i < len(vals) and np.isfinite(vals[i])]
            if len(samples) == 0:
                return np.nan
            flags = [1.0 if s >= near_clip_threshold else 0.0 for s in samples]
            return float(np.mean(np.array(flags, dtype=np.float64)))

        def _mean_signed(vals: List[float], idx: List[int]) -> float:
            if len(idx) == 0:
                return np.nan
            samples = [float(vals[i]) for i in idx if i < len(vals) and np.isfinite(vals[i])]
            if len(samples) == 0:
                return np.nan
            return float(np.mean(np.array(samples, dtype=np.float64)))

        idx_0_8 = _band_indices(0.0, 8.0)
        idx_8_12 = _band_indices(8.0, 12.0)
        idx_12_20 = _band_indices(12.0, 20.0)
        pre_inv = 1.0 if len(input_y) > 1 and input_y[0] > input_y[1] else 0.0
        post_inv = 1.0 if len(post_y) > 1 and post_y[0] > post_y[1] else 0.0
        introduced = 1.0 if (pre_inv < 0.5 and post_inv > 0.5) else 0.0

        self.last_generation_diagnostics = {
            "diag_available": 1.0,
            "diag_generated_by_fallback": 0.0,
            "diag_points_generated": float(len(points)),
            "diag_x_clip_count": float(debug_trace.get("clip_count", 0)),
            "diag_pre_y0": self._diag_value_at(input_y, 0),
            "diag_pre_y1": self._diag_value_at(input_y, 1),
            "diag_pre_y2": self._diag_value_at(input_y, 2),
            "diag_post_y0": self._diag_value_at(post_y, 0),
            "diag_post_y1": self._diag_value_at(post_y, 1),
            "diag_post_y2": self._diag_value_at(post_y, 2),
            "diag_used_provided_distance0": self._diag_value_at(used_provided, 0),
            "diag_used_provided_distance1": self._diag_value_at(used_provided, 1),
            "diag_used_provided_distance2": self._diag_value_at(used_provided, 2),
            "diag_post_minus_pre_y0": self._diag_value_at(post_minus_pre, 0),
            "diag_post_minus_pre_y1": self._diag_value_at(post_minus_pre, 1),
            "diag_post_minus_pre_y2": self._diag_value_at(post_minus_pre, 2),
            "diag_preclip_x0": self._diag_value_at(pre_x, 0),
            "diag_preclip_x1": self._diag_value_at(pre_x, 1),
            "diag_preclip_x2": self._diag_value_at(pre_x, 2),
            "diag_preclip_x_abs_max": float(np.max(pre_x_abs)) if len(pre_x_abs) > 0 else np.nan,
            "diag_preclip_x_abs_p95": float(np.percentile(np.array(pre_x_abs, dtype=np.float64), 95)) if len(pre_x_abs) > 0 else np.nan,
            "diag_preclip_mean_12_20m_lane_source_x": _mean_signed(pre_x_lane_source, idx_12_20),
            "diag_preclip_mean_12_20m_distance_scale_delta_x": _mean_signed(pre_x_distance_scale_delta, idx_12_20),
            "diag_preclip_mean_12_20m_camera_offset_delta_x": _mean_signed(pre_x_camera_offset_delta, idx_12_20),
            "diag_preclip_abs_mean_12_20m_lane_source_x": _mean_abs(pre_x_lane_source, idx_12_20),
            "diag_preclip_abs_mean_12_20m_distance_scale_delta_x": _mean_abs(pre_x_distance_scale_delta, idx_12_20),
            "diag_preclip_abs_mean_12_20m_camera_offset_delta_x": _mean_abs(pre_x_camera_offset_delta, idx_12_20),
            "diag_preclip_abs_mean_0_8m": _mean_abs(pre_x, idx_0_8),
            "diag_preclip_abs_mean_8_12m": _mean_abs(pre_x, idx_8_12),
            "diag_preclip_abs_mean_12_20m": _mean_abs(pre_x, idx_12_20),
            "diag_postclip_x0": self._diag_value_at(post_x, 0),
            "diag_postclip_x1": self._diag_value_at(post_x, 1),
            "diag_postclip_x2": self._diag_value_at(post_x, 2),
            "diag_postclip_abs_mean_0_8m": _mean_abs(post_x, idx_0_8),
            "diag_postclip_abs_mean_8_12m": _mean_abs(post_x, idx_8_12),
            "diag_postclip_abs_mean_12_20m": _mean_abs(post_x, idx_12_20),
            "diag_postclip_near_clip_frac_12_20m": _near_clip_frac(post_x, idx_12_20),
            "diag_first_segment_y0_gt_y1_pre": pre_inv if len(input_y) > 1 else np.nan,
            "diag_first_segment_y0_gt_y1_post": post_inv if len(post_y) > 1 else np.nan,
            "diag_inversion_introduced_after_conversion": introduced if len(input_y) > 1 and len(post_y) > 1 else np.nan,
            "diag_far_band_contribution_limited_active": (
                1.0 if np.any(np.asarray(far_band_limited_any_flags, dtype=np.float64) > 0.5) else 0.0
            ),
            "diag_far_band_contribution_limit_start_m": far_band_limit_start_m,
            "diag_far_band_contribution_limit_gain": far_band_limit_gain,
            "diag_far_band_contribution_scale_mean_12_20m": (
                float(np.mean(np.asarray(far_band_scale_samples, dtype=np.float64)))
                if len(far_band_scale_samples) > 0
                else np.nan
            ),
            "diag_far_band_contribution_limited_frac_12_20m": (
                float(np.mean(np.asarray(far_band_limited_flags, dtype=np.float64)))
                if len(far_band_limited_flags) > 0
                else np.nan
            ),
        }

        return points
    
    def _compute_curvature(self, lane_coeffs: np.ndarray, y: float) -> float:
        """
        Compute curvature at given y position (image-space, kept for backward compatibility).
        
        Args:
            lane_coeffs: Lane polynomial coefficients
            y: Y position
        
        Returns:
            Curvature (1/m)
        """
        # For polynomial x = ay^2 + by + c
        # Curvature = |d^2x/dy^2| / (1 + (dx/dy)^2)^(3/2)
        if len(lane_coeffs) < 2:
            return 0.0
        
        # First derivative
        dx_dy = 2 * lane_coeffs[0] * y + lane_coeffs[1]
        
        # Second derivative
        d2x_dy2 = 2 * lane_coeffs[0]
        
        # Curvature
        denominator = (1 + dx_dy ** 2) ** 1.5
        if denominator < 1e-6:
            return 0.0
        
        curvature = abs(d2x_dy2) / denominator
        return curvature
    
    def _compute_curvature_from_points(self, points: List[TrajectoryPoint], point_idx: int) -> float:
        """
        Compute curvature from vehicle-space trajectory points.
        This is more accurate than image-space calculation.
        
        Args:
            points: List of trajectory points in vehicle coordinates
            point_idx: Index of point to compute curvature for
        
        Returns:
            Curvature (1/m) in vehicle space
        """
        if len(points) < 2 or point_idx < 0 or point_idx >= len(points):
            return 0.0
        
        # For the first point, use forward difference
        if point_idx == 0:
            if len(points) < 2:
                return 0.0
            p0 = points[0]
            p1 = points[1]
            ds = np.sqrt((p1.x - p0.x)**2 + (p1.y - p0.y)**2)
            if ds < 1e-6:
                return 0.0
            d_theta = p1.heading - p0.heading
            # Normalize heading difference to [-pi, pi]
            while d_theta > np.pi:
                d_theta -= 2 * np.pi
            while d_theta < -np.pi:
                d_theta += 2 * np.pi
            curvature = abs(d_theta / ds)
            return curvature
        
        # For the last point, use backward difference
        if point_idx == len(points) - 1:
            p0 = points[point_idx - 1]
            p1 = points[point_idx]
            ds = np.sqrt((p1.x - p0.x)**2 + (p1.y - p0.y)**2)
            if ds < 1e-6:
                return 0.0
            d_theta = p1.heading - p0.heading
            # Normalize heading difference to [-pi, pi]
            while d_theta > np.pi:
                d_theta -= 2 * np.pi
            while d_theta < -np.pi:
                d_theta += 2 * np.pi
            curvature = abs(d_theta / ds)
            return curvature
        
        # For middle points, use central difference (more accurate)
        p_prev = points[point_idx - 1]
        p_curr = points[point_idx]
        p_next = points[point_idx + 1]
        
        # Compute arc lengths
        ds1 = np.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)
        ds2 = np.sqrt((p_next.x - p_curr.x)**2 + (p_next.y - p_curr.y)**2)
        ds_total = ds1 + ds2
        
        if ds_total < 1e-6:
            return 0.0
        
        # Compute heading change
        d_theta = p_next.heading - p_prev.heading
        # Normalize heading difference to [-pi, pi]
        while d_theta > np.pi:
            d_theta -= 2 * np.pi
        while d_theta < -np.pi:
            d_theta += 2 * np.pi
        
        # Curvature = d(heading)/d(arc_length)
        curvature = abs(d_theta / ds_total)
        
        return curvature
    
    def compute_reference_point_direct(self, left_coeffs: np.ndarray, 
                                       right_coeffs: np.ndarray,
                                       lookahead: float = 10.0) -> Optional[Dict]:
        """
        Compute reference point directly from lane coefficients at lookahead distance.
        This is simpler and more direct than going through center lane polynomial.
        
        Args:
            left_coeffs: Left lane polynomial coefficients
            right_coeffs: Right lane polynomial coefficients
            lookahead: Lookahead distance in meters
        
        Returns:
            Reference point dict with x, y, heading, or None if invalid
        """
        # Convert lookahead distance to image y coordinate
        # y_vehicle = 0 at vehicle, increases forward
        # y_image: bottom (y=height) is closest, top (y=0) is farthest
        y_image = self.image_height - (lookahead / self.pixel_to_meter_y)
        y_image = max(0, min(self.image_height - 1, y_image))
        
        # Evaluate each lane at lookahead distance
        left_x_image = np.polyval(left_coeffs, y_image)
        right_x_image = np.polyval(right_coeffs, y_image)
        
        # Compute midpoint (lane center) in image space
        center_x_image = (left_x_image + right_x_image) / 2.0
        
        # Convert to vehicle coordinates
        center_x_vehicle, center_y_vehicle = self._convert_image_to_vehicle_coords(
            center_x_image, y_image, lookahead_distance=lookahead
        )
        
        # Compute heading from lane direction
        # FIXED: Use a larger distance (2m) between points to avoid numerical issues
        # The previous 50 pixels (~0.7m) was too small, causing large heading errors
        # For a 50m radius curve, over 2m forward, lateral change = 2/50 = 0.04m
        # Heading = arctan(0.04/2.0) = 0.02 rad = 1.15° (correct!)
        # But with 0.7m dy, same dx gives arctan(0.04/0.7) = 0.057 rad = 3.3° (3x too high!)
        distance_ahead = 2.0  # meters - larger distance for more accurate heading
        lookahead_further = lookahead + distance_ahead
        
        # Convert lookahead_further to image y coordinate
        y_image_further = self.image_height - (lookahead_further / self.pixel_to_meter_y)
        y_image_further = max(0, min(self.image_height - 1, y_image_further))
        
        # Evaluate lanes at further point
        left_x_further = np.polyval(left_coeffs, y_image_further)
        right_x_further = np.polyval(right_coeffs, y_image_further)
        center_x_further_image = (left_x_further + right_x_further) / 2.0
        
        # Convert further point to vehicle coords
        center_x_further_vehicle, center_y_further_vehicle = self._convert_image_to_vehicle_coords(
            center_x_further_image, y_image_further, lookahead_distance=lookahead_further
        )
        
        # Compute heading: direction FROM lookahead TO further ahead
        # This gives us the heading at the lookahead point
        dx = center_x_further_vehicle - center_x_vehicle
        dy = center_y_further_vehicle - center_y_vehicle
        
        # CRITICAL FIX: Check for invalid dy (negative = pointing backwards, which shouldn't happen)
        # If dy is negative, the reference point is behind the vehicle, which is invalid
        # This can cause heading to jump from 0° to 180° (arctan2 with negative dy)
        if dy < 0:
            # Reference point is behind vehicle - this is invalid, use heading = 0
            # Log warning for debugging (but don't spam logs)
            import logging
            logger = logging.getLogger(__name__)
            if not hasattr(self, '_heading_warning_logged') or not self._heading_warning_logged:
                logger.warning(f"[TRAJECTORY] Invalid reference point: dy={dy:.3f}m (negative = behind vehicle). "
                             f"center_y_vehicle={center_y_vehicle:.3f}m, center_y_further={center_y_further_vehicle:.3f}m. "
                             f"Setting heading=0.0 to prevent 180° jump.")
                self._heading_warning_logged = True
            heading = 0.0
        elif abs(dy) < 0.1:  # FIXED: Increased threshold from 0.01 to 0.1 (2m distance should give ~2m dy)
            heading = 0.0
        else:
            heading = np.arctan2(dx, dy)
        
        # CRITICAL: Determine straight vs curve based on vehicle-space geometry, not image-space coeffs.
        # Image-space coefficients from segmentation can appear "small" even on real curves.
        if abs(heading) > np.radians(90.0):
            # Heading > 90° indicates invalid coordinate conversion (dy was negative or very small)
            import logging
            logger = logging.getLogger(__name__)
            if not hasattr(self, '_heading_180_warning_logged') or not self._heading_180_warning_logged:
                logger.warning(f"[TRAJECTORY] Invalid heading: {np.degrees(heading):.1f}° (should be -90° to +90°). "
                             f"dx={dx:.3f}m, dy={dy:.3f}m. Setting heading=0.0 to prevent 180° jump.")
                self._heading_180_warning_logged = True
            heading = 0.0
        else:
            center_coeffs_early = (np.asarray(left_coeffs) + np.asarray(right_coeffs)) / 2.0
            curvature_early = self._compute_curvature(center_coeffs_early, y_image)
            if abs(heading) < np.radians(1.0) and abs(curvature_early) < self.heading_zero_curvature_guard:
                heading = 0.0
        
        center_coeffs = (np.asarray(left_coeffs) + np.asarray(right_coeffs)) / 2.0
        curvature = self._compute_curvature(center_coeffs, y_image)
        
        return {
            'x': center_x_vehicle,
            'y': center_y_vehicle,
            'heading': heading,
            'velocity': self.target_speed,
            'curvature': curvature
        }
    
    def _create_straight_trajectory(self, vehicle_state: Optional[dict]) -> Trajectory:
        """
        Create straight trajectory when no lanes detected.
        
        Args:
            vehicle_state: Current vehicle state
        
        Returns:
            Straight trajectory
        """
        points = []
        num_points = int(self.lookahead_distance / self.point_spacing)
        
        heading = 0.0
        if vehicle_state and 'heading' in vehicle_state:
            heading = vehicle_state['heading']
        
        for i in range(num_points + 1):
            y = i * self.point_spacing
            x = 0.0  # Straight ahead
            
            point = TrajectoryPoint(
                x=x,
                y=y,
                heading=heading,
                velocity=self.target_speed * 0.5,  # Slower when no lanes
                curvature=0.0
            )
            points.append(point)
        
        self.last_generation_diagnostics = {
            **self._empty_generation_diagnostics(),
            "diag_available": 1.0,
            "diag_generated_by_fallback": 1.0,
            "diag_points_generated": float(len(points)),
            "diag_x_clip_count": 0.0,
        }
        return Trajectory(points=points, length=self.lookahead_distance)


class MLTrajectoryPlanner:
    """
    ML-based trajectory planner (placeholder for future implementation).
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML trajectory planner.
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        # TODO: Load model when implementing
    
    def plan(self, lane_coeffs: List[Optional[np.ndarray]], 
             vehicle_state: Optional[dict] = None) -> Trajectory:
        """
        Plan trajectory using ML model.
        
        Args:
            lane_coeffs: Lane polynomial coefficients
            vehicle_state: Current vehicle state
        
        Returns:
            Planned trajectory
        """
        # TODO: Implement ML-based planning
        # For now, fallback to rule-based
        planner = RuleBasedTrajectoryPlanner()
        return planner.plan(lane_coeffs, vehicle_state)

