"""
Trajectory planning inference pipeline.
"""

from typing import List, Optional, Dict
import numpy as np
import logging

from .models.trajectory_planner import (
    RuleBasedTrajectoryPlanner, MLTrajectoryPlanner, Trajectory
)

logger = logging.getLogger(__name__)


class TrajectoryPlanningInference:
    """Trajectory planning inference pipeline."""
    
    def __init__(self, planner_type: str = "rule_based", **kwargs):
        """
        Initialize trajectory planning inference.
        
        Args:
            planner_type: Type of planner ("rule_based" or "ml")
            **kwargs: Additional arguments for planner (image_width, image_height, etc.)
        """
        # Extract smoothing parameters before passing to planner
        reference_smoothing = kwargs.pop('reference_smoothing', 0.85)
        lane_smoothing_alpha = kwargs.pop('lane_smoothing_alpha', 0.7)  # NEW: Lane coefficient smoothing
        center_spline_enabled = kwargs.pop('center_spline_enabled', False)
        center_spline_degree = kwargs.pop('center_spline_degree', 2)
        center_spline_samples = kwargs.pop('center_spline_samples', 20)
        center_spline_alpha = kwargs.pop('center_spline_alpha', 0.7)
        lane_position_smoothing_alpha = kwargs.pop('lane_position_smoothing_alpha', 0.1)
        ref_point_jump_clamp_x = kwargs.pop('ref_point_jump_clamp_x', 0.0)
        ref_point_jump_clamp_heading = kwargs.pop('ref_point_jump_clamp_heading', 0.0)
        reference_x_max_abs = kwargs.pop('reference_x_max_abs', 0.0)
        target_lane = kwargs.pop('target_lane', 'center')
        target_lane_width_m = kwargs.pop('target_lane_width_m', 0.0)
        ref_sign_flip_threshold = kwargs.pop('ref_sign_flip_threshold', 0.0)
        ref_x_rate_limit = kwargs.pop('ref_x_rate_limit', 0.0)
        multi_lookahead_enabled = kwargs.pop('multi_lookahead_enabled', False)
        multi_lookahead_far_scale = kwargs.pop('multi_lookahead_far_scale', 1.5)
        multi_lookahead_blend_alpha = kwargs.pop('multi_lookahead_blend_alpha', 0.35)
        multi_lookahead_curvature_threshold = kwargs.pop('multi_lookahead_curvature_threshold', 0.01)
        
        if planner_type == "rule_based":
            # Pass lane smoothing to planner
            kwargs['lane_smoothing_alpha'] = lane_smoothing_alpha
            kwargs['center_spline_enabled'] = center_spline_enabled
            kwargs['center_spline_degree'] = center_spline_degree
            kwargs['center_spline_samples'] = center_spline_samples
            kwargs['center_spline_alpha'] = center_spline_alpha
            # Extract camera offset if provided
            camera_offset_x = kwargs.pop('camera_offset_x', 0.0)
            kwargs['camera_offset_x'] = camera_offset_x
            # Extract distance scaling factor if provided
            distance_scaling_factor = kwargs.pop('distance_scaling_factor', 0.875)
            kwargs['distance_scaling_factor'] = distance_scaling_factor
            self.planner = RuleBasedTrajectoryPlanner(**kwargs)
        elif planner_type == "ml":
            self.planner = MLTrajectoryPlanner(**kwargs)
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
        
        # Store smoothing parameter for reference point smoothing
        self.reference_smoothing_alpha = reference_smoothing
        self.lane_position_smoothing_alpha = float(np.clip(lane_position_smoothing_alpha, 0.0, 1.0))
        self.ref_point_jump_clamp_x = float(np.clip(ref_point_jump_clamp_x, 0.0, 5.0))
        self.ref_point_jump_clamp_heading = float(np.clip(ref_point_jump_clamp_heading, 0.0, 1.0))
        self.reference_x_max_abs = float(np.clip(reference_x_max_abs, 0.0, 10.0))
        self.target_lane = str(target_lane).lower()
        self.target_lane_width_m = float(np.clip(target_lane_width_m, 0.0, 10.0))
        self.ref_sign_flip_threshold = float(np.clip(ref_sign_flip_threshold, 0.0, 1.0))
        self.ref_x_rate_limit = float(np.clip(ref_x_rate_limit, 0.0, 1.0))
        self.multi_lookahead_enabled = bool(multi_lookahead_enabled)
        self.multi_lookahead_far_scale = float(np.clip(multi_lookahead_far_scale, 1.0, 3.0))
        self.multi_lookahead_blend_alpha = float(np.clip(multi_lookahead_blend_alpha, 0.0, 1.0))
        self.multi_lookahead_curvature_threshold = float(
            np.clip(multi_lookahead_curvature_threshold, 0.0, 0.1)
        )
        
        # NEW: Solution 5 - Reference Point Smoothing with Bias Compensation
        # Track smoothed reference point bias for compensation
        self.smoothed_bias_history = []  # History of smoothed reference point biases
        self.smoothed_bias_history_size = 30  # Track last 30 frames
        self.bias_compensation_threshold = 0.02  # 2cm persistent bias triggers compensation
        
        # Track last timestamp for time gap detection
        self.last_timestamp = None
        self.bias_compensation_persistence = 15  # Need 15 frames of persistent bias
        # FIXED: Relaxed std threshold from 0.05 to 0.06m to match current variance (0.0545m)
        # This allows adaptive corrections to trigger even with higher variance
        self.bias_compensation_std_threshold = 0.06  # Allow higher variance (was 0.05, current = 0.0545m)
        
        # For reference point smoothing
        self.last_smoothed_ref_point = None
        self.ref_debug_counter = 0

    def _compute_target_ref_from_coeffs(
        self,
        lane_coeffs: List[Optional[np.ndarray]],
        lookahead: float,
    ) -> Optional[Dict[str, float]]:
        """Compute a target-lane reference from lane coefficients at a lookahead distance."""
        valid_lanes = [coeffs for coeffs in lane_coeffs if coeffs is not None]
        if len(valid_lanes) < 2:
            return None

        lane1 = valid_lanes[0]
        lane2 = valid_lanes[1]
        image_height = getattr(self.planner, "image_height", 480.0)
        y_eval = float(image_height) - 1.0
        lane1_x = float(np.polyval(lane1, y_eval))
        lane2_x = float(np.polyval(lane2, y_eval))
        if lane1_x < lane2_x:
            left_coeffs, right_coeffs = lane1, lane2
        else:
            left_coeffs, right_coeffs = lane2, lane1

        if not hasattr(self.planner, "pixel_to_meter_y"):
            return None
        pixel_to_meter_y = float(self.planner.pixel_to_meter_y)
        if pixel_to_meter_y <= 0.0:
            return None
        y_image = float(image_height) - (float(lookahead) / pixel_to_meter_y)
        y_image = max(0.0, min(float(image_height) - 1.0, y_image))

        left_x_image = float(np.polyval(left_coeffs, y_image))
        right_x_image = float(np.polyval(right_coeffs, y_image))

        left_x_vehicle, _ = self.planner._convert_image_to_vehicle_coords(
            left_x_image, y_image, lookahead_distance=lookahead
        )
        right_x_vehicle, _ = self.planner._convert_image_to_vehicle_coords(
            right_x_image, y_image, lookahead_distance=lookahead
        )

        lane_width = right_x_vehicle - left_x_vehicle
        use_target_width = self.target_lane_width_m > 0.0
        if lane_width <= 0.0:
            center_x = (left_x_vehicle + right_x_vehicle) / 2.0
        elif self.target_lane == "right":
            if use_target_width:
                center_x = right_x_vehicle - (self.target_lane_width_m * 0.5)
            else:
                center_x = left_x_vehicle + (0.75 * lane_width)
        elif self.target_lane == "left":
            if use_target_width:
                center_x = left_x_vehicle + (self.target_lane_width_m * 0.5)
            else:
                center_x = left_x_vehicle + (0.25 * lane_width)
        else:
            center_x = (left_x_vehicle + right_x_vehicle) / 2.0

        heading = 0.0
        curvature = 0.0
        direct_ref = self.planner.compute_reference_point_direct(
            left_coeffs, right_coeffs, lookahead
        )
        if direct_ref is not None:
            heading = direct_ref.get("heading", 0.0)
            curvature = direct_ref.get("curvature", 0.0)

        return {
            "x": float(center_x),
            "y": float(lookahead),
            "heading": float(heading),
            "velocity": float(self.planner.target_speed),
            "curvature": float(curvature),
        }

    def get_curvature_at_lookahead(
        self,
        lane_coeffs: Optional[List[Optional[np.ndarray]]],
        lookahead: float,
    ) -> Optional[float]:
        """Return curvature at lookahead using lane coefficients (if available)."""
        if lane_coeffs is None:
            return None
        ref = self._compute_target_ref_from_coeffs(lane_coeffs, lookahead)
        if ref is None:
            return None
        return float(ref.get("curvature", 0.0))
    
    def plan(self, lane_coeffs: List[Optional[np.ndarray]], 
             vehicle_state: Optional[Dict] = None) -> Trajectory:
        """
        Plan trajectory from lane lines.
        
        Args:
            lane_coeffs: List of lane polynomial coefficients from perception
            vehicle_state: Current vehicle state
        
        Returns:
            Planned trajectory
        """
        return self.planner.plan(lane_coeffs, vehicle_state)
    
    def get_reference_point(self, trajectory: Trajectory, 
                           lookahead: float = 10.0,
                           lane_coeffs: Optional[List[Optional[np.ndarray]]] = None,
                           lane_positions: Optional[Dict[str, float]] = None,
                           use_direct: bool = True,
                           timestamp: Optional[float] = None,
                           confidence: Optional[float] = None) -> Optional[Dict]:
        """
        Get reference point at specified lookahead distance.
        Can use direct midpoint computation (simpler, more accurate) or trajectory-based.
        Applies exponential smoothing to reduce instability.
        
        Args:
            trajectory: Planned trajectory
            lookahead: Lookahead distance in meters
            lane_coeffs: Lane coefficients (for direct computation)
            lane_positions: Dict with 'left_lane_line_x' and 'right_lane_line_x' in vehicle coords (preferred)
            use_direct: If True, compute directly from lane coefficients (simpler, more accurate)
        
        Returns:
            Reference point with x, y, heading, velocity, curvature
            Also includes raw (unsmoothed) values for analysis
        """
        # PREFERRED: Use lane positions in vehicle coordinates if available (most accurate)
        if use_direct and lane_positions is not None:
            left_lane_line_x = lane_positions.get('left_lane_line_x') or lane_positions.get('left_lane_x')  # Backward compatibility
            right_lane_line_x = lane_positions.get('right_lane_line_x') or lane_positions.get('right_lane_x')  # Backward compatibility
            if left_lane_line_x is not None and right_lane_line_x is not None:
                # CRITICAL FIX: If both lanes are 0.0 (perception failed), return None
                # Don't drive when perception fails - car should not move
                if abs(left_lane_line_x) < 0.001 and abs(right_lane_line_x) < 0.001:
                    # Perception failed - return None to prevent movement
                    logger.warning(f"[TRAJECTORY] Perception failed (both lanes = 0.0) - returning None to prevent movement")
                    return None
                
                # Road center in vehicle coordinates (matches cyan line calculation in visualizer)
                # This is what ref_x should be - the center of the detected road
                lane_width = right_lane_line_x - left_lane_line_x
                
                if lane_width <= 0:
                    center_x = (left_lane_line_x + right_lane_line_x) / 2.0
                    logger.debug(f"[TRAJECTORY] Invalid lane_width={lane_width:.3f}, using road center")
                else:
                    # Road center = midpoint of detected lane lines (same as cyan line)
                    center_x = (left_lane_line_x + right_lane_line_x) / 2.0
                    logger.debug(f"[TRAJECTORY] Road center: left={left_lane_line_x:.3f}, right={right_lane_line_x:.3f}, center={center_x:.3f}")
                
                # CRITICAL FIX: For straight roads, heading should be 0° regardless of lateral offset
                # When using lane_positions (vehicle coordinates), we don't have curvature information
                # Tests expect heading ~0° for straight roads, so default to 0° when using lane_positions
                # Only use lane_coeffs to compute heading if available (which has curvature information)
                heading = 0.0
                curvature = 0.0
                if lane_coeffs is not None:
                    # Use lane_coeffs to compute heading if available (has curvature information)
                    valid_lanes = [coeffs for coeffs in lane_coeffs if coeffs is not None]
                    if len(valid_lanes) >= 2:
                        # Use the trajectory planner's direct computation to get heading
                        # This properly handles curvature and computes heading from lane geometry
                        direct_ref = self.planner.compute_reference_point_direct(
                            valid_lanes[0], valid_lanes[1], lookahead
                        )
                        if direct_ref is not None:
                            heading = direct_ref.get('heading', 0.0)
                            curvature = direct_ref.get('curvature', 0.0)
                
                raw_ref_point = {
                    'x': center_x,
                    'y': lookahead,
                    'heading': heading,  # Use computed heading, not hardcoded 0°
                    'velocity': self.planner.target_speed,
                    'curvature': curvature,
                    'method': 'lane_positions',  # NEW: Track which method was used
                    'perception_center_x': center_x  # NEW: Store perception center for comparison
                }
                
                # Confidence-aware speed scaling (lower confidence => slower)
                if confidence is not None:
                    conf = float(np.clip(confidence, 0.0, 1.0))
                    if conf < 0.4:
                        scale = 0.5 + 0.5 * (conf / 0.4)  # 0.5 to 1.0
                        raw_ref_point['velocity'] *= scale
                # CRITICAL: When using lane_positions (most accurate), still apply jump detection
                # to prevent large jumps when Unity pauses or perception fails
                # But use minimal smoothing to avoid preserving bias
                raw_ref_point['method'] = 'lane_positions'  # Preserve method
                raw_ref_point['perception_center_x'] = center_x  # Preserve perception center
                # Store raw values for analysis
                raw_ref_point['raw_x'] = center_x
                raw_ref_point['raw_y'] = lookahead
                raw_ref_point['raw_heading'] = heading
                # DEBUG: Log calculation details for specific frames
                if hasattr(self, 'frame_count') and self.frame_count in [134, 135, 136]:
                    logger.warning(
                        f"[TRAJECTORY DEBUG Frame {getattr(self, 'frame_count', '?')}] "
                        f"left={left_lane_line_x:.3f} right={right_lane_line_x:.3f} "
                        f"lane_width={lane_width:.3f} road_center={center_x:.3f} "
                        f"raw_x={raw_ref_point['raw_x']:.3f}"
                    )
                # Optional: multi-lookahead blend using lane coefficients (better straight/curve transitions)
                if (
                    self.multi_lookahead_enabled
                    and lane_coeffs is not None
                    and abs(raw_ref_point.get('curvature', 0.0)) <= self.multi_lookahead_curvature_threshold
                ):
                    far_ref = self._compute_target_ref_from_coeffs(lane_coeffs, lookahead * self.multi_lookahead_far_scale)
                    if far_ref is not None:
                        alpha = self.multi_lookahead_blend_alpha
                        raw_ref_point['heading'] = (
                            (1.0 - alpha) * raw_ref_point['heading'] + alpha * far_ref['heading']
                        )
                        raw_ref_point['curvature'] = (
                            (1.0 - alpha) * raw_ref_point['curvature'] + alpha * far_ref['curvature']
                        )
                        raw_ref_point['velocity'] = (
                            (1.0 - alpha) * raw_ref_point['velocity'] + alpha * far_ref['velocity']
                        )
                        raw_ref_point['method'] = 'multi_lookahead_heading_blend'
                        raw_ref_point['raw_heading'] = raw_ref_point['heading']
                # Apply jump detection even for lane_positions (but with minimal smoothing)
                smoothed = self._apply_smoothing(
                    raw_ref_point,
                    use_light_smoothing=True,
                    minimal_smoothing=True,
                    timestamp=timestamp,
                    confidence=confidence,
                )
                smoothed['method'] = 'lane_positions'  # Preserve method
                self.ref_debug_counter += 1
                if self.ref_debug_counter % 10 == 0:
                    logger.info(
                        "[TRAJECTORY REF DEBUG] lane_width=%.3f left=%.3f right=%.3f "
                        "road_center=%.3f raw_x=%.3f smoothed_x=%.3f",
                        lane_width,
                        left_lane_line_x,
                        right_lane_line_x,
                        center_x,
                        raw_ref_point['x'],
                        smoothed.get('x', raw_ref_point['x'])
                    )
                return smoothed
        
        # FALLBACK: Use direct midpoint computation from lane coefficients
        if use_direct and lane_coeffs is not None:
            valid_lanes = [coeffs for coeffs in lane_coeffs if coeffs is not None]
            if len(valid_lanes) >= 2:
                # Determine left and right lanes
                lane1 = valid_lanes[0]
                lane2 = valid_lanes[1]
                
                # Check order at bottom of image
                # Access image_height from planner (it's a RuleBasedTrajectoryPlanner attribute)
                if hasattr(self.planner, 'image_height'):
                    h = self.planner.image_height
                else:
                    h = 480  # Default
                x1 = np.polyval(lane1, h)
                x2 = np.polyval(lane2, h)
                
                if x1 < x2:
                    left_coeffs = lane1
                    right_coeffs = lane2
                else:
                    left_coeffs = lane2
                    right_coeffs = lane1
                
                # Compute reference point directly from lane coefficients
                raw_ref_point = self.planner.compute_reference_point_direct(
                    left_coeffs, right_coeffs, lookahead
                )
                
                if raw_ref_point is not None:
                    # NEW: Track which method was used
                    raw_ref_point['method'] = 'lane_coeffs'
                    # Calculate perception center from coeffs for comparison
                    # (approximate - would need to convert from image coords)
                    raw_ref_point['perception_center_x'] = None  # Not available when using coeffs
                    smoothed = self._apply_smoothing(raw_ref_point, timestamp=timestamp, confidence=confidence)
                    smoothed['method'] = 'lane_coeffs'  # Preserve method
                    return smoothed
        
        # Fallback to trajectory-based method
        if not trajectory.points:
            return None
        
        # Find point closest to lookahead distance
        target_point = None
        min_dist = float('inf')
        
        for point in trajectory.points:
            dist = np.sqrt(point.x ** 2 + point.y ** 2)
            if abs(dist - lookahead) < min_dist:
                min_dist = abs(dist - lookahead)
                target_point = point
        
        if target_point is None:
            # Use last point
            target_point = trajectory.points[-1]
        
        # Store raw (unsmoothed) reference point
        raw_ref_point = {
            'x': target_point.x,
            'y': target_point.y,
            'heading': target_point.heading,
            'velocity': target_point.velocity,
            'curvature': target_point.curvature,
            'method': 'trajectory',  # NEW: Track which method was used
            'perception_center_x': None  # Not available when using trajectory points
        }
        
        # Apply smoothing (trajectory-based method uses normal smoothing, not light smoothing)
        smoothed = self._apply_smoothing(raw_ref_point, use_light_smoothing=False, confidence=confidence)
        smoothed['method'] = 'trajectory'  # Preserve method
        return smoothed
        if self.last_smoothed_ref_point is not None:
            if is_fallback_trajectory:
                # Lane detection failed - use fallback trajectory directly (no smoothing to preserve bias)
                # This prevents preserving biased ref_x when lanes fail
                smoothed_x = raw_ref_point['x']  # Use centered trajectory (0.0) directly
                smoothed_y = raw_ref_point['y']
                smoothed_heading = raw_ref_point['heading']
                smoothed_velocity = raw_ref_point['velocity']
            else:
                # Normal smoothing
                # Use lighter smoothing for x when use_light_smoothing is True
                smoothed_x = alpha_x * self.last_smoothed_ref_point['x'] + (1 - alpha_x) * raw_ref_point['x']
                smoothed_y = alpha * self.last_smoothed_ref_point['y'] + (1 - alpha) * raw_ref_point['y']
                smoothed_heading = alpha * self.last_smoothed_ref_point['heading'] + (1 - alpha) * raw_ref_point['heading']
                smoothed_velocity = alpha * self.last_smoothed_ref_point['velocity'] + (1 - alpha) * raw_ref_point['velocity']
        else:
            smoothed_x = raw_ref_point['x']
            smoothed_y = raw_ref_point['y']
            smoothed_heading = raw_ref_point['heading']
            smoothed_velocity = raw_ref_point['velocity']
        
        # NEW: Solution 5 - Bias Compensation in Smoothing
        # Track smoothed reference point bias
        self.smoothed_bias_history.append(smoothed_x)
        if len(self.smoothed_bias_history) > self.smoothed_bias_history_size:
            self.smoothed_bias_history.pop(0)
        
        # Check if smoothed reference point has persistent bias
        bias_compensation = 0.0
        if len(self.smoothed_bias_history) >= self.bias_compensation_persistence:
            bias_array = np.array(self.smoothed_bias_history[-self.bias_compensation_persistence:])
            bias_mean = np.mean(bias_array)
            bias_std = np.std(bias_array)
            bias_abs_mean = abs(bias_mean)
            
            # NEW: Alternative correction mechanisms (mean-based, not variance-dependent)
            # Mechanism 1: Variance-based (original) - for low variance biases
            if bias_std < self.bias_compensation_std_threshold and bias_abs_mean > self.bias_compensation_threshold:
                # Persistent bias detected - apply compensation
                bias_compensation = -bias_mean  # Compensate for bias (opposite direction)
            # Mechanism 2: Mean-based correction (for high variance biases)
            # If bias is significant (|mean| > threshold), apply correction regardless of variance
            elif bias_abs_mean > self.bias_compensation_threshold:
                # Significant bias detected - apply compensation with variance adjustment
                # Use reduced compensation factor if variance is high (to avoid overcorrection)
                variance_factor = min(1.0, self.bias_compensation_std_threshold / max(bias_std, 0.01))
                bias_compensation = -bias_mean * variance_factor
            # Mechanism 3: Time-based persistence (if bias persists over time)
            # Check if bias has been consistently in same direction for multiple frames
            if len(self.smoothed_bias_history) >= 20:
                recent_bias = bias_array[-20:] if len(bias_array) >= 20 else bias_array
                consistent_direction = np.all(np.sign(recent_bias) == np.sign(bias_mean))
                if consistent_direction and bias_abs_mean > self.bias_compensation_threshold:
                    # Bias consistently in same direction - apply compensation
                    # Use mean-based compensation (not variance-dependent)
                    if abs(bias_compensation) < abs(bias_mean * 0.3):  # Only if not already compensated
                        bias_compensation = -bias_mean * 0.5  # Partial compensation for high variance
        
        # Apply bias compensation
        smoothed_x += bias_compensation

        if self.reference_x_max_abs > 0.0:
            smoothed_x = float(np.clip(
                smoothed_x, -self.reference_x_max_abs, self.reference_x_max_abs
            ))
        
        # Update raw_ref_point with smoothed and compensated values
        raw_ref_point['x'] = smoothed_x
        raw_ref_point['y'] = smoothed_y
        raw_ref_point['heading'] = smoothed_heading
        raw_ref_point['velocity'] = smoothed_velocity
        
        # Store smoothed values for next iteration
        self.last_smoothed_ref_point = {
            'x': raw_ref_point['x'],
            'y': raw_ref_point['y'],
            'heading': raw_ref_point['heading'],
            'velocity': raw_ref_point['velocity'],
            'curvature': raw_ref_point['curvature']
        }
        
        # Store raw values for analysis (before smoothing)
        raw_ref_point['raw_x'] = target_point.x
        raw_ref_point['raw_y'] = target_point.y
        raw_ref_point['raw_heading'] = target_point.heading
        
        return self._apply_smoothing(raw_ref_point, timestamp=timestamp)
    
    def _apply_smoothing(
        self,
        raw_ref_point: Dict,
        use_light_smoothing: bool = False,
        minimal_smoothing: bool = False,
        timestamp: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Dict:
        """
        Apply exponential smoothing to reference point.
        
        Args:
            raw_ref_point: Raw reference point dict
            use_light_smoothing: If True, use lighter smoothing for x coordinate (for lane_positions)
            minimal_smoothing: If True, use minimal smoothing (almost no smoothing) - for accurate lane_positions
        
        Returns:
            Smoothed reference point dict
        """
        # FIXED: Detect when trajectory planner used fallback (straight trajectory from lane failure)
        # CRITICAL FIX: Also detect when reference point jumps dramatically (perception failure)
        if minimal_smoothing and self.last_smoothed_ref_point is not None:
            raw_x = raw_ref_point.get('raw_x', raw_ref_point['x'])
            last_x = self.last_smoothed_ref_point.get('x', 0.0)
            if self.ref_sign_flip_threshold > 0.0:
                if raw_x * last_x < 0 and abs(raw_x) < self.ref_sign_flip_threshold:
                    raw_ref_point['raw_x'] = raw_x
                    raw_ref_point['x'] = last_x
            if self.ref_x_rate_limit > 0.0:
                delta_x = raw_ref_point['x'] - last_x
                if abs(delta_x) > self.ref_x_rate_limit:
                    raw_ref_point['raw_x'] = raw_ref_point.get('raw_x', raw_x)
                    raw_ref_point['x'] = last_x + np.sign(delta_x) * self.ref_x_rate_limit

        is_fallback_trajectory = (abs(raw_ref_point['x']) < 0.01 and 
                                  abs(raw_ref_point['heading']) < 0.01 and
                                  self.last_smoothed_ref_point is not None and
                                  abs(self.last_smoothed_ref_point['x']) > 0.05)
        
        # CRITICAL FIX: Detect large jumps in reference point (indicates perception failure)
        # When ref_x jumps >0.2m, it's likely a perception failure, not a real trajectory change
        # Also detect time gaps (Unity pauses) - be more conservative with jump detection
        is_large_jump = False
        has_time_gap = False
        if timestamp is not None and self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0.5:  # Large time gap (>0.5s) indicates Unity pause
                has_time_gap = True
                # After time gap, reset last_smoothed_ref_point to allow fresh start
                # But still check for jumps to prevent perception failures
                logger.warning(f"Large time gap detected: {dt:.3f}s - resetting reference point state")
                self.last_smoothed_ref_point = None
        
        if self.last_smoothed_ref_point is not None:
            ref_x_change = abs(raw_ref_point['x'] - self.last_smoothed_ref_point['x'])
            ref_heading_change = abs(raw_ref_point['heading'] - self.last_smoothed_ref_point['heading'])
            # Large jump: >0.2m lateral change OR >0.1 rad heading change
            # CRITICAL FIX: When using lane_positions (minimal_smoothing=True), perception is accurate
            # and large changes are likely real (car turning, perception improving), not failures
            # Use a more lenient threshold for lane_positions to prevent freezing
            # If time gap exists, use stricter threshold (0.15m instead of 0.2m)
            if minimal_smoothing:
                # Using lane_positions - be more lenient (0.5m instead of 0.2m)
                # Large changes are likely real trajectory changes, not perception failures
                jump_threshold_x = 0.5 if has_time_gap else 0.5  # Same threshold regardless of time gap
                jump_threshold_heading = 0.2 if has_time_gap else 0.2  # More lenient for heading too
            else:
                # Using trajectory-based method - use normal thresholds
                jump_threshold_x = 0.15 if has_time_gap else 0.2
                jump_threshold_heading = 0.08 if has_time_gap else 0.1
            is_large_jump = ref_x_change > jump_threshold_x or ref_heading_change > jump_threshold_heading
        
        # Apply exponential smoothing
        # CRITICAL: When using lane_positions (use_light_smoothing=True), use lighter smoothing
        # to avoid preserving bias. Lane positions are already accurate, so we don't need heavy smoothing.
        # If minimal_smoothing=True, use almost no smoothing (0.02 alpha) to allow fast correction
        # FIXED: Reduced from 0.05 to 0.02 to allow even faster correction and prevent freezing
        # With alpha_x = 0.02, we get 98% new value, 2% old value - should update quickly
        alpha = self.reference_smoothing_alpha
        if minimal_smoothing:
            alpha_x = self.lane_position_smoothing_alpha
        else:
            alpha_x = alpha * 0.2 if use_light_smoothing else alpha  # Lighter smoothing for x when using lane_positions (was 0.3)
        
        # Confidence-aware smoothing: lower confidence => more smoothing
        if confidence is not None:
            conf = float(np.clip(confidence, 0.0, 1.0))
            if conf < 0.4:
                boost = (0.4 - conf) / 0.4  # 0..1
                alpha = min(0.95, alpha + boost * 0.4)
                if minimal_smoothing:
                    alpha_x = min(0.8, alpha_x + boost * 0.6)
                else:
                    alpha_x = min(0.9, alpha_x + boost * 0.4)
        if self.last_smoothed_ref_point is not None:
            # Optional jump clamp to reduce reference jitter before smoothing.
            if not has_time_gap:
                if self.ref_point_jump_clamp_x > 0.0:
                    raw_x = raw_ref_point.get('raw_x', raw_ref_point['x'])
                    raw_ref_point['raw_x'] = raw_x
                    last_x = self.last_smoothed_ref_point['x']
                    delta_x = np.clip(raw_ref_point['x'] - last_x,
                                      -self.ref_point_jump_clamp_x,
                                      self.ref_point_jump_clamp_x)
                    raw_ref_point['x'] = last_x + delta_x
                if self.ref_point_jump_clamp_heading > 0.0:
                    raw_heading = raw_ref_point.get('raw_heading', raw_ref_point['heading'])
                    raw_ref_point['raw_heading'] = raw_heading
                    last_heading = self.last_smoothed_ref_point['heading']
                    delta_heading = np.clip(raw_ref_point['heading'] - last_heading,
                                            -self.ref_point_jump_clamp_heading,
                                            self.ref_point_jump_clamp_heading)
                    raw_ref_point['heading'] = last_heading + delta_heading
            if is_fallback_trajectory:
                # Lane detection failed - use fallback trajectory directly
                smoothed_x = raw_ref_point['x']
                smoothed_y = raw_ref_point['y']
                smoothed_heading = raw_ref_point['heading']
                smoothed_velocity = raw_ref_point['velocity']
            elif is_large_jump:
                # CRITICAL FIX: Large jump detected - could be perception failure OR Unity pause recovery OR real trajectory change
                # When using lane_positions (minimal_smoothing=True), perception is accurate and large changes
                # are likely real (car turning, perception improving), not failures
                # Only reject if using trajectory-based method AND no time gap
                if has_time_gap:
                    # After Unity pause, accept the new reference point (it's expected to be different)
                    logger.info(f"[TRAJECTORY] Large jump after Unity pause - accepting new reference point")
                    smoothed_x = raw_ref_point['x']
                    smoothed_y = raw_ref_point['y']
                    smoothed_heading = raw_ref_point['heading']
                    smoothed_velocity = raw_ref_point['velocity']
                elif minimal_smoothing:
                    # Using lane_positions - accept the jump (perception is accurate, changes are real)
                    logger.info(f"[TRAJECTORY] Large jump detected with lane_positions - accepting (perception is accurate)")
                    smoothed_x = raw_ref_point['x']
                    smoothed_y = raw_ref_point['y']
                    smoothed_heading = raw_ref_point['heading']
                    smoothed_velocity = raw_ref_point['velocity']
                else:
                    # Using trajectory-based method AND no time gap - likely perception failure, keep using last known good reference
                    logger.warning(f"[TRAJECTORY] Large jump detected (trajectory-based, no time gap) - keeping last known good reference")
                    smoothed_x = self.last_smoothed_ref_point['x']
                    smoothed_y = self.last_smoothed_ref_point['y']
                    smoothed_heading = self.last_smoothed_ref_point['heading']
                    smoothed_velocity = self.last_smoothed_ref_point['velocity']
            else:
                # Normal smoothing
                # Use lighter smoothing for x when use_light_smoothing is True
                smoothed_x = alpha_x * self.last_smoothed_ref_point['x'] + (1 - alpha_x) * raw_ref_point['x']
                smoothed_y = alpha * self.last_smoothed_ref_point['y'] + (1 - alpha) * raw_ref_point['y']
                smoothed_heading = alpha * self.last_smoothed_ref_point['heading'] + (1 - alpha) * raw_ref_point['heading']
                smoothed_velocity = alpha * self.last_smoothed_ref_point['velocity'] + (1 - alpha) * raw_ref_point['velocity']
        else:
            smoothed_x = raw_ref_point['x']
            smoothed_y = raw_ref_point['y']
            smoothed_heading = raw_ref_point['heading']
            smoothed_velocity = raw_ref_point['velocity']
        
        # Bias compensation
        self.smoothed_bias_history.append(smoothed_x)
        if len(self.smoothed_bias_history) > self.smoothed_bias_history_size:
            self.smoothed_bias_history.pop(0)
        
        # CRITICAL FIX: Disable bias compensation when using lane_positions (minimal_smoothing=True)
        # The bias compensation is trying to correct for perception bias, but it's overcorrecting
        # and causing the reference point to be wrong in the opposite direction.
        # When using lane_positions, we rely on perception directly, so bias compensation
        # should be handled at the perception level, not here.
        # FIXED: Only apply bias compensation when NOT using minimal_smoothing (i.e., when using trajectory-based method)
        bias_compensation = 0.0
        if not minimal_smoothing and len(self.smoothed_bias_history) >= self.bias_compensation_persistence:
            bias_array = np.array(self.smoothed_bias_history[-self.bias_compensation_persistence:])
            bias_mean = np.mean(bias_array)
            bias_std = np.std(bias_array)
            bias_abs_mean = abs(bias_mean)
            
            if bias_std < self.bias_compensation_std_threshold and bias_abs_mean > self.bias_compensation_threshold:
                bias_compensation = -bias_mean
            elif bias_abs_mean > self.bias_compensation_threshold:
                variance_factor = min(1.0, self.bias_compensation_std_threshold / max(bias_std, 0.01))
                bias_compensation = -bias_mean * variance_factor
        
        smoothed_x += bias_compensation
        
        # Update timestamp
        if timestamp is not None:
            self.last_timestamp = timestamp
        
        # Update and return
        result = {
            'x': smoothed_x,
            'y': smoothed_y,
            'heading': smoothed_heading,
            'velocity': smoothed_velocity,
            'curvature': raw_ref_point.get('curvature', 0.0),
            'raw_x': raw_ref_point.get('raw_x', raw_ref_point.get('x', smoothed_x)),  # FIX: Read from 'raw_x' first, not 'x'
            'raw_y': raw_ref_point.get('raw_y', raw_ref_point.get('y', smoothed_y)),
            'raw_heading': raw_ref_point.get('raw_heading', raw_ref_point.get('heading', smoothed_heading)),
            'perception_center_x': raw_ref_point.get('perception_center_x'),  # FIX: Preserve perception center for analysis
            'method': raw_ref_point.get('method', 'unknown')  # Preserve method for debugging
        }
        
        self.last_smoothed_ref_point = {
            'x': smoothed_x,
            'y': smoothed_y,
            'heading': smoothed_heading,
            'velocity': smoothed_velocity,
            'curvature': raw_ref_point.get('curvature', 0.0)
        }
        
        return result
