"""
PID controller for vehicle control.
Handles both lateral (steering) and longitudinal (throttle/brake) control.
"""

import numpy as np
from typing import Optional, Tuple


class PIDController:
    """
    PID controller with integral windup protection.
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 integral_limit: Optional[float] = None, output_limit: Optional[Tuple[float, float]] = None):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            integral_limit: Limit for integral term (anti-windup)
            output_limit: (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller.
        
        Args:
            error: Current error
            dt: Time step
        
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term
        if self.prev_time is not None and dt > 0:
            d_error = (error - self.prev_error) / dt
            d_term = self.kd * d_error
        else:
            d_term = 0.0
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limit is not None:
            output = np.clip(output, self.output_limit[0], self.output_limit[1])
        
        # Update state
        self.prev_error = error
        self.prev_time = (self.prev_time or 0.0) + dt
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class LateralController:
    """
    Lateral control (steering) using PID controller.
    """
    
    def __init__(self, kp: float = 0.3, ki: float = 0.0, kd: float = 0.1,
                 lookahead_distance: float = 10.0, max_steering: float = 0.5,
                 deadband: float = 0.02, heading_weight: float = 0.5,
                 lateral_weight: float = 0.5, error_clip: float = np.pi / 4,
                 integral_limit: float = 0.3):
        """
        Initialize lateral controller.
        
        Args:
            kp: Proportional gain (reduced for smoother control)
            ki: Integral gain
            kd: Derivative gain (reduced for less oscillation)
            lookahead_distance: Lookahead distance for reference point (meters)
            max_steering: Maximum steering angle (-1.0 to 1.0, reduced for stability)
        """
        self.lookahead_distance = lookahead_distance
        self.max_steering = max_steering
        self.base_deadband = deadband
        self.deadband = deadband
        self.heading_weight = heading_weight
        self.lateral_weight = lateral_weight
        self.error_clip = error_clip
        self.pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            integral_limit=integral_limit,  # Use from parameter (can be set from config)
            output_limit=(-max_steering, max_steering)  # Reduced steering range for stability
        )
        # Integral reset tracking
        self.small_error_count = 0
        self.last_error_sign = None
        self.recent_large_error = False  # Track if we recently had large error (on curve)
        self.large_error_frames = 0  # Count consecutive frames with large error
        
        # NEW: Error smoothing to reduce sensitivity to perception noise
        # Smooth lateral error before feeding to PID to prevent control oscillations from perception instability
        self.smoothed_lateral_error = None
        self.base_error_smoothing_alpha = 0.7  # Exponential smoothing: 70% new value, 30% old value (can be tuned)
        self.error_smoothing_alpha = self.base_error_smoothing_alpha

        # Straight-away adaptive tuning (deadband + smoothing)
        self.straight_curvature_threshold = 0.02
        self.straight_window = 60  # ~2 seconds at 30 FPS
        self.straight_oscillation_high = 0.20
        self.straight_oscillation_low = 0.05
        self.deadband_min = 0.01
        self.deadband_max = 0.08
        self.smoothing_min = 0.60
        self.smoothing_max = 0.90
        self._straight_frames = 0
        self._straight_sign_changes = 0
        self._last_straight_sign = 0
        self.straight_oscillation_rate = 0.0
    
    def compute_steering(self, current_heading: float, reference_point: dict,
                        vehicle_position: Optional[np.ndarray] = None, 
                        return_metadata: bool = False) -> float:
        """
        Compute steering command.
        
        Args:
            current_heading: Current vehicle heading (radians)
            reference_point: Reference point from trajectory (x, y, heading)
            vehicle_position: Current vehicle position [x, y] (optional)
            return_metadata: If True, return dict with steering and internal state
        
        Returns:
            Steering command (-1.0 to 1.0) if return_metadata=False
            Dict with steering, errors, and PID state if return_metadata=True
        """
        if vehicle_position is None:
            vehicle_position = np.array([0.0, 0.0])
        
        # Get reference point data
        ref_x = reference_point['x']  # Already in vehicle frame (meters, lateral offset)
        ref_y = reference_point['y']  # Already in vehicle frame (meters, forward distance)
        desired_heading = reference_point['heading']  # Path curvature (not correction needed)
        
        # Compute heading error
        # CRITICAL FIX: On curves, ref_heading represents path curvature, not correction needed
        # This causes conflicts: path curves right (ref_heading > 0) but car heading more right
        # → heading_error < 0 (suggests steer left) conflicts with lateral_error > 0 (steer right)
        #
        # Solution: Calculate heading_error as the heading needed to point TOWARD the reference point
        # This represents the correction needed, not the path curvature
        # Heading needed to point toward reference point
        # This is the correction needed, not the path curvature
        heading_to_ref = np.arctan2(ref_x, ref_y) if ref_y > 0.1 else 0.0
        
        # Use path curvature (ref_heading) only for feedforward, not for error calculation
        # For error calculation, use heading_to_ref (correction needed)
        # But blend with ref_heading for smooth curves
        path_curvature = desired_heading  # This is the path curvature
        
        # On curves, we need both:
        # 1. Heading to point toward reference (correction)
        # 2. Path curvature (for feedforward)
        # CRITICAL FIX: On curves, use heading_to_ref directly (don't subtract current_heading)
        # Subtracting current_heading causes conflicts: car heading more right → negative heading_error
        # But lateral_error is positive → conflict!
        # Solution: heading_error = heading_to_ref (correction needed to point toward reference)
        # NOTE: heading_to_ref is already computed in vehicle frame.
        # Subtracting current_heading (world frame) causes large sign flips on straights.
        # Use heading_to_ref directly for both straights and curves.
        heading_error = heading_to_ref
        
        # Normalize error to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Compute lateral error (cross-track error)
        # CRITICAL FIX: ref_x is already in VEHICLE frame (lateral offset from vehicle center)
        # vehicle_position is in WORLD frame (Unity world coordinates)
        # We CANNOT subtract them directly - they're in different coordinate systems!
        #
        # Solution: Use ref_x directly as lateral error (it's already the lateral offset)
        # For small headings, this is correct. For large headings, we need heading correction.
        
        # FIXED: ref_x is already the lateral offset in vehicle frame
        # We don't need to subtract vehicle_position[0] because:
        #   - ref_x is relative to vehicle center (vehicle frame)
        #   - vehicle_position[0] is absolute world position (world frame)
        #   - They're in different coordinate systems!
        #
        # For zero heading: lateral_error = ref_x (directly)
        # For non-zero heading: need to account for heading effect on lateral offset
        #
        # The heading effect: if vehicle is rotated, the lateral offset needs adjustment
        # But ref_x is already in vehicle frame, so for small headings it's correct
        # For large headings, we might need to account for how heading affects perception
        
        # FIXED: Use ref_x directly as lateral error (for recording/analysis)
        # But we need to think about the sign convention for control:
        # - ref_x > 0 means target is to the RIGHT of vehicle center
        # - If target is RIGHT, car is LEFT of target
        # - To move RIGHT (toward target), need to steer RIGHT (positive steering)
        # - PID: negative error → negative output, but we need positive output
        # - So we need to negate the error OR negate the output
        # - We'll negate total_error before PID to keep lateral_error correct for recording
        base_lateral_error = ref_x  # Keep original sign for recording/analysis
        
        # For large headings, the perception might be affected by vehicle rotation
        # But since ref_x is already in vehicle frame, we use it directly
        # The heading error is handled separately in total_error calculation
        heading_abs = abs(current_heading)
        heading_deg = np.degrees(heading_abs)
        
        # For very large headings, perception might be unreliable
        # But ref_x is still the best estimate we have
        lateral_error = base_lateral_error
        
        # NEW: Smooth lateral error to reduce sensitivity to perception noise
        # Perception instability can cause small changes in lane position that propagate to control
        # Smoothing the error prevents control oscillations from perception noise
        if self.smoothed_lateral_error is None:
            self.smoothed_lateral_error = lateral_error
        else:
            # Exponential smoothing: blend new error with previous smoothed error
            self.smoothed_lateral_error = (self.error_smoothing_alpha * lateral_error + 
                                         (1.0 - self.error_smoothing_alpha) * self.smoothed_lateral_error)
        
        # Use smoothed error for control (but store original for recording/analysis)
        # Steering sign convention: positive ref_x means target is to the RIGHT, so steering should be positive.
        lateral_error_for_control = self.smoothed_lateral_error
        lateral_error_for_recording = lateral_error  # Store original for analysis
        
        # FIXED: Prioritize lateral_error when heading_error conflicts
        # Analysis showed 67% of frames have heading/lateral conflicts causing 24% sign errors
        # Solution: When conflicts occur, prioritize lateral_error (more reliable for steering)
        # Only use heading_error for large headings (>10°) or when errors agree
        heading_abs_deg = abs(np.degrees(heading_error))
        
        # Check if heading_error and lateral_error have opposite signs (conflict)
        has_conflict = (heading_error > 0 and lateral_error < 0) or (heading_error < 0 and lateral_error > 0)
        
        if heading_abs_deg >= 10.0:
            # Large headings: use heading error primarily (80% heading, 20% lateral)
            # This prevents heading effect from dominating
            total_error = 0.8 * heading_error + 0.2 * lateral_error
        elif has_conflict:
            # CRITICAL FIX: When conflicts occur, use ONLY lateral_error (smoothed)
            # On curves, heading_error can conflict with lateral_error
            # Lateral error is more reliable for steering direction
            # Use 100% lateral_error to prevent wrong steering sign
            # This is especially important on curves where heading_error might be noisy
            total_error = lateral_error_for_control
        else:
            # Normal operation: use configurable weights (errors agree)
            # Use smoothed lateral error for control to reduce perception noise sensitivity
            total_error = self.heading_weight * heading_error + self.lateral_weight * lateral_error_for_control
        
        # Scale error to prevent overreaction
        total_error = np.clip(total_error, -self.error_clip, self.error_clip)
        stored_total_error = total_error
        
        # Add deadband to prevent constant small corrections
        # CRITICAL FIX: Don't apply deadband when on curves - we need continuous steering
        # On curves, even small errors need correction to maintain path following
        is_on_curve = abs(path_curvature) > 0.05  # On a curve (curvature > ~3°)
        if abs(total_error) < self.deadband and not is_on_curve:
            # Only apply deadband on straight roads, not on curves
            total_error = 0.0
        
        # Store errors for metadata (use original, not smoothed, for analysis)
        stored_lateral_error = lateral_error_for_recording  # Store original for analysis
        stored_heading_error = heading_error
        
        # FIXED: More aggressive integral reset to prevent accumulation (was 11.63x increase after 6s)
        # Strategy: Multiple reset mechanisms with stricter thresholds
        
        # Mechanism 1: Reset if error is small for extended period (0.3 seconds = 9 frames)
        # CRITICAL FIX: Don't reset if we recently had large error (transition period)
        # When steering maxes out and error becomes small, we need the integral to maintain steering
        # FIXED: Increased threshold and delay to prevent jerky steering after curves
        if abs(total_error) < 0.1:  # Small error threshold
            # Only count small errors if we're not in transition from large error
            # AND we're not in the middle of a curve-to-straight transition
            in_transition = hasattr(self, 'straight_transition_counter') and self.straight_transition_counter < 15
            if not self.recent_large_error and not in_transition:
                self.small_error_count += 1
                if self.small_error_count > 9:  # Slower reset (0.3s instead of 0.2s) for smoother behavior
                    self.pid.reset()
                    self.small_error_count = 0
            else:
                # In transition period - don't count small errors (protect integral)
                self.small_error_count = 0
        else:
            self.small_error_count = 0
        
        # Mechanism 2: Reset if error is large (immediate reset)
        # CRITICAL FIX: Don't reset if steering would be maxed out OR if we're on a curve
        # When steering maxes out or we're on a curve, error is large, but we need to keep integral
        # to maintain steering through the curve
        # FIXED: Disable resets entirely when on curves to prevent jerky steering
        if abs(total_error) > 0.3:  # Large error (likely on curve)
            # Check if we're on a curve (path curvature > threshold)
            is_on_curve = abs(path_curvature) > 0.05  # On a curve (curvature > ~3°)
            
            # Estimate if steering would be maxed out
            estimated_steering = abs(self.pid.kp * total_error + self.pid.integral)
            steering_would_be_maxed = estimated_steering > 0.9 * self.max_steering
            
            # CRITICAL FIX: Don't reset if:
            # 1. Steering is maxed out (we need the integral)
            # 2. We're on a curve (error oscillation is normal, don't reset)
            # Only reset if steering is NOT maxed AND we're NOT on a curve
            if not steering_would_be_maxed and not is_on_curve:
                # Reset immediately on large errors (but only if not maxed and not on curve)
                if abs(self.pid.integral) > 0.03:  # Stricter threshold (was 0.05)
                    self.pid.reset()
                    self.small_error_count = 0
            # If steering IS maxed out OR we're on a curve, keep the integral (don't reset)
        
        # Mechanism 3: Reset on sign change (oscillation detection)
        # CRITICAL FIX: Don't reset on sign change if we need constant steering (curves)
        # On curved roads, error naturally oscillates around zero, but we need constant steering
        # FIXED: Disable resets entirely when on curves to prevent jerky steering
        if self.last_error_sign is not None and np.sign(total_error) != self.last_error_sign:
            # Check if we're on a curve (path curvature > threshold)
            is_on_curve = abs(path_curvature) > 0.05  # On a curve (curvature > ~3°)
            
            # CRITICAL FIX: Never reset on sign change when on a curve
            # Error oscillation is normal on curves - we need constant steering, not resets
            if is_on_curve:
                # On curve - don't reset, keep integral to maintain steering
                pass  # Keep integral
            else:
                # Not on curve - check if this is true oscillation (small errors) or overshooting
                error_magnitude = abs(total_error)
                last_error_magnitude = abs(self.last_error) if hasattr(self, 'last_error') else 0.0
                
                # CRITICAL FIX: Don't reset if we just had a large error and are now correcting
                # This prevents reset when steering maxes out and then error becomes small
                was_large_error = last_error_magnitude > 0.3  # Had large error (steering was maxed)
                is_small_error = error_magnitude < 0.15  # Now small error (correcting)
                is_overshooting = (last_error_magnitude > 0.2 and error_magnitude > 0.2)  # Large → large opposite
                
                # Only reset if:
                # 1. Both errors are small (< 0.15) - true oscillation around zero
                # 2. OR we're overshooting (large → large opposite, not correcting)
                # DO NOT reset if: large error → small error (we're successfully correcting!)
                should_reset = False
                if last_error_magnitude < 0.15 and error_magnitude < 0.15:
                    # Both small - true oscillation, safe to reset
                    should_reset = True
                elif is_overshooting:
                    # Large → large opposite - overshooting, reset
                    should_reset = True
                # Otherwise: large → small means we're correcting, KEEP integral!
                
                if should_reset:
                    # Error changed sign and it's true oscillation or overshooting - reset integral
                    self.pid.reset()
                    self.small_error_count = 0
                # Otherwise, keep integral to maintain constant steering
        self.last_error_sign = np.sign(total_error) if abs(total_error) > 0.01 else None
        if not hasattr(self, 'last_error'):
            self.last_error = 0.0
        self.last_error = total_error  # Store for next frame comparison
        
        # Mechanism 4: Periodic reset with integral decay (every 0.67 seconds = 20 frames)
        # CRITICAL FIX: Don't decay if steering is maxed or we recently had large error
        # When steering maxes out or error is large, we need the integral to maintain steering
        if not hasattr(self, 'reset_counter'):
            self.reset_counter = 0
        self.reset_counter += 1
        if self.reset_counter > 20:  # Every 0.67 seconds (was 1 second = 30 frames)
            # Estimate if steering would be maxed out
            estimated_steering = abs(self.pid.kp * total_error + self.pid.integral)
            steering_would_be_maxed = estimated_steering > 0.9 * self.max_steering
            error_is_large = abs(total_error) > 0.3  # On a curve (large error)
            
            # Only apply decay if:
            # 1. Steering is NOT maxed out
            # 2. Error is NOT large (not on a curve)
            # 3. We did NOT recently have large error (not in transition period)
            if not steering_would_be_maxed and not error_is_large and not self.recent_large_error:
                if abs(self.pid.integral) > 0.03:  # Stricter threshold (was 0.05)
                    # NEW: Apply exponential decay instead of full reset (smoother)
                    decay_factor = 0.5  # Reduce integral by 50%
                    self.pid.integral *= decay_factor
            # If steering is maxed, error is large, or we recently had large error, don't decay
            self.reset_counter = 0
        
        # Track if we recently had large error (on a curve)
        # This helps protect integral during transition from large error to small error
        if abs(total_error) > 0.3:
            self.large_error_frames = min(30, self.large_error_frames + 2)  # Build up quickly, max 30 frames
            self.recent_large_error = True
        else:
            # Decay the "recent large error" flag slowly
            # Keep it for ~1 second (30 frames) after error becomes small (transition period)
            # This protects integral when steering maxes out and then error becomes small
            if self.large_error_frames > 0:
                self.large_error_frames = max(0, self.large_error_frames - 1)  # Decay slowly (1 frame per update)
            if self.large_error_frames == 0:
                self.recent_large_error = False
        
        # Mechanism 5: FIXED - More aggressive integral decay to prevent accumulation
        # CRITICAL FIX: Don't decay integral when on a curve (large error) or when steering is maxed
        # When steering maxes out or error is large, we need the integral to maintain steering
        # Also protect during transition from large error to small error (recent_large_error)
        # Estimate if steering would be maxed out
        estimated_steering = abs(self.pid.kp * total_error + self.pid.integral)
        steering_would_be_maxed = estimated_steering > 0.9 * self.max_steering
        error_is_large = abs(total_error) > 0.3  # On a curve (large error)
        
        # Only apply decay if:
        # 1. Steering is NOT maxed out
        # 2. Error is NOT large (not on a curve)
        # 3. We did NOT recently have large error (not in transition period)
        # If steering is maxed, error is large, or we recently had large error, we need the integral
        if not steering_would_be_maxed and not error_is_large and not self.recent_large_error:
            # REVERTED: Back to 0.90 decay (was working better than 0.95)
            # Analysis showed 0.95 decay made things WORSE (13.19x vs 5.54x)
            # 0.90 decay (10% per frame) is stronger and was working better
            if abs(self.pid.integral) > 0.01:  # Restored threshold (was 0.005)
                # Apply decay factor (0.90 = 10% decay per frame, was 0.95 = 5%)
                # Stronger decay prevents accumulation more effectively
                decay_factor = 0.90
                self.pid.integral *= decay_factor
            elif abs(self.pid.integral) > 0.005:  # Light decay threshold
                # Light decay (0.99 = 1% decay per frame) for smaller integrals
                decay_factor = 0.99
                self.pid.integral *= decay_factor
        # If steering IS maxed out OR error is large (on curve), don't decay the integral
        
        # INDUSTRY STANDARD: Feedforward + Feedback Control
        # Real AV companies use geometric feedforward based on path curvature
        # This is better than adaptive kp - it's the proper way to handle curves
        # Feedforward: Calculate required steering from path geometry (bicycle model)
        # Feedback: PID corrects for errors and disturbances
        
        # Calculate feedforward steering from path curvature
        # Using bicycle model: steering = atan(wheelbase * curvature)
        # For normalized steering (-1 to 1), we need to scale by max steering angle
        wheelbase = 2.5  # meters (typical car wheelbase)
        
        # FIXED: Add smoothing to feedforward to prevent jerky steering from noisy curvature
        # Smooth the path curvature to prevent rapid feedforward changes
        if not hasattr(self, 'smoothed_path_curvature'):
            self.smoothed_path_curvature = 0.0
        smoothing_alpha = 0.7  # Exponential smoothing factor (0.7 = 70% new, 30% old)
        self.smoothed_path_curvature = (smoothing_alpha * path_curvature + 
                                        (1.0 - smoothing_alpha) * self.smoothed_path_curvature)
        
        feedforward_steering = 0.0
        
        if abs(self.smoothed_path_curvature) > 0.01:  # Only apply feedforward on curves
            # Convert curvature (1/m) to steering angle using bicycle model
            # curvature = 1/radius, steering = atan(wheelbase / radius) = atan(wheelbase * curvature)
            # For small angles: steering ≈ wheelbase * curvature
            # Normalize to [-1, 1] range
            raw_steering = wheelbase * self.smoothed_path_curvature  # Use smoothed curvature
            # Clamp to reasonable range (max steering angle in radians)
            max_steering_rad = np.radians(30.0)  # 30 degrees max steering
            feedforward_steering = np.clip(raw_steering / max_steering_rad, -1.0, 1.0)
            # Scale by max_steering to match our steering range
            feedforward_steering *= self.max_steering
        
        # Track curve-to-straight transition for smooth integral decay
        is_on_curve = abs(path_curvature) > 0.05  # On a curve (curvature > ~3°)
        if not hasattr(self, 'was_on_curve'):
            self.was_on_curve = False
        if not hasattr(self, 'straight_transition_counter'):
            self.straight_transition_counter = 0
        
        # Detect transition from curve to straight
        transition_to_straight = self.was_on_curve and not is_on_curve
        
        # CRITICAL FIX: Gradually decay integral when transitioning from curve to straight
        # Instead of immediately resetting (which causes jerky steering), gradually decay
        # The integral accumulates during curves to maintain steering
        # When entering straight road, we need to smoothly reduce the integral
        if transition_to_straight:
            # Start transition counter
            self.straight_transition_counter = 0
        
        # Gradually decay integral over ~0.5 seconds (15 frames) when on straight after curve
        if not is_on_curve and self.straight_transition_counter < 15:
            # Only decay if error is small (we're actually on straight, not just low curvature)
            if abs(total_error) < 0.15:  # Small error = actually on straight road
                # Exponential decay: reduce integral by 30% each frame
                # After 15 frames: 0.7^15 ≈ 0.005 (essentially zero)
                decay_factor = 0.70  # 30% decay per frame
                self.pid.integral *= decay_factor
                self.straight_transition_counter += 1
            else:
                # Error is still large - we're not fully on straight yet, keep integral
                self.straight_transition_counter = 0  # Reset counter
        elif is_on_curve:
            # Back on curve - reset transition counter
            self.straight_transition_counter = 0
        
        # Update state
        self.was_on_curve = is_on_curve
        
        # Update PID controller (feedback term)
        dt = 0.033  # Assume ~30 Hz update rate
        # Note: The steering direction issue (23.7% wrong) is caused by reference point smoothing
        # preserving bias when lane detection fails, not a sign error in the controller
        # The fix is in trajectory/inference.py to not preserve bias during lane failures
        feedback_steering = self.pid.update(total_error, dt)
        
        # Combine feedforward + feedback
        # Feedforward handles the curve, feedback corrects for errors
        steering_before_limits = feedforward_steering + feedback_steering
        
        # Get PID internal state
        pid_integral = self.pid.integral
        pid_derivative = 0.0
        if self.pid.prev_time is not None and dt > 0:
            pid_derivative = self.pid.kd * ((total_error - self.pid.prev_error) / dt)
        
        # FIXED: Add steering rate limiting to prevent sudden changes and oscillation
        # CRITICAL FIX: Adaptive rate limiting - allow faster changes when error is large
        # Analysis showed rate limiting prevents recovery from large errors
        # Solution: Increase rate limit when error is large (need quick recovery)
        if not hasattr(self, 'last_steering'):
            self.last_steering = 0.0
        
        # Adaptive rate limiting based on error magnitude
        # Small errors: Slow rate limit (smoothness)
        # Large errors: Fast rate limit (recovery)
        error_magnitude = abs(total_error)
        if error_magnitude > 0.6:
            # Very large error (> 0.6) - allow fast steering changes for recovery
            max_steering_rate = 0.5  # Fast recovery (was 0.2)
        elif error_magnitude > 0.3:
            # Moderate error (0.3-0.6) - moderate rate limit
            max_steering_rate = 0.3  # Moderate (was 0.2)
        else:
            # Small error (< 0.3) - slow rate limit for smoothness
            max_steering_rate = 0.2  # Smooth (original)
        
        # FIXED: Increased steering rate limit for better curve tracking
        # Required steering for 50m curve is only 0.095 normalized, but need to reach it quickly
        # Increased to 0.2 per frame (6.0 per second at 30 FPS) for more responsive steering
        # This allows reaching required steering (0.095) in ~0.5 frames instead of ~0.6 frames
        # FIXED: On first frame, allow larger initial steering to respond to initial state
        # This is important when starting on a curve - we need immediate steering response
        if hasattr(self, '_steering_rate_limit_active'):
            steering_change = steering_before_limits - self.last_steering
            steering_change = np.clip(steering_change, -max_steering_rate, max_steering_rate)
            steering_before_limits = self.last_steering + steering_change
        else:
            # First frame - allow larger initial steering (up to 0.5) to respond to initial state
            # This is critical when starting on a curve where we need immediate steering
            max_initial_steering_rate = 0.5  # FIXED: Increased from 0.4 to 0.5 for better initial response
            steering_change = steering_before_limits - self.last_steering
            if abs(steering_change) > max_initial_steering_rate:
                steering_change = np.clip(steering_change, -max_initial_steering_rate, max_initial_steering_rate)
                steering_before_limits = self.last_steering + steering_change
            # Activate normal rate limiting for next frame
            self._steering_rate_limit_active = True
        
        self.last_steering = steering_before_limits
        
        # Apply additional smoothing to prevent oscillation
        steering = np.clip(steering_before_limits, -self.max_steering, self.max_steering)

        # Straight-away adaptive tuning (deadband + smoothing)
        is_straight = abs(path_curvature) < self.straight_curvature_threshold
        if is_straight:
            steering_sign = np.sign(steering) if abs(steering) > 0.02 else 0
            if self._last_straight_sign != 0 and steering_sign != 0 and steering_sign != self._last_straight_sign:
                self._straight_sign_changes += 1
            if steering_sign != 0:
                self._last_straight_sign = steering_sign
            self._straight_frames += 1

            if self._straight_frames >= self.straight_window:
                self.straight_oscillation_rate = self._straight_sign_changes / max(1, self._straight_frames - 1)
                if self.straight_oscillation_rate > self.straight_oscillation_high:
                    # Too much weave on straight: increase deadband and smoothing
                    self.deadband = min(self.deadband_max, self.deadband + 0.005)
                    self.error_smoothing_alpha = min(self.smoothing_max, self.error_smoothing_alpha + 0.02)
                elif self.straight_oscillation_rate < self.straight_oscillation_low:
                    # Stable: relax toward baseline
                    if self.deadband > self.base_deadband:
                        self.deadband = max(self.base_deadband, self.deadband - 0.003)
                    if self.error_smoothing_alpha > self.base_error_smoothing_alpha:
                        self.error_smoothing_alpha = max(self.base_error_smoothing_alpha, self.error_smoothing_alpha - 0.01)
                # Reset window counters
                self._straight_frames = 0
                self._straight_sign_changes = 0
        else:
            # Off straight: decay toward baseline
            if self.deadband > self.base_deadband:
                self.deadband = max(self.base_deadband, self.deadband - 0.002)
            elif self.deadband < self.base_deadband:
                self.deadband = min(self.base_deadband, self.deadband + 0.001)
            if self.error_smoothing_alpha > self.base_error_smoothing_alpha:
                self.error_smoothing_alpha = max(self.base_error_smoothing_alpha, self.error_smoothing_alpha - 0.005)
            elif self.error_smoothing_alpha < self.base_error_smoothing_alpha:
                self.error_smoothing_alpha = min(self.base_error_smoothing_alpha, self.error_smoothing_alpha + 0.003)
            self._straight_frames = 0
            self._straight_sign_changes = 0
            self._last_straight_sign = 0
        
        if return_metadata:
            return {
                'steering': steering,
                'steering_before_limits': steering_before_limits,
                'lateral_error': stored_lateral_error,
                'heading_error': stored_heading_error,
                'total_error': stored_total_error,
                'pid_integral': pid_integral,
                'pid_derivative': pid_derivative,
                'is_straight': is_straight,
                'straight_oscillation_rate': self.straight_oscillation_rate,
                'tuned_deadband': self.deadband,
                'tuned_error_smoothing_alpha': self.error_smoothing_alpha
            }
        
        return steering
    
    def reset(self):
        """Reset controller."""
        self.pid.reset()
        self.small_error_count = 0
        self.last_error_sign = None
        if hasattr(self, 'last_steering'):
            self.last_steering = 0.0


class LongitudinalController:
    """
    Longitudinal control (throttle/brake) using PID controller.
    """
    
    def __init__(self, kp: float = 0.3, ki: float = 0.05, kd: float = 0.02,
                 target_speed: float = 8.0, max_speed: float = 10.0):
        """
        Initialize longitudinal controller.
        
        Args:
            kp: Proportional gain (reduced for smoother control)
            ki: Integral gain
            kd: Derivative gain
            target_speed: Target speed (m/s)
            max_speed: Maximum allowed speed (m/s) - hard limit
        """
        self.target_speed = target_speed
        self.max_speed = max_speed
        self.last_speed = 0.0  # For speed smoothing
        self.pid_throttle = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            integral_limit=0.8,  # Reduced to prevent windup
            output_limit=(0.0, 1.0)  # Throttle range
        )
        self.pid_brake = PIDController(
            kp=kp * 3.0,  # More aggressive braking
            ki=ki * 0.3,
            kd=kd * 3.0,
            integral_limit=1.0,
            output_limit=(0.0, 1.0)  # Brake range
        )
    
    def compute_control(self, current_speed: float, reference_velocity: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute throttle and brake commands with speed smoothing and limiting.
        
        Args:
            current_speed: Current vehicle speed (m/s)
            reference_velocity: Desired velocity from trajectory (m/s)
        
        Returns:
            Tuple of (throttle, brake) commands (0.0 to 1.0)
        """
        if reference_velocity is None:
            reference_velocity = self.target_speed
        
        # Enforce hard speed limit - check FIRST before any other logic
        if current_speed > self.max_speed:
            # Over speed limit - EMERGENCY BRAKING
            speed_excess = current_speed - self.max_speed
            # Very aggressive: minimum 0.8 brake, up to 1.0 for extreme speeds
            brake = min(1.0, 0.8 + (speed_excess / 5.0))
            throttle = 0.0
            self.pid_throttle.reset()
            # Update brake controller for smooth braking
            dt = 0.033
            self.pid_brake.update(speed_excess, dt)
            self.last_speed = current_speed
            return throttle, brake
        
        # Also limit reference velocity to prevent requesting high speeds
        if reference_velocity > self.max_speed:
            reference_velocity = self.max_speed * 0.9  # Cap at 90% of max
        
        # Speed smoothing - filter rapid changes
        alpha = 0.7  # Smoothing factor
        smoothed_speed = alpha * self.last_speed + (1 - alpha) * current_speed
        self.last_speed = smoothed_speed
        
        # Compute speed error using smoothed speed
        speed_error = reference_velocity - smoothed_speed
        
        dt = 0.033  # Assume ~30 Hz update rate
        
        # FIXED: Add hysteresis to prevent throttle/brake oscillation
        # Use different thresholds for throttle vs brake to create a "dead zone"
        # This prevents rapid switching when speed is near target
        throttle_threshold = 0.15  # Need 0.15 m/s below target to throttle
        brake_threshold = -0.15   # Need 0.15 m/s above target to brake
        # Dead zone: -0.15 to +0.15 m/s = no throttle, no brake (coast)
        
        if speed_error > throttle_threshold:  # Need to accelerate
            # Compute base throttle from PID
            base_throttle = self.pid_throttle.update(speed_error, dt)
            
            # FIXED: Add initial throttle limiting when starting from rest
            # CRITICAL FIX: Limit throttle to prevent unrealistic acceleration
            # Real cars accelerate at 2-4 m/s², not 500+ m/s²!
            # Prevents aggressive acceleration from 0 m/s (throttle would be 1.0 instantly)
            if smoothed_speed < 2.0:  # Very slow or at rest (increased from 1.0 to 2.0)
                # Limit initial throttle to prevent aggressive acceleration
                # Ramp up from 0.2 to full throttle as speed increases (reduced from 0.3)
                initial_throttle_limit = 0.2 + (smoothed_speed / 2.0) * 0.8  # 0.2 to 1.0 over 0-2 m/s
                base_throttle = min(base_throttle, initial_throttle_limit)
            
            # ADDITIONAL FIX: Cap maximum throttle to prevent extreme acceleration
            # Even at higher speeds, limit throttle to prevent overspeed
            # This helps prevent the car from accelerating too quickly
            max_throttle_limit = 0.8  # Never exceed 80% throttle (prevents extreme acceleration)
            base_throttle = min(base_throttle, max_throttle_limit)
            
            # Progressive throttle limiting as we approach max speed
            # This prevents overshoot and reduces need for emergency braking
            if smoothed_speed > self.max_speed * 0.80:
                # Very close to max - aggressive reduction
                throttle = base_throttle * 0.2
            elif smoothed_speed > self.max_speed * 0.75:
                # Close to max - moderate reduction
                throttle = base_throttle * 0.4
            elif smoothed_speed > self.max_speed * 0.65:
                # Approaching max - light reduction
                throttle = base_throttle * 0.6
            else:
                # Normal operation - full throttle
                throttle = base_throttle
            brake = 0.0
            self.pid_brake.reset()  # Reset brake controller
        elif speed_error < brake_threshold:  # Need to decelerate
            throttle = 0.0
            brake = self.pid_brake.update(-speed_error, dt)
            self.pid_throttle.reset()  # Reset throttle controller
        else:  # Within dead zone - maintain current speed (coast)
            throttle = 0.0
            brake = 0.0
            # Don't reset controllers in dead zone - allows smooth transitions
        
        return throttle, brake
    
    def reset(self):
        """Reset controllers."""
        self.pid_throttle.reset()
        self.pid_brake.reset()


class VehicleController:
    """
    Combined vehicle controller (lateral + longitudinal).
    """
    
    def __init__(self, lateral_kp: float = 0.3, lateral_ki: float = 0.0, lateral_kd: float = 0.1,
                 longitudinal_kp: float = 0.3, longitudinal_ki: float = 0.05, longitudinal_kd: float = 0.02,
                 lookahead_distance: float = 10.0, target_speed: float = 8.0, max_speed: float = 10.0, max_steering: float = 0.5,
                 lateral_deadband: float = 0.02, lateral_heading_weight: float = 0.5, lateral_lateral_weight: float = 0.5,
                 lateral_error_clip: float = None, lateral_integral_limit: float = 0.3):
        """
        Initialize vehicle controller.
        
        Args:
            lateral_kp: Lateral proportional gain
            lateral_ki: Lateral integral gain
            lateral_kd: Lateral derivative gain
            longitudinal_kp: Longitudinal proportional gain
            longitudinal_ki: Longitudinal integral gain
            longitudinal_kd: Longitudinal derivative gain
            lookahead_distance: Lookahead distance (meters)
            target_speed: Target speed (m/s)
        """
        # Set error_clip default if not provided
        if lateral_error_clip is None:
            import numpy as np
            lateral_error_clip = np.pi / 4
        
        self.lateral_controller = LateralController(
            kp=lateral_kp,
            ki=lateral_ki,
            kd=lateral_kd,
            lookahead_distance=lookahead_distance,
            max_steering=max_steering,
            deadband=lateral_deadband,
            heading_weight=lateral_heading_weight,
            lateral_weight=lateral_lateral_weight,
            error_clip=lateral_error_clip,
            integral_limit=lateral_integral_limit
        )
        self.longitudinal_controller = LongitudinalController(
            kp=longitudinal_kp,
            ki=longitudinal_ki,
            kd=longitudinal_kd,
            target_speed=target_speed,
            max_speed=max_speed
        )
    
    def compute_control(self, current_state: dict, reference_point: dict, 
                       return_metadata: bool = False) -> dict:
        """
        Compute control commands.
        
        Args:
            current_state: Current vehicle state (heading, speed, position)
            reference_point: Reference point from trajectory
            return_metadata: If True, include PID internal state and errors
        
        Returns:
            Control commands (steering, throttle, brake)
            If return_metadata=True, also includes: steering_before_limits, 
            lateral_error, heading_error, total_error, pid_integral, pid_derivative
        """
        # Lateral control
        steering_result = self.lateral_controller.compute_steering(
            current_state.get('heading', 0.0),
            reference_point,
            current_state.get('position'),
            return_metadata=return_metadata
        )
        
        if return_metadata:
            steering = steering_result['steering']
            lateral_metadata = steering_result
        else:
            steering = steering_result
            lateral_metadata = {}
        
        # Longitudinal control
        throttle, brake = self.longitudinal_controller.compute_control(
            current_state.get('speed', 0.0),
            reference_point.get('velocity')
        )
        
        result = {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }
        
        # FIXED: Always include lateral_error in control_command for safety mechanisms
        # Safety mechanisms in av_stack.py need lateral_error to trigger
        lateral_error = lateral_metadata.get('lateral_error', 0.0)
        result['lateral_error'] = lateral_error
        
        if return_metadata:
            result.update(lateral_metadata)
            # For throttle/brake, we don't track before_limits separately yet
            result['throttle_before_limits'] = throttle
            result['brake_before_limits'] = brake
        
        return result
    
    def reset(self):
        """Reset all controllers."""
        self.lateral_controller.reset()
        self.longitudinal_controller.reset()

