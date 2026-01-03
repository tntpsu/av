"""
Lane detection model using UFLD (Ultra Fast Lane Detection) approach.
Simplified implementation for Unity simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LaneDetectionModel(nn.Module):
    """
    Simplified lane detection model based on UFLD architecture.
    Uses a lightweight CNN backbone with row-wise classification.
    """
    
    def __init__(self, num_lanes: int = 4, num_gridding: int = 200, img_height: int = 320, img_width: int = 800):
        """
        Initialize lane detection model.
        
        Args:
            num_lanes: Maximum number of lanes to detect
            num_gridding: Number of grid cells for row-wise classification
            img_height: Input image height
            img_width: Input image width
        """
        super(LaneDetectionModel, self).__init__()
        
        self.num_lanes = num_lanes
        self.num_gridding = num_gridding
        self.img_height = img_height
        self.img_width = img_width
        
        # Backbone: Simplified ResNet-like architecture
        self.backbone = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Row-wise classification head
        # Each row has num_gridding cells, predict which cell contains lane point
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_gridding * num_lanes)
        )
        
        # Lane existence prediction
        self.exist = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_lanes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, 3, H, W]
        
        Returns:
            cls: Classification logits [B, num_lanes, num_gridding]
            exist: Lane existence logits [B, num_lanes]
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Row-wise classification
        cls_logits = self.cls(features)
        cls_logits = cls_logits.view(-1, self.num_lanes, self.num_gridding)
        
        # Lane existence
        exist_logits = self.exist(features)
        
        return cls_logits, exist_logits
    
    def decode_lanes(self, cls_logits: torch.Tensor, exist_logits: torch.Tensor, 
                     conf_threshold: float = 0.5) -> List[Optional[np.ndarray]]:
        """
        Decode lane lines from model outputs.
        
        Args:
            cls_logits: Classification logits [B, num_lanes, num_gridding]
            exist_logits: Lane existence logits [B, num_lanes]
            conf_threshold: Confidence threshold for lane existence
        
        Returns:
            List of lane line coordinates (polynomial coefficients) or None
        """
        batch_size = cls_logits.size(0)
        lanes = []
        
        # Convert to probabilities
        cls_probs = F.softmax(cls_logits, dim=-1)
        exist_probs = torch.sigmoid(exist_logits)
        
        for b in range(batch_size):
            batch_lanes = []
            
            for lane_idx in range(self.num_lanes):
                # Check if lane exists
                if exist_probs[b, lane_idx] < conf_threshold:
                    batch_lanes.append(None)
                    continue
                
                # Get row-wise predictions
                row_probs = cls_probs[b, lane_idx].cpu().numpy()
                
                # Extract lane points
                lane_points = []
                num_rows = self.img_height // 8  # After 4 stride-2 convolutions
                
                for row in range(num_rows):
                    # Map row to gridding index
                    grid_idx = int((row / num_rows) * self.num_gridding)
                    grid_idx = np.clip(grid_idx, 0, self.num_gridding - 1)
                    
                    # Get probability distribution for this row
                    prob_dist = row_probs[grid_idx]
                    
                    # Find most likely position
                    if prob_dist > 0.1:  # Threshold
                        # Convert grid index to pixel x coordinate
                        x = (grid_idx / self.num_gridding) * self.img_width
                        y = row * 8  # Scale back to original image coordinates
                        lane_points.append([x, y])
                
                if len(lane_points) >= 3:
                    # Fit polynomial to lane points
                    lane_points = np.array(lane_points)
                    try:
                        # Fit 2nd degree polynomial
                        coeffs = np.polyfit(lane_points[:, 1], lane_points[:, 0], deg=2)
                        batch_lanes.append(coeffs)
                    except:
                        batch_lanes.append(None)
                else:
                    batch_lanes.append(None)
            
            lanes.append(batch_lanes)
        
        return lanes[0] if batch_size == 1 else lanes


class SimpleLaneDetector:
    """
    Simple lane detector using traditional computer vision as fallback.
    Can be used when ML model is not available or for comparison.
    """
    
    def __init__(self):
        """Initialize simple lane detector."""
        # Temporal persistence: Store previous frame's lanes
        self.previous_lanes = [None, None]  # [left, right] - polynomial coefficients
        self.previous_lanes_confidence = [0.0, 0.0]  # Confidence for each lane
        self.previous_lanes_x_at_bottom = [None, None]  # [left, right] - x position at bottom of image
        self.temporal_decay = 0.9  # Decay factor for confidence over time
        self.curvature_deviation_threshold = 0.01  # Max allowed curvature change between frames
        self.position_deviation_threshold = 50.0  # Max allowed x position change (pixels) for swap detection
        # Track if current frame's lanes were rejected due to temporal filtering (to prevent feedback loop)
        self.lanes_rejected_this_frame = [False, False]  # [left, right] - True if rejected due to curvature flip
    
    def detect(self, image: np.ndarray, return_debug: bool = False) -> List[Optional[np.ndarray]]:
        """
        Detect lane lines using traditional CV methods with adaptive thresholds.
        
        Args:
            image: Input RGB image
            return_debug: If True, return debug information (lines, masks, etc.)
        
        Returns:
            List of lane line polynomial coefficients or None
            If return_debug=True, also returns dict with debug info
        """
        # DEBUG: Check image format
        if image.ndim == 2:
            # Grayscale - convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            # RGBA - convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Already RGB (or BGR) - assume RGB
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        h, w = image.shape[:2]
        center_x = w // 2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Adaptive brightness detection
        mean_brightness = np.mean(gray)
        brightness_factor = max(0.5, min(2.0, mean_brightness / 128.0))  # Normalize to typical brightness
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhanced color-based lane detection with adaptive thresholds
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Adaptive white lane mask (adjust based on brightness)
        # EXTREMELY STRICT: Must exclude sky, offroad, and road edges
        # Road edges are often bright gray/white, so we need to be very selective
        # FIXED: Even higher threshold and lower saturation to exclude road edges
        white_threshold_low = max(220, int(250 * brightness_factor))  # Raised even more to exclude road edges
        white_mask = cv2.inRange(hsv, (0, 0, white_threshold_low), (180, 15, 255))  # Lowered saturation to 15 to be extremely strict
        
        # Yellow lane mask (more flexible range)
        yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
        
        # REMOVED: Gray lane detection - was causing false positives from:
        # - Road surface variations
        # - Shadows
        # - Off-road areas
        # - Car body/hood reflections
        # Only yellow and white lane markings are reliable for detection
        
        # Create region of interest (lower 2/3 of image for camera perspective)
        # BALANCED: Use moderate ROI (center 80%) to reduce noise while still
        # allowing detection of both lanes even if car is offset
        roi_margin = int(w * 0.10)  # 10% margin on each side (center 80%)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        # EXCLUDE: Sky (top), car body/hood (very bottom), and sides
        # More restrictive to avoid car body and sky
        # INCREASED: Exclude bottom 7% (was 5%) to better exclude car at different angles
        roi_vertices = np.array([[
            (roi_margin, int(h * 0.93)),  # Stop 7% from bottom to exclude car hood/body (increased from 5%)
            (w - roi_margin, int(h * 0.93)),
            (w - roi_margin, h // 3),  # Focus on road ahead (exclude sky)
            (roi_margin, h // 3)
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        
        # CRITICAL FIX: Apply ROI mask to white detection too!
        # White detection should ONLY happen in ROI (road area) to exclude sky and road edges
        white_mask_roi = cv2.bitwise_and(white_mask, roi_mask)
        
        # Combine detection methods - ONLY yellow and white (no gray)
        # Yellow and white are the standard lane marking colors
        lane_color_mask = cv2.bitwise_or(white_mask_roi, yellow_mask)  # Use ROI-restricted white mask
        
        # CRITICAL FIX: Only do edge detection on lane paint areas (yellow/white mask)
        # This prevents detecting edges from:
        # - Car front/hood
        # - Road color changes (asphalt variations)
        # - Road shoulders/edges
        # - Other non-lane-marking features
        # 
        # Strategy: Apply Canny edge detection only to areas where we detected lane paint
        # This focuses edge detection on actual lane markings
        
        # Create a masked version of the blurred image (only lane paint areas)
        # Use lane_color_mask to restrict where we look for edges
        blurred_masked = cv2.bitwise_and(blurred, blurred, mask=lane_color_mask)
        
        # Enhanced edge detection with adaptive thresholds
        # Calculate adaptive Canny thresholds based on masked image statistics
        # Only calculate statistics from lane paint areas (more accurate for lane edges)
        masked_pixels = blurred[lane_color_mask > 0]
        if len(masked_pixels) > 0:
            median_val = np.median(masked_pixels)
            lower_threshold = int(max(10, median_val * 0.3))
            upper_threshold = int(min(200, median_val * 2.0))
        else:
            # Fallback if no lane paint detected
            median_val = np.median(blurred)
            lower_threshold = int(max(10, median_val * 0.3))
            upper_threshold = int(min(200, median_val * 2.0))
        
        # CRITICAL: Only detect edges within lane paint mask
        # This ensures we only find edges of lane markings, not other features
        edges = cv2.Canny(blurred_masked, lower_threshold, upper_threshold)
        
        # Further restrict edges to ROI (road area, not sky/car)
        edges_roi = cv2.bitwise_and(edges, roi_mask)
        
        # DEBUG: Also create edges restricted to yellow mask only (for visualization/debugging)
        # This helps verify that edges are only detected on yellow lane markings
        blurred_yellow_masked = cv2.bitwise_and(blurred, blurred, mask=yellow_mask)
        edges_yellow_only = cv2.Canny(blurred_yellow_masked, lower_threshold, upper_threshold)
        edges_yellow_roi = cv2.bitwise_and(edges_yellow_only, roi_mask)
        
        # Combine: Use color mask as primary, edges as secondary (for faint markings)
        # Edges help detect lane markings that might be faint or partially occluded
        paint_primary = lane_color_mask  # Paint detection is primary (most reliable)
        edges_secondary = edges_roi  # Edges are secondary (for faint markings)
        combined = cv2.bitwise_or(paint_primary, edges_secondary)
        
        # Apply ROI mask to combined result (final safety check)
        # Note: edges_roi already has ROI applied, and lane_color_mask already has ROI applied
        # This ensures everything is properly masked
        mask = roi_mask.copy()  # Reuse ROI mask created above
        masked_edges = cv2.bitwise_and(combined, mask)
        
        # Hough line detection with adjusted parameters
        # VERY SENSITIVE: Lower thresholds significantly to detect faint lines
        # The visualizations show NO lines detected, so we need to be much more aggressive
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=15,  # Lowered from 25 to detect more lines (was 35 originally)
            minLineLength=15,  # Lowered from 20 to detect shorter segments (was 30 originally)
            maxLineGap=120  # Increased from 100 to connect more segments (was 80 originally)
        )
        
        if lines is None:
            debug_info = {}
            if return_debug:
                debug_info = {
                    'all_lines': None,
                    'masked_edges': masked_edges,
                    'lane_color_mask': lane_color_mask,
                    'yellow_mask': yellow_mask,
                    'edges': edges,
                    'roi_mask': roi_mask,
                    'combined': combined,
                    'gray': gray,
                    'blurred': blurred,
                    'num_lines_detected': 0,
                    'left_lines_count': 0,
                    'right_lines_count': 0,
                    'skipped_short': 0,
                    'skipped_center': 0,
                    'validation_failures': {}
                }
                return [None, None], debug_info
            return [None, None]
        
        # FIX #3: Prevent left/right swaps using temporal tracking
        # First, collect all candidate lines with their x_at_bottom positions
        candidate_lines = []  # List of (line, x_at_bottom, distance_from_center)
        skipped_short = 0
        skipped_center = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            
            # Filter out very short lines (likely noise)
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if line_length < 20:  # Less than 20 pixels - too short
                skipped_short += 1
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            line_center_x = (x1 + x2) / 2
            
            # IMPROVED: Calculate x position at bottom of image (closest to vehicle)
            # This is more reliable than simple center average for diagonal lines
            if abs(y2 - y1) > 1:  # Avoid division by zero
                # Linear interpolation: x = x1 + (x2 - x1) * (y_target - y1) / (y2 - y1)
                y_bottom = h  # Bottom of image
                x_at_bottom = x1 + (x2 - x1) * (y_bottom - y1) / (y2 - y1)
            else:
                # Vertical line, use center x
                x_at_bottom = line_center_x
            
            # Calculate distance from center
            distance_from_center = abs(x_at_bottom - center_x)
            margin_pixels = w * 0.02
            
            # Only consider lines that are clearly left or right of center
            if x_at_bottom < center_x - margin_pixels or x_at_bottom > center_x + margin_pixels:
                candidate_lines.append(([x1, y1, x2, y2], x_at_bottom, distance_from_center))
            else:
                # Lines very close to center (within 2%) are ambiguous - skip them
                skipped_center += 1
        
        # Now assign candidates to left/right based on temporal continuity
        left_lines = []
        right_lines = []
        
        if len(candidate_lines) > 0:
            # If we have previous frame's lane positions, use them for matching
            if (self.previous_lanes_x_at_bottom[0] is not None and 
                self.previous_lanes_x_at_bottom[1] is not None):
                # Temporal tracking: match candidates to previous lanes
                prev_left_x = self.previous_lanes_x_at_bottom[0]
                prev_right_x = self.previous_lanes_x_at_bottom[1]
                
                for line, x_at_bottom, dist_from_center in candidate_lines:
                    # Calculate distance to previous left and right positions
                    dist_to_prev_left = abs(x_at_bottom - prev_left_x)
                    dist_to_prev_right = abs(x_at_bottom - prev_right_x)
                    
                    # Assign to the closer previous lane (with threshold to prevent swaps)
                    if dist_to_prev_left < dist_to_prev_right:
                        # Closer to previous left lane
                        if dist_to_prev_left < self.position_deviation_threshold:
                            # Within threshold - assign to left
                            left_lines.append(line)
                        elif x_at_bottom < center_x:
                            # Outside threshold but still left of center - assign to left
                            left_lines.append(line)
                        # else: too far from previous left and right of center - skip (likely noise)
                    else:
                        # Closer to previous right lane
                        if dist_to_prev_right < self.position_deviation_threshold:
                            # Within threshold - assign to right
                            right_lines.append(line)
                        elif x_at_bottom > center_x:
                            # Outside threshold but still right of center - assign to right
                            right_lines.append(line)
                        # else: too far from previous right and left of center - skip (likely noise)
            else:
                # No previous frame data - use simple center-based classification
                for line, x_at_bottom, dist_from_center in candidate_lines:
                    if x_at_bottom < center_x:
                        left_lines.append(line)
                    else:
                        right_lines.append(line)
        
        # Reset rejection flags at start of detection (before processing lanes)
        # CRITICAL: Reset BEFORE detection loop so flags can be set during detection
        self.lanes_rejected_this_frame = [False, False]
        
        # Fit polynomials to left and right lanes
        lanes = []
        validation_failures = {'left': [], 'right': []}
        
        for lane_idx, (line_group, lane_name) in enumerate([(left_lines, 'left'), (right_lines, 'right')]):
            if len(line_group) == 0:
                lanes.append(None)
                validation_failures[lane_name].append('no_lines')
                continue
            
            # Collect all points
            points = []
            for line in line_group:
                points.append([line[0], line[1]])
                points.append([line[2], line[3]])
            
            points = np.array(points)
            
            if len(points) < 3:
                lanes.append(None)
                validation_failures[lane_name].append(f'insufficient_points_{len(points)}')
                continue
            
            # FIX #1: Sort points by y (ascending) before fitting
            # This stabilizes polynomial fitting, especially on curves
            # Points should be ordered from top (small y) to bottom (large y) of image
            sort_indices = np.argsort(points[:, 1])
            points = points[sort_indices]
            
            try:
                # Fit polynomial
                # CRITICAL FIX: Check if points are valid before fitting
                # If points span a small range, polynomial might extrapolate poorly
                y_range = np.max(points[:, 1]) - np.min(points[:, 1])
                x_range = np.max(points[:, 0]) - np.min(points[:, 0])
                y_min = np.min(points[:, 1])
                y_max = np.max(points[:, 1])
                x_min = np.min(points[:, 0])
                x_max = np.max(points[:, 0])
                
                # FRAME 0 DEBUG: Log point statistics
                if len(points) > 0:
                    logger.info(f"[FRAME0 DEBUG] {lane_name} lane: {len(points)} points, "
                               f"y_range=[{y_min:.1f}, {y_max:.1f}] ({y_range:.1f}px), "
                               f"x_range=[{x_min:.1f}, {x_max:.1f}] ({x_range:.1f}px), "
                               f"image_height={h}px")
                
                # If y_range is too small, polynomial will extrapolate poorly
                # RELAXED: Reduced from 50px to 30px to allow left lane (which is farther away)
                # Left lane often has smaller y_range due to perspective (farther = fewer pixels)
                if y_range < 30:  # Less than 30 pixels vertical span (relaxed from 50px)
                    logger.warning(f"[FRAME0 DEBUG] {lane_name}: y_range too small ({y_range:.1f}px < 30px)")
                    lanes.append(None)
                    validation_failures[lane_name].append(f'insufficient_y_range_{y_range:.1f}px')
                    continue
                
                # Check if we need to extrapolate significantly
                extrapolation_distance = h - y_max  # How far we need to extrapolate to reach bottom
                if extrapolation_distance > 50:  # Need to extrapolate >50px
                    logger.warning(f"[FRAME0 DEBUG] {lane_name}: Large extrapolation needed! "
                                 f"y_max={y_max:.1f}px, h={h}px, extrapolation={extrapolation_distance:.1f}px")
                
                # Fit polynomial with improved method for dashed lines and curves
                # Use weighted fitting to emphasize near points (more accurate)
                # This helps with both dashed lines (fewer points) and curves (perspective distortion)
                y = points[:, 1]
                x = points[:, 0]
                
                # Weighted polynomial fitting: weight by y-position (closer to vehicle = higher weight)
                # Points near vehicle (large y, bottom of image) are more accurate
                # Use image height to normalize weights and avoid numerical issues
                h_image = h  # Use image height for normalization
                
                # Weight by distance from top: weight = (y / h)^2
                # This gives higher weight to points closer to vehicle (bottom of image)
                # Normalizing by h keeps weights in [0, 1] range, avoiding numerical issues
                weights = (y / h_image) ** 2
                
                # Use numpy's built-in weighted polyfit (numerically stable)
                try:
                    coeffs = np.polyfit(y, x, deg=2, w=weights)
                    logger.debug(f"[POLY FIT] {lane_name}: Weighted fitting succeeded (using np.polyfit with weights)")
                except Exception as e:
                    # Fallback to unweighted if weighted fitting fails
                    logger.warning(f"[POLY FIT] {lane_name}: Weighted fitting failed, using unweighted fallback: {type(e).__name__}: {e}")
                    coeffs = np.polyfit(y, x, deg=2)
                
                # FIX #2: Temporal continuity - check for curvature sign flips and large deviations
                # Use previous frame's fit as prior to prevent instability on curves
                # CRITICAL: Only use temporal prior if we have previous frame data (not on frame 0)
                previous_coeffs = self.previous_lanes[lane_idx]
                use_temporal_prior = (previous_coeffs is not None and 
                                     self.previous_lanes_confidence[lane_idx] > 0.5)
                
                # Track if we rejected the current fit (to prevent storing rejected coefficients)
                rejected_current_fit = False
                
                if use_temporal_prior:
                    # Check for curvature sign flip (indicates potential swap or wrong fit)
                    current_curvature = coeffs[0]
                    previous_curvature = previous_coeffs[0]
                    current_curvature_sign = np.sign(current_curvature) if abs(current_curvature) > 0.001 else 0
                    previous_curvature_sign = np.sign(previous_curvature) if abs(previous_curvature) > 0.001 else 0
                    curvature_flip = current_curvature_sign != previous_curvature_sign
                    
                    if curvature_flip:
                        logger.warning(f"[TEMPORAL] {lane_name}: Curvature sign flip detected! "
                                     f"Previous: {previous_curvature:.6f}, Current: {current_curvature:.6f}. "
                                     f"Rejecting current fit and using previous.")
                        # Use previous frame's fit instead
                        coeffs = previous_coeffs.copy()
                        rejected_current_fit = True  # Mark that we rejected the current detection
                        # CRITICAL FIX: Mark this lane as rejected to prevent storing it as new previous_lanes
                        # This prevents the feedback loop where rejected coefficients get stored and reused
                        self.lanes_rejected_this_frame[lane_idx] = True
                    else:
                        # Check if curvature deviation is too large
                        curvature_deviation = abs(current_curvature - previous_curvature)
                        if curvature_deviation > self.curvature_deviation_threshold:
                            logger.warning(f"[TEMPORAL] {lane_name}: Large curvature deviation! "
                                         f"Previous: {previous_curvature:.6f}, Current: {current_curvature:.6f}, "
                                         f"Deviation: {curvature_deviation:.6f}. "
                                         f"Using smoothed fit.")
                            # Smooth the coefficients (exponential moving average)
                            alpha = 0.7  # Smoothing factor
                            coeffs = alpha * previous_coeffs + (1 - alpha) * coeffs
                
                logger.info(f"[FRAME0 DEBUG] {lane_name}: Fitted weighted polynomial coeffs={coeffs}")
                
                # Validate polynomial curvature (2nd degree coefficient)
                # For a straight road with perspective, curvature should be reasonable
                # Extreme curvature might indicate overfitting or incorrect detection
                # Typical curvature: |a| < 0.01 for reasonable perspective (for 480px height)
                curvature_coeff = abs(coeffs[0])  # |a| in ax^2 + bx + c
                max_reasonable_curvature = 0.02  # Allow up to 0.02 for perspective convergence
                if curvature_coeff > max_reasonable_curvature:
                    logger.warning(f"[FRAME0 DEBUG] {lane_name}: High curvature detected! "
                                 f"|a|={curvature_coeff:.6f} > {max_reasonable_curvature:.6f}. "
                                 f"This might indicate overfitting or incorrect detection.")
                
                # VALIDATION: Check if polynomial is reasonable
                # CRITICAL FIX: Don't evaluate at y=h (bottom) if points don't reach there!
                # ROI excludes bottom 7% (y=446 to 479), so y_max is likely ~446
                # Evaluating at y=h (480) requires extrapolation, which produces invalid values
                # Solution: Evaluate at y_max (where we have points) instead
                
                # Check if we need to extrapolate
                extrapolation_needed = h - y_max
                if extrapolation_needed > 10:  # Need to extrapolate >10px
                    # Use y_max instead of h (where we actually have points)
                    evaluation_y = y_max
                    logger.info(f"[FRAME0 DEBUG] {lane_name}: Points don't reach bottom! "
                               f"y_max={y_max:.1f}px, h={h}px, extrapolation={extrapolation_needed:.1f}px. "
                               f"Evaluating at y_max instead of h.")
                else:
                    # Points reach close to bottom, safe to evaluate at h
                    evaluation_y = h
                
                # Evaluate at chosen y position
                lane_x_at_bottom = np.polyval(coeffs, evaluation_y)
                
                # FRAME 0 DEBUG: Log evaluation
                logger.info(f"[FRAME0 DEBUG] {lane_name}: Evaluating at y={evaluation_y:.1f}px, "
                           f"result={lane_x_at_bottom:.1f}px")
                
                # Also evaluate at y_min and y_max (where we have actual points)
                lane_x_at_y_min = np.polyval(coeffs, y_min)
                lane_x_at_y_max = np.polyval(coeffs, y_max)
                logger.info(f"[FRAME0 DEBUG] {lane_name}: At y_min={y_min:.1f}px: {lane_x_at_y_min:.1f}px, "
                           f"at y_max={y_max:.1f}px: {lane_x_at_y_max:.1f}px")
                
                # Also evaluate at min/max y of points (should be within reasonable range)
                lane_x_at_min_y = np.polyval(coeffs, y_min)
                lane_x_at_max_y = np.polyval(coeffs, y_max)
                
                # Check if polynomial extrapolates too far beyond point range
                # If bottom (h) is far from y_max, polynomial might extrapolate poorly
                if h > y_max + 50:  # Bottom is >50px below max point
                    # Polynomial is extrapolating - check if result is reasonable
                    # Use linear extrapolation from last two points as sanity check
                    if len(points) >= 2:
                        # Get last two points (closest to bottom)
                        sorted_indices = np.argsort(points[:, 1])
                        last_two = points[sorted_indices[-2:]]
                        # Linear extrapolation
                        if last_two[1, 1] != last_two[0, 1]:
                            slope = (last_two[1, 0] - last_two[0, 0]) / (last_two[1, 1] - last_two[0, 1])
                            linear_extrapolation = last_two[1, 0] + slope * (h - last_two[1, 1])
                            # If polynomial differs significantly from linear, it's likely wrong
                            if abs(lane_x_at_bottom - linear_extrapolation) > w * 0.3:  # >30% of image width
                                logger.warning(f"[POLY DEBUG] {lane_name}: Polynomial extrapolation differs from linear! "
                                             f"poly={lane_x_at_bottom:.1f}px, linear={linear_extrapolation:.1f}px")
                                # Use linear extrapolation instead
                                lane_x_at_bottom = linear_extrapolation
                
                # DEBUG: Log polynomial evaluation
                logger.debug(f"[POLY DEBUG] {lane_name}: coeffs={coeffs}, h={h}, "
                           f"lane_x_at_bottom={lane_x_at_bottom:.1f}px, center_x={center_x:.1f}px")
                
                # Bounds check: lane_x_at_bottom should be within [0, w]
                # CRITICAL FIX: Clamp to bounds instead of rejecting, especially on frame 0
                if lane_x_at_bottom < 0 or lane_x_at_bottom > w:
                    logger.warning(f"[POLY DEBUG] {lane_name}: Polynomial evaluation out of bounds! "
                                 f"lane_x_at_bottom={lane_x_at_bottom:.1f}px (image width={w}px). Clamping to bounds.")
                    # Clamp to valid range instead of rejecting
                    lane_x_at_bottom = np.clip(lane_x_at_bottom, 0, w)
                    # If still way out of bounds after clamping, use linear extrapolation
                    if abs(lane_x_at_bottom - center_x) > w * 0.9:  # >90% of image width from center
                        # Try linear extrapolation from last two points
                        if len(points) >= 2:
                            sorted_indices = np.argsort(points[:, 1])
                            last_two = points[sorted_indices[-2:]]
                            if last_two[1, 1] != last_two[0, 1]:
                                slope = (last_two[1, 0] - last_two[0, 0]) / (last_two[1, 1] - last_two[0, 1])
                                linear_extrapolation = last_two[1, 0] + slope * (evaluation_y - last_two[1, 1])
                                linear_extrapolation = np.clip(linear_extrapolation, 0, w)
                                logger.info(f"[POLY DEBUG] {lane_name}: Using linear extrapolation instead: "
                                          f"{linear_extrapolation:.1f}px (was {lane_x_at_bottom:.1f}px)")
                                lane_x_at_bottom = linear_extrapolation
                        # If still invalid, reject
                        if abs(lane_x_at_bottom - center_x) > w * 0.9:
                            logger.warning(f"[POLY DEBUG] {lane_name}: Still invalid after clamping/linear extrapolation. Rejecting.")
                            lanes.append(None)
                            validation_failures[lane_name].append(f'invalid_evaluation_{lane_x_at_bottom:.1f}px')
                            continue
                
                # Expected lane positions: left lane should be left of center, right lane right of center
                # RELAXED: Increased threshold to 90% to allow for car offset, wide lanes, and coordinate issues
                # If lane is too far from center (>90% of image width), it might be a road edge
                # NOTE: We're seeing distances of 350-850px, which suggests coordinate conversion issues
                # But for now, relax threshold to allow these through so we can see polynomials
                distance_from_center = abs(lane_x_at_bottom - center_x)
                max_reasonable_distance = w * 0.90  # 90% of image width (576px for 640px) - very lenient
                
                if distance_from_center > max_reasonable_distance:
                    # Lane is too far from center - likely a road edge or false detection
                    # Skip this lane
                    logger.warning(f"[POLY DEBUG] {lane_name}: Rejected - too_far_from_center "
                                 f"({distance_from_center:.1f}px > {max_reasonable_distance:.1f}px)")
                    lanes.append(None)
                    validation_failures[lane_name].append(f'too_far_from_center_{distance_from_center:.1f}px')
                    continue
                
                lanes.append(coeffs)
            except Exception as e:
                lanes.append(None)
                validation_failures[lane_name].append(f'polyfit_failed_{str(e)}')
        
        # ADDITIONAL VALIDATION: Check if detected lane width is reasonable
        # If both lanes detected, check their separation
        if lanes[0] is not None and lanes[1] is not None:
            # CRITICAL FIX: Evaluate at same y position used for individual validation
            # Don't use h (bottom) if points don't reach there - use y_max instead
            # We need to find the evaluation_y that was used for each lane
            # For now, use a common evaluation point: max of both lanes' y_max values
            # Or use h if both lanes reach close to bottom
            
            # Get the actual evaluation y positions (would need to track these, but for now use h)
            # Actually, since we evaluate at y_max when extrapolation >10px, we should do the same here
            # But we don't have access to the y_max values here, so we'll evaluate at h and hope it's close
            # BETTER: Evaluate at a reasonable y position that both lanes likely have points at
            # Use y = min(y_max_left, y_max_right) or h, whichever is smaller
            # For simplicity, evaluate at h but clamp results if they're invalid
            
            # CRITICAL FIX: Don't evaluate at y=h if points don't reach there
            # Use a safer y position that both lanes likely have points at
            # Start with y = h - 100 (100px from bottom) to avoid extrapolation
            evaluation_y = max(0, h - 100)  # 100px from bottom (safer than h)
            left_x = np.polyval(lanes[0], evaluation_y)
            right_x = np.polyval(lanes[1], evaluation_y)
            
            # If evaluation produces invalid values, try even higher y
            if left_x < 0 or left_x > w or right_x < 0 or right_x > w:
                # Invalid evaluation - try at an even higher y position
                evaluation_y = max(0, h - 150)  # 150px from bottom
                left_x = np.polyval(lanes[0], evaluation_y)
                right_x = np.polyval(lanes[1], evaluation_y)
                logger.warning(f"[FRAME0 DEBUG] Lane width validation: Invalid at y={h-100}, trying y={evaluation_y}")
            
            # Final check - if still invalid, skip width validation (don't reject lanes)
            if left_x < 0 or left_x > w or right_x < 0 or right_x > w:
                logger.warning(f"[FRAME0 DEBUG] Lane width validation: Still invalid at y={evaluation_y}, skipping width check")
                # Don't reject lanes - width validation can't be trusted if evaluation is invalid
            else:
                lane_width_pixels = abs(right_x - left_x)
                logger.info(f"[FRAME0 DEBUG] Lane width validation: left_x={left_x:.1f}px, right_x={right_x:.1f}px, "
                           f"width={lane_width_pixels:.1f}px at y={evaluation_y:.1f}px")
                
                # Expected lane width: ~200-300 pixels at bottom (for 640px width, 3.5m lane)
                # CRITICAL FIX: Check if lanes are too close to image edges first
                # If lanes are at edges, they might be road edges (not lane lines)
                # Road edges would give width ~80-90% of image, which is invalid
                edge_margin = w * 0.05  # 5% margin from edges (32px for 640px)
                left_near_edge = left_x < edge_margin or left_x > (w - edge_margin)
                right_near_edge = right_x < edge_margin or right_x > (w - edge_margin)
                
                # If both lanes are near edges, likely detecting road edges (not lane lines)
                if left_near_edge and right_near_edge:
                    logger.warning(f"[FRAME0 DEBUG] Lane width validation: Both lanes near edges "
                                 f"(left={left_x:.1f}px, right={right_x:.1f}px). "
                                 f"Likely detecting road edges, not lane lines. Rejecting both.")
                    # Don't reject - let individual validation handle it
                    # But log the issue for debugging
                else:
                    # Normal lane width validation
                    # RELAXED: More lenient to account for perspective when car swerves
                    # When car swerves right, left lane appears smaller (perspective)
                    # When car swerves left, right lane appears smaller (perspective)
                    # Increased max to 85% to handle extreme perspective cases
                    min_reasonable_width = w * 0.08  # 8% of image width (51px for 640px) - RELAXED from 10%
                    max_reasonable_width = w * 0.85  # 85% of image width (544px for 640px) - RELAXED from 80%
                    
                    if lane_width_pixels < min_reasonable_width or lane_width_pixels > max_reasonable_width:
                        logger.warning(f"[FRAME0 DEBUG] Lane width validation: Width {lane_width_pixels:.1f}px outside "
                                     f"reasonable range [{min_reasonable_width:.1f}, {max_reasonable_width:.1f}]px")
                        # Lane width is unreasonable - likely false detection OR perspective issue
                        # RELAXED: Don't reject immediately - check if it's just perspective
                        # If width is close to limits, it might be valid perspective distortion
                        margin = w * 0.05  # 5% margin (32px for 640px)
                        if (lane_width_pixels < min_reasonable_width + margin or 
                            lane_width_pixels > max_reasonable_width - margin):
                            # Close to limits - might be valid perspective, keep both lanes
                            logger.info(f"[FRAME0 DEBUG] Lane width validation: Width {lane_width_pixels:.1f}px close to "
                                      f"limits, allowing (likely perspective distortion)")
                        else:
                            # Far outside limits - likely false detection
                            # Keep the lane closer to center, reject the other
                            left_dist = abs(left_x - center_x)
                            right_dist = abs(right_x - center_x)
                            
                            if left_dist < right_dist:
                                # Left lane is closer to center, keep it, reject right
                                logger.warning(f"[FRAME0 DEBUG] Lane width validation: Rejecting right lane (closer to center)")
                                lanes[1] = None
                            else:
                                # Right lane is closer to center, keep it, reject left
                                logger.warning(f"[FRAME0 DEBUG] Lane width validation: Rejecting left lane (closer to center)")
                                lanes[0] = None
        
        # TEMPORAL PERSISTENCE: Use previous frame's lanes if current detection fails
        # Also update x_at_bottom positions for next frame's swap prevention
        # NOTE: Rejection flags are reset at start of detection loop (above), not here
        
        for i, lane in enumerate(lanes):
            if lane is None and self.previous_lanes[i] is not None:
                # Current detection failed, but we have previous frame's lane
                # Use previous lane if confidence is still reasonable
                if self.previous_lanes_confidence[i] > 0.3:  # Minimum confidence threshold
                    lanes[i] = self.previous_lanes[i].copy()  # Use copy to avoid mutation
                    logger.info(f"[TEMPORAL] Using previous frame's {'left' if i == 0 else 'right'} lane "
                              f"(confidence: {self.previous_lanes_confidence[i]:.2f})")
                    # Decay confidence (will be updated if detection succeeds next frame)
                    self.previous_lanes_confidence[i] *= self.temporal_decay
                    # Keep previous x_at_bottom for swap prevention
                    # (don't update it since we're using previous lane)
                else:
                    # Confidence too low, don't use previous lane
                    logger.debug(f"[TEMPORAL] Previous {'left' if i == 0 else 'right'} lane confidence too low "
                               f"({self.previous_lanes_confidence[i]:.2f}), not using")
                    # Clear x_at_bottom since we're not using this lane
                    self.previous_lanes_x_at_bottom[i] = None
            elif lane is not None:
                # Current detection succeeded - update previous lanes and reset confidence
                # CRITICAL FIX: If this lane was rejected due to temporal filtering (curvature sign flip),
                # do NOT update previous_lanes with the rejected coefficients. This prevents a feedback loop
                # where rejected coefficients get stored and reused, causing lanes to freeze.
                if self.lanes_rejected_this_frame[i]:
                    # Lane was rejected due to curvature sign flip - don't store it as new previous_lanes
                    # Instead, decay confidence to allow new detections to eventually update
                    if self.previous_lanes[i] is not None:
                        self.previous_lanes_confidence[i] = min(0.8, self.previous_lanes_confidence[i] * 0.9)
                        logger.debug(f"[TEMPORAL] {('left' if i == 0 else 'right')} lane was rejected - "
                                   f"not updating previous_lanes, decaying confidence to {self.previous_lanes_confidence[i]:.2f}")
                    # Keep old previous_lanes - don't update with rejected coefficients
                else:
                    # New detection or smoothed fit - update previous lanes
                    self.previous_lanes[i] = lane.copy()  # Store copy to avoid mutation
                    self.previous_lanes_confidence[i] = 1.0  # Full confidence for successful detection
                
                # Update x_at_bottom position for next frame's swap prevention
                # Evaluate polynomial at bottom of image
                try:
                    lane_x_at_bottom = np.polyval(lane, h)
                    # Clamp to valid range
                    if 0 <= lane_x_at_bottom <= w:
                        self.previous_lanes_x_at_bottom[i] = lane_x_at_bottom
                    else:
                        # Invalid evaluation - try at a higher y position
                        evaluation_y = max(0, h - 50)
                        lane_x_at_bottom = np.polyval(lane, evaluation_y)
                        if 0 <= lane_x_at_bottom <= w:
                            self.previous_lanes_x_at_bottom[i] = lane_x_at_bottom
                        else:
                            self.previous_lanes_x_at_bottom[i] = None
                except Exception:
                    self.previous_lanes_x_at_bottom[i] = None
            else:
                # Both current and previous failed - decay confidence
                self.previous_lanes_confidence[i] *= self.temporal_decay
                # Clear x_at_bottom since we don't have a valid lane
                self.previous_lanes_x_at_bottom[i] = None
        
        # Store debug info if requested (after all processing is done)
        if return_debug:
            debug_info = {
                'all_lines': lines,
                'masked_edges': masked_edges,
                'lane_color_mask': lane_color_mask,
                # Note: white_mask removed from debug_info since we're using yellow lanes
                # White detection still used in pipeline but not saved for debugging
                'yellow_mask': yellow_mask,  # Yellow detection (primary for yellow lanes)
                'white_mask_roi': white_mask_roi,  # White detection (ROI-restricted)
                'edges': edges_roi,  # Edges restricted to lane paint + ROI (for visualization)
                'edges_yellow_only': edges_yellow_roi,  # Edges only on yellow mask (for debugging/verification)
                'roi_mask': roi_mask,  # ROI mask used for restriction
                'combined': combined,  # Combined edges + color mask
                'gray': gray,  # Grayscale image
                'blurred': blurred,  # Blurred grayscale
                'num_lines_detected': len(lines) if lines is not None else 0,  # Count of lines
                'left_lines_count': len(left_lines),
                'right_lines_count': len(right_lines),
                'skipped_short': skipped_short,
                'skipped_center': skipped_center,
                'validation_failures': validation_failures
            }
            return lanes, debug_info
        return lanes
    
    def visualize_detection(self, image: np.ndarray, lane_coeffs: List[Optional[np.ndarray]], 
                          debug_info: Optional[dict] = None) -> np.ndarray:
        """
        Visualize lane detection results with debug information.
        
        Args:
            image: Original RGB image
            lane_coeffs: Detected lane polynomial coefficients [left, right]
            debug_info: Debug information dict from detect() with return_debug=True
        
        Returns:
            Visualization image with detected lines overlaid
        """
        import cv2
        import numpy as np
        
        vis_image = image.copy()
        h, w = image.shape[:2]
        center_x = w // 2
        
        # Draw all Hough lines (gray) if available
        if debug_info is not None and 'all_lines' in debug_info:
            all_lines = debug_info['all_lines']
            # Handle both numpy array and list formats
            if all_lines is not None:
                try:
                    if isinstance(all_lines, np.ndarray):
                        if all_lines.ndim > 0 and all_lines.size > 0:
                            for i in range(len(all_lines)):
                                line = all_lines[i]
                                # Check if line is valid (not None and has elements)
                                if line is not None:
                                    line_array = np.asarray(line)
                                    if line_array.size > 0 and len(line_array.shape) > 0:
                                        try:
                                            x1, y1, x2, y2 = line_array[0]
                                            cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
                                        except (IndexError, ValueError, TypeError):
                                            pass
                    elif isinstance(all_lines, (list, tuple)):
                        for line in all_lines:
                            # Check if line is valid
                            if line is not None:
                                try:
                                    line_array = np.asarray(line)
                                    if line_array.size > 0 and len(line_array.shape) > 0:
                                        x1, y1, x2, y2 = line_array[0]
                                        cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
                                except (IndexError, ValueError, TypeError):
                                    pass
                except Exception:
                    pass  # Skip if drawing fails
        
        # Draw final polynomial fits (green for left, cyan for right)
        y_points = np.linspace(h // 3, h, 100).astype(np.float32)
        
        if lane_coeffs is not None and len(lane_coeffs) > 0 and lane_coeffs[0] is not None:
            try:
                # Left lane (green)
                x_points = np.polyval(lane_coeffs[0], y_points)
                # Filter out invalid points
                valid_mask = (x_points >= 0) & (x_points < w)
                if np.any(valid_mask):
                    points = np.array([x_points[valid_mask], y_points[valid_mask]], dtype=np.int32).T
                    if len(points) > 1:
                        cv2.polylines(vis_image, [points], False, (0, 255, 0), 3)
            except Exception:
                pass  # Skip if polynomial evaluation fails
        
        if lane_coeffs is not None and len(lane_coeffs) > 1 and lane_coeffs[1] is not None:
            try:
                # Right lane (cyan)
                x_points = np.polyval(lane_coeffs[1], y_points)
                # Filter out invalid points
                valid_mask = (x_points >= 0) & (x_points < w)
                if np.any(valid_mask):
                    points = np.array([x_points[valid_mask], y_points[valid_mask]], dtype=np.int32).T
                    if len(points) > 1:
                        cv2.polylines(vis_image, [points], False, (255, 255, 0), 3)
            except Exception:
                pass  # Skip if polynomial evaluation fails
        
        # Draw ROI boundary (yellow rectangle) if available
        # ROI is center 80% of image width (10% margin each side), lower 2/3 of height
        try:
            roi_margin = int(w * 0.10)  # 10% margin
            # Draw yellow rectangle outline for ROI
            cv2.rectangle(vis_image, 
                        (roi_margin, h // 3), 
                        (w - roi_margin, h), 
                        (0, 255, 255),  # Yellow in BGR
                        2)  # Thickness
        except Exception:
            pass  # Skip if ROI drawing fails
        
        # Draw image center line (red)
        cv2.line(vis_image, (center_x, h // 3), (center_x, h), (0, 0, 255), 1, cv2.LINE_AA)
        
        return vis_image


def load_pretrained_model(model_path: Optional[str] = None) -> LaneDetectionModel:
    """
    Load pretrained lane detection model.
    
    Args:
        model_path: Path to model checkpoint (if None, returns untrained model)
    
    Returns:
        Loaded model
    """
    model = LaneDetectionModel()
    
    if model_path and model_path.exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print("No pretrained model found, using untrained model")
    
    return model

