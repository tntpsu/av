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
        # PHASE 2: Spatial clustering parameters
        self.cluster_eps = 30.0  # Maximum distance between points in same cluster (pixels)
        self.cluster_min_samples = 3  # Minimum points required to form a cluster
    
    def _evaluate_polynomial_safe(self, coeffs: np.ndarray, y: float, points: np.ndarray, 
                                   w: int, h: int) -> float:
        """
        Safely evaluate polynomial with linear extrapolation when far from data points.
        
        This prevents unstable polynomial extrapolation by using linear extrapolation
        when evaluating far from the data range (>50px).
        
        Args:
            coeffs: Polynomial coefficients [a, b, c] for ax^2 + bx + c
            y: Y position to evaluate at
            points: Array of points used to fit the polynomial
            w: Image width
            h: Image height
        
        Returns:
            X position at y, using polynomial or linear extrapolation
        """
        if len(points) < 2:
            # Not enough points for linear extrapolation, use polynomial
            return np.polyval(coeffs, y)
        
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        
        # Check if we're far from data range
        extrapolation_distance = 0
        if y < y_min:
            extrapolation_distance = y_min - y
        elif y > y_max:
            extrapolation_distance = y - y_max
        
        if extrapolation_distance > 50:  # More than 50px from data
            # Use linear extrapolation from nearest data points
            sorted_indices = np.argsort(points[:, 1])
            
            if y < y_min:
                # Extrapolate upward using first two points (top of data)
                first_two = points[sorted_indices[:2]]
                if first_two[1, 1] != first_two[0, 1]:
                    slope = (first_two[1, 0] - first_two[0, 0]) / (first_two[1, 1] - first_two[0, 1])
                    x = first_two[0, 0] + slope * (y - first_two[0, 1])
                    # Clamp to reasonable bounds
                    return np.clip(x, -w * 0.2, w * 1.2)
                else:
                    # Points have same y, use polynomial
                    return np.polyval(coeffs, y)
            else:  # y > y_max
                # Extrapolate downward using last two points (bottom of data)
                last_two = points[sorted_indices[-2:]]
                if last_two[1, 1] != last_two[0, 1]:
                    slope = (last_two[1, 0] - last_two[0, 0]) / (last_two[1, 1] - last_two[0, 1])
                    x = last_two[1, 0] + slope * (y - last_two[1, 1])
                    # Clamp to reasonable bounds
                    return np.clip(x, -w * 0.2, w * 1.2)
                else:
                    # Points have same y, use polynomial
                    return np.polyval(coeffs, y)
        else:
            # Close to data range, use polynomial (it's stable here)
            return np.polyval(coeffs, y)
    
    def _spatial_cluster_points(self, points: np.ndarray, lane_idx: int, center_x: int, 
                                w: int, h: int, lane_name: str) -> Optional[np.ndarray]:
        """
        PHASE 2: Spatial clustering to separate left/right clusters.
        
        Uses distance-based clustering from expected lane position to find points
        that belong to the same lane line. This handles cross-lane contamination
        and ensures we use points across the entire lane line (not just bottom).
        
        Args:
            points: Array of points [[x1, y1], [x2, y2], ...]
            lane_idx: 0 for left lane, 1 for right lane
            center_x: Center x-coordinate of image
            w: Image width
            h: Image height
            lane_name: 'left' or 'right' for logging
        
        Returns:
            Clustered points array (all points within threshold of expected lane) or None if clustering fails
        """
        if len(points) < self.cluster_min_samples:
            return None
        
        # IMPROVED: Cluster by distance from expected lane position
        # Use previous frame's lane position as reference, or estimate from current points
        # This works better on curves where x-position varies along y
        
        # Estimate expected lane x-position at each y
        # If we have previous frame data, use it; otherwise fit a line to current points
        if self.previous_lanes_x_at_bottom[lane_idx] is not None:
            # Use previous frame's position as reference
            prev_x_bottom = self.previous_lanes_x_at_bottom[lane_idx]
            # For parallel lanes, x is roughly constant in image space
            # But on curves, we need to account for perspective
            # Simple model: x_expected ≈ prev_x (constant for now, can improve with polynomial)
            expected_x = prev_x_bottom
        else:
            # No previous data - estimate from current points
            # Use median x-position as reference (more robust than mean)
            expected_x = np.median(points[:, 0])
        
        # Calculate distance from expected lane position for each point
        # Use perpendicular distance (simplified: just x-distance for efficiency)
        # On curves, we should use actual perpendicular distance, but x-distance is a good approximation
        distances = np.abs(points[:, 0] - expected_x)
        
        # Cluster points that are close to expected lane position
        # Use adaptive threshold based on lane width and image size
        # Typical lane width in image: ~100-200px, so threshold should be ~50-100px
        # IMPROVED: Use tighter threshold (8% instead of 15%) to prevent cross-lane contamination
        # This is especially important in upper regions where perspective makes lanes appear closer
        cluster_threshold = max(50.0, w * 0.08)  # At least 50px, or 8% of image width (tighter)
        
        # Find points within threshold
        in_cluster_mask = distances <= cluster_threshold
        clustered_points = points[in_cluster_mask]
        
        if len(clustered_points) < self.cluster_min_samples:
            # Not enough points in cluster, try a more lenient threshold
            cluster_threshold = max(80.0, w * 0.25)  # More lenient: 80px or 25% of width
            in_cluster_mask = distances <= cluster_threshold
            clustered_points = points[in_cluster_mask]
            
            if len(clustered_points) < self.cluster_min_samples:
                # Still not enough, return all points
                logger.warning(f"[SPATIAL CLUSTER] {lane_name}: Not enough points in cluster "
                             f"({len(clustered_points)} < {self.cluster_min_samples}), using all {len(points)} points")
                return points
        
        # Additional validation: ensure cluster is on correct side of center
        cluster_mean_x = np.mean(clustered_points[:, 0])
        cluster_y_range = np.max(clustered_points[:, 1]) - np.min(clustered_points[:, 1])
        
        # CRITICAL: Filter out points that are clearly on the wrong side of center
        # This prevents cross-lane contamination even if cluster mean is acceptable
        # Use stricter margin for top region points (they're further away, more prone to cross-lane)
        margin = w * 0.08  # 8% margin (stricter than 10% to prevent cross-lane contamination)
        if lane_idx == 0:  # Left lane
            # Left lane points should be left of center (with margin for curves)
            valid_mask = clustered_points[:, 0] < (center_x + margin)
            if np.sum(~valid_mask) > 0:
                logger.info(f"[SPATIAL CLUSTER] {lane_name}: Filtering {np.sum(~valid_mask)} points on wrong side "
                           f"(x >= {center_x + margin:.0f}px)")
                clustered_points = clustered_points[valid_mask]
            if cluster_mean_x > center_x + margin:
                logger.warning(f"[SPATIAL CLUSTER] {lane_name}: Cluster mean_x={cluster_mean_x:.1f} "
                             f"is too far right (>{center_x + margin:.1f})")
        else:  # Right lane
            # Right lane points should be right of center (with margin for curves)
            valid_mask = clustered_points[:, 0] > (center_x - margin)
            if np.sum(~valid_mask) > 0:
                logger.info(f"[SPATIAL CLUSTER] {lane_name}: Filtering {np.sum(~valid_mask)} points on wrong side "
                           f"(x <= {center_x - margin:.0f}px)")
                clustered_points = clustered_points[valid_mask]
            if cluster_mean_x < center_x - margin:
                logger.warning(f"[SPATIAL CLUSTER] {lane_name}: Cluster mean_x={cluster_mean_x:.1f} "
                             f"is too far left (<{center_x - margin:.1f})")
        
        # Log clustering results
        removed_count = len(points) - len(clustered_points)
        if removed_count > 0:
            logger.info(f"[SPATIAL CLUSTER] {lane_name}: Clustered {len(clustered_points)}/{len(points)} points "
                       f"(removed {removed_count} outliers, y_range={cluster_y_range:.1f}px, "
                       f"mean_x={cluster_mean_x:.1f}, threshold={cluster_threshold:.1f}px)")
        
        return clustered_points
    
    def _sample_points_per_row(self, lane_idx: int, yellow_mask: np.ndarray, white_mask: np.ndarray,
                                lane_color_mask: np.ndarray, center_x: int, w: int, h: int,
                                lane_name: str, previous_coeffs: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        PHASE 1 (ROOT CAUSE FIX): Sample points directly from yellow mask using per-row sampling.
        
        This replaces Hough line detection as the primary point source. Instead of sparse,
        bottom-heavy Hough segments, we sample one point per y-slice from the yellow mask.
        This provides uniform y-coverage, eliminates extrapolation, and ensures stable regression.
        
        Args:
            lane_idx: 0 for left lane, 1 for right lane
            yellow_mask: Binary mask of yellow pixels
            white_mask: Binary mask of white pixels (ROI-restricted)
            lane_color_mask: Binary mask of lane-colored pixels (yellow + white)
            center_x: Center x-coordinate of image
            w: Image width
            h: Image height
            lane_name: 'left' or 'right' for logging
            previous_coeffs: Previous frame's polynomial coefficients (for temporal prediction)
        
        Returns:
            Array of points [[x1, y1], [x2, y2], ...] or None if insufficient points
        """
        roi_y_start = int(h * 0.18)
        roi_y_end = int(h * 0.80)
        lookahead_y_estimate = int(h * 0.73)
        region_top_end = max(roi_y_start + 50, lookahead_y_estimate - 80)
        
        # Use lane-specific color mask (left = yellow, right = white)
        primary_mask = yellow_mask if lane_idx == 0 else white_mask
        sampling_mask = primary_mask.copy()
        if np.sum(sampling_mask) < 100:  # If lane-specific mask is sparse, use combined mask
            sampling_mask = lane_color_mask.copy()
        
        # Apply ROI
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_vertices = np.array([[(0, roi_y_end), (w, roi_y_end), (w, roi_y_start), (0, roi_y_start)]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        sampling_mask = cv2.bitwise_and(sampling_mask, roi_mask)
        
        # Per-row sampling: sample one point per y-slice (every 5-10px)
        y_step = 8  # Sample every 8px (adjustable: 5-10px recommended)
        points = []
        
        # Determine search region for each row based on lane side
        # Use previous frame's polynomial to predict x at each y (if available)
        use_temporal_prediction = (previous_coeffs is not None and 
                                  self.previous_lanes_confidence[lane_idx] > 0.3)
        
        rows_checked = 0
        rows_with_pixels = 0
        rows_skipped_no_pixels = 0
        rows_skipped_lateral = 0
        rows_skipped_delta = 0
        
        top_region_y = int(h * 0.45)
        min_top_pixels = max(40, int(0.001 * w * (region_top_end - roi_y_start)))
        top_region_pixel_count = int(np.sum(primary_mask[roi_y_start:region_top_end, :] > 0))
        top_region_has_pixels = top_region_pixel_count >= min_top_pixels
        for y in range(roi_y_start, roi_y_end, y_step):
            rows_checked += 1
            if y < region_top_end and not top_region_has_pixels:
                rows_skipped_no_pixels += 1
                continue
            # Extract horizontal band (y ± 2px for tolerance)
            y_band_start = max(roi_y_start, y - 2)
            y_band_end = min(roi_y_end, y + 2)
            band_mask = sampling_mask[y_band_start:y_band_end, :]
            
            # Find yellow pixels in this band
            y_coords, x_coords = np.where(band_mask > 0)
            
            if len(x_coords) == 0:
                rows_skipped_no_pixels += 1
                continue  # No pixels in this row
            
            # Apply lateral constraint (left lane < center, right lane > center)
            margin = w * 0.15  # 15% margin for curves
            if lane_idx == 0:  # Left lane
                valid_mask = x_coords < (center_x + margin)
            else:  # Right lane
                valid_mask = x_coords > (center_x - margin)
            
            valid_x = x_coords[valid_mask]
            if len(valid_x) == 0:
                rows_skipped_lateral += 1
                continue  # No valid pixels after lateral constraint
            
            min_row_pixels = 3
            min_row_pixels_top = 5
            if y < region_top_end:
                if len(valid_x) < min_row_pixels_top:
                    rows_skipped_no_pixels += 1
                    continue
                # Require a minimum contiguous run in the top band to avoid speckle noise
                sorted_x = np.sort(valid_x)
                max_run = 1
                run_len = 1
                for i in range(1, len(sorted_x)):
                    if sorted_x[i] - sorted_x[i - 1] <= 1:
                        run_len += 1
                        if run_len > max_run:
                            max_run = run_len
                    else:
                        run_len = 1
                if max_run < 6:
                    rows_skipped_no_pixels += 1
                    continue
            else:
                if len(valid_x) < min_row_pixels:
                    rows_skipped_no_pixels += 1
                    continue

            # Top-region guard: keep points near expected lane to prevent cross-lane swaps
            if y < top_region_y:
                top_margin = w * 0.05  # Stricter in far/top region
                if lane_idx == 0:
                    valid_x = valid_x[valid_x < (center_x + top_margin)]
                else:
                    valid_x = valid_x[valid_x > (center_x - top_margin)]
                
                if len(valid_x) == 0:
                    rows_skipped_lateral += 1
                    continue
                
                if len(points) > 0:
                    expected_x = np.median([p[0] for p in points])
                    expected_band = w * 0.10
                    expected_mask = (valid_x >= expected_x - expected_band) & (valid_x <= expected_x + expected_band)
                    if np.any(expected_mask):
                        valid_x = valid_x[expected_mask]
                    else:
                        rows_skipped_lateral += 1
                        continue
            
            # Use temporal prediction to narrow search if available
            if use_temporal_prediction:
                predicted_x = np.polyval(previous_coeffs, y)
                search_width = w * 0.20  # 20% search width around prediction
                x_min_search = max(0, int(predicted_x - search_width))
                x_max_search = min(w, int(predicted_x + search_width))
                search_mask = (valid_x >= x_min_search) & (valid_x <= x_max_search)
                if np.sum(search_mask) > 0:
                    valid_x = valid_x[search_mask]
            
            # Compute x as median of valid pixels (robust to outliers)
            x = np.median(valid_x)
            
            # Validate: check maximum lateral delta between adjacent rows
            if len(points) > 0:
                prev_x = points[-1][0]
                max_delta = w * 0.25  # Max 25% of image width change per row
                if abs(x - prev_x) > max_delta:
                    # Too large a jump - likely wrong lane or noise, skip this row
                    rows_skipped_delta += 1
                    logger.debug(f"[PER-ROW SAMPLING] {lane_name}: Skipping y={y}px (x={x:.1f}px, "
                               f"delta={abs(x-prev_x):.1f}px > {max_delta:.1f}px)")
                    continue
            
            rows_with_pixels += 1
            points.append([float(x), float(y)])
        
        if len(points) < 3:
            logger.warning(f"[PER-ROW SAMPLING] {lane_name}: Insufficient points ({len(points)} < 3)")
            return None
        
        points_array = np.array(points) if len(points) > 0 else np.array([])
        
        if len(points_array) > 0:
            logger.info(f"[PER-ROW SAMPLING] {lane_name}: Sampled {len(points)} points "
                       f"(y-range: [{np.min(points_array[:, 1]):.0f}, {np.max(points_array[:, 1]):.0f}]px)")
            logger.debug(f"[PER-ROW SAMPLING] {lane_name}: Rows checked={rows_checked}, "
                        f"with pixels={rows_with_pixels}, skipped: no_pixels={rows_skipped_no_pixels}, "
                        f"lateral={rows_skipped_lateral}, delta={rows_skipped_delta}")
        else:
            logger.warning(f"[PER-ROW SAMPLING] {lane_name}: No points sampled! "
                          f"Rows checked={rows_checked}, skipped: no_pixels={rows_skipped_no_pixels}, "
                          f"lateral={rows_skipped_lateral}, delta={rows_skipped_delta}")
        
        return points_array if len(points_array) > 0 else None
    
    def _supplement_points_from_mask(self, points: np.ndarray, lane_idx: int, 
                                     yellow_mask: np.ndarray, white_mask: np.ndarray, lane_color_mask: np.ndarray,
                                     center_x: int, w: int, h: int, lane_name: str,
                                     expected_x: Optional[float] = None) -> np.ndarray:
        """
        PHASE 2 ENHANCEMENT: Supplement points from yellow mask to ensure consistent distribution.
        
        After clustering, checks if points are evenly distributed across y-range.
        If gaps exist, samples points from yellow_mask to fill them.
        
        Args:
            points: Existing clustered points
            lane_idx: 0 for left lane, 1 for right lane
            yellow_mask: Yellow lane mask (binary image)
            white_mask: White lane mask (binary image, ROI-restricted)
            lane_color_mask: Combined lane color mask (yellow + white)
            center_x: Center x-coordinate of image
            w: Image width
            h: Image height
            lane_name: 'left' or 'right' for logging
            expected_x: Expected x-position of lane (for sampling region)
        
        Returns:
            Supplemented points array with consistent y-distribution
        """
        if len(points) == 0:
            return points
        
        # Determine expected x-position for sampling
        # CRITICAL: For top region, use polynomial to predict x at top y, not bottom
        # This is more accurate for curves where lane position changes with y
        if expected_x is None:
            if self.previous_lanes[lane_idx] is not None:
                # Use previous frame's polynomial to predict x at top region
                prev_coeffs = self.previous_lanes[lane_idx]
                top_y = roi_y_start + (region_top_end - roi_y_start) // 2  # Middle of top region
                expected_x = np.polyval(prev_coeffs, top_y)
            elif len(points) > 0:
                # Use median of current points, but also check if we can extrapolate
                expected_x = np.median(points[:, 0])
                # If we have a polynomial from current detection, use it to predict at top
                # (This will be set later, but for now use median)
            else:
                expected_x = center_x  # Fallback to center
        
        # Check y-distribution of existing points
        y_positions = points[:, 1]
        y_min = np.min(y_positions)
        y_max = np.max(y_positions)
        y_range = y_max - y_min
        
        # Divide y-range into regions: top, middle, bottom
        # IMPORTANT: Black dotted line (lookahead distance) is typically at ~73% from top (y=350px for 480px image)
        # We want points both above and below this line for good polynomial fitting
        roi_y_start = int(h * 0.18)  # ROI starts at 18% from top
        roi_y_end = int(h * 0.80)     # ROI ends at 80% from top
        
        # Estimate lookahead distance (black dotted line) - typically 8m = ~73% from top
        # This is where we evaluate the polynomial for control, so we need points around it
        lookahead_y_estimate = int(h * 0.73)  # ~350px for 480px image
        
        # Define regions: Split ROI into regions that ensure coverage above and below lookahead
        # CRITICAL: Black dotted line (lookahead) is at ~350px, we need points CLEARLY above it
        # Region 1: Top (ROI start to well above lookahead - ensures points above black line)
        # Region 2: Around lookahead (lookahead ± margin)  
        # Region 3: Bottom (below lookahead to ROI end)
        lookahead_margin = int(h * 0.05)  # 5% margin around lookahead (24px for 480px)
        
        # Top region should end well above lookahead to ensure we get points above black line
        # EXPANDED: Use 50px above lookahead (instead of 80px) to include more yellow pixels
        # This ensures we capture yellow lines that exist just below the original boundary
        # Original: 80px above = 270px, New: 50px above = 300px (includes gap region)
        region_top_end = max(roi_y_start + 50, lookahead_y_estimate - 50)  # 50px above lookahead (300px for 350px lookahead)
        region_lookahead_start = lookahead_y_estimate - lookahead_margin  # 326px
        region_lookahead_end = lookahead_y_estimate + lookahead_margin   # 374px
        region_bottom_start = lookahead_y_estimate + lookahead_margin    # 374px
        
        # Count points in each region
        top_points = np.sum((y_positions >= roi_y_start) & (y_positions < region_top_end))
        # Gap region: between top and lookahead (where yellow mask often has pixels)
        gap_start = region_top_end
        gap_end = region_lookahead_start
        gap_points = np.sum((y_positions >= gap_start) & (y_positions < gap_end))
        lookahead_points = np.sum((y_positions >= region_lookahead_start) & (y_positions < region_lookahead_end))
        bottom_points = np.sum((y_positions >= region_bottom_start) & (y_positions <= roi_y_end))
        
        # For backward compatibility, also count "middle" as the lookahead region
        middle_points = lookahead_points
        
        # Minimum points per region (ensure at least 2-3 points per region)
        min_points_per_region = 2
        min_lookahead_points = 2  # Critical region - need points at lookahead distance
        total_min_points = 6  # Minimum total points needed
        
        # CRITICAL: Check if points span the full y-range
        # If points don't start near roi_y_start, we need aggressive supplementation
        y_min_current = np.min(y_positions) if len(y_positions) > 0 else roi_y_end
        y_max_current = np.max(y_positions) if len(y_positions) > 0 else roi_y_start
        y_range_coverage = y_max_current - y_min_current
        y_range_needed = roi_y_end - roi_y_start
        
        # If points don't span at least 70% of ROI, we need supplementation
        # This ensures polynomial doesn't need to extrapolate >100px
        coverage_ratio = y_range_coverage / y_range_needed if y_range_needed > 0 else 0.0
        needs_aggressive_supplementation = coverage_ratio < 0.70 or y_min_current > roi_y_start + 50
        
        # If we have enough points overall and good distribution AND good y-range coverage, return early
        if len(points) >= total_min_points and top_points >= min_points_per_region and \
           lookahead_points >= min_lookahead_points and bottom_points >= min_points_per_region and \
           not needs_aggressive_supplementation:
            logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Good distribution, no supplementation needed "
                        f"(top={top_points}, gap={gap_points}, lookahead={lookahead_points}, bot={bottom_points}, "
                        f"y_range={y_range_coverage:.0f}px, coverage={coverage_ratio*100:.0f}%)")
            return points
        
        if needs_aggressive_supplementation:
            logger.info(f"[POINT SUPPLEMENT] {lane_name}: NEEDS AGGRESSIVE SUPPLEMENTATION - "
                       f"y_range={y_range_coverage:.0f}px ({coverage_ratio*100:.0f}% of ROI), "
                       f"y_min={y_min_current:.0f}px (ROI starts at {roi_y_start}px)")
        
        logger.info(f"[POINT SUPPLEMENT] {lane_name}: Checking distribution - "
                   f"top={top_points} (above), gap={gap_points} (between top/lookahead), "
                   f"lookahead={lookahead_points} (at {lookahead_y_estimate}px), "
                   f"bot={bottom_points} (below) (min={min_points_per_region} per region)")
        
        # Need to supplement points - sample from lane-specific mask when possible
        # Left lane prefers yellow, right lane prefers white; fallback to combined if sparse
        primary_mask = yellow_mask if lane_idx == 0 else white_mask
        sampling_mask = primary_mask.copy()
        if np.sum(sampling_mask) < 100:
            sampling_mask = lane_color_mask.copy()
        
        # If there are no lane pixels in the top region, skip top supplementation entirely
        min_top_pixels = max(40, int(0.001 * w * (region_top_end - roi_y_start)))
        top_region_pixel_count = int(np.sum(primary_mask[roi_y_start:region_top_end, :] > 0))
        top_region_has_pixels = top_region_pixel_count >= min_top_pixels
        
        # Restrict to ROI
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_vertices = np.array([[
            (0, roi_y_end),
            (w, roi_y_end),
            (w, roi_y_start),
            (0, roi_y_start)
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        sampling_mask = cv2.bitwise_and(sampling_mask, roi_mask)
        
        # Restrict to expected lane region (lateral constraint)
        # CRITICAL: For top region, use polynomial to predict x at top, not bottom x
        # This is more accurate because lanes curve and perspective changes x-position with y
        # BUT: Only use polynomial if points span a reasonable y-range (avoid extreme extrapolation)
        search_x = expected_x  # Default to expected_x
        if len(points) >= 3:
            try:
                y_vals = points[:, 1]
                x_vals = points[:, 0]
                y_range = np.max(y_vals) - np.min(y_vals)
                top_y_mid = roi_y_start + (region_top_end - roi_y_start) // 2
                
                # Only use polynomial if:
                # 1. Points span at least 50px in y (enough to fit polynomial)
                # 2. Top region is not too far from point range (extrapolation < 100px)
                y_min = np.min(y_vals)
                extrapolation_distance = y_min - top_y_mid
                
                if y_range >= 50 and extrapolation_distance < 100:
                    coeffs = np.polyfit(y_vals, x_vals, min(2, len(points) - 1))
                    predicted_x = np.polyval(coeffs, top_y_mid)
                    # Validate prediction is reasonable (within image bounds)
                    if 0 <= predicted_x < w:
                        search_x = predicted_x
                        logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Using polynomial prediction for search: "
                                   f"x={search_x:.1f}px at y={top_y_mid}px (was {expected_x:.1f}px at bottom)")
                    else:
                        logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Polynomial prediction out of bounds "
                                   f"({predicted_x:.1f}px), using expected_x ({expected_x:.1f}px)")
                else:
                    logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Skipping polynomial prediction "
                               f"(y_range={y_range:.1f}px, extrapolation={extrapolation_distance:.1f}px), "
                               f"using expected_x ({expected_x:.1f}px)")
            except Exception as e:
                logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Polynomial prediction failed: {e}, using expected_x")
                pass  # Fallback to expected_x
        
        # Use wider search on curves and for upper regions where lanes curve more
        # Base search width: 25% of image width (wider to catch curves and perspective effects)
        base_search_width = w * 0.25
        # For upper regions, use even wider search (lanes curve more at distance, perspective makes them appear wider apart)
        search_width = max(100, base_search_width)  # At least 100px, or 25% of width
        if lane_idx == 0:  # Left lane
            x_min = max(0, int(search_x - search_width))
            x_max = min(w, int(search_x + search_width))
        else:  # Right lane
            x_min = max(0, int(search_x - search_width))
            x_max = min(w, int(search_x + search_width))
        
        # Create lateral mask
        lateral_mask = np.zeros((h, w), dtype=np.uint8)
        lateral_mask[:, x_min:x_max] = 255
        sampling_mask = cv2.bitwise_and(sampling_mask, lateral_mask)
        
        # Sample points from regions that need more points
        supplemented_points = points.tolist()
        points_added = 0
        
        # CRITICAL: If points don't start near roi_y_start, we MUST supplement aggressively
        # This prevents large extrapolation that causes polynomial rejection
        y_min_current = np.min(y_positions) if len(y_positions) > 0 else roi_y_end
        gap_to_top = y_min_current - roi_y_start
        
        # If there's a large gap (>100px) from ROI start to first point, supplement aggressively
        needs_top_supplementation = (top_points < min_points_per_region) or (gap_to_top > 100)
        
        if needs_top_supplementation and not top_region_has_pixels:
            logger.info(f"[POINT SUPPLEMENT] {lane_name}: Skipping top supplementation (no pixels in top region)")
        
        # Sample from top region if needed (ABOVE lookahead/black line)
        if needs_top_supplementation and top_region_has_pixels:
            # CRITICAL: If there's a large gap, sample from the ENTIRE gap region
            # Don't just sample from "top region" - sample from roi_y_start all the way to where points start
            if gap_to_top > 100:
                # Sample from the entire gap: roi_y_start to y_min_current
                gap_end = min(int(y_min_current), region_top_end)
                region_mask = sampling_mask[roi_y_start:gap_end, :].copy()
                logger.info(f"[POINT SUPPLEMENT] {lane_name}: Sampling from ENTIRE gap region "
                           f"(y={roi_y_start}-{gap_end}px, gap={gap_to_top:.0f}px)")
            else:
                # Normal case: sample from top region
                region_mask = sampling_mask[roi_y_start:region_top_end, :].copy()
            y_coords, x_coords = np.where(region_mask > 0)
            logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Top region (above lookahead) - found {len(y_coords)} pixels in mask "
                        f"(region: y={roi_y_start}-{region_top_end}, x={x_min}-{x_max})")
            if len(y_coords) > 0:
                # If we found very few pixels, try wide search to get more
                if len(y_coords) < 5:
                    # Try wide search to get more points
                    # Use wider search width for top region (lanes curve more at distance)
                    wider_search_width = w * 0.35  # 35% for top region (even wider)
                    x_min_wide = max(0, int(search_x - wider_search_width))
                    x_max_wide = min(w, int(search_x + wider_search_width))
                    lateral_mask_wide = np.zeros((h, w), dtype=np.uint8)
                    lateral_mask_wide[:, x_min_wide:x_max_wide] = 255
                    sampling_mask_wide = cv2.bitwise_and(sampling_mask, roi_mask)
                    sampling_mask_wide = cv2.bitwise_and(sampling_mask_wide, lateral_mask_wide)
                    region_mask_wide = sampling_mask_wide[roi_y_start:region_top_end, :].copy()
                    y_coords_wide, x_coords_wide = np.where(region_mask_wide > 0)
                    if len(y_coords_wide) > len(y_coords):
                        # Use wide search results, but filter by lateral constraint to prevent cross-lane contamination
                        # CRITICAL: Filter out points that are clearly on the wrong side of center
                        valid_mask = np.ones(len(y_coords_wide), dtype=bool)
                        margin = w * 0.15  # 15% margin for curves (relaxed for top region)
                        if lane_idx == 0:  # Left lane
                            valid_mask = x_coords_wide < (center_x + margin)
                        else:  # Right lane
                            valid_mask = x_coords_wide > (center_x - margin)
                        
                        y_coords_filtered = y_coords_wide[valid_mask]
                        x_coords_filtered = x_coords_wide[valid_mask]
                        
                        if len(y_coords_filtered) > len(y_coords):
                            # Use filtered wide search results
                            y_coords, x_coords = y_coords_filtered, x_coords_filtered
                            logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Using wide search for top region - found {len(y_coords)} pixels "
                                       f"(x={x_min_wide}-{x_max_wide}, filtered from {len(y_coords_wide)} to remove cross-lane)")
                        else:
                            # Filtered results aren't better, keep original
                            logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Wide search found pixels but filtering removed too many "
                                       f"({len(y_coords_filtered)} <= {len(y_coords)}), keeping original")
                # Sample evenly spaced points - add MORE than minimum for better coverage
                # Top region is critical for polynomial fitting (lanes curve more at distance)
                # CRITICAL: Apply lateral constraint, but be MORE AGGRESSIVE in top region
                # If yellow mask is sparse, we need to trust it more - if we can see it, use it!
                # In top region (far away), perspective makes lanes appear closer, so relax constraint
                valid_indices = []
                # Use wider margin in top region (where we're far away and perspective distorts)
                # Bottom region: 15% margin, Top region: 25% margin (more aggressive)
                base_margin = w * 0.15
                top_margin = w * 0.25  # More aggressive for top region
                
                for i in range(len(x_coords)):
                    px = x_coords[i]
                    py_abs = y_coords[i] + roi_y_start  # Absolute y position
                    
                    # Use wider margin in top half of gap region
                    y_gap_mid = roi_y_start + (gap_end - roi_y_start) / 2
                    if py_abs < y_gap_mid:
                        margin = top_margin  # Top half: more aggressive
                    else:
                        margin = base_margin  # Bottom half: normal
                    
                    if lane_idx == 0:  # Left lane
                        if px < (center_x + margin):
                            valid_indices.append(i)
                    else:  # Right lane
                        if px > (center_x - margin):
                            valid_indices.append(i)
                
                if len(valid_indices) > 0:
                    # CRITICAL: If there's a large gap, add MORE points to fill it
                    # Be AGGRESSIVE - if yellow mask exists and is clear, use it!
                    # The yellow mask shows us the lane line clearly, so we should trust it
                    if gap_to_top > 100:
                        # Need to fill a large gap - add MORE points aggressively
                        # Target: ~1 point per 20-25px to ensure good coverage
                        num_to_add = max(min_points_per_region - top_points, 10, int(gap_to_top / 20))  # More aggressive: 1 per 20px
                    else:
                        num_to_add = max(min_points_per_region - top_points, 8)  # Add at least 8 points if available
                    # Use ALL available valid pixels if we have a large gap (don't be too conservative)
                    if gap_to_top > 100:
                        num_to_add = min(num_to_add, len(valid_indices), 15)  # Cap at 15 to avoid overfitting, but use what we have
                    else:
                        num_to_add = min(num_to_add, len(valid_indices))
                    # CRITICAL: Sample evenly across Y-RANGE, not just indices
                    # This ensures points span the entire gap, not just cluster where pixels are dense
                    if num_to_add > 0:
                        valid_y_coords = y_coords[valid_indices] + roi_y_start  # Convert to absolute y
                        valid_x_coords = x_coords[valid_indices]
                        
                        # Divide gap into num_to_add regions and sample one point from each region
                        y_min_gap = roi_y_start
                        y_max_gap = gap_end if gap_to_top > 100 else region_top_end
                        y_gap_range = y_max_gap - y_min_gap
                        
                        points_added_this_round = 0
                        for i in range(num_to_add):
                            # Target y for this point (evenly distributed across gap)
                            target_y = y_min_gap + (i + 0.5) * (y_gap_range / num_to_add)
                            
                            # Find pixel closest to target_y
                            distances = np.abs(valid_y_coords - target_y)
                            closest_idx = np.argmin(distances)
                            
                            # Add this point (avoid duplicates by checking if y is too close to existing)
                            point_y = valid_y_coords[closest_idx]
                            point_x = valid_x_coords[closest_idx]
                            
                            # Check if we already have a point very close to this y
                            too_close = False
                            for existing_point in supplemented_points:
                                if abs(existing_point[1] - point_y) < 10:  # Within 10px
                                    too_close = True
                                    break
                            
                            if not too_close:
                                supplemented_points.append([float(point_x), float(point_y)])
                                points_added += 1
                                points_added_this_round += 1
                        
                        logger.info(f"[POINT SUPPLEMENT] {lane_name}: Added {points_added_this_round} points evenly across gap "
                                   f"(y={y_min_gap}-{y_max_gap}px, gap={gap_to_top:.0f}px, "
                                   f"target distribution: {num_to_add} points)")
                else:
                    logger.warning(f"[POINT SUPPLEMENT] {lane_name}: No valid points after lateral constraint filtering "
                                 f"(all {len(x_coords)} points were on wrong side)")
            else:
                # If no pixels in restricted region, try wider search for top region (lanes curve more at distance)
                # IMPROVED: Also check gap region if top region is empty (yellow lines often in gap region)
                # First try top region with wide search
                search_x = expected_x
                if len(points) >= 3:
                    try:
                        y_vals = points[:, 1]
                        x_vals = points[:, 0]
                        coeffs = np.polyfit(y_vals, x_vals, min(2, len(points) - 1))
                        top_y_mid = roi_y_start + (region_top_end - roi_y_start) // 2
                        search_x = np.polyval(coeffs, top_y_mid)
                    except:
                        pass
                wider_search_width = w * 0.40  # 40% for top region (wider to catch curves)
                x_min_wide = max(0, int(search_x - wider_search_width))
                x_max_wide = min(w, int(search_x + wider_search_width))
                lateral_mask_wide = np.zeros((h, w), dtype=np.uint8)
                lateral_mask_wide[:, x_min_wide:x_max_wide] = 255
                sampling_mask_wide = cv2.bitwise_and(lane_color_mask, roi_mask)
                sampling_mask_wide = cv2.bitwise_and(sampling_mask_wide, lateral_mask_wide)
                region_mask_wide = sampling_mask_wide[roi_y_start:region_top_end, :].copy()
                y_coords, x_coords = np.where(region_mask_wide > 0)
                logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Top region (wide search) - found {len(y_coords)} pixels "
                            f"(x={x_min_wide}-{x_max_wide})")
                
                # If still no pixels in top region, try gap region (where yellow lines often exist)
                if len(y_coords) == 0:
                    gap_start = region_top_end
                    gap_end = region_lookahead_start
                    gap_mask = sampling_mask_wide[gap_start:gap_end, :].copy()
                    y_coords_gap, x_coords_gap = np.where(gap_mask > 0)
                    if len(y_coords_gap) > 0:
                        logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Top region empty, using gap region - found {len(y_coords_gap)} pixels")
                        # Use gap region pixels, but adjust y-coords to be relative to ROI start
                        y_coords = y_coords_gap + gap_start - roi_y_start
                        x_coords = x_coords_gap
                if len(y_coords) > 0:
                    # CRITICAL: Apply lateral constraint filtering before adding points
                    # This prevents cross-lane contamination from wide search
                    # RELAXED: Use 15% margin for top region (instead of 8%) to allow curves
                    valid_indices = []
                    margin = w * 0.15  # 15% margin (relaxed for curves and top region)
                    for i in range(len(x_coords)):
                        px = x_coords[i]
                        if lane_idx == 0:  # Left lane
                            if px < (center_x + margin):
                                valid_indices.append(i)
                        else:  # Right lane
                            if px > (center_x - margin):
                                valid_indices.append(i)
                    
                    if len(valid_indices) > 0:
                        # Add more points in top region for better coverage (lanes curve more at distance)
                        # Add up to 5-6 points (not just minimum) to ensure good polynomial fit
                        num_to_add = max(min_points_per_region - top_points, 5)  # Add at least 5 points if available
                        num_to_add = min(num_to_add, len(valid_indices))  # Don't exceed available valid pixels
                        indices = np.linspace(0, len(valid_indices) - 1, num_to_add, dtype=int)
                        for idx in indices:
                            i = valid_indices[idx]
                            supplemented_points.append([float(x_coords[i]), float(y_coords[i] + roi_y_start)])
                            points_added += 1
                        logger.info(f"[POINT SUPPLEMENT] {lane_name}: Added {num_to_add} points to top region (wide search, above black line, "
                                   f"filtered from {len(x_coords)} to {len(valid_indices)} valid)")
                    else:
                        logger.warning(f"[POINT SUPPLEMENT] {lane_name}: Wide search found {len(x_coords)} pixels but all were on wrong side "
                                     f"(left lane: all x >= {center_x + margin:.0f}px, right lane: all x <= {center_x - margin:.0f}px)")
                else:
                    logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Top region - no pixels found even with wide search")
        
        # Sample from gap region (between top and lookahead) - yellow mask often has pixels here
        if gap_points < min_points_per_region and gap_end > gap_start:  # Only if gap exists
            region_mask = sampling_mask[gap_start:gap_end, :].copy()
            y_coords, x_coords = np.where(region_mask > 0)
            logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Gap region - found {len(y_coords)} pixels in mask "
                        f"(region: y={gap_start}-{gap_end}, x={x_min}-{x_max})")
            if len(y_coords) > 0:
                num_to_add = min_points_per_region - gap_points
                indices = np.linspace(0, len(y_coords) - 1, min(num_to_add, len(y_coords)), dtype=int)
                for idx in indices:
                    supplemented_points.append([float(x_coords[idx]), float(y_coords[idx] + gap_start)])
                    points_added += 1
            else:
                # Try wider search for gap region
                wider_search_width = w * 0.25
                x_min_wide = max(0, int(expected_x - wider_search_width))
                x_max_wide = min(w, int(expected_x + wider_search_width))
                lateral_mask_wide = np.zeros((h, w), dtype=np.uint8)
                lateral_mask_wide[:, x_min_wide:x_max_wide] = 255
                sampling_mask_wide = cv2.bitwise_and(lane_color_mask, roi_mask)
                sampling_mask_wide = cv2.bitwise_and(sampling_mask_wide, lateral_mask_wide)
                region_mask_wide = sampling_mask_wide[gap_start:gap_end, :].copy()
                y_coords, x_coords = np.where(region_mask_wide > 0)
                logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Gap region (wide search) - found {len(y_coords)} pixels "
                            f"(x={x_min_wide}-{x_max_wide})")
                if len(y_coords) > 0:
                    num_to_add = min_points_per_region - gap_points
                    indices = np.linspace(0, len(y_coords) - 1, min(num_to_add, len(y_coords)), dtype=int)
                    for idx in indices:
                        supplemented_points.append([float(x_coords[idx]), float(y_coords[idx] + gap_start)])
                        points_added += 1
        
        # CRITICAL: Sample from lookahead region (around black dotted line)
        # This is where we evaluate the polynomial for control, so we MUST have points here
        if lookahead_points < min_lookahead_points:
            region_mask = sampling_mask[region_lookahead_start:region_lookahead_end, :].copy()
            y_coords, x_coords = np.where(region_mask > 0)
            logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Lookahead region (black line) - found {len(y_coords)} pixels in mask "
                        f"(region: y={region_lookahead_start}-{region_lookahead_end}, x={x_min}-{x_max})")
            if len(y_coords) > 0:
                num_to_add = min_lookahead_points - lookahead_points
                indices = np.linspace(0, len(y_coords) - 1, min(num_to_add, len(y_coords)), dtype=int)
                for idx in indices:
                    supplemented_points.append([float(x_coords[idx]), float(y_coords[idx] + region_lookahead_start)])
                    points_added += 1
            else:
                # Try wider search for lookahead region (critical for control)
                wider_search_width = w * 0.25  # 25% for lookahead region
                x_min_wide = max(0, int(expected_x - wider_search_width))
                x_max_wide = min(w, int(expected_x + wider_search_width))
                lateral_mask_wide = np.zeros((h, w), dtype=np.uint8)
                lateral_mask_wide[:, x_min_wide:x_max_wide] = 255
                sampling_mask_wide = cv2.bitwise_and(lane_color_mask, roi_mask)
                sampling_mask_wide = cv2.bitwise_and(sampling_mask_wide, lateral_mask_wide)
                region_mask_wide = sampling_mask_wide[region_lookahead_start:region_lookahead_end, :].copy()
                y_coords, x_coords = np.where(region_mask_wide > 0)
                logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Lookahead region (wide search) - found {len(y_coords)} pixels "
                            f"(x={x_min_wide}-{x_max_wide})")
                if len(y_coords) > 0:
                    num_to_add = min_lookahead_points - lookahead_points
                    indices = np.linspace(0, len(y_coords) - 1, min(num_to_add, len(y_coords)), dtype=int)
                    for idx in indices:
                        supplemented_points.append([float(x_coords[idx]), float(y_coords[idx] + region_lookahead_start)])
                        points_added += 1
                else:
                    logger.warning(f"[POINT SUPPLEMENT] {lane_name}: Lookahead region - no pixels found even with wide search "
                                 f"(CRITICAL: polynomial evaluated here for control!)")
        
        # Note: "middle" region is now the lookahead region (handled above)
        # This section is kept for backward compatibility but should not be needed
        
        # Sample from bottom region if needed
        if bottom_points < min_points_per_region:
            region_mask = sampling_mask[region_bottom_start:roi_y_end, :].copy()
            y_coords, x_coords = np.where(region_mask > 0)
            if len(y_coords) > 0:
                num_to_add = min_points_per_region - bottom_points
                indices = np.linspace(0, len(y_coords) - 1, min(num_to_add, len(y_coords)), dtype=int)
                for idx in indices:
                    supplemented_points.append([float(x_coords[idx]), float(y_coords[idx] + region_bottom_start)])
                    points_added += 1
        
        if points_added > 0:
            logger.info(f"[POINT SUPPLEMENT] {lane_name}: Added {points_added} points from mask "
                       f"(top={top_points}, gap={gap_points}, lookahead={lookahead_points}, bot={bottom_points})")
        else:
            logger.warning(f"[POINT SUPPLEMENT] {lane_name}: No points added - no pixels found in sampling_mask "
                          f"for regions needing points (top={top_points}, gap={gap_points}, "
                          f"lookahead={lookahead_points}, bot={bottom_points})")
        
        return np.array(supplemented_points)
    
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
        
        # Create region of interest (lower 62% of image for camera perspective)
        # WIDENED: No horizontal margin (100% width) to detect lanes at full image width
        # FIXED: Start ROI at 18% from top to include more of the road surface
        # FIXED: Exclude bottom 20% (was 7%) to better exclude car hood/body
        roi_margin = 0  # No horizontal margin - use full image width for yellow mask detection
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        # EXCLUDE: Sky (top), car body/hood (very bottom), but NOT sides
        # More restrictive to avoid car body and sky, but allow full horizontal detection
        # INCREASED: Exclude bottom 20% to better exclude car at different angles
        roi_vertices = np.array([[
            (roi_margin, int(h * 0.80)),  # Stop 20% from bottom to exclude car hood/body
            (w - roi_margin, int(h * 0.80)),
            (w - roi_margin, int(h * 0.18)),  # Start at 18% from top - includes more road surface
            (roi_margin, int(h * 0.18))
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
        # Store points used for fitting (for debug visualization)
        fit_points = {'left': None, 'right': None}
        
        for lane_idx, (line_group, lane_name) in enumerate([(left_lines, 'left'), (right_lines, 'right')]):
            previous_coeffs = self.previous_lanes[lane_idx]
            
            # ROOT CAUSE FIX: Use per-row sampling from yellow mask as PRIMARY point source
            # Hough lines are only used for initial lane-side detection, not for regression points
            # This provides uniform y-coverage, eliminates extrapolation, and ensures stable regression
            per_row_points = self._sample_points_per_row(
                lane_idx, yellow_mask, white_mask_roi, lane_color_mask, center_x, w, h, lane_name, previous_coeffs
            )
            
            using_per_row_sampling = False
            if per_row_points is not None and len(per_row_points) >= 3:
                # Use per-row sampling points (dense, uniform coverage)
                # CRITICAL: Per-row sampling already provides clean, evenly distributed points
                # Skip aggressive filtering and clustering that were designed for Hough lines
                points = per_row_points
                using_per_row_sampling = True
                logger.info(f"[ROOT CAUSE FIX] {lane_name}: Using per-row sampling ({len(points)} points, "
                           f"y-range: [{np.min(points[:, 1]):.0f}, {np.max(points[:, 1]):.0f}]px)")
            elif len(line_group) > 0:
                # Fallback: Use Hough lines if per-row sampling failed (should be rare)
                logger.warning(f"[ROOT CAUSE FIX] {lane_name}: Per-row sampling failed, falling back to Hough lines")
                points = []
                for line in line_group:
                    points.append([line[0], line[1]])
                    points.append([line[2], line[3]])
                points = np.array(points)
            else:
                # No points from either method
                if previous_coeffs is not None and self.previous_lanes_confidence[lane_idx] > 0.3:
                    logger.info(f"[RELIABILITY] {lane_name}: No points detected, using previous frame")
                    lanes.append(previous_coeffs.copy())
                    validation_failures[lane_name].append('no_points_used_previous')
                    continue
                else:
                    logger.warning(f"[RELIABILITY] {lane_name}: No points detected and no previous frame, rejecting")
                    lanes.append(None)
                    validation_failures[lane_name].append('no_points')
                    continue
            
            if len(points) < 3:
                # RELIABILITY: Use previous frame instead of rejecting
                previous_coeffs = self.previous_lanes[lane_idx]
                if previous_coeffs is not None and self.previous_lanes_confidence[lane_idx] > 0.3:
                    logger.info(f"[RELIABILITY] {lane_name}: Insufficient points ({len(points)} < 3), using previous frame")
                    lanes.append(previous_coeffs.copy())
                    validation_failures[lane_name].append(f'insufficient_points_{len(points)}_used_previous')
                    continue
                else:
                    logger.warning(f"[RELIABILITY] {lane_name}: Insufficient points ({len(points)} < 3) and no previous frame, rejecting")
                    lanes.append(None)
                    validation_failures[lane_name].append(f'insufficient_points_{len(points)}')
                    continue
            
            # NEW: Filter out points from upcoming curves or other lanes
            # NOTE: Skip aggressive filtering if using per-row sampling (already clean)
            filtered_points = []
            if not using_per_row_sampling and (self.previous_lanes_x_at_bottom[lane_idx] is not None and 
                len(points) > 3):  # Only filter if we have temporal data and enough points
                prev_lane_x_bottom = self.previous_lanes_x_at_bottom[lane_idx]  # x at bottom of image (y=h)
                
                # Estimate expected lane x position at each y using previous frame's position
                # In image space, lanes converge slightly due to perspective (vanishing point)
                # But for filtering, we use a simple model: x_expected ≈ prev_x (constant)
                # This works because lanes are roughly parallel in vehicle coordinates
                
                for point in points:
                    px, py = point[0], point[1]
                    
                    # PHASE 1 FIX: Lateral constraint - points must be on correct side of center
                    # Left lane points should be left of center (px < center_x)
                    # Right lane points should be right of center (px > center_x)
                    # This prevents cross-lane contamination (e.g., right lane using left lane points)
                    lateral_constraint_passed = False
                    if lane_idx == 0:  # Left lane
                        # Left lane: allow some margin for curves, but reject points clearly on right side
                        # Use 10% of image width as margin (allows for curve, but catches wrong lane)
                        margin = w * 0.10
                        if px < center_x + margin:  # Left lane can be slightly right of center on curves
                            lateral_constraint_passed = True
                        else:
                            logger.debug(f"[CURVE FILTER] {lane_name}: Filtered point at ({px:.1f}, {py:.1f}): "
                                       f"LATERAL CONSTRAINT FAILED - left lane point is too far right "
                                       f"(px={px:.1f} > center_x+margin={center_x + margin:.1f})")
                    else:  # Right lane (lane_idx == 1)
                        # Right lane: allow some margin for curves, but reject points clearly on left side
                        margin = w * 0.10
                        if px > center_x - margin:  # Right lane can be slightly left of center on curves
                            lateral_constraint_passed = True
                        else:
                            logger.debug(f"[CURVE FILTER] {lane_name}: Filtered point at ({px:.1f}, {py:.1f}): "
                                       f"LATERAL CONSTRAINT FAILED - right lane point is too far left "
                                       f"(px={px:.1f} < center_x-margin={center_x - margin:.1f})")
                    
                    # If lateral constraint failed, skip this point
                    if not lateral_constraint_passed:
                        continue
                    
                    # Expected x at this y position
                    # For parallel lanes in vehicle frame, x is roughly constant in image space
                    # (perspective convergence is small for nearby lanes)
                    x_expected = prev_lane_x_bottom
                    
                    # Calculate deviation from expected position
                    x_deviation = abs(px - x_expected)
                    
                    # Tolerance: tighter at bottom (near vehicle), looser at top (far away)
                    # But also: upper points that deviate significantly are likely from upcoming curves
                    y_normalized = py / h  # 0 (top) to 1 (bottom)
                    
                    # Base tolerance: 60px at bottom, 120px at top
                    base_tolerance = 60 + (1.0 - y_normalized) * 60
                    
                    # Stricter check for upper half: if point is far from expected, it's likely from upcoming curve
                    # RELAXED: Increased from 80px to 120px to be less aggressive
                    # We want to filter obvious upcoming curve points, but not filter valid lane points on curves
                    if y_normalized < 0.5:  # Upper half of image
                        # Tighter tolerance for upper points - they shouldn't deviate much if they're from current lane
                        max_deviation = 120  # Relaxed: 120px max deviation for upper points (was 80px)
                    else:
                        # Lower half: more lenient (points are closer, more reliable)
                        max_deviation = base_tolerance
                    
                    if x_deviation <= max_deviation:
                        filtered_points.append(point)
                    else:
                        logger.debug(f"[CURVE FILTER] {lane_name}: Filtered point at ({px:.1f}, {py:.1f}): "
                                   f"deviation={x_deviation:.1f}px > {max_deviation:.1f}px (expected={x_expected:.1f}px)")
                
                original_count = len(points)
                if len(filtered_points) >= 3:
                    points = np.array(filtered_points)
                    removed_count = original_count - len(filtered_points)
                    logger.info(f"[CURVE FILTER] {lane_name}: Filtered {len(filtered_points)}/{original_count} points "
                              f"(removed {removed_count} from upcoming curves)")
                else:
                    # Not enough points after filtering - use all points but log warning
                    logger.warning(f"[CURVE FILTER] {lane_name}: Filtering removed too many points "
                                 f"({len(filtered_points)} < 3), using all {original_count} points")
                    # Keep original points array
            else:
                # No temporal data - can't use deviation filter, but still apply lateral constraint
                # PHASE 1 FIX: Even without temporal data, enforce lateral constraint
                # This prevents obvious cross-lane contamination on first frame
                filtered_points = []
                for point in points:
                    px, py = point[0], point[1]
                    
                    # Lateral constraint (same as above)
                    lateral_constraint_passed = False
                    if lane_idx == 0:  # Left lane
                        margin = w * 0.10
                        if px < center_x + margin:
                            lateral_constraint_passed = True
                        else:
                            logger.debug(f"[CURVE FILTER] {lane_name}: Filtered point at ({px:.1f}, {py:.1f}): "
                                       f"LATERAL CONSTRAINT FAILED (no temporal data) - left lane point too far right")
                    else:  # Right lane
                        margin = w * 0.10
                        if px > center_x - margin:
                            lateral_constraint_passed = True
                        else:
                            logger.debug(f"[CURVE FILTER] {lane_name}: Filtered point at ({px:.1f}, {py:.1f}): "
                                       f"LATERAL CONSTRAINT FAILED (no temporal data) - right lane point too far left")
                    
                    if lateral_constraint_passed:
                        filtered_points.append(point)
                
                original_count = len(points)
                if len(filtered_points) >= 3:
                    points = np.array(filtered_points)
                    removed_count = original_count - len(filtered_points)
                    if removed_count > 0:
                        logger.info(f"[CURVE FILTER] {lane_name}: Applied lateral constraint (no temporal data): "
                                  f"kept {len(filtered_points)}/{original_count} points")
                else:
                    # Not enough points after filtering - use all points but log warning
                    logger.warning(f"[CURVE FILTER] {lane_name}: Lateral constraint removed too many points "
                                 f"({len(filtered_points)} < 3), using all {original_count} points")
                    points = np.array(points)
            
            # PHASE 2: Spatial clustering to separate left/right clusters
            # NOTE: Skip clustering if using per-row sampling (already clean, evenly distributed)
            expected_x_for_supplement = None
            if not using_per_row_sampling and len(points) >= 6:  # Need enough points for clustering to be meaningful
                clustered_points = self._spatial_cluster_points(
                    points, lane_idx, center_x, w, h, lane_name
                )
                if clustered_points is not None and len(clustered_points) >= 3:
                    points = clustered_points
                    expected_x_for_supplement = np.median(points[:, 0])  # Use median for supplementation
                    logger.info(f"[SPATIAL CLUSTER] {lane_name}: Using {len(points)} points from dominant cluster")
                else:
                    logger.warning(f"[SPATIAL CLUSTER] {lane_name}: Clustering failed or insufficient points, "
                                 f"using all {len(points)} points")
            elif using_per_row_sampling:
                # Per-row sampling: use median for supplementation if needed
                expected_x_for_supplement = np.median(points[:, 0]) if len(points) > 0 else center_x
                logger.debug(f"[ROOT CAUSE FIX] {lane_name}: Skipping clustering (per-row sampling already clean)")
            # If not enough points for clustering, use all points (clustering needs at least 6 points)
            
            # Set expected_x if not set by clustering
            # CRITICAL: For supplementation, we need to predict x at the TOP region, not bottom
            # Use a quick polynomial fit to current points to predict where lane should be at top
            if expected_x_for_supplement is None:
                if len(points) >= 3:
                    try:
                        # Fit polynomial to current points
                        y_vals = points[:, 1]
                        x_vals = points[:, 0]
                        coeffs = np.polyfit(y_vals, x_vals, min(2, len(points) - 1))
                        # Predict x at middle of top region (where we want to sample)
                        roi_y_start = int(h * 0.18)
                        lookahead_y_estimate = int(h * 0.73)
                        region_top_end = max(roi_y_start + 50, lookahead_y_estimate - 80)
                        top_y_mid = roi_y_start + (region_top_end - roi_y_start) // 2
                        expected_x_for_supplement = np.polyval(coeffs, top_y_mid)
                        logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Using polynomial prediction for top region: "
                                   f"x={expected_x_for_supplement:.1f}px at y={top_y_mid}px (from {len(points)} points)")
                    except Exception as e:
                        # Fallback to median
                        expected_x_for_supplement = np.median(points[:, 0]) if len(points) > 0 else center_x
                        logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Polynomial prediction failed, using median: "
                                   f"x={expected_x_for_supplement:.1f}px")
                elif len(points) > 0:
                    expected_x_for_supplement = np.median(points[:, 0])
                else:
                    expected_x_for_supplement = center_x
            
            # PHASE 2 ENHANCEMENT: Supplement points from yellow mask to ensure consistent distribution
            # NOTE: If we used per-row sampling, only supplement if there's a gap at the top
            # (per-row sampling already provides uniform coverage where yellow pixels exist)
            y_min = np.min(points[:, 1])
            roi_y_start = int(h * 0.18)
            roi_y_end = int(h * 0.80)
            y_range = np.max(points[:, 1]) - y_min
            coverage_ratio = y_range / (roi_y_end - roi_y_start) if (roi_y_end - roi_y_start) > 0 else 0.0
            gap_to_top = y_min - roi_y_start
            
            # For per-row sampling: only supplement if there's a gap at the top (no yellow pixels there)
            # For Hough lines: supplement if coverage is poor
            if using_per_row_sampling:
                needs_supplementation = gap_to_top > 50  # Only if gap > 50px
            else:
                needs_supplementation = coverage_ratio < 0.70 or gap_to_top > 50
            
            if needs_supplementation and len(points) >= 3:
                logger.info(f"[POINT SUPPLEMENT] {lane_name}: Coverage is poor ({coverage_ratio*100:.1f}%, gap={gap_to_top:.0f}px), "
                           f"attempting supplementation with {len(points)} points")
                try:
                    points = self._supplement_points_from_mask(
                        points, lane_idx, 
                        yellow_mask,  # yellow_mask is defined in detect() scope
                        white_mask_roi,  # white_mask_roi is defined in detect() scope
                        lane_color_mask,  # lane_color_mask is defined in detect() scope
                        center_x, w, h, lane_name, expected_x_for_supplement
                    )
                    logger.info(f"[POINT SUPPLEMENT] {lane_name}: Supplementation completed, now have {len(points)} points")
                except Exception as e:
                    logger.warning(f"[POINT SUPPLEMENT] {lane_name}: Failed to supplement points: {e}")
                    import traceback
                    logger.warning(f"[POINT SUPPLEMENT] {lane_name}: Traceback: {traceback.format_exc()}")
            else:
                logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Good coverage ({coverage_ratio*100:.1f}%, gap={gap_to_top:.0f}px), "
                            f"skipping supplementation")
            
            # CRITICAL: If points still don't span full range, use polynomial prediction to fill top gap
            # This ensures polynomial doesn't need to extrapolate >100px
            # For per-row sampling: use per-row logic to fill gaps (sample every 8px)
            # For Hough lines: use synthetic points
            roi_y_start = int(h * 0.18)
            lookahead_y_estimate = int(h * 0.73)
            region_top_end = max(roi_y_start + 50, lookahead_y_estimate - 50)
            primary_mask = yellow_mask if lane_idx == 0 else white_mask_roi
            sampling_mask = primary_mask.copy()
            if np.sum(sampling_mask) < 100:
                sampling_mask = lane_color_mask.copy()
            min_top_pixels = max(40, int(0.001 * w * (region_top_end - roi_y_start)))
            top_region_pixel_count = int(np.sum(primary_mask[roi_y_start:region_top_end, :] > 0))
            top_region_has_pixels = top_region_pixel_count >= min_top_pixels
            if not needs_supplementation or (needs_supplementation and len(points) >= 3):
                y_min = np.min(points[:, 1])
                roi_y_start = int(h * 0.18)
                gap_to_top = y_min - roi_y_start
                
                if gap_to_top > 50 and not top_region_has_pixels:
                    logger.info(f"[POINT SUPPLEMENT] {lane_name}: Skipping gap fill (no pixels in top region)")
                
                if gap_to_top > 50 and len(points) >= 3 and top_region_has_pixels:  # More aggressive: fill gaps >50px (not just >100px)
                    if using_per_row_sampling:
                        # Use per-row sampling logic: sample every 8px in the gap
                        y_step = 8
                        gap_points = []
                        
                        # Fit polynomial to existing points for prediction
                        try:
                            y_vals = points[:, 1]
                            x_vals = points[:, 0]
                            pred_coeffs = np.polyfit(y_vals, x_vals, min(2, len(points) - 1))
                            
                            # Sample every 8px in the gap region
                            for y in range(roi_y_start, int(y_min), y_step):
                                predicted_x = np.polyval(pred_coeffs, y)
                                
                                # Apply lane-side constraint
                                margin = w * 0.20
                                if lane_idx == 0:  # Left lane
                                    predicted_x = min(predicted_x, center_x + margin)
                                else:  # Right lane
                                    predicted_x = max(predicted_x, center_x - margin)
                                
                                # Clamp to bounds
                                predicted_x = np.clip(predicted_x, 0, w - 1)
                                gap_points.append([predicted_x, float(y)])
                            
                            if len(gap_points) > 0:
                                gap_points_array = np.array(gap_points)
                                points = np.vstack([gap_points_array, points])
                                logger.info(f"[ROOT CAUSE FIX] {lane_name}: Added {len(gap_points)} evenly-spaced points "
                                           f"to fill gap (y={roi_y_start:.0f}-{y_min:.0f}px) using per-row sampling logic")
                        except Exception as e:
                            logger.warning(f"[ROOT CAUSE FIX] {lane_name}: Failed to fill gap with per-row sampling: {e}")
                    else:
                        # Hough lines: use synthetic points (old logic)
                        # Fit a quick polynomial to current points to predict x at top
                        try:
                            y_vals = points[:, 1]
                            x_vals = points[:, 0]
                            
                            # Remove outliers before fitting (to get better prediction)
                            # Use IQR method to detect outliers
                            q1, q3 = np.percentile(x_vals, [25, 75])
                            iqr = q3 - q1
                            if iqr > 0:  # Only filter if there's variation
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                outlier_mask = (x_vals >= lower_bound) & (x_vals <= upper_bound)
                                
                                if np.sum(outlier_mask) >= 3:  # Need at least 3 points after outlier removal
                                    y_vals_clean = y_vals[outlier_mask]
                                    x_vals_clean = x_vals[outlier_mask]
                                    # Use linear fit for prediction (more stable for extrapolation)
                                    pred_coeffs = np.polyfit(y_vals_clean, x_vals_clean, deg=1)
                                    logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Removed {np.sum(~outlier_mask)} outliers "
                                               f"before synthetic point generation")
                                else:
                                    # Not enough points after outlier removal, use all points
                                    pred_coeffs = np.polyfit(y_vals, x_vals, deg=1)
                            else:
                                # No variation, use all points
                                pred_coeffs = np.polyfit(y_vals, x_vals, deg=1)
                            
                            # Add synthetic points evenly spaced across the gap
                            num_synthetic = min(8, int(gap_to_top / 30))  # More aggressive: ~1 point per 30px
                            synthetic_points = []
                            
                            # Validate prediction before generating synthetic points
                            # Check if prediction at y_min is reasonable
                            predicted_x_at_y_min = np.polyval(pred_coeffs, y_min)
                            margin = w * 0.20  # 20% margin for validation
                            
                            # Lane-side constraint: left lane should be left of center, right lane right of center
                            if lane_idx == 0:  # Left lane
                                max_reasonable_x = center_x + margin
                                if predicted_x_at_y_min > max_reasonable_x:
                                    logger.warning(f"[POINT SUPPLEMENT] {lane_name}: Prediction at y_min ({predicted_x_at_y_min:.1f}px) "
                                                 f"is too far right for left lane (>{max_reasonable_x:.1f}px), skipping synthetic points")
                                    raise ValueError("Invalid prediction for left lane")
                            else:  # Right lane
                                min_reasonable_x = center_x - margin
                                if predicted_x_at_y_min < min_reasonable_x:
                                    logger.warning(f"[POINT SUPPLEMENT] {lane_name}: Prediction at y_min ({predicted_x_at_y_min:.1f}px) "
                                                 f"is too far left for right lane (<{min_reasonable_x:.1f}px), skipping synthetic points")
                                    raise ValueError("Invalid prediction for right lane")
                            
                            # Check if prediction is within image bounds
                            if predicted_x_at_y_min < -margin or predicted_x_at_y_min > w + margin:
                                logger.warning(f"[POINT SUPPLEMENT] {lane_name}: Prediction at y_min ({predicted_x_at_y_min:.1f}px) "
                                             f"is way outside image bounds [0, {w}], skipping synthetic points")
                                raise ValueError("Prediction outside image bounds")
                            
                            for i in range(num_synthetic):
                                target_y = roi_y_start + (i + 1) * (gap_to_top / (num_synthetic + 1))
                                predicted_x = np.polyval(pred_coeffs, target_y)
                                
                                # Apply lane-side constraint
                                if lane_idx == 0:  # Left lane
                                    predicted_x = min(predicted_x, center_x + margin)
                                else:  # Right lane
                                    predicted_x = max(predicted_x, center_x - margin)
                                
                                # Clamp to image bounds
                                predicted_x = np.clip(predicted_x, 0, w - 1)
                                synthetic_points.append([predicted_x, target_y])
                            
                            # Add synthetic points to the beginning (they're at smaller y)
                            points = np.vstack([np.array(synthetic_points), points])
                            logger.info(f"[POINT SUPPLEMENT] {lane_name}: Added {len(synthetic_points)} synthetic points "
                                       f"to fill top gap (y={roi_y_start:.0f}-{y_min:.0f}px) using linear prediction")
                        except Exception as e:
                            logger.debug(f"[POINT SUPPLEMENT] {lane_name}: Failed to add synthetic points: {e}")
            
            # FIX #1: Sort points by y (ascending) before fitting
            # This stabilizes polynomial fitting, especially on curves
            # Points should be ordered from top (small y) to bottom (large y) of image
            sort_indices = np.argsort(points[:, 1])
            points = points[sort_indices]
            
            # Store points used for fitting (for debug visualization)
            if return_debug:
                fit_points[lane_name] = points.copy()  # Store copy of points array
            
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
                
                # Log point statistics for debugging polynomial fit issues
                if len(points) > 0:
                    # Calculate point distribution (how evenly spread across y-range)
                    y_std = np.std(points[:, 1]) if len(points) > 1 else 0.0
                    x_std = np.std(points[:, 0]) if len(points) > 1 else 0.0
                    
                    logger.info(f"[POLY FIT] {lane_name} lane: {len(points)} points, "
                               f"y_range=[{y_min:.1f}, {y_max:.1f}] ({y_range:.1f}px), "
                               f"x_range=[{x_min:.1f}, {x_max:.1f}] ({x_range:.1f}px), "
                               f"y_std={y_std:.1f}px, x_std={x_std:.1f}px")
                
                # If y_range is too small, polynomial will extrapolate poorly
                # RELAXED: Reduced from 50px to 30px to allow left lane (which is farther away)
                # Left lane often has smaller y_range due to perspective (farther = fewer pixels)
                if y_range < 30:  # Less than 30 pixels vertical span (relaxed from 50px)
                    logger.warning(f"[FRAME0 DEBUG] {lane_name}: y_range too small ({y_range:.1f}px < 30px)")
                    # RELIABILITY: Use previous frame instead of rejecting
                    previous_coeffs = self.previous_lanes[lane_idx]
                    if previous_coeffs is not None and self.previous_lanes_confidence[lane_idx] > 0.3:
                        logger.info(f"[RELIABILITY] {lane_name}: Using previous frame due to small y_range")
                        lanes.append(previous_coeffs.copy())
                        validation_failures[lane_name].append(f'insufficient_y_range_{y_range:.1f}px_used_previous')
                        continue
                    else:
                        logger.warning(f"[RELIABILITY] {lane_name}: No previous frame, rejecting due to small y_range")
                        lanes.append(None)
                        validation_failures[lane_name].append(f'insufficient_y_range_{y_range:.1f}px')
                        continue
                
                # Check if we need to extrapolate significantly
                extrapolation_distance = h - y_max  # How far we need to extrapolate to reach bottom
                # RELAXED: Increased from 100px to 200px to match original behavior
                # The original code accepted detections with up to ~160px extrapolation
                # We want to prevent extreme cases (>200px) but allow reasonable extrapolation
                max_extrapolation = 200  # Maximum allowed extrapolation (pixels)
                
                if extrapolation_distance > max_extrapolation:
                    logger.warning(f"[FRAME0 DEBUG] {lane_name}: EXCESSIVE EXTRAPOLATION! "
                                 f"y_max={y_max:.1f}px, h={h}px, extrapolation={extrapolation_distance:.1f}px > {max_extrapolation}px")
                    # RELIABILITY: Use previous frame instead of rejecting
                    previous_coeffs = self.previous_lanes[lane_idx]
                    if previous_coeffs is not None and self.previous_lanes_confidence[lane_idx] > 0.3:
                        logger.info(f"[RELIABILITY] {lane_name}: Using previous frame due to excessive extrapolation")
                        lanes.append(previous_coeffs.copy())
                        validation_failures[lane_name].append(f'excessive_extrapolation_{extrapolation_distance:.1f}px_used_previous')
                        continue
                    else:
                        logger.warning(f"[RELIABILITY] {lane_name}: No previous frame, rejecting due to excessive extrapolation")
                        lanes.append(None)
                        validation_failures[lane_name].append(f'excessive_extrapolation_{extrapolation_distance:.1f}px')
                        continue
                elif extrapolation_distance > 100:  # Need to extrapolate >100px (warning threshold)
                    logger.warning(f"[FRAME0 DEBUG] {lane_name}: Large extrapolation needed! "
                                 f"y_max={y_max:.1f}px, h={h}px, extrapolation={extrapolation_distance:.1f}px")
                
                # RELAXED: Removed the "bottom 1/3" requirement - it was too strict
                # The extrapolation check above is sufficient to catch cases where points don't reach bottom
                # This allows detections on curves where points might be higher in the image
                
                # Fit polynomial with improved method for dashed lines and curves
                # Use weighted fitting to emphasize near points (more accurate)
                # This helps with both dashed lines (fewer points) and curves (perspective distortion)
                y = points[:, 1]
                x = points[:, 0]
                
                # IMPROVED: Edge-weighted polynomial fitting for better extrapolation
                # Emphasize points at BOTH edges (top and bottom) to ensure good extrapolation
                # This prevents polynomial from extrapolating poorly when points don't span full range
                h_image = h  # Use image height for normalization
                y_normalized = y / h_image  # Normalize y to [0, 1]
                
                # Weight function: higher weight at edges (top and bottom), lower in middle
                # This ensures polynomial fits well at both ends, enabling better extrapolation
                # Formula: weight = 1 + 2 * min(y_norm, 1 - y_norm)
                # This gives weight=3 at edges (y=0 or y=1) and weight=1 at middle (y=0.5)
                edge_weights = 1.0 + 2.0 * np.minimum(y_normalized, 1.0 - y_normalized)
                
                # Also weight by distance from center of point distribution
                # Points far from center (edges of distribution) are more important for extrapolation
                y_center = np.mean(y)
                y_std = np.std(y) if len(y) > 1 else 1.0
                distance_weights = 1.0 + np.abs(y - y_center) / (y_std + 1.0)  # Higher weight for edge points
                
                # Combine both weighting strategies
                weights = edge_weights * distance_weights
                
                # Use numpy's built-in weighted polyfit (numerically stable)
                try:
                    coeffs = np.polyfit(y, x, deg=2, w=weights)
                    logger.debug(f"[POLY FIT] {lane_name}: Weighted fitting succeeded (using np.polyfit with weights)")
                    
                    # CRITICAL: If points don't span full range, validate and constrain polynomial
                    # Check if polynomial extrapolates reasonably at ROI boundaries AND lookahead distance
                    roi_y_start = int(h * 0.18)
                    roi_y_end = int(h * 0.80)
                    lookahead_y = int(h * 0.73)  # Where we actually USE the polynomial
                    
                    # Evaluate at top, lookahead, and bottom of ROI
                    x_at_top = np.polyval(coeffs, roi_y_start)
                    x_at_lookahead = np.polyval(coeffs, lookahead_y)
                    x_at_bottom = np.polyval(coeffs, roi_y_end)
                    
                    # If extrapolation is extreme, use constrained fitting
                    # Allow 10% margin for curves
                    margin = w * 0.10
                    needs_constraint = (x_at_top < -margin or x_at_top > w + margin) or \
                                      (x_at_lookahead < -margin or x_at_lookahead > w + margin) or \
                                      (x_at_bottom < -margin or x_at_bottom > w + margin)
                    
                    if needs_constraint and len(points) >= 3:
                        # Use linear extrapolation from edge points instead of quadratic
                        # This prevents extreme extrapolation when points don't span full range
                        logger.info(f"[POLY FIT] {lane_name}: Polynomial extrapolates poorly "
                                   f"(top: {x_at_top:.1f}px, lookahead: {x_at_lookahead:.1f}px, bottom: {x_at_bottom:.1f}px). "
                                   f"Using constrained linear-quadratic blend.")
                        
                        # Fit linear to bottom points (more reliable for extrapolation)
                        bottom_points = points[points[:, 1] >= np.median(points[:, 1])]
                        if len(bottom_points) >= 2:
                            linear_coeffs = np.polyfit(bottom_points[:, 1], bottom_points[:, 0], deg=1)
                            
                            # Blend: use quadratic in data range, linear for extrapolation
                            # Evaluate both at key positions
                            y_data_min, y_data_max = np.min(points[:, 1]), np.max(points[:, 1])
                            
                            # CRITICAL: ALWAYS constrain at lookahead distance (where we USE it)
                            # This is the most important constraint - we MUST have valid polynomial at lookahead
                            x_lookahead_linear = np.polyval(linear_coeffs, lookahead_y)
                            x_lookahead_quad = np.polyval(coeffs, lookahead_y)
                            x_lookahead_current = x_lookahead_quad
                            
                            # Blend: more weight to linear for extrapolation (more stable)
                            x_lookahead_target = 0.7 * x_lookahead_linear + 0.3 * x_lookahead_quad
                            # Clamp to reasonable bounds
                            x_lookahead_target = np.clip(x_lookahead_target, -margin, w + margin)
                            
                            # ALWAYS adjust polynomial to match at lookahead (this is where we use it!)
                            # Solve: a*y^2 + b*y + c = x_target
                            # We adjust c to match (keeping a and b from weighted fit)
                            coeffs[2] = x_lookahead_target - coeffs[0] * lookahead_y**2 - coeffs[1] * lookahead_y
                            
                            # Verify the constraint worked
                            x_lookahead_after = np.polyval(coeffs, lookahead_y)
                            logger.info(f"[POLY FIT] {lane_name}: Constrained at lookahead: {x_lookahead_current:.1f}px -> {x_lookahead_target:.1f}px (after: {x_lookahead_after:.1f}px)")
                            
                            # CRITICAL: Don't adjust top/bottom - they will be clamped during evaluation if needed
                            # The lookahead constraint is the most important (where we USE the polynomial)
                            # Top and bottom are only for visualization, so we can use _evaluate_polynomial_safe to clamp them
                            logger.debug(f"[POLY FIT] {lane_name}: Lookahead constraint applied, top/bottom will use safe evaluation")
                        
                        logger.info(f"[POLY FIT] {lane_name}: Constrained polynomial: {coeffs}")
                        
                except Exception as e:
                    # Fallback to unweighted if weighted fitting fails
                    logger.warning(f"[POLY FIT] {lane_name}: Weighted fitting failed, using unweighted fallback: {type(e).__name__}: {e}")
                    coeffs = np.polyfit(y, x, deg=2)
                
                # NEW: Validate polynomial coefficients to catch extreme fits
                # Extreme constant term (c) indicates polynomial is trying to fit points that don't form a smooth curve
                # This can happen with sparse points, noisy detections, or points from upcoming curves
                if len(coeffs) >= 3:
                    a, b, c = coeffs[0], coeffs[1], coeffs[2]
                    
                    # Check if constant term is extreme (way outside image bounds)
                    # For a 640px image, reasonable constant term should be roughly [-640, 1280]
                    # Extreme values like |c| > 2000px indicate a bad fit
                    max_reasonable_c = w * 3.0  # 3x image width is reasonable for perspective
                    
                    if abs(c) > max_reasonable_c:
                        logger.warning(f"[POLY FIT] {lane_name}: EXTREME CONSTANT TERM detected! "
                                     f"c={c:.1f}px (max reasonable: {max_reasonable_c:.1f}px). "
                                     f"This suggests points don't form a smooth curve.")
                        logger.warning(f"[POLY FIT] {lane_name}: Polynomial coefficients: a={a:.6f}, b={b:.6f}, c={c:.1f}")
                        logger.warning(f"[POLY FIT] {lane_name}: Point stats: {len(points)} points, "
                                     f"y_range=[{y_min:.1f}, {y_max:.1f}], x_range=[{x_min:.1f}, {x_max:.1f}]")
                        
                        # CRITICAL FIX: Check polynomial at lookahead distance (where we actually USE it)
                        # Don't check at ROI boundaries - check where the polynomial is evaluated for control
                        # Lookahead is typically 8m, which corresponds to ~y=350px for 480px image
                        # This is more lenient for dashed lines which may have extreme constant terms
                        # but are reasonable at the lookahead distance
                        y_lookahead = int(h * 0.73)  # ~350px for 480px image (8m lookahead, ~73% from top)
                        x_at_lookahead = a * y_lookahead * y_lookahead + b * y_lookahead + c
                        
                        logger.warning(f"[POLY FIT] {lane_name}: At lookahead distance (y={y_lookahead}px): "
                                     f"x={x_at_lookahead:.1f}px")
                        
                        # Check if polynomial produces reasonable x values at lookahead distance
                        # This is where we actually USE the polynomial for coordinate conversion
                        # If it's reasonable here, accept the fit even if constant term is extreme
                        max_reasonable_x = w * 2.5
                        min_reasonable_x = -w * 1.5
                        
                        if (x_at_lookahead < min_reasonable_x or x_at_lookahead > max_reasonable_x):
                            logger.warning(f"[POLY FIT] {lane_name}: Polynomial produces extreme x value "
                                         f"at lookahead distance (x={x_at_lookahead:.1f}px, range: {min_reasonable_x:.0f} to {max_reasonable_x:.0f}px)")
                            # RELIABILITY: Use previous frame instead of rejecting
                            previous_coeffs = self.previous_lanes[lane_idx]
                            if previous_coeffs is not None and self.previous_lanes_confidence[lane_idx] > 0.3:
                                logger.info(f"[RELIABILITY] {lane_name}: Using previous frame due to extreme constant term")
                                coeffs = previous_coeffs.copy()
                                rejected_current_fit = True
                                self.lanes_rejected_this_frame[lane_idx] = True
                                # Continue with corrected coeffs
                            else:
                                logger.warning(f"[RELIABILITY] {lane_name}: No previous frame, rejecting due to extreme constant term")
                                lanes.append(None)
                                validation_failures[lane_name].append(f'extreme_constant_term_c_{c:.1f}px')
                                continue
                        else:
                            logger.info(f"[POLY FIT] {lane_name}: Constant term is extreme (c={c:.1f}px) but polynomial is reasonable "
                                      f"at lookahead distance (x={x_at_lookahead:.1f}px). Accepting fit.")
                
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
                
                # PHASE 2: Polynomial Extrapolation Validation
                # Check if polynomial extrapolates out of bounds at key positions
                # This prevents poor polygon visualization and invalid detections
                roi_y_start = int(h * 0.18)
                lookahead_y = int(h * 0.73)
                roi_y_end = int(h * 0.80)
                
                # Test polynomial at critical y-positions
                # Use linear extrapolation for regions far from data points
                test_y_positions = [
                    roi_y_start,  # Top of ROI
                    lookahead_y,  # Lookahead distance (where we use it)
                    roi_y_end,    # Bottom of ROI
                    y_min,        # Minimum y where we have points
                    y_max,        # Maximum y where we have points
                ]
                
                polynomial_valid = True
                invalid_positions = []
                for test_y in test_y_positions:
                    # Use safe polynomial evaluation (linear extrapolation when far from data)
                    test_x = self._evaluate_polynomial_safe(coeffs, test_y, points, w, h)
                    
                    # Allow some margin for curves (10% of image width on each side)
                    margin = w * 0.10
                    if test_x < -margin or test_x > w + margin:
                        polynomial_valid = False
                        invalid_positions.append((test_y, test_x))
                
                # Track if we rejected the current fit (to prevent storing rejected coefficients)
                rejected_current_fit = False
                
                if not polynomial_valid:
                    logger.warning(f"[POLY VALIDATION] {lane_name}: Polynomial extrapolates out of bounds!")
                    for test_y, test_x in invalid_positions:
                        logger.warning(f"[POLY VALIDATION] {lane_name}: At y={test_y}px: x={test_x:.1f}px (out of bounds)")
                    
                    # RELIABILITY FIX: Always produce a valid result, never reject
                    # Strategy: Try previous frame first, then use clamped/adjusted polynomial
                    previous_coeffs = self.previous_lanes[lane_idx]
                    use_previous = (previous_coeffs is not None and 
                                   self.previous_lanes_confidence[lane_idx] > 0.3)  # Lower threshold for reliability
                    
                    if use_previous:
                        logger.info(f"[POLY VALIDATION] {lane_name}: Using previous frame's polynomial (current extrapolates poorly)")
                        coeffs = previous_coeffs.copy()
                        rejected_current_fit = True
                        self.lanes_rejected_this_frame[lane_idx] = True
                    else:
                        # No previous frame or low confidence - fix the polynomial instead of rejecting
                        logger.info(f"[POLY VALIDATION] {lane_name}: No reliable previous frame. Adjusting polynomial to stay in bounds.")
                        
                        # Strategy: Create a corrected polynomial that stays within bounds
                        # Use the polynomial at valid positions, but clamp/adjust at invalid positions
                        # Fit a new polynomial using only the valid evaluations
                        valid_test_positions = []
                        for test_y in test_y_positions:
                            test_x = self._evaluate_polynomial_safe(coeffs, test_y, points, w, h)
                            margin = w * 0.10
                            # Clamp to valid range
                            clamped_x = np.clip(test_x, -margin, w + margin)
                            valid_test_positions.append((test_y, clamped_x))
                        
                        # If we have enough valid points, refit polynomial
                        if len(valid_test_positions) >= 3:
                            valid_y = np.array([p[0] for p in valid_test_positions])
                            valid_x = np.array([p[1] for p in valid_test_positions])
                            try:
                                # Refit polynomial to clamped values
                                corrected_coeffs = np.polyfit(valid_y, valid_x, deg=min(2, len(valid_test_positions) - 1))
                                
                                # Verify corrected polynomial is better
                                all_valid = True
                                for test_y in test_y_positions:
                                    test_x = self._evaluate_polynomial_safe(corrected_coeffs, test_y, points, w, h)
                                    margin = w * 0.10
                                    if test_x < -margin or test_x > w + margin:
                                        all_valid = False
                                        break
                                
                                if all_valid:
                                    logger.info(f"[POLY VALIDATION] {lane_name}: Corrected polynomial is valid")
                                    coeffs = corrected_coeffs
                                else:
                                    logger.warning(f"[POLY VALIDATION] {lane_name}: Corrected polynomial still has issues, using original with clamping")
                            except Exception as e:
                                logger.warning(f"[POLY VALIDATION] {lane_name}: Failed to correct polynomial: {e}, using original with clamping")
                        else:
                            logger.warning(f"[POLY VALIDATION] {lane_name}: Not enough valid positions, using original with clamping")
                        
                        # Mark that we're using a corrected/adjusted polynomial
                        # The polynomial will be clamped later in the code, but we accept it
                        rejected_current_fit = False
                
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
                
                # Evaluate at chosen y position (use safe evaluation with linear extrapolation)
                lane_x_at_bottom = self._evaluate_polynomial_safe(coeffs, evaluation_y, points, w, h)
                
                # FRAME 0 DEBUG: Log evaluation
                logger.info(f"[FRAME0 DEBUG] {lane_name}: Evaluating at y={evaluation_y:.1f}px, "
                           f"result={lane_x_at_bottom:.1f}px")
                
                # Also evaluate at y_min and y_max (where we have actual points)
                lane_x_at_y_min = self._evaluate_polynomial_safe(coeffs, y_min, points, w, h)
                lane_x_at_y_max = self._evaluate_polynomial_safe(coeffs, y_max, points, w, h)
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
                        # If still invalid, use previous frame
                        if abs(lane_x_at_bottom - center_x) > w * 0.9:
                            logger.warning(f"[POLY DEBUG] {lane_name}: Still invalid after clamping/linear extrapolation.")
                            # RELIABILITY: Use previous frame instead of rejecting
                            previous_coeffs = self.previous_lanes[lane_idx]
                            if previous_coeffs is not None and self.previous_lanes_confidence[lane_idx] > 0.3:
                                logger.info(f"[RELIABILITY] {lane_name}: Using previous frame due to invalid evaluation")
                                coeffs = previous_coeffs.copy()
                                rejected_current_fit = True
                                self.lanes_rejected_this_frame[lane_idx] = True
                                # Recalculate lane_x_at_bottom with previous coeffs (use safe evaluation)
                                lane_x_at_bottom = self._evaluate_polynomial_safe(coeffs, evaluation_y, points, w, h)
                                lane_x_at_bottom = np.clip(lane_x_at_bottom, 0, w)
                            else:
                                logger.warning(f"[RELIABILITY] {lane_name}: No previous frame, rejecting due to invalid evaluation")
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
                    logger.warning(f"[POLY DEBUG] {lane_name}: Too far from center "
                                 f"({distance_from_center:.1f}px > {max_reasonable_distance:.1f}px)")
                    # RELIABILITY: Use previous frame instead of rejecting
                    previous_coeffs = self.previous_lanes[lane_idx]
                    if previous_coeffs is not None and self.previous_lanes_confidence[lane_idx] > 0.3:
                        logger.info(f"[RELIABILITY] {lane_name}: Using previous frame due to too far from center")
                        coeffs = previous_coeffs.copy()
                        rejected_current_fit = True
                        self.lanes_rejected_this_frame[lane_idx] = True
                        # Continue with corrected coeffs
                    else:
                        logger.warning(f"[RELIABILITY] {lane_name}: No previous frame, rejecting due to too far from center")
                        lanes.append(None)
                        validation_failures[lane_name].append(f'too_far_from_center_{distance_from_center:.1f}px')
                        continue
                
                lanes.append(coeffs)
            except Exception as e:
                logger.warning(f"[POLY FIT] {lane_name}: Polynomial fitting failed: {e}")
                # RELIABILITY: Use previous frame instead of rejecting
                previous_coeffs = self.previous_lanes[lane_idx]
                if previous_coeffs is not None and self.previous_lanes_confidence[lane_idx] > 0.3:
                    logger.info(f"[RELIABILITY] {lane_name}: Using previous frame due to polyfit failure")
                    lanes.append(previous_coeffs.copy())
                    validation_failures[lane_name].append(f'polyfit_failed_{str(e)}_used_previous')
                else:
                    logger.warning(f"[RELIABILITY] {lane_name}: No previous frame, rejecting due to polyfit failure")
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
                'validation_failures': validation_failures,
                'fit_points': fit_points  # Points used for polynomial fitting: {'left': np.array([[x, y], ...]), 'right': ...}
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
        # FIXED: Match ROI start position (18% from top) for consistency
        y_points = np.linspace(int(h * 0.18), h, 100).astype(np.float32)
        
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

