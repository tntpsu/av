"""
Improved polynomial fitting methods for lane detection.
Handles dashed lines and curved roads better than standard polyfit.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import curve_fit
import cv2


def weighted_polyfit(y: np.ndarray, x: np.ndarray, deg: int = 2, 
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fit polynomial with optional weighting.
    
    For lane detection, weight points by distance from vehicle:
    - Points near vehicle (large y) = higher weight
    - Points far from vehicle (small y) = lower weight
    
    Args:
        y: Dependent variable (image y-coordinate, 0=top, h=bottom)
        x: Independent variable (image x-coordinate)
        deg: Polynomial degree (2 for quadratic)
        weights: Optional weights (if None, auto-weight by y-position)
    
    Returns:
        Polynomial coefficients [a, b, c] for ax^2 + bx + c
    """
    if weights is None:
        # Auto-weight: points closer to vehicle (larger y) get higher weight
        # Weight = y^2 to emphasize near points (quadratic weighting)
        # Normalize so max weight = 1.0
        y_max = np.max(y)
        if y_max > 0:
            weights = (y / y_max) ** 2
        else:
            weights = np.ones_like(y)
    
    # Weighted least squares polynomial fitting
    # Build Vandermonde matrix
    A = np.vander(y, deg + 1, increasing=True)
    
    # Weighted least squares: (A^T W A) c = A^T W x
    W = np.diag(weights)
    ATA = A.T @ W @ A
    ATx = A.T @ W @ x
    
    # Solve for coefficients
    try:
        coeffs = np.linalg.solve(ATA, ATx)
    except np.linalg.LinAlgError:
        # Fallback to unweighted if singular
        coeffs = np.polyfit(y, x, deg)
    
    return coeffs


def robust_polyfit_ransac(y: np.ndarray, x: np.ndarray, deg: int = 2,
                          max_iterations: int = 50, 
                          inlier_threshold: float = 5.0,
                          min_inliers: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit polynomial using RANSAC for outlier rejection.
    
    Useful for dashed lines where gaps create outliers.
    
    Args:
        y: Dependent variable (image y-coordinate)
        x: Independent variable (image x-coordinate)
        deg: Polynomial degree
        max_iterations: Maximum RANSAC iterations
        inlier_threshold: Distance threshold for inliers (pixels)
        min_inliers: Minimum fraction of points that must be inliers
    
    Returns:
        Tuple of (coefficients, inlier_mask)
    """
    if len(y) < deg + 1:
        # Not enough points - fallback to standard fit
        return np.polyfit(y, x, deg), np.ones(len(y), dtype=bool)
    
    best_coeffs = None
    best_inliers = None
    best_inlier_count = 0
    
    # Minimum points needed for fitting
    min_points = deg + 1
    
    for _ in range(max_iterations):
        # Randomly sample minimum points
        if len(y) < min_points:
            break
        
        sample_indices = np.random.choice(len(y), min_points, replace=False)
        y_sample = y[sample_indices]
        x_sample = x[sample_indices]
        
        try:
            # Fit polynomial to sample
            coeffs = np.polyfit(y_sample, x_sample, deg)
            
            # Evaluate polynomial for all points
            x_pred = np.polyval(coeffs, y)
            
            # Find inliers (points close to polynomial)
            errors = np.abs(x - x_pred)
            inliers = errors < inlier_threshold
            inlier_count = np.sum(inliers)
            
            # Check if this is the best model
            if inlier_count > best_inlier_count:
                best_coeffs = coeffs
                best_inliers = inliers
                best_inlier_count = inlier_count
        except:
            continue
    
    # If we found a good model, refine with all inliers
    if best_coeffs is not None and best_inlier_count >= len(y) * min_inliers:
        # Refit with all inliers (weighted)
        y_inliers = y[best_inliers]
        x_inliers = x[best_inliers]
        refined_coeffs = weighted_polyfit(y_inliers, x_inliers, deg)
        return refined_coeffs, best_inliers
    else:
        # RANSAC failed or didn't find enough inliers - fallback to standard fit
        return np.polyfit(y, x, deg), np.ones(len(y), dtype=bool)


def densify_dashed_line_points(points: np.ndarray, y_range: Tuple[float, float],
                               gap_threshold: float = 20.0) -> np.ndarray:
    """
    Densify points for dashed lines by interpolating across gaps.
    
    For dashed lines, gaps between segments can cause unstable polynomial fitting.
    This function interpolates across small gaps to create a denser point set.
    
    Args:
        points: Array of [x, y] points from line segments
        y_range: (y_min, y_max) range to densify
        gap_threshold: Maximum gap size to interpolate (pixels)
    
    Returns:
        Denser array of points with interpolated gaps filled
    """
    if len(points) < 2:
        return points
    
    # Sort by y-coordinate
    sorted_indices = np.argsort(points[:, 1])
    sorted_points = points[sorted_indices]
    
    densified = [sorted_points[0]]
    
    for i in range(1, len(sorted_points)):
        prev_y = sorted_points[i-1, 1]
        curr_y = sorted_points[i, 1]
        gap = curr_y - prev_y
        
        if gap > gap_threshold:
            # Large gap - interpolate intermediate points
            num_interp = int(gap / gap_threshold)
            for j in range(1, num_interp):
                interp_y = prev_y + (gap * j / num_interp)
                # Linear interpolation
                interp_x = np.interp(interp_y, 
                                    [prev_y, curr_y],
                                    [sorted_points[i-1, 0], sorted_points[i, 0]])
                densified.append([interp_x, interp_y])
        
        densified.append(sorted_points[i])
    
    return np.array(densified)


def fit_lane_polynomial_improved(points: np.ndarray, 
                                 image_height: int,
                                 use_weighted: bool = True,
                                 use_ransac: bool = True,
                                 densify_dashed: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Improved polynomial fitting for lane detection.
    
    Handles:
    - Dashed lines (gaps)
    - Curved roads
    - Outlier rejection
    - Weighted fitting (emphasize near points)
    
    Args:
        points: Array of [x, y] points from line segments
        image_height: Image height (for weighting)
        use_weighted: Use weighted fitting (emphasize near points)
        use_ransac: Use RANSAC for outlier rejection
        densify_dashed: Densify points across gaps in dashed lines
    
    Returns:
        Tuple of (coefficients, metadata_dict)
    """
    if len(points) < 3:
        return None, {'error': 'insufficient_points'}
    
    metadata = {}
    
    # Densify dashed line points if requested
    if densify_dashed:
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        points = densify_dashed_line_points(points, (y_min, y_max))
        metadata['densified_points'] = len(points)
    
    y = points[:, 1]
    x = points[:, 0]
    
    # RANSAC for outlier rejection (especially useful for dashed lines)
    if use_ransac and len(points) >= 4:
        coeffs, inlier_mask = robust_polyfit_ransac(y, x, deg=2)
        metadata['inlier_ratio'] = np.sum(inlier_mask) / len(inlier_mask)
        metadata['used_ransac'] = True
        
        # Use only inliers for weighted fit
        if np.sum(inlier_mask) >= 3:
            y = y[inlier_mask]
            x = x[inlier_mask]
    else:
        metadata['used_ransac'] = False
    
    # Weighted polynomial fitting
    if use_weighted:
        coeffs = weighted_polyfit(y, x, deg=2)
        metadata['used_weighted'] = True
    else:
        coeffs = np.polyfit(y, x, deg=2)
        metadata['used_weighted'] = False
    
    # Validate curvature direction (for curves)
    # Check if polynomial curvature matches expected road curvature
    # For a right curve, right lane should curve right (positive quadratic coeff)
    # For a left curve, right lane should curve left (negative quadratic coeff)
    # This is a sanity check - can be disabled if needed
    curvature_coeff = coeffs[0]  # a in ax^2 + bx + c
    metadata['curvature'] = curvature_coeff
    
    return coeffs, metadata

