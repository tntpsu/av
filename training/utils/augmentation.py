"""
Data augmentation utilities for training.
"""

import numpy as np
import cv2
from typing import Tuple


def augment_image(image: np.ndarray, brightness_range: Tuple[float, float] = (0.7, 1.3),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 flip_prob: float = 0.5) -> np.ndarray:
    """
    Apply data augmentation to image.
    
    Args:
        image: Input RGB image
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment
        flip_prob: Probability of horizontal flip
    
    Returns:
        Augmented image
    """
    aug_image = image.copy()
    
    # Brightness adjustment
    if np.random.rand() > 0.5:
        brightness_factor = np.random.uniform(*brightness_range)
        aug_image = (aug_image * brightness_factor).clip(0, 255).astype(np.uint8)
    
    # Contrast adjustment
    if np.random.rand() > 0.5:
        contrast_factor = np.random.uniform(*contrast_range)
        mean = aug_image.mean()
        aug_image = ((aug_image - mean) * contrast_factor + mean).clip(0, 255).astype(np.uint8)
    
    # Horizontal flip
    if np.random.rand() < flip_prob:
        aug_image = np.fliplr(aug_image)
    
    # Gaussian noise
    if np.random.rand() > 0.7:
        noise = np.random.normal(0, 10, aug_image.shape).astype(np.int16)
        aug_image = (aug_image.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
    
    return aug_image


def augment_lane_labels(lane_coeffs: np.ndarray, image_width: int, flip: bool = False) -> np.ndarray:
    """
    Augment lane labels when image is flipped.
    
    Args:
        lane_coeffs: Lane polynomial coefficients
        image_width: Image width
        flip: Whether image was flipped
    
    Returns:
        Augmented lane coefficients
    """
    if not flip:
        return lane_coeffs
    
    # Flip lane positions
    # For polynomial x = ay^2 + by + c, flipping means x' = W - x
    # So: x' = W - (ay^2 + by + c) = -ay^2 - by + (W - c)
    flipped_coeffs = lane_coeffs.copy()
    flipped_coeffs[0] = -flipped_coeffs[0]  # a term
    flipped_coeffs[1] = -flipped_coeffs[1]   # b term
    flipped_coeffs[2] = image_width - flipped_coeffs[2]  # c term
    
    return flipped_coeffs

