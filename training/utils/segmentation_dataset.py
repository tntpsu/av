"""
Utilities for preparing lane segmentation datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class MaskStats:
    total_lane_pixels: int
    left_lane_pixels: int
    right_lane_pixels: int


def build_label_mask(yellow_mask: np.ndarray, white_mask: np.ndarray) -> np.ndarray:
    """
    Build a single-channel label mask.

    0 = background, 1 = left lane (yellow), 2 = right lane (white).
    """
    if yellow_mask.shape != white_mask.shape:
        raise ValueError("yellow_mask and white_mask must have the same shape")
    if yellow_mask.ndim != 2 or white_mask.ndim != 2:
        raise ValueError("yellow_mask and white_mask must be 2D arrays")

    label_mask = np.zeros_like(yellow_mask, dtype=np.uint8)
    label_mask[yellow_mask > 0] = 1
    label_mask[white_mask > 0] = 2
    return label_mask


def compute_mask_stats(label_mask: np.ndarray) -> MaskStats:
    """Return pixel counts for each class."""
    total = int(np.sum(label_mask > 0))
    left = int(np.sum(label_mask == 1))
    right = int(np.sum(label_mask == 2))
    return MaskStats(total_lane_pixels=total, left_lane_pixels=left, right_lane_pixels=right)


def should_keep_frame(
    label_mask: np.ndarray,
    min_total_pixels: int = 400,
    min_lane_pixels: int = 150,
    require_both_lanes: bool = True,
) -> bool:
    """
    Decide if a frame should be included based on mask pixel counts.
    """
    stats = compute_mask_stats(label_mask)
    if stats.total_lane_pixels < min_total_pixels:
        return False
    if stats.left_lane_pixels < min_lane_pixels:
        return False
    if stats.right_lane_pixels < min_lane_pixels:
        return False
    if require_both_lanes and (stats.left_lane_pixels == 0 or stats.right_lane_pixels == 0):
        return False
    return True


def build_white_mask_from_combined(
    lane_color_mask: np.ndarray,
    yellow_mask: np.ndarray,
    image_center_x: int,
) -> np.ndarray:
    """
    Approximate white mask when only combined mask is available.

    Pixels in the combined mask that are not yellow are assigned to white.
    If that is still empty, split by image center.
    """
    if lane_color_mask.shape != yellow_mask.shape:
        raise ValueError("lane_color_mask and yellow_mask must have the same shape")
    if lane_color_mask.ndim != 2 or yellow_mask.ndim != 2:
        raise ValueError("lane_color_mask and yellow_mask must be 2D arrays")

    white_mask = (lane_color_mask > 0) & (yellow_mask == 0)
    if np.any(white_mask):
        return white_mask.astype(np.uint8) * 255

    # Fallback: split combined mask by center
    h, w = lane_color_mask.shape
    x_coords = np.arange(w)[None, :].repeat(h, axis=0)
    right_side = (x_coords >= image_center_x) & (lane_color_mask > 0)
    return right_side.astype(np.uint8) * 255


def compute_overlay(
    image: np.ndarray, label_mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay label mask on image for quick inspection.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    overlay = image.copy()
    yellow = np.array([255, 255, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)

    left_mask = label_mask == 1
    right_mask = label_mask == 2

    overlay[left_mask] = (alpha * overlay[left_mask] + (1 - alpha) * yellow).astype(np.uint8)
    overlay[right_mask] = (alpha * overlay[right_mask] + (1 - alpha) * white).astype(np.uint8)
    return overlay
