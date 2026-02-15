import numpy as np

from perception.inference import LaneDetectionInference


def test_mask_to_lane_data_returns_fit_points():
    height, width = 100, 200
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        mask[y, 50] = 1
        mask[y, 51] = 1
        mask[y, 150] = 2
        mask[y, 151] = 2

    inference = LaneDetectionInference(segmentation_mode=True)
    lane_coeffs, fit_left, fit_right, mask_resized = inference._mask_to_lane_data(
        mask, mask.shape
    )

    assert mask_resized.shape == mask.shape
    assert lane_coeffs[0] is not None
    assert lane_coeffs[1] is not None
    assert len(fit_left) > 0
    assert len(fit_right) > 0
    left_x = np.mean([p[0] for p in fit_left])
    right_x = np.mean([p[0] for p in fit_right])
    assert 49.0 <= left_x <= 52.0
    assert 149.0 <= right_x <= 152.0


def test_mask_to_lane_data_respects_row_limit():
    height, width = 100, 200
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        mask[y, 50] = 1
        mask[y, 51] = 1
        mask[y, 150] = 2
        mask[y, 151] = 2

    inference = LaneDetectionInference(
        segmentation_mode=True,
        segmentation_fit_min_row_ratio=0.45,
        segmentation_fit_max_row_ratio=0.5,
    )
    _, fit_left, fit_right, _ = inference._mask_to_lane_data(mask, mask.shape)

    assert fit_left and fit_right
    assert min(p[1] for p in fit_left) >= height * 0.45
    assert min(p[1] for p in fit_right) >= height * 0.45
    assert max(p[1] for p in fit_left) < height * 0.5
    assert max(p[1] for p in fit_right) < height * 0.5


def test_mask_to_lane_data_ransac_rejects_outliers():
    height, width = 120, 200
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        mask[y, 50] = 1
        mask[y, 51] = 1
        mask[y, 150] = 2
        mask[y, 151] = 2
    # Add stray outliers for left lane
    for y in range(55, 85, 3):
        mask[y, 5] = 1

    inference = LaneDetectionInference(
        segmentation_mode=True,
        segmentation_ransac_enabled=True,
        segmentation_ransac_residual_px=2.0,
        segmentation_ransac_min_inliers=15,
        segmentation_ransac_max_trials=50,
    )
    lane_coeffs, _, _, _ = inference._mask_to_lane_data(mask, mask.shape)

    assert lane_coeffs[0] is not None
    x_mid = np.polyval(lane_coeffs[0], 60.0)
    assert 45.0 <= x_mid <= 55.0


def test_mask_to_lane_data_ransac_fallback():
    height, width = 120, 200
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        mask[y, 50] = 1
        mask[y, 51] = 1

    inference = LaneDetectionInference(
        segmentation_mode=True,
        segmentation_ransac_enabled=True,
        segmentation_ransac_min_inliers=500,  # force RANSAC failure
    )
    lane_coeffs, _, _, _ = inference._mask_to_lane_data(mask, mask.shape)

    assert inference.last_segmentation_ransac_used["left"] is False
    assert lane_coeffs[0] is not None
