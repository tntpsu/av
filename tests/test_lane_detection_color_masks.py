import numpy as np

from perception.models.lane_detection import SimpleLaneDetector


def test_sample_points_use_lane_specific_masks():
    detector = SimpleLaneDetector()
    h, w = 100, 100
    center_x = w // 2

    yellow_mask = np.zeros((h, w), dtype=np.uint8)
    white_mask = np.zeros((h, w), dtype=np.uint8)

    # Left lane (yellow) at x=20..22, right lane (white) at x=78..80
    for y in range(h):
        yellow_mask[y, 20:23] = 255
        white_mask[y, 78:81] = 255

    lane_color_mask = np.maximum(yellow_mask, white_mask)

    left_points = detector._sample_points_per_row(
        lane_idx=0,
        yellow_mask=yellow_mask,
        white_mask=white_mask,
        lane_color_mask=lane_color_mask,
        center_x=center_x,
        w=w,
        h=h,
        lane_name="left",
        previous_coeffs=None,
    )

    right_points = detector._sample_points_per_row(
        lane_idx=1,
        yellow_mask=yellow_mask,
        white_mask=white_mask,
        lane_color_mask=lane_color_mask,
        center_x=center_x,
        w=w,
        h=h,
        lane_name="right",
        previous_coeffs=None,
    )

    assert left_points is not None and len(left_points) > 0
    assert right_points is not None and len(right_points) > 0

    assert np.max(left_points[:, 0]) < center_x
    assert np.min(right_points[:, 0]) > center_x


def test_no_top_supplement_when_no_pixels_in_top_region():
    detector = SimpleLaneDetector()
    h, w = 100, 100
    center_x = w // 2

    yellow_mask = np.zeros((h, w), dtype=np.uint8)
    white_mask = np.zeros((h, w), dtype=np.uint8)

    # Only bottom region has lane pixels (no pixels in top region)
    for y in range(70, h):
        yellow_mask[y, 20] = 255

    lane_color_mask = yellow_mask.copy()

    points = np.array([[20.0, 75.0], [20.0, 85.0], [20.0, 95.0]])
    roi_y_start = int(h * 0.18)
    lookahead_y_estimate = int(h * 0.73)
    region_top_end = max(roi_y_start + 50, lookahead_y_estimate - 50)

    supplemented = detector._supplement_points_from_mask(
        points=points,
        lane_idx=0,
        yellow_mask=yellow_mask,
        white_mask=white_mask,
        lane_color_mask=lane_color_mask,
        center_x=center_x,
        w=w,
        h=h,
        lane_name="left",
        expected_x=None,
    )

    assert np.min(supplemented[:, 1]) >= region_top_end


def test_sparse_top_rows_are_skipped_in_sampling():
    detector = SimpleLaneDetector()
    h, w = 120, 120
    center_x = w // 2

    yellow_mask = np.zeros((h, w), dtype=np.uint8)
    white_mask = np.zeros((h, w), dtype=np.uint8)

    # Sparse single-pixel lane in top region (should be ignored)
    roi_y_start = int(h * 0.18)
    lookahead_y_estimate = int(h * 0.73)
    region_top_end = max(roi_y_start + 50, lookahead_y_estimate - 50)
    for y in range(roi_y_start, region_top_end, 8):
        yellow_mask[y, 20] = 255

    # Dense lane in bottom region (should be used)
    for y in range(region_top_end, int(h * 0.80), 2):
        yellow_mask[y, 20:23] = 255

    lane_color_mask = yellow_mask.copy()

    points = detector._sample_points_per_row(
        lane_idx=0,
        yellow_mask=yellow_mask,
        white_mask=white_mask,
        lane_color_mask=lane_color_mask,
        center_x=center_x,
        w=w,
        h=h,
        lane_name="left",
        previous_coeffs=None,
    )

    assert points is not None and len(points) > 0
    assert np.min(points[:, 1]) >= region_top_end


def test_top_rows_with_contiguous_run_are_kept():
    detector = SimpleLaneDetector()
    h, w = 120, 120
    center_x = w // 2

    yellow_mask = np.zeros((h, w), dtype=np.uint8)
    white_mask = np.zeros((h, w), dtype=np.uint8)

    roi_y_start = int(h * 0.18)
    lookahead_y_estimate = int(h * 0.73)
    region_top_end = max(roi_y_start + 50, lookahead_y_estimate - 50)

    # Contiguous run in top region (length >= 6)
    for y in range(roi_y_start, region_top_end, 8):
        yellow_mask[y, 20:27] = 255

    # Dense lane in bottom region
    for y in range(region_top_end, int(h * 0.80), 2):
        yellow_mask[y, 20:27] = 255

    lane_color_mask = yellow_mask.copy()

    points = detector._sample_points_per_row(
        lane_idx=0,
        yellow_mask=yellow_mask,
        white_mask=white_mask,
        lane_color_mask=lane_color_mask,
        center_x=center_x,
        w=w,
        h=h,
        lane_name="left",
        previous_coeffs=None,
    )

    assert points is not None and len(points) > 0
    assert np.min(points[:, 1]) < region_top_end
