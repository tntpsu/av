import numpy as np

from training.utils.segmentation_dataset import (
    build_label_mask,
    compute_mask_stats,
    should_keep_frame,
)


def test_build_label_mask_assigns_classes():
    yellow = np.zeros((4, 4), dtype=np.uint8)
    white = np.zeros((4, 4), dtype=np.uint8)
    yellow[0, 0] = 255
    white[1, 1] = 255

    label = build_label_mask(yellow, white)

    assert label[0, 0] == 1
    assert label[1, 1] == 2
    assert label[2, 2] == 0


def test_should_keep_frame_filters_small_masks():
    label = np.zeros((10, 10), dtype=np.uint8)
    label[0, :5] = 1
    label[1, :5] = 2

    assert not should_keep_frame(label, min_total_pixels=20, min_lane_pixels=10)


def test_compute_mask_stats_counts_pixels():
    label = np.zeros((4, 4), dtype=np.uint8)
    label[0, :2] = 1
    label[1, :3] = 2

    stats = compute_mask_stats(label)

    assert stats.total_lane_pixels == 5
    assert stats.left_lane_pixels == 2
    assert stats.right_lane_pixels == 3
