from tools.segmentation.preview_segmentation_masks import parse_frame_indices


def test_parse_frame_indices_handles_ranges_and_commas():
    assert parse_frame_indices("0,2,5-7") == [0, 2, 5, 6, 7]


def test_parse_frame_indices_deduplicates_and_sorts():
    assert parse_frame_indices("3,1,2,2,1") == [1, 2, 3]
