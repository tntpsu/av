import pytest

from tools.ground_truth_follower import resolve_segmentation_settings


def test_resolve_segmentation_settings_use_cv(tmp_path):
    use_segmentation, checkpoint = resolve_segmentation_settings(
        use_cv=True, segmentation_checkpoint=str(tmp_path / "missing.pt")
    )
    assert use_segmentation is False
    assert checkpoint == str(tmp_path / "missing.pt")


def test_resolve_segmentation_settings_requires_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_segmentation_settings(use_cv=False, segmentation_checkpoint=None)

    with pytest.raises(FileNotFoundError):
        resolve_segmentation_settings(
            use_cv=False, segmentation_checkpoint=str(tmp_path / "missing.pt")
        )


def test_resolve_segmentation_settings_accepts_existing(tmp_path):
    checkpoint = tmp_path / "segnet_best.pt"
    checkpoint.write_text("checkpoint")
    use_segmentation, path = resolve_segmentation_settings(
        use_cv=False, segmentation_checkpoint=str(checkpoint)
    )
    assert use_segmentation is True
    assert path == str(checkpoint)
