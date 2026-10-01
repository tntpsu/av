"""Camera stride in the recorder + nightly recording retention (2026-10-01).

Camera images were ~75 % of every 150 MB recording and the pool grew 3.4 GB/day until
the disk hit 99 %. The stride keeps camera/images 1:1 with frames (zero placeholders
compress to ~nothing) so index-based consumers keep working; the prune script keeps
goldens/referenced, recent, and newest-per-track files.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.formats.data_format import CameraFrame  # noqa: E402
from data.recorder import DataRecorder  # noqa: E402
from tools.nightly import prune_recordings as pr  # noqa: E402


def _record(tmp_path: Path, stride: str | None, n: int = 12) -> Path:
    if stride is None:
        os.environ.pop("AV_RECORD_CAMERA_STRIDE", None)
    else:
        os.environ["AV_RECORD_CAMERA_STRIDE"] = stride
    try:
        rec = DataRecorder(str(tmp_path), recording_name=f"stride_{stride or 1}")
        rng = np.random.default_rng(0)
        for i in range(n):
            img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
            rec.record_camera_frame(CameraFrame(image=img, timestamp=i * 0.05, frame_id=i))
        rec.flush()
        rec.close()
    finally:
        os.environ.pop("AV_RECORD_CAMERA_STRIDE", None)
    return rec.output_file


class TestCameraStride:
    def test_default_is_full_rate_and_flags_all_zero(self, tmp_path):
        out = _record(tmp_path, None)
        with h5py.File(out, "r") as f:
            assert f["camera/images"].shape[0] == 12
            assert int(f["camera/image_is_placeholder"][:].sum()) == 0
            assert json.loads(f.attrs["metadata"])["recording_provenance"]["camera_stride"] == 1

    def test_stride_keeps_every_nth_and_stays_one_to_one(self, tmp_path):
        out = _record(tmp_path, "4")
        with h5py.File(out, "r") as f:
            imgs = f["camera/images"]; flags = f["camera/image_is_placeholder"][:]
            assert imgs.shape[0] == 12 == len(flags)          # still 1:1 with frames
            assert flags.tolist() == [0, 1, 1, 1] * 3
            assert imgs[1].max() == 0 and imgs[4].max() > 0   # placeholder vs kept
            assert json.loads(f.attrs["metadata"])["recording_provenance"]["camera_stride"] == 4

    def test_stride_shrinks_the_file(self, tmp_path):
        full = _record(tmp_path / "a", None); strided = _record(tmp_path / "b", "4")
        assert strided.stat().st_size < 0.5 * full.stat().st_size

    def test_bad_env_value_falls_back_to_full_rate(self, tmp_path):
        out = _record(tmp_path, "banana")
        with h5py.File(out, "r") as f:
            assert int(f["camera/image_is_placeholder"][:].sum()) == 0


class TestPrunePlan:
    def _mk(self, d: Path, name: str, age_days: float, track: str = "s_loop", size: int = 10) -> Path:
        p = d / name
        with h5py.File(p, "w") as f:
            f.attrs["metadata"] = json.dumps({"recording_provenance": {"track_id": track}})
            f.create_dataset("pad", data=np.zeros(size, dtype=np.uint8))
        t = time.time() - age_days * 86400
        os.utime(p, (t, t))
        return p

    def test_rules(self, tmp_path):
        d = tmp_path / "rec"; d.mkdir()
        self._mk(d, "recording_20260101_000000.h5", 100)                    # old, unreferenced → delete
        self._mk(d, "recording_20260102_000000.h5", 100)                    # old but referenced → keep
        self._mk(d, "recording_20260103_000000.h5", 5)                      # recent → keep
        self._mk(d, "recording_20260104_000000.h5", 40, track="hairpin_15")  # old, only one of its track → keep (newest per track)
        self._mk(d, "recording_20260105_000000.h5", 50, track="hairpin_15")  # older sibling → delete when keep_per_track=1
        self._mk(d, "recording_20260106_000000.h5", 0.001)                  # just written → keep
        res = pr.plan(d, keep_days=14, keep_per_track=1, refs={"recording_20260102_000000.h5"})
        deleted = {e["file"] for e in res["delete"]}
        kept = {e["file"]: e["reason"] for e in res["keep"]}
        assert deleted == {"recording_20260101_000000.h5", "recording_20260105_000000.h5"}
        assert kept["recording_20260102_000000.h5"] == "referenced"
        assert kept["recording_20260103_000000.h5"] == "recent"
        assert kept["recording_20260104_000000.h5"] == "newest_per_track"
        assert kept["recording_20260106_000000.h5"] == "being_written"

    def test_referenced_scan_finds_goldens(self):
        refs = pr.referenced_recordings()
        goldens = json.loads((REPO_ROOT / "tests/fixtures/golden_recordings.json").read_text())["tracks"]
        for p in goldens.values():
            assert Path(p).name in refs

    def test_dry_run_deletes_nothing(self, tmp_path, monkeypatch):
        d = tmp_path / "rec"; d.mkdir()
        self._mk(d, "recording_20260101_000000.h5", 100)
        monkeypatch.setattr(pr, "MANIFEST_DIR", tmp_path / "man")
        assert pr.main(["prune", "--recordings-dir", str(d), "--keep-per-track", "0"]) == 0
        assert (d / "recording_20260101_000000.h5").exists()
        assert list((tmp_path / "man").glob("prune_*.json"))
