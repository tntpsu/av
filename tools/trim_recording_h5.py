#!/usr/bin/env python3
"""
Trim an AV stack HDF5 recording for smaller size (e.g. GitHub-friendly fixtures).

- Slices 1D time-series and image stacks along the leading time/frame dimension.
- Optionally drops camera/topdown image tensors (largest contributors).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def _infer_frame_count(f: h5py.File) -> int:
    for key in (
        "control/steering",
        "vehicle/timestamps",
        "control/timestamps",
        "camera/timestamps",
    ):
        if key in f:
            return int(f[key].shape[0])
    # Fallback: max first dim of any 1D dataset under control/
    n = 0
    if "control" in f:
        for name in f["control"].keys():
            ds = f["control"][name]
            if isinstance(ds, h5py.Dataset) and ds.ndim >= 1:
                n = max(n, int(ds.shape[0]))
    return n


def trim_recording(
    src: Path,
    dst: Path,
    *,
    max_frames: int,
    drop_camera_images: bool = True,
    drop_topdown_images: bool = True,
) -> dict:
    """
    Copy ``src`` to ``dst`` with time truncation and optional image removal.

    Returns stats dict with byte sizes.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)

    n_full = 0
    with h5py.File(src, "r") as f_in:
        n_full = _infer_frame_count(f_in)
        n_use = max(1, min(max_frames, n_full))

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()

        with h5py.File(dst, "w") as f_out:
            # File-level metadata
            for k, v in f_in.attrs.items():
                f_out.attrs[k] = v

            def visit_copy(parent_in: h5py.Group, parent_out: h5py.Group, loc: str) -> None:
                for key in parent_in.keys():
                    item = parent_in[key]
                    path = f"{loc}/{key}" if loc else key
                    if isinstance(item, h5py.Group):
                        g = parent_out.require_group(key)
                        visit_copy(item, g, path)
                    else:
                        _copy_dataset(
                            item,
                            parent_out,
                            key,
                            path,
                            n_use,
                            n_full,
                            drop_camera_images,
                            drop_topdown_images,
                        )

            visit_copy(f_in, f_out, "")

    return {
        "source_frames": n_full,
        "output_frames": n_use,
        "source_bytes": src.stat().st_size,
        "output_bytes": dst.stat().st_size,
    }


def _copy_dataset(
    ds_in: h5py.Dataset,
    parent_out: h5py.Group,
    key: str,
    path: str,
    n_use: int,
    n_full: int,
    drop_camera: bool,
    drop_topdown: bool,
) -> None:
    if drop_camera and path == "camera/images":
        return
    if drop_topdown and path == "camera/topdown_images":
        return

    shape = ds_in.shape
    if len(shape) == 0:
        parent_out.create_dataset(
            key, data=ds_in[()], compression=ds_in.compression, compression_opts=ds_in.compression_opts
        )
        return

    lead = int(shape[0])
    # Align to recording frame count when first dim matches full length
    if lead == n_full:
        sl = slice(0, n_use)
        if len(shape) == 1:
            data = ds_in[sl]
        else:
            idx: tuple = (sl,) + tuple(slice(None) for _ in range(len(shape) - 1))
            data = ds_in[idx]
    elif lead > n_full:
        # e.g. topdown with different cadence — scale slice roughly
        ratio = n_use / float(n_full) if n_full else 1.0
        n2 = max(1, min(lead, int(round(lead * ratio))))
        sl = slice(0, n2)
        if len(shape) == 1:
            data = ds_in[sl]
        else:
            idx = (sl,) + tuple(slice(None) for _ in range(len(shape) - 1))
            data = ds_in[idx]
    else:
        # Shorter series — keep as-is
        data = ds_in[()]

    kwargs = {}
    if ds_in.compression:
        kwargs["compression"] = ds_in.compression
        if ds_in.compression_opts is not None:
            kwargs["compression_opts"] = ds_in.compression_opts

    parent_out.create_dataset(key, data=np.asarray(data), **kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description="Trim HDF5 recording for smaller fixtures.")
    p.add_argument("input", type=Path, help="Source .h5")
    p.add_argument("output", type=Path, help="Destination .h5")
    p.add_argument("--max-frames", type=int, default=400, help="Max frames to keep (default 400)")
    p.add_argument("--keep-camera", action="store_true", help="Keep camera/images (large)")
    p.add_argument("--keep-topdown", action="store_true", help="Keep camera/topdown_images")
    args = p.parse_args()

    stats = trim_recording(
        args.input,
        args.output,
        max_frames=args.max_frames,
        drop_camera_images=not args.keep_camera,
        drop_topdown_images=not args.keep_topdown,
    )
    mb = stats["output_bytes"] / (1024 * 1024)
    print(
        f"Wrote {args.output} — {stats['output_frames']} frames "
        f"({stats['source_frames']} src), {mb:.3f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
