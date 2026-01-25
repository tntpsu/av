"""
Prepare a lane segmentation dataset from AV stack recordings.

This uses the current CV lane detector to generate pseudo-labels:
0 = background, 1 = left (yellow), 2 = right (white).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import sys
from datetime import datetime

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from perception.models.lane_detection import SimpleLaneDetector
from training.utils.segmentation_dataset import (
    build_label_mask,
    build_white_mask_from_combined,
    compute_mask_stats,
    compute_overlay,
    should_keep_frame,
)


def _iter_recordings(recording_path: Path) -> List[Path]:
    if recording_path.is_dir():
        return sorted(recording_path.glob("*.h5"))
    return [recording_path]


def _recording_date_ok(recording: Path, min_date: Optional[str]) -> bool:
    if not min_date:
        return True
    try:
        min_dt = datetime.strptime(min_date, "%Y%m%d")
    except ValueError as exc:
        raise ValueError("min_date must be in YYYYMMDD format") from exc
    
    name = recording.stem
    if name.startswith("recording_") and len(name) >= len("recording_YYYYMMDD"):
        date_str = name.split("_")[1]
        try:
            rec_dt = datetime.strptime(date_str, "%Y%m%d")
            return rec_dt >= min_dt
        except ValueError:
            return True
    return True


def _iter_frame_indices(frame_count: int, stride: int, max_frames: Optional[int]) -> Iterable[int]:
    indices = range(0, frame_count, stride)
    if max_frames is None:
        return indices
    return list(indices)[:max_frames]


def _load_ground_truth_lane_center(file_handle: h5py.File, frame_idx: int) -> Optional[float]:
    if "ground_truth/lane_center_x" not in file_handle:
        return None
    try:
        return float(file_handle["ground_truth/lane_center_x"][frame_idx])
    except (IndexError, OSError, ValueError):
        return None


def prepare_dataset(
    recordings_path: Path,
    output_dir: Path,
    stride: int = 1,
    max_frames: Optional[int] = None,
    min_total_pixels: int = 400,
    min_lane_pixels: int = 150,
    require_both_lanes: bool = True,
    save_overlays: bool = True,
    overlay_alpha: float = 0.5,
    use_ground_truth_filter: bool = True,
    max_lane_center_abs: float = 1.5,
    min_date: Optional[str] = None,
) -> Tuple[int, int]:
    output_images = output_dir / "images"
    output_masks = output_dir / "masks"
    output_overlays = output_dir / "overlays"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        output_overlays.mkdir(parents=True, exist_ok=True)

    detector = SimpleLaneDetector()

    metadata_path = output_dir / "metadata.csv"
    skip_summary_path = output_dir / "skip_summary.json"
    skip_examples_path = output_dir / "skip_examples.csv"

    skip_counts = {
        "date_filtered": 0,
        "recording_error": 0,
        "frame_read_error": 0,
        "gt_index_out_of_range": 0,
        "gt_out_of_bounds": 0,
        "missing_masks": 0,
        "min_pixels": 0,
    }
    skip_examples: dict[str, list[str]] = {key: [] for key in skip_counts}

    def _add_skip_example(reason: str, text: str) -> None:
        if len(skip_examples[reason]) < 5:
            skip_examples[reason].append(text)

    with metadata_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "recording",
                "frame_idx",
                "timestamp",
                "total_lane_pixels",
                "left_lane_pixels",
                "right_lane_pixels",
                "lane_center_x",
            ]
        )

        kept = 0
        skipped = 0
        for recording in _iter_recordings(recordings_path):
            if not _recording_date_ok(recording, min_date):
                skip_counts["date_filtered"] += 1
                _add_skip_example("date_filtered", recording.name)
                continue
            try:
                with h5py.File(recording, "r") as f:
                    images = f["camera/images"]
                    timestamps = f["camera/timestamps"]
                    frame_count = min(images.shape[0], timestamps.shape[0])
                    gt_len = None
                    if "ground_truth/lane_center_x" in f:
                        gt_len = f["ground_truth/lane_center_x"].shape[0]

                    for idx in _iter_frame_indices(frame_count, stride, max_frames):
                        try:
                            image = images[idx]
                            timestamp = float(timestamps[idx])
                        except (OSError, ValueError) as exc:
                            skipped += 1
                            skip_counts["frame_read_error"] += 1
                            _add_skip_example("frame_read_error", f"{recording.name}:{idx}")
                            print(f"Skipping frame {idx} from {recording.name}: {exc}")
                            continue

                        lane_center = _load_ground_truth_lane_center(f, idx)
                        if use_ground_truth_filter and gt_len is not None and idx >= gt_len:
                            skipped += 1
                            skip_counts["gt_index_out_of_range"] += 1
                            _add_skip_example("gt_index_out_of_range", f"{recording.name}:{idx}")
                            continue
                        if use_ground_truth_filter and lane_center is not None:
                            if abs(lane_center) > max_lane_center_abs:
                                skipped += 1
                                skip_counts["gt_out_of_bounds"] += 1
                                _add_skip_example("gt_out_of_bounds", f"{recording.name}:{idx}")
                                continue

                        _, debug_info = detector.detect(image, return_debug=True)
                        lane_color_mask = debug_info.get("lane_color_mask")
                        yellow_mask = debug_info.get("yellow_mask")

                        if lane_color_mask is None or yellow_mask is None:
                            skipped += 1
                            skip_counts["missing_masks"] += 1
                            _add_skip_example("missing_masks", f"{recording.name}:{idx}")
                            continue

                        h, w = lane_color_mask.shape
                        white_mask = build_white_mask_from_combined(
                            lane_color_mask, yellow_mask, image_center_x=w // 2
                        )
                        label_mask = build_label_mask(yellow_mask, white_mask)

                        if not should_keep_frame(
                            label_mask,
                            min_total_pixels=min_total_pixels,
                            min_lane_pixels=min_lane_pixels,
                            require_both_lanes=require_both_lanes,
                        ):
                            skipped += 1
                            skip_counts["min_pixels"] += 1
                            _add_skip_example("min_pixels", f"{recording.name}:{idx}")
                            continue

                        stats = compute_mask_stats(label_mask)
                        image_path = output_images / f"{recording.stem}_frame_{idx:06d}.png"
                        mask_path = output_masks / f"{recording.stem}_frame_{idx:06d}.png"
                        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(mask_path), label_mask)

                        if save_overlays:
                            overlay = compute_overlay(image, label_mask, alpha=overlay_alpha)
                            overlay_path = output_overlays / f"{recording.stem}_frame_{idx:06d}.png"
                            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                        writer.writerow(
                            [
                                recording.name,
                                idx,
                                timestamp,
                                stats.total_lane_pixels,
                                stats.left_lane_pixels,
                                stats.right_lane_pixels,
                                lane_center if lane_center is not None else "",
                            ]
                        )
                        kept += 1
            except (OSError, ValueError) as exc:
                print(f"Skipping recording {recording.name}: {exc}")
                skipped += 1
                skip_counts["recording_error"] += 1
                _add_skip_example("recording_error", recording.name)

    with skip_summary_path.open("w") as summary_file:
        json.dump(skip_counts, summary_file, indent=2)

    with skip_examples_path.open("w", newline="") as examples_file:
        writer = csv.writer(examples_file)
        writer.writerow(["reason", "examples"])
        for reason, examples in skip_examples.items():
            writer.writerow([reason, "; ".join(examples)])

    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare lane segmentation dataset from recordings")
    parser.add_argument(
        "--recordings",
        type=str,
        default="data/recordings",
        help="Path to a recording file or directory of recordings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/segmentation_dataset",
        help="Output directory for dataset",
    )
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for extraction")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per recording")
    parser.add_argument("--min-total-pixels", type=int, default=400, help="Min total lane pixels")
    parser.add_argument("--min-lane-pixels", type=int, default=150, help="Min pixels per lane")
    parser.add_argument(
        "--no-require-both",
        action="store_true",
        help="Allow frames with only one lane present",
    )
    parser.add_argument("--no-overlays", action="store_true", help="Skip overlay images")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Overlay alpha")
    parser.add_argument(
        "--no-gt-filter",
        action="store_true",
        help="Disable ground truth lane center filtering",
    )
    parser.add_argument(
        "--max-lane-center-abs",
        type=float,
        default=1.5,
        help="Max abs(ground truth lane center) to keep a frame",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help="Skip recordings before this date (YYYYMMDD)",
    )

    args = parser.parse_args()
    kept, skipped = prepare_dataset(
        recordings_path=Path(args.recordings),
        output_dir=Path(args.output_dir),
        stride=args.stride,
        max_frames=args.max_frames,
        min_total_pixels=args.min_total_pixels,
        min_lane_pixels=args.min_lane_pixels,
        require_both_lanes=not args.no_require_both,
        save_overlays=not args.no_overlays,
        overlay_alpha=args.overlay_alpha,
        use_ground_truth_filter=not args.no_gt_filter,
        max_lane_center_abs=args.max_lane_center_abs,
        min_date=args.min_date,
    )
    print(f"Prepared dataset: kept={kept}, skipped={skipped}, output={args.output_dir}")
    print(f"Skip summary saved to: {Path(args.output_dir) / 'skip_summary.json'}")


if __name__ == "__main__":
    main()
