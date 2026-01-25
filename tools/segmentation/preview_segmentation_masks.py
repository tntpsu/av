"""
Preview lane segmentation masks on a few frames for quick validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from perception.models.lane_detection import SimpleLaneDetector


def parse_frame_indices(spec: str) -> List[int]:
    """Parse a comma-separated list of frame indices."""
    indices: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            indices.extend(list(range(start, end + 1)))
        else:
            indices.append(int(chunk))
    return sorted(set(indices))


def get_mask_from_debug(debug_info: dict, mask_type: str) -> np.ndarray:
    """Select the requested mask from debug_info."""
    if mask_type == "yellow":
        return debug_info["yellow_mask"]
    if mask_type == "white":
        return debug_info["white_mask_roi"]
    if mask_type == "lane_color":
        return debug_info["lane_color_mask"]
    raise ValueError(f"Unsupported mask type: {mask_type}")


def iter_frames(recording_path: Path, indices: Iterable[int]) -> Iterable[tuple[int, np.ndarray]]:
    """Yield (index, image) for requested frames from a recording."""
    with h5py.File(recording_path, "r") as f:
        images = f["camera/images"]
        for idx in indices:
            if idx < 0 or idx >= len(images):
                continue
            yield idx, images[idx]


def main() -> None:
    """Run segmentation mask preview and save outputs."""
    parser = argparse.ArgumentParser(description="Preview lane segmentation masks.")
    parser.add_argument("--recording", type=str, required=True, help="Path to .h5 recording")
    parser.add_argument("--frames", type=str, default="0,1", help="Frame indices, e.g. 0,1,5-8")
    parser.add_argument("--mask", type=str, default="lane_color", choices=["lane_color", "yellow", "white"])
    parser.add_argument("--output_dir", type=str, default="tmp/segmentation_preview")
    parser.add_argument("--save_overlay", action="store_true", help="Save overlay image")
    args = parser.parse_args()

    recording_path = Path(args.recording)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = SimpleLaneDetector()
    frame_indices = parse_frame_indices(args.frames)

    for idx, image in iter_frames(recording_path, frame_indices):
        _, debug_info = detector.detect(image, return_debug=True)
        mask = get_mask_from_debug(debug_info, args.mask)
        mask_path = output_dir / f"frame_{idx:06d}_{args.mask}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        if args.save_overlay:
            # Overlay mask in green on the raw image for quick inspection
            overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mask_color = np.zeros_like(overlay)
            mask_color[:, :, 1] = mask  # Green channel
            blended = cv2.addWeighted(overlay, 0.85, mask_color, 0.15, 0)
            overlay_path = output_dir / f"frame_{idx:06d}_{args.mask}_overlay.png"
            cv2.imwrite(str(overlay_path), blended)

        print(f"Saved: {mask_path}")


if __name__ == "__main__":
    main()
