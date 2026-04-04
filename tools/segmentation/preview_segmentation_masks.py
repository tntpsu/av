"""
Preview lane segmentation masks on a few frames for quick validation.

Supports two modes:
  --segmentation  Run the trained SimpleSegNet model (yellow=left, white=right)
  (default)       Run the CV-based SimpleLaneDetector (green channel overlay)

Usage:
  # Seg model on recording frames 480-520:
  python preview_segmentation_masks.py --recording rec.h5 --frames 480-520 --segmentation --save_overlay

  # CV detector (original behaviour):
  python preview_segmentation_masks.py --recording rec.h5 --frames 0-10 --save_overlay
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


def _seg_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay segmentation mask: yellow=left lane (1), white=right lane (2)."""
    overlay = image.copy()
    yellow = np.array([255, 255, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    left_mask = mask == 1
    right_mask = mask == 2
    overlay[left_mask] = (alpha * overlay[left_mask] + (1 - alpha) * yellow).astype(np.uint8)
    overlay[right_mask] = (alpha * overlay[right_mask] + (1 - alpha) * white).astype(np.uint8)
    return overlay


def iter_frames(recording_path: Path, indices: Iterable[int]) -> Iterable[tuple[int, np.ndarray]]:
    """Yield (index, image) for requested frames from a recording."""
    with h5py.File(recording_path, "r") as f:
        images = f["camera/images"]
        for idx in indices:
            if idx < 0 or idx >= len(images):
                continue
            yield idx, images[idx]


def _get_frame_count(recording_path: Path) -> int:
    """Return total number of camera frames in a recording."""
    with h5py.File(recording_path, "r") as f:
        return f["camera/images"].shape[0]


def main() -> None:
    """Run segmentation mask preview and save outputs."""
    parser = argparse.ArgumentParser(description="Preview lane segmentation masks.")
    parser.add_argument("--recording", type=str, required=True, help="Path to .h5 recording")
    parser.add_argument("--frames", type=str, default="0,1", help="Frame indices, e.g. 0,1,5-8")
    parser.add_argument("--all", action="store_true", help="Process all frames in the recording")
    parser.add_argument("--mask", type=str, default="lane_color", choices=["lane_color", "yellow", "white"])
    parser.add_argument("--output_dir", type=str, default="tmp/segmentation_preview")
    parser.add_argument("--save_overlay", action="store_true", help="Save overlay image")
    # Segmentation model mode
    parser.add_argument("--segmentation", action="store_true", help="Use segmentation model instead of CV detector")
    parser.add_argument("--checkpoint", type=str, default="data/segmentation_dataset/checkpoints/segnet_best.pt",
                        help="Segmentation model checkpoint path")
    args = parser.parse_args()

    recording_path = Path(args.recording)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        n = _get_frame_count(recording_path)
        frame_indices = list(range(n))
    else:
        frame_indices = parse_frame_indices(args.frames)

    if args.segmentation:
        import torch
        from perception.models.segmentation_model import load_segmentation_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_segmentation_model(args.checkpoint, device)
        target_h, target_w = 320, 640

        for idx, image in iter_frames(recording_path, frame_indices):
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            left_px = int(np.sum(pred == 1))
            right_px = int(np.sum(pred == 2))
            total_px = pred.shape[0] * pred.shape[1]
            coverage = (left_px + right_px) / total_px * 100

            # Save colorized mask
            mask_color = np.zeros((*pred.shape, 3), dtype=np.uint8)
            mask_color[pred == 1] = [255, 255, 0]  # yellow = left
            mask_color[pred == 2] = [255, 255, 255]  # white = right
            mask_path = output_dir / f"frame_{idx:06d}_seg_mask.png"
            cv2.imwrite(str(mask_path), cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR))

            if args.save_overlay:
                overlay = _seg_overlay(resized, pred)
                overlay_path = output_dir / f"frame_{idx:06d}_seg_overlay.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            print(f"Frame {idx:5d}: left={left_px:5d}px  right={right_px:5d}px  coverage={coverage:.1f}%  -> {mask_path.name}")
    else:
        detector = SimpleLaneDetector()
        for idx, image in iter_frames(recording_path, frame_indices):
            _, debug_info = detector.detect(image, return_debug=True)
            mask = get_mask_from_debug(debug_info, args.mask)
            mask_path = output_dir / f"frame_{idx:06d}_{args.mask}_mask.png"
            cv2.imwrite(str(mask_path), mask)

            if args.save_overlay:
                overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mask_color = np.zeros_like(overlay)
                mask_color[:, :, 1] = mask  # Green channel
                blended = cv2.addWeighted(overlay, 0.85, mask_color, 0.15, 0)
                overlay_path = output_dir / f"frame_{idx:06d}_{args.mask}_overlay.png"
                cv2.imwrite(str(overlay_path), blended)

            print(f"Saved: {mask_path}")


if __name__ == "__main__":
    main()
