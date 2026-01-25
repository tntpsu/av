"""
Build side-by-side comparison images (CV vs model prediction).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from perception.models.lane_detection import SimpleLaneDetector
from perception.models.segmentation_model import SimpleSegNet


def _overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = image.copy()
    yellow = np.array([255, 255, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    left_mask = mask == 1
    right_mask = mask == 2
    overlay[left_mask] = (alpha * overlay[left_mask] + (1 - alpha) * yellow).astype(np.uint8)
    overlay[right_mask] = (alpha * overlay[right_mask] + (1 - alpha) * white).astype(np.uint8)
    return overlay


def _cv_overlay(
    detector: SimpleLaneDetector,
    image: np.ndarray,
    size: Tuple[int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    _, debug_info = detector.detect(image, return_debug=True)
    lane_color_mask = debug_info.get("lane_color_mask")
    yellow_mask = debug_info.get("yellow_mask")
    if lane_color_mask is None or yellow_mask is None:
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return resized
    h, w = lane_color_mask.shape
    white_mask = (lane_color_mask > 0) & (yellow_mask == 0)
    label_mask = np.zeros((h, w), dtype=np.uint8)
    label_mask[yellow_mask > 0] = 1
    label_mask[white_mask] = 2
    overlay = _overlay(image, label_mask, alpha=alpha)
    resized = cv2.resize(overlay, size, interpolation=cv2.INTER_LINEAR)
    return resized


def _model_overlay(
    model: nn.Module, image: np.ndarray, size: Tuple[int, int], device: torch.device, alpha: float
) -> np.ndarray:
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return _overlay(resized, pred, alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CV vs model comparison images")
    parser.add_argument("--dataset-dir", type=str, default="data/segmentation_dataset")
    parser.add_argument(
        "--checkpoint", type=str, default="data/segmentation_dataset/checkpoints/segnet_best.pt"
    )
    parser.add_argument("--output-dir", type=str, default="tmp/segmentation_comparison")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--img-height", type=int, default=320)
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted((dataset_dir / "images").glob("*.png"))
    if not images:
        raise SystemExit("No images found in dataset.")

    random.seed(args.seed)
    samples = random.sample(images, k=min(args.num_samples, len(images)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSegNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    detector = SimpleLaneDetector()
    size = (args.img_width, args.img_height)

    for image_path in samples:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        cv_overlay = _cv_overlay(detector, image, size, alpha=args.overlay_alpha)
        model_overlay = _model_overlay(model, image, size, device, alpha=args.overlay_alpha)
        combined = np.concatenate([image_resized, cv_overlay, model_overlay], axis=1)
        out_path = output_dir / f"{image_path.stem}_compare.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"Saved comparisons to {output_dir}")


if __name__ == "__main__":
    main()
