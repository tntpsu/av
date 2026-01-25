"""
Run segmentation model on a sample of images and save overlays.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from perception.models.segmentation_model import SimpleSegNet


def _collect_images(dataset_dir: Path) -> List[Path]:
    return sorted((dataset_dir / "images").glob("*.png"))


def _overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = image.copy()
    yellow = np.array([255, 255, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)

    left_mask = mask == 1
    right_mask = mask == 2
    overlay[left_mask] = (alpha * overlay[left_mask] + (1 - alpha) * yellow).astype(np.uint8)
    overlay[right_mask] = (alpha * overlay[right_mask] + (1 - alpha) * white).astype(np.uint8)
    return overlay


def _prepare_image(image_path: Path, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return image, resized, tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview segmentation model outputs")
    parser.add_argument("--dataset-dir", type=str, default="data/segmentation_dataset")
    parser.add_argument(
        "--checkpoint", type=str, default="data/segmentation_dataset/checkpoints/segnet_best.pt"
    )
    parser.add_argument("--output-dir", type=str, default="tmp/segmentation_inference_preview")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--img-height", type=int, default=320)
    parser.add_argument("--img-width", type=int, default=640)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(dataset_dir)
    if not images:
        raise SystemExit("No images found in dataset.")

    random.seed(args.seed)
    samples = random.sample(images, k=min(args.num_samples, len(images)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSegNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    size = (args.img_width, args.img_height)
    for image_path in samples:
        image, resized, tensor = _prepare_image(image_path, size)
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        overlay = _overlay(resized, pred)
        out_name = image_path.stem + "_pred.png"
        cv2.imwrite(str(output_dir / out_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(samples)} previews to {output_dir}")


if __name__ == "__main__":
    main()
