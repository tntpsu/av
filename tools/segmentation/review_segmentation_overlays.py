"""
Create a small review set for segmentation masks and overlays.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _collect_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    overlays = sorted((dataset_dir / "overlays").glob("*.png"))
    masks_dir = dataset_dir / "masks"
    pairs = []
    for overlay in overlays:
        mask = masks_dir / overlay.name
        if mask.exists():
            pairs.append((overlay, mask))
    return pairs


def _mask_stats(mask_path: Path) -> Tuple[int, int, int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0, 0, 0
    total = int(np.sum(mask > 0))
    left = int(np.sum(mask == 1))
    right = int(np.sum(mask == 2))
    return total, left, right


def _write_index(output_dir: Path, entries: List[Tuple[str, str, int, int, int]]) -> None:
    html_path = output_dir / "index.html"
    rows = []
    for overlay_name, mask_name, total, left, right in entries:
        rows.append(
            f"<tr><td>{overlay_name}</td>"
            f"<td><img src='{overlay_name}' width='320'></td>"
            f"<td><img src='{mask_name}' width='320'></td>"
            f"<td>{total}</td><td>{left}</td><td>{right}</td></tr>"
        )
    html = (
        "<html><body>"
        "<h2>Segmentation Review Set</h2>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>File</th><th>Overlay</th><th>Mask</th>"
        "<th>Total</th><th>Left</th><th>Right</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>"
    )
    html_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample segmentation overlays for review")
    parser.add_argument("--dataset-dir", type=str, default="data/segmentation_dataset")
    parser.add_argument("--output-dir", type=str, default="tmp/segmentation_review")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(dataset_dir)
    if not pairs:
        raise SystemExit("No overlay/mask pairs found. Run dataset prep with overlays enabled.")

    random.seed(args.seed)
    samples = random.sample(pairs, k=min(args.num_samples, len(pairs)))

    entries = []
    for overlay_path, mask_path in samples:
        total, left, right = _mask_stats(mask_path)
        overlay_dest = output_dir / overlay_path.name
        mask_dest = output_dir / f"mask_{mask_path.name}"
        shutil.copy2(overlay_path, overlay_dest)
        shutil.copy2(mask_path, mask_dest)
        entries.append((overlay_dest.name, mask_dest.name, total, left, right))

    _write_index(output_dir, entries)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["overlay", "mask", "total_lane_pixels", "left_lane_pixels", "right_lane_pixels"])
        for overlay_name, mask_name, total, left, right in entries:
            writer.writerow([overlay_name, mask_name, total, left, right])

    print(f"Review set ready: {output_dir}")
    print(f"Open {output_dir / 'index.html'} in a browser to inspect samples.")


if __name__ == "__main__":
    main()
