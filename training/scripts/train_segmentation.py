"""
Train a lightweight lane segmentation model on prepared datasets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class TrainConfig:
    data_dir: Path
    batch_size: int
    epochs: int
    learning_rate: float
    num_workers: int
    img_height: int
    img_width: int
    val_split: float
    device: str


class SegmentationDataset(Dataset):
    def __init__(self, images: List[Path], masks: List[Path], size: Tuple[int, int]) -> None:
        self.images = images
        self.masks = masks
        self.size = size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.masks[idx])
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)

        return torch.from_numpy(image), torch.from_numpy(mask)


class SimpleSegNet(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def _collect_paths(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    images = sorted((data_dir / "images").glob("*.png"))
    masks = sorted((data_dir / "masks").glob("*.png"))
    if len(images) != len(masks):
        raise ValueError("Number of images and masks must match")
    return images, masks


def train(config: TrainConfig) -> Path:
    images, masks = _collect_paths(config.data_dir)
    if not images:
        raise ValueError("No images found. Run prepare_segmentation_dataset.py first.")

    dataset = SegmentationDataset(images, masks, size=(config.img_width, config.img_height))
    val_count = int(len(dataset) * config.val_split)
    train_count = len(dataset) - val_count
    train_set, val_set = random_split(dataset, [train_count, val_count])

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    device = torch.device(config.device)
    model = SimpleSegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    checkpoint_dir = config.data_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "segnet_best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for images_batch, masks_batch in train_loader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

            optimizer.zero_grad()
            logits = model(images_batch)
            loss = criterion(logits, masks_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        val_loss = _evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{config.epochs} - train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)

    return best_path


def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images_batch, masks_batch in loader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)
            logits = model(images_batch)
            loss = criterion(logits, masks_batch)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight lane segmentation model")
    parser.add_argument("--data-dir", type=str, default="data/segmentation_dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-height", type=int, default=320)
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    config = TrainConfig(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        img_height=args.img_height,
        img_width=args.img_width,
        val_split=args.val_split,
        device=args.device,
    )

    best_path = train(config)
    print(f"Training complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
