"""
Training script for lane detection model.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from perception.models.lane_detection import LaneDetectionModel
from perception.training.dataset import LaneDetectionDataset


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion_cls: nn.Module,
                criterion_exist: nn.Module, optimizer: optim.Optimizer, 
                device: torch.device, epoch: int) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_exist_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, cls_targets, exist_targets) in enumerate(pbar):
        images = images.to(device)
        cls_targets = cls_targets.to(device)
        exist_targets = exist_targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        cls_logits, exist_logits = model(images)
        
        # Calculate losses
        # Classification loss (cross-entropy)
        cls_loss = criterion_cls(cls_logits.view(-1, cls_logits.size(-1)), 
                                 cls_targets.view(-1))
        
        # Existence loss (binary cross-entropy)
        exist_loss = criterion_exist(exist_logits, exist_targets)
        
        # Total loss
        loss = cls_loss + exist_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_exist_loss += exist_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'exist': f'{exist_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'exist_loss': total_exist_loss / len(dataloader)
    }


def validate(model: nn.Module, dataloader: DataLoader, criterion_cls: nn.Module,
             criterion_exist: nn.Module, device: torch.device) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_exist_loss = 0.0
    
    with torch.no_grad():
        for images, cls_targets, exist_targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            cls_targets = cls_targets.to(device)
            exist_targets = exist_targets.to(device)
            
            # Forward pass
            cls_logits, exist_logits = model(images)
            
            # Calculate losses
            cls_loss = criterion_cls(cls_logits.view(-1, cls_logits.size(-1)), 
                                    cls_targets.view(-1))
            exist_loss = criterion_exist(exist_logits, exist_targets)
            loss = cls_loss + exist_loss
            
            # Update metrics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_exist_loss += exist_loss.item()
    
    return {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'exist_loss': total_exist_loss / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(description='Train lane detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    
    # Dataset and dataloader
    print("Loading dataset...")
    train_dataset = LaneDetectionDataset(args.data_dir, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # For now, use same dataset for validation (should be split)
    val_dataset = LaneDetectionDataset(args.data_dir, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model
    model = LaneDetectionModel()
    model.to(device)
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_exist = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion_cls, criterion_exist,
            optimizer, device, epoch + 1
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion_cls, criterion_exist, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Train/ClsLoss', train_metrics['cls_loss'], epoch)
        writer.add_scalar('Train/ExistLoss', train_metrics['exist_loss'], epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/ClsLoss', val_metrics['cls_loss'], epoch)
        writer.add_scalar('Val/ExistLoss', val_metrics['exist_loss'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss']
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pth')
        
        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(checkpoint, output_dir / 'best.pth')
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")
    
    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

