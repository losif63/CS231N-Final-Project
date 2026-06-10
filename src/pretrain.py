#!/usr/bin/env python3
"""
Trajectory pretraining with SlowFast backbone.

Trains a model to predict 32 future frames of player height given 16 input frames.
Uses a SlowFast backbone with an MLP trajectory head.

Usage:
    conda run -n cv_final_proj python src/pretrain.py \
        --backbone slowfast \
        --epochs 50 \
        --batch-size 32 \
        --learning-rate 1e-3 \
        --save-dir checkpoints/pretrain

Written using Claude Code
"""

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.pretraining_dataset import PretrainingDataset
from losses.trajectory import TrajectoryLoss
from models.trajectory_model import TrajectoryModel
from models.factory import build_backbone


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(backbone_name: str, device):
    """Create trajectory model with backbone."""
    backbone = build_backbone(backbone_name, pretrained=True)
    backbone = backbone.to(device)

    # Probe backbone output dimension
    probe_input = torch.randn(1, 3, 16, 224, 224).to(device)
    with torch.no_grad():
        backbone_out = backbone(probe_input)
        if backbone_out.dim() > 1:
            head_input_dim = backbone_out.view(backbone_out.size(0), -1).shape[1]
        else:
            head_input_dim = backbone_out.shape[-1]

    logger.info(f"Backbone output dimension: {head_input_dim}")

    model = TrajectoryModel(backbone, head_input_dim=head_input_dim)
    model = model.to(device)

    return model


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        frames = batch['frames'].to(device)  # (B, 16, 3, 224, 224)
        heights = batch['heights'].to(device)  # (B, 32)
        occlusion = batch['occlusion'].to(device)  # (B, 32)

        optimizer.zero_grad()

        # Forward pass
        pred_heights = model(frames)  # (B, 32)

        # Compute loss
        loss = loss_fn(pred_heights, heights, occlusion)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            logger.info(f"  Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.6f}")

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    """Validate the model."""
    model.eval()

    total_loss = 0
    num_batches = 0

    for batch in val_loader:
        frames = batch['frames'].to(device)
        heights = batch['heights'].to(device)
        occlusion = batch['occlusion'].to(device)

        pred_heights = model(frames)
        loss = loss_fn(pred_heights, heights, occlusion)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Trajectory pretraining with SlowFast")
    parser.add_argument("--backbone", type=str, default="slowfast",
                       choices=["slowfast", "r2plus1d"],
                       help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                       help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--dataset-path", type=str,
                       default="yolo_player_height/pretraining_dataset.hdf5",
                       help="Path to pretraining dataset")
    parser.add_argument("--save-dir", type=str, default="checkpoints/pretrain",
                       help="Directory to save checkpoints")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="DataLoader workers")

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    # Create checkpoint directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    logger.info(f"Creating {args.backbone} model...")
    model = create_model(args.backbone, device)

    # Create datasets and loaders
    logger.info(f"Loading dataset from {args.dataset_path}...")
    train_dataset = PretrainingDataset(args.dataset_path, split='train', augment=True)
    val_dataset = PretrainingDataset(args.dataset_path, split='val', augment=False)

    # pin_memory not supported on MPS
    use_pin_memory = device.type != "mps"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )

    # Setup loss, optimizer, scheduler
    loss_fn = TrajectoryLoss(beta=0.05, vel_lambda=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    # Warmup + cosine annealing scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress))))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()

        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)

        # Log
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                   f"Time: {epoch_time:.2f}s | "
                   f"Train Loss: {train_loss:.6f} | "
                   f"Val Loss: {val_loss:.6f} | "
                   f"LR: {current_lr:.2e}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }
            checkpoint_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1} "
                       f"(best: epoch {best_epoch}, val_loss: {best_val_loss:.6f})")
            break

    # Save final checkpoint
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
        'args': vars(args),
    }
    final_checkpoint_path = save_dir / 'final_model.pt'
    torch.save(final_checkpoint, final_checkpoint_path)
    logger.info(f"Saved final model to {final_checkpoint_path}")

    # Save history
    history_path = save_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    logger.info(f"\nTraining complete!")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best val loss: {best_val_loss:.6f}")
    logger.info(f"  Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
