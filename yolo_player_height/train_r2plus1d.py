#!/usr/bin/env python3
"""
Fine-tune pretrained r2plus1d_18 on Geometry Dash player prediction.

Uses 5-frame 320×320 clips preprocessed by preprocess_r2plus1d_clips.py.
Loads pretrained r2plus1d_18 from Kinetics-400 and adapts final layer.

Outputs:
  player_height     [0, 1]  normalized vertical position
  player_occlusion  [0, 1]  probability player is not visible
  player_mode       [0, 1]  0=left side (regular), 1=right side (mirror)

Design decisions:
  - Fine-tune all layers (no freezing) — faster convergence with small dataset
  - Low learning rate (1e-4) — preserve pretrained knowledge
  - Dropout in head (0.3) — regularization for 4400 samples
  - Weight decay (1e-4) — prevent overfitting
  - Video-level split — no data leakage
  - Early stopping on val loss — avoid overfitting to small dataset

Usage:
  conda run -n cv_final_proj python train_r2plus1d.py \
    --dataset-dir r2plus1d_dataset_10000_0 \
    --labels-file r2plus1d_dataset_10000_0/labels_4567.json \
    --epochs 50 --batch-size 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.video import r2plus1d_18


# ── Constants ──────────────────────────────────────────────────────────────
CLIP_SIZE = 5
INPUT_RESOLUTION = 320
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────
class R2Plus1DClipDataset(Dataset):
    """
    Loads 5-frame clip npy arrays (preprocessed by preprocess_r2plus1d_clips.py).
    Returns (5, 3, 320, 320) tensors with labels from center frame.
    """

    def __init__(self, samples, dataset_dir, augment=False, enable_blur=True):
        self.samples = samples
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment
        self.enable_blur = enable_blur

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load preprocessed clip npy: (5, 320, 320, 3) uint8 RGB
        npy_path = self.dataset_dir / s["filename"]
        clip_array = np.load(npy_path)  # (5, 320, 320, 3)

        frames = [clip_array[i].copy() for i in range(CLIP_SIZE)]
        label_y = s["label_y"]
        label_x = s["label_x"]
        occlusion = s["occlusion"]

        # ── Augmentations (applied uniformly to all frames) ───────────────
        if self.augment:
            # 1. Horizontal flip (p=0.5)
            if random.random() < 0.5:
                frames = [f[:, ::-1, :].copy() for f in frames]
                label_x = INPUT_RESOLUTION - label_x - 1

            # 2. Vertical flip (p=0.3)
            if random.random() < 0.3:
                frames = [f[::-1, :, :].copy() for f in frames]
                label_y = INPUT_RESOLUTION - label_y - 1

            # 3. Gaussian blur mild (p=0.2, sigma=1)
            if self.enable_blur and random.random() < 0.2:
                frames = [cv2.GaussianBlur(f, (3, 3), 1.0) for f in frames]

            # 4. Gaussian blur strong (p=0.15, sigma=3)
            if self.enable_blur and random.random() < 0.15:
                frames = [cv2.GaussianBlur(f, (5, 5), 3.0) for f in frames]

        # ── Normalize to float32 ──────────────────────────────────────────
        frames = [f.astype(np.float32) / 255.0 for f in frames]
        frames = [(f - IMAGENET_MEAN) / IMAGENET_STD for f in frames]

        # Stack into (T, H, W, C) then transpose to (T, C, H, W)
        frames_array = np.stack(frames, axis=0)  # (5, 320, 320, 3)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)  # (5, 3, 320, 320)

        # ── Compute targets ────────────────────────────────────────────────
        height_norm = label_y / INPUT_RESOLUTION
        mode = float(1.0 if label_x >= INPUT_RESOLUTION / 2.0 else 0.0)

        targets = torch.tensor([height_norm, float(occlusion), mode], dtype=torch.float32)

        return frames_tensor, targets


# ── Model ──────────────────────────────────────────────────────────────────
class R2Plus1DPlayer(nn.Module):
    """
    Pretrained r2plus1d_18 adapted for player prediction.

    Takes 5-frame clips and outputs height, occlusion, mode.
    Replaces final classification head with custom regression head.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained r2plus1d_18 from Kinetics-400
        self.backbone = r2plus1d_18(pretrained=True)

        # Get feature dimension from backbone
        in_features = self.backbone.fc.in_features

        # Replace classification head with regression head
        self.backbone.fc = nn.Identity()  # Remove fc layer, we'll add custom head

        # Custom head: 512 → 128 → 3
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

        # Initialize head weights
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Bias initialization for final output layer
        with torch.no_grad():
            self.head[-1].bias[0] = 0.0    # height: sigmoid(0) = 0.5
            self.head[-1].bias[1] = -2.0   # occlusion: sigmoid(-2) ≈ 0.12
            self.head[-1].bias[2] = 0.0    # mode: sigmoid(0) = 0.5

    def forward(self, x):
        """
        Forward pass.

        Input: x of shape (B, T, C, H, W) = (B, 5, 3, 320, 320)
        Permute to (B, C, T, H, W) for r2plus1d model
        """
        # r2plus1d expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, 3, 5, 320, 320)

        # Backbone: (B, 3, 5, 320, 320) → (B, 512)
        features = self.backbone.stem(x)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        features = self.backbone.avgpool(features)
        features = features.flatten(1)  # (B, 512)

        # Custom head: (B, 512) → (B, 3)
        logits = self.head(features)

        return logits


# ── Loss ───────────────────────────────────────────────────────────────────
def compute_loss(logits, targets):
    """
    Compute weighted loss.

    height: Smooth L1 loss, only for non-occluded samples
    occlusion: BCE loss
    mode: BCE loss
    """
    height_logit = logits[:, 0]
    occ_logit = logits[:, 1]
    mode_logit = logits[:, 2]

    height_target = targets[:, 0]
    occ_target = targets[:, 1]
    mode_target = targets[:, 2]

    # Height loss: Smooth L1, only for in-frame samples
    in_frame_mask = occ_target < 0.5
    if in_frame_mask.any():
        height_pred = torch.sigmoid(height_logit)
        height_loss = F.smooth_l1_loss(
            height_pred[in_frame_mask],
            height_target[in_frame_mask],
            beta=0.25
        )
    else:
        height_loss = torch.tensor(0.0, device=logits.device)

    # Occlusion loss: BCE
    occ_loss = F.binary_cross_entropy_with_logits(occ_logit, occ_target)

    # Mode loss: BCE
    mode_loss = F.binary_cross_entropy_with_logits(mode_logit, mode_target)

    total_loss = height_loss + occ_loss + mode_loss

    return total_loss, {
        'height': height_loss.item() if isinstance(height_loss, torch.Tensor) else height_loss,
        'occlusion': occ_loss.item(),
        'mode': mode_loss.item(),
    }


# ── Data preparation ───────────────────────────────────────────────────────
def prepare_data(dataset_dir, labels_file):
    """
    Load preprocessed clip dataset and split into train/val.
    Uses video-level split.
    """
    dataset_dir = Path(dataset_dir)

    with open(dataset_dir / "metadata.json") as f:
        metadata = json.load(f)

    with open(labels_file) as f:
        raw_labels = json.load(f)

    # Build sample list with labels
    all_samples = []
    for meta in metadata:
        clip_id = str(meta["sample_id"])

        if clip_id not in raw_labels:
            continue

        lbl = raw_labels[clip_id]

        if lbl.get("removed", False):
            continue

        not_in_frame = lbl.get("not_in_frame", False)
        has_height = "height_y" in lbl

        if not has_height and not not_in_frame:
            continue

        sample = {
            "sample_id": meta["sample_id"],
            "filename": meta["filename"],
            "video_path": meta.get("video_path", "unknown"),
            "label_y": float(lbl.get("height_y", INPUT_RESOLUTION / 2)),
            "label_x": float(lbl.get("height_x", INPUT_RESOLUTION / 2)),
            "occlusion": 1.0 if not_in_frame else 0.0,
        }
        all_samples.append(sample)

    # Group by video for video-level split
    video_samples = {}
    for sample in all_samples:
        video_path = sample["video_path"]
        if video_path not in video_samples:
            video_samples[video_path] = []
        video_samples[video_path].append(sample)

    # Video-level random split
    video_paths = list(video_samples.keys())
    random.shuffle(video_paths)
    split_idx = int(0.9 * len(video_paths))
    train_videos = set(video_paths[:split_idx])
    val_videos = set(video_paths[split_idx:])

    train_samples = []
    val_samples = []

    for video_path, samples in video_samples.items():
        if video_path in train_videos:
            train_samples.extend(samples)
        else:
            val_samples.extend(samples)

    print(f"Prepared data:")
    print(f"  Videos: {len(video_paths)} total ({len(train_videos)} train, {len(val_videos)} val)")
    print(f"  Clips: {len(train_samples)} train, {len(val_samples)} val")

    return train_samples, val_samples


# ── Training loop ──────────────────────────────────────────────────────────
def run_epoch(model, dataloader, optimizer, device, train=True):
    """Run one epoch of training or validation."""
    model.train(train)

    total_loss = 0.0
    total_height_loss = 0.0
    total_occ_loss = 0.0
    total_mode_loss = 0.0

    height_mae = 0.0
    occ_correct = 0
    occ_tp = 0
    occ_fn = 0
    mode_correct = 0

    count = 0
    n_in_frame = 0

    with torch.set_grad_enabled(train):
        for frames, targets in dataloader:
            frames = frames.to(device)
            targets = targets.to(device)

            # Forward
            logits = model(frames)

            # Loss
            loss, loss_dict = compute_loss(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_height_loss += loss_dict['height']
            total_occ_loss += loss_dict['occlusion']
            total_mode_loss += loss_dict['mode']

            # Height MAE (only for non-occluded)
            in_frame_mask = targets[:, 1] < 0.5
            if in_frame_mask.any():
                height_pred = torch.sigmoid(logits[in_frame_mask, 0])
                height_mae += torch.abs(height_pred - targets[in_frame_mask, 0]).sum().item()
                n_in_frame += in_frame_mask.sum().item()

            # Occlusion accuracy and recall
            occ_pred_binary = (torch.sigmoid(logits[:, 1]) > 0.5).float()
            occ_target = targets[:, 1]
            occ_correct += (occ_pred_binary == occ_target).sum().item()
            occ_tp += ((occ_pred_binary == 1) & (occ_target == 1)).sum().item()
            occ_fn += ((occ_pred_binary == 0) & (occ_target == 1)).sum().item()

            # Mode accuracy
            mode_pred_binary = (torch.sigmoid(logits[:, 2]) > 0.5).float()
            mode_target = targets[:, 2]
            mode_correct += (mode_pred_binary == mode_target).sum().item()

            count += frames.shape[0]

    if n_in_frame > 0:
        height_mae /= n_in_frame
    else:
        height_mae = 0.0

    avg_loss = total_loss / len(dataloader)
    occ_acc = occ_correct / count
    occ_recall = occ_tp / max(1, occ_tp + occ_fn)
    mode_acc = mode_correct / count

    return {
        'loss': avg_loss,
        'height_loss': total_height_loss / len(dataloader),
        'occ_loss': total_occ_loss / len(dataloader),
        'mode_loss': total_mode_loss / len(dataloader),
        'height_mae': height_mae,
        'occ_acc': occ_acc,
        'occ_recall': occ_recall,
        'mode_acc': mode_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune r2plus1d_18 for GD player prediction")
    parser.add_argument("--dataset-dir", default="r2plus1d_dataset_10000_0")
    parser.add_argument("--labels-file", default="r2plus1d_dataset_10000_0/labels_4567.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-blur", action="store_true", help="Disable Gaussian blur augmentations")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Preparing data...")
    train_samples, val_samples = prepare_data(args.dataset_dir, args.labels_file)

    train_dataset = R2Plus1DClipDataset(
        train_samples, args.dataset_dir, augment=True, enable_blur=not args.no_blur
    )
    val_dataset = R2Plus1DClipDataset(
        val_samples, args.dataset_dir, augment=False, enable_blur=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Building model...")
    model = R2Plus1DPlayer().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create run directory
    run_dir = Path("runs") / f"train_r2plus1d_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}/")

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    print("\n Epoch   Tr-Loss   Tr-H-Loss Tr-Occ-Loss Tr-Mode-Loss Va-Loss   Va-H-MAE  Va-Occ-Acc Va-Occ-Rec Va-Mode-Acc")
    print("─" * 110)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, None, device, train=False)

        scheduler.step()

        # Check early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }, run_dir / 'best.pt')
        else:
            patience_counter += 1

        # Save last checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, run_dir / 'last.pt')

        print(f"{epoch:6d}  {train_metrics['loss']:7.4f}  {train_metrics['height_loss']:7.4f}  "
              f"{train_metrics['occ_loss']:7.4f}  {train_metrics['mode_loss']:7.4f}  "
              f"{val_metrics['loss']:7.4f}  {val_metrics['height_mae']:7.1f}  "
              f"{val_metrics['occ_acc']:7.1%}  {val_metrics['occ_recall']:7.1%}  {val_metrics['mode_acc']:7.1%}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {run_dir / 'best.pt'} and {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
