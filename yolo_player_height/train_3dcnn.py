#!/usr/bin/env python3
"""
Lightweight 3D CNN for Geometry Dash player height/occlusion/mode prediction.

Uses 8-frame clips with RGB-only input. 3D convolutions learn temporal patterns directly.
No synthetic frame difference needed — real temporal context is enough.

Architecture: Lightweight 3D CNN backbone
  - Input: (B, T=8, C=3, H=620, W=620) uint8 RGB frames
  - 3D conv blocks with spatial and temporal downsampling
  - Global average pooling over spatial and temporal dimensions
  - MLP head: feature_dim → 128 → 3 outputs (height, occlusion, mode)

Outputs:
  player_height     [0, 1]  normalized vertical position in 620×620 image
  player_occlusion  [0, 1]  probability player is not visible
  player_mode       [0, 1]  0=left side (regular), 1=right side (mirror)

Training:
  - Smooth L1 loss (beta=0.25) for height (in-frame samples only)
  - BCE loss for occlusion and mode (all samples)
  - Video-level train/val split (entire videos go to train or val)
  - Robust augmentations applied uniformly across all frames in clip

Usage:
  conda run -n cv_final_proj python train_3dcnn.py \
    --dataset-dir dataset_10000_0 \
    --labels-file dataset_10000_0/labels_4567.json \
    --epochs 20 --batch-size 16
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


# ── Constants ──────────────────────────────────────────────────────────────
DATASET_SIZE  = 660    # stored npy image size
CROP_SIZE     = 620    # model input size
MAX_TRANSLATE = 20     # ±pixels of translation
NUM_FRAMES    = 8      # frames per clip

# ImageNet normalization for RGB
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────
class GDClipDataset(Dataset):
    """
    Loads 8-frame clip npy arrays (preprocessed by preprocess_clips.py).
    Returns (8, 3, 620, 620) tensors with labels from center frame.

    Input npy files: (8, 660, 660, 3) uint8 RGB frames from preprocess_clips.py
    Output tensor: (8, 3, 620, 620) float32 normalized frames
    Targets: [height, occlusion, mode]
    """

    def __init__(self, samples, dataset_dir, augment=False):
        self.samples = samples
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load preprocessed clip npy: (8, 660, 660, 3)
        npy_path = self.dataset_dir / s["filename"]
        clip_array = np.load(npy_path)  # (8, 660, 660, 3) uint8 RGB

        frames = [clip_array[i] for i in range(NUM_FRAMES)]
        label_y = s["label_y"]
        label_x = s["label_x"]
        occlusion = s["occlusion"]

        # ── Determine crop window (same for all frames in clip) ──────────────
        if self.augment:
            scale_h = random.uniform(0.9, 1.1)
            scale_w = random.uniform(0.9, 1.1)
        else:
            scale_h = scale_w = 1.0

        window_h = min(int(CROP_SIZE * scale_h), DATASET_SIZE)
        window_w = min(int(CROP_SIZE * scale_w), DATASET_SIZE)

        cy = DATASET_SIZE // 2
        cx = DATASET_SIZE // 2
        if self.augment:
            cy += random.randint(-MAX_TRANSLATE, MAX_TRANSLATE)
            cx += random.randint(-MAX_TRANSLATE, MAX_TRANSLATE)

        oy = max(0, min(cy - window_h // 2, DATASET_SIZE - window_h))
        ox = max(0, min(cx - window_w // 2, DATASET_SIZE - window_w))

        # ── Crop all frames (same window for temporal consistency) ──
        frames = [f[oy:oy + window_h, ox:ox + window_w, :].copy() for f in frames]
        frames = [cv2.resize(f, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                  if (window_h != CROP_SIZE or window_w != CROP_SIZE) else f
                  for f in frames]

        # ── Transform label into resized 620×620 space ──────────────────────
        if not occlusion:
            crop_y = (label_y - oy) * (CROP_SIZE / window_h)
            crop_x = (label_x - ox) * (CROP_SIZE / window_w)
        else:
            crop_y = CROP_SIZE / 2.0
            crop_x = CROP_SIZE / 2.0

        # ── Apply augmentations uniformly to all frames ────────────────────
        if self.augment:
            # 1. Vertical flip (p=0.3) → update crop_y
            if random.random() < 0.3:
                frames = [f[::-1, :, :].copy() for f in frames]
                crop_y = CROP_SIZE - crop_y - 1

            # 2. Horizontal flip (p=0.5) → update crop_x
            if random.random() < 0.5:
                frames = [f[:, ::-1, :].copy() for f in frames]
                crop_x = CROP_SIZE - crop_x - 1

            # 3. Aspect ratio augmentation (p=0.15, ±5%)
            if random.random() < 0.15:
                axis = random.choice(['horizontal', 'vertical'])
                scale = random.uniform(0.95, 1.05)

                if axis == 'horizontal':
                    new_width = int(CROP_SIZE * scale)
                    new_frames = []
                    for f in frames:
                        resized = cv2.resize(f, (new_width, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                        new_crop = np.zeros_like(f, dtype=np.uint8)
                        if scale < 1.0:
                            left_pad = (CROP_SIZE - new_width) // 2
                            right_pad = CROP_SIZE - new_width - left_pad
                            new_crop[:, left_pad:left_pad+new_width] = resized
                            if random.random() < 0.7:
                                for i in range(left_pad):
                                    new_crop[:, i] = resized[:, 0]
                                for i in range(right_pad):
                                    new_crop[:, CROP_SIZE - right_pad + i] = resized[:, -1]
                            crop_x = crop_x * scale + left_pad
                        else:
                            overflow = new_width - CROP_SIZE
                            left_crop = overflow // 2
                            new_crop = resized[:, left_crop:left_crop+CROP_SIZE]
                            crop_x = crop_x * scale - left_crop
                        new_frames.append(new_crop)
                    frames = new_frames
                else:  # vertical
                    new_height = int(CROP_SIZE * scale)
                    new_frames = []
                    for f in frames:
                        resized = cv2.resize(f, (CROP_SIZE, new_height), interpolation=cv2.INTER_LINEAR)
                        new_crop = np.zeros_like(f, dtype=np.uint8)
                        if scale < 1.0:
                            top_pad = (CROP_SIZE - new_height) // 2
                            bottom_pad = CROP_SIZE - new_height - top_pad
                            new_crop[top_pad:top_pad+new_height] = resized
                            if random.random() < 0.7:
                                for i in range(top_pad):
                                    new_crop[i] = resized[0]
                                for i in range(bottom_pad):
                                    new_crop[CROP_SIZE - bottom_pad + i] = resized[-1]
                            crop_y = crop_y * scale + top_pad
                        else:
                            overflow = new_height - CROP_SIZE
                            top_crop = overflow // 2
                            new_crop = resized[top_crop:top_crop+CROP_SIZE]
                            crop_y = crop_y * scale - top_crop
                        new_frames.append(new_crop)
                    frames = new_frames

            # 4. Horizontal motion blur (p=0.3)
            if random.random() < 0.3:
                kernel_size = random.choice([3, 5, 7])
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, 1))
                frames = [cv2.filter2D(f, -1, kernel) for f in frames]

            # 5. Gaussian blur mild (p=0.2, sigma=1)
            if random.random() < 0.2:
                frames = [cv2.GaussianBlur(f, (3, 3), 1.0) for f in frames]

            # 6. Gaussian blur strong (p=0.15, sigma=3)
            if random.random() < 0.15:
                frames = [cv2.GaussianBlur(f, (5, 5), 3.0) for f in frames]

        # ── Convert to float and normalize (ImageNet) ──────────────────────
        frames = [f.astype(np.float32) / 255.0 for f in frames]
        frames = [(f - IMAGENET_MEAN) / IMAGENET_STD for f in frames]

        # Stack into (T, H, W, C) then transpose to (T, C, H, W)
        frames_array = np.stack(frames, axis=0)  # (8, 620, 620, 3)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)  # (8, 3, 620, 620)

        # ── Compute targets ────────────────────────────────────────────────
        height_norm = crop_y / CROP_SIZE
        mode = float(1.0 if crop_x >= CROP_SIZE / 2.0 else 0.0)

        targets = torch.tensor([height_norm, float(occlusion), mode], dtype=torch.float32)

        return frames_tensor, targets


# ── Model ──────────────────────────────────────────────────────────────────
class Lightweight3DCNN(nn.Module):
    """
    Lightweight 3D CNN for video understanding.

    Input: (B, T, C, H, W) = (batch, 8 frames, 3 RGB, 620, 620)
    Internal: permuted to (B, C, T, H, W) for Conv3d
    Output: (B, 3) logits for height, occlusion, mode
    """

    def __init__(self):
        super().__init__()

        # 3D convolution blocks
        # Input: (B, 3, T=8, 620, 620)

        # Block 1: (3, 8, 620, 620) → (32, 8, 310, 310)
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)

        # Block 2: (32, 8, 310, 310) → (64, 8, 155, 155)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)

        # Block 3: (64, 8, 155, 155) → (128, 8, 77, 77)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)

        # Block 4: (128, 8, 77, 77) → (256, 4, 38, 38) with temporal downsampling
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(256)

        # Global average pooling over T, H, W → (256,)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3),  # height, occlusion, mode
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with reasonable defaults."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Bias initialization for final layer
        with torch.no_grad():
            self.head[-1].bias[0] = 0.0    # height: sigmoid(0) = 0.5
            self.head[-1].bias[1] = -2.0   # occlusion: sigmoid(-2) ≈ 0.12
            self.head[-1].bias[2] = 0.0    # mode: sigmoid(0) = 0.5

    def forward(self, x):
        """
        Forward pass.

        Input: x of shape (B, T, C, H, W)
        """
        # Permute to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)  # (B, 3, 8, 620, 620)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Global average pooling
        x = self.pool(x).flatten(1)  # (B, 256)

        # MLP head
        x = self.head(x)  # (B, 3)

        return x


# ── Training ───────────────────────────────────────────────────────────────
def prepare_data(dataset_dir, labels_file):
    """
    Load preprocessed clip dataset and split into train/val.
    Uses video-level split (all clips from same video go to train or val).
    """
    dataset_dir = Path(dataset_dir)

    # Load metadata and labels from preprocessed dataset
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
            "label_y": float(lbl.get("height_y", DATASET_SIZE / 2)),
            "label_x": float(lbl.get("height_x", DATASET_SIZE / 2)),
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


def compute_loss(height_pred, occlusion_pred, mode_pred, targets, occlusion_labels):
    """
    Compute weighted loss.

    height: Smooth L1 loss, only for non-occluded samples
    occlusion: BCE loss
    mode: BCE loss
    """
    height_target, occlusion_target, mode_target = targets[:, 0], targets[:, 1], targets[:, 2]

    # Height loss: Smooth L1, only for non-occluded
    in_frame_mask = occlusion_target < 0.5
    if in_frame_mask.any():
        height_loss = F.smooth_l1_loss(
            height_pred[in_frame_mask],
            height_target[in_frame_mask],
            beta=0.25
        )
    else:
        height_loss = torch.tensor(0.0, device=height_pred.device)

    # Occlusion loss: BCE (sigmoid already applied)
    occlusion_loss = F.binary_cross_entropy_with_logits(
        occlusion_pred, occlusion_target
    )

    # Mode loss: BCE
    mode_loss = F.binary_cross_entropy_with_logits(
        mode_pred, mode_target
    )

    total_loss = height_loss + occlusion_loss + mode_loss

    return total_loss, {
        'height': height_loss.item(),
        'occlusion': occlusion_loss.item(),
        'mode': mode_loss.item(),
    }


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
    occ_fp = 0
    occ_fn = 0
    mode_correct = 0

    count = 0

    with torch.set_grad_enabled(train):
        for batch_idx, (frames, targets) in enumerate(dataloader):
            frames = frames.to(device)
            targets = targets.to(device)

            # Forward
            logits = model(frames)  # (B, 3)
            height_pred = torch.sigmoid(logits[:, 0])
            occlusion_pred = logits[:, 1]
            mode_pred = logits[:, 2]

            # Loss
            loss, loss_dict = compute_loss(
                logits[:, 0], occlusion_pred, mode_pred,
                targets, targets[:, 1]
            )

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
                height_mae += torch.abs(
                    height_pred[in_frame_mask] - targets[in_frame_mask, 0]
                ).sum().item()

            # Occlusion accuracy and recall
            occ_pred_binary = (torch.sigmoid(occlusion_pred) > 0.5).float()
            occ_target = targets[:, 1]
            occ_correct += (occ_pred_binary == occ_target).sum().item()
            occ_tp += ((occ_pred_binary == 1) & (occ_target == 1)).sum().item()
            occ_fp += ((occ_pred_binary == 1) & (occ_target == 0)).sum().item()
            occ_fn += ((occ_pred_binary == 0) & (occ_target == 1)).sum().item()

            # Mode accuracy
            mode_pred_binary = (torch.sigmoid(mode_pred) > 0.5).float()
            mode_target = targets[:, 2]
            mode_correct += (mode_pred_binary == mode_target).sum().item()

            count += frames.shape[0]

    n_in_frame = max(1, sum(1 for t in dataloader.dataset.samples if t["occlusion"] < 0.5))
    height_mae /= max(1, n_in_frame)

    avg_loss = total_loss / len(dataloader)
    avg_height_loss = total_height_loss / len(dataloader)
    avg_occ_loss = total_occ_loss / len(dataloader)
    avg_mode_loss = total_mode_loss / len(dataloader)

    occ_acc = occ_correct / count
    occ_recall = occ_tp / max(1, occ_tp + occ_fn)
    mode_acc = mode_correct / count

    return {
        'loss': avg_loss,
        'height_loss': avg_height_loss,
        'occ_loss': avg_occ_loss,
        'mode_loss': avg_mode_loss,
        'height_mae': height_mae,
        'occ_acc': occ_acc,
        'occ_recall': occ_recall,
        'mode_acc': mode_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train 3D CNN for GD player prediction")
    parser.add_argument("--dataset-dir", default="dataset_10000_0")
    parser.add_argument("--labels-file", default="dataset_10000_0/labels_4567.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Preparing data...")
    train_samples, val_samples = prepare_data(args.dataset_dir, args.labels_file)

    train_dataset = GDClipDataset(train_samples, args.dataset_dir, augment=True)
    val_dataset = GDClipDataset(val_samples, args.dataset_dir, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Building model...")
    model = Lightweight3DCNN().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create run directory
    run_dir = Path("runs") / f"train_3dcnn_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}/")

    best_val_loss = float('inf')

    print("\n Epoch   Tr-Loss   Tr-H-Loss Tr-Occ-Loss Tr-Mode-Loss Va-Loss   Va-H-MAE  Va-Occ-Acc Va-Occ-Rec Va-Mode-Acc")
    print("─" * 100)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, None, device, train=False)

        scheduler.step()

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }, run_dir / 'best.pt')

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, run_dir / 'last.pt')

        print(f"{epoch:6d}  {train_metrics['loss']:7.4f}  {train_metrics['height_loss']:7.4f}  {train_metrics['occ_loss']:7.4f}  {train_metrics['mode_loss']:7.4f}  "
              f"{val_metrics['loss']:7.4f}  {val_metrics['height_mae']:7.1f}  {val_metrics['occ_acc']:7.1%}  {val_metrics['occ_recall']:7.1%}  {val_metrics['mode_acc']:7.1%}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {run_dir / 'best.pt'} and {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
