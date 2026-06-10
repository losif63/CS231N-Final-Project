#!/usr/bin/env python3
"""
DETECTION BACKBONE variant: Uses YOLOv8n-det backbone instead of classification.

Compared to train_robust_occlusions_mlp.py, uses:
  - YOLOv8n detection backbone (richer multi-scale features) instead of classification
  - Global average pooling to aggregate features
  - Same MLP regression head for height + occlusion prediction

Architecture: YOLOv8n-det backbone + custom MLP head:
  - 4-channel input (BGR channels + greyscale frame diff as channel 4)
  - Detection backbone extracts multi-scale features
  - MLP head: feature_vector → 128 → 2 (height, occlusion logits)

Outputs:
  player_height   [0, 1]  normalized vertical position in 620×620 image
  player_occlusion [0, 1]  probability player is not visible

Usage:
  python train_robust_occlusions_mlp_detection_bb.py \
    --dataset-dir dataset_10000_0 \
    --labels-file dataset_10000_0/labels_4567.json

Written using Claude Code
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
MAX_TRANSLATE = 20     # ±pixels of translation (660 - 620 = 40 → ±20)

# ImageNet normalization in RGB order; applied after converting BGR→RGB
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Diff channel: empirical rough normalization (motion is usually small)
DIFF_MEAN = 0.05
DIFF_STD  = 0.15


# ── Dataset ────────────────────────────────────────────────────────────────
class GDDataset(Dataset):
    """
    Loads 660×660 4-channel numpy arrays, crops/augments to 620×620,
    and returns (image_tensor [4, 620, 620], targets [2]).

    targets[0] = player_height   in [0, 1] normalized to 620-px image height
    targets[1] = player_occlusion  0.0 (in frame) or 1.0 (not in frame)

    Coordinate notes:
      - label_y / label_x are pixel coords in the 660×660 stored image.
      - Default crop: center 620×620, i.e. offset (20, 20).
      - After translation/scale augmentation, coords are transformed accordingly.
    """

    def __init__(self, samples, augment=False):
        self.samples = samples  # list of dicts; see prepare_data()
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        arr = np.load(s["npy_path"])    # (660, 660, 4) uint8
        label_y  = s["label_y"]        # float, pixel row in 660×660 space
        label_x  = s["label_x"]        # float, pixel col in 660×660 space
        occlusion = s["occlusion"]     # 0.0 or 1.0

        # ── Determine crop window in 660×660 source space ─────────────────
        # window_h / window_w: the region we'll crop then resize to 620×620.
        # Scale augmentation: independently vary h and w by ±10%.
        if self.augment:
            scale_h = random.uniform(0.9, 1.1)
            scale_w = random.uniform(0.9, 1.1)
        else:
            scale_h = scale_w = 1.0

        window_h = min(int(CROP_SIZE * scale_h), DATASET_SIZE)
        window_w = min(int(CROP_SIZE * scale_w), DATASET_SIZE)

        # Center of window, with optional translation
        cy = DATASET_SIZE // 2
        cx = DATASET_SIZE // 2
        if self.augment:
            cy += random.randint(-MAX_TRANSLATE, MAX_TRANSLATE)
            cx += random.randint(-MAX_TRANSLATE, MAX_TRANSLATE)

        # Top-left corner, clamped so window fits within 660×660
        oy = max(0, min(cy - window_h // 2, DATASET_SIZE - window_h))
        ox = max(0, min(cx - window_w // 2, DATASET_SIZE - window_w))

        # Crop
        crop = arr[oy:oy + window_h, ox:ox + window_w].copy()  # (wh, ww, 4)

        # Resize to 620×620 (no-op when window is already 620×620)
        if window_h != CROP_SIZE or window_w != CROP_SIZE:
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)

        # Transform label into resized 620×620 space
        if not occlusion:
            crop_y = (label_y - oy) * (CROP_SIZE / window_h)
            crop_x = (label_x - ox) * (CROP_SIZE / window_w)
        else:
            # Placeholder — masked out in loss, value doesn't matter
            crop_y = CROP_SIZE / 2.0
            crop_x = CROP_SIZE / 2.0

        # ── Robust augmentations (uint8, all channels, spatial first) ─────
        if self.augment:
            # 1. Vertical flip (p=0.3) → update crop_y
            if random.random() < 0.3:
                crop = crop[::-1, :, :].copy()
                crop_y = CROP_SIZE - crop_y - 1

            # 2. Horizontal flip (p=0.3) → update crop_x
            if random.random() < 0.5:
                crop  = crop[:, ::-1, :].copy()
                crop_x = CROP_SIZE - crop_x - 1

            # 2a. Aspect ratio augmentation (p=0.15, ±5% stretch/shrink on one axis)
            # Expands robustness to different video aspect ratios
            if random.random() < 0.15:
                axis = random.choice(['horizontal', 'vertical'])
                scale = random.uniform(0.95, 1.05)  # ±5%

                if axis == 'horizontal':
                    new_width = int(CROP_SIZE * scale)
                    resized = cv2.resize(crop, (new_width, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                    new_crop = np.zeros_like(crop, dtype=np.uint8)

                    if scale < 1.0:  # Shrunk - fill sides
                        left_pad = (CROP_SIZE - new_width) // 2
                        right_pad = CROP_SIZE - new_width - left_pad
                        new_crop[:, left_pad:left_pad+new_width] = resized

                        # Fill gaps: copy edges or interpolate (70% copy, 30% interpolate)
                        if random.random() < 0.7:
                            for i in range(left_pad):
                                new_crop[:, i] = resized[:, 0]
                            for i in range(right_pad):
                                new_crop[:, CROP_SIZE - right_pad + i] = resized[:, -1]
                        else:
                            # Fade from edge toward neutral gray
                            for i in range(left_pad):
                                alpha = (left_pad - i) / max(1, left_pad)
                                new_crop[:, i] = (resized[:, 0].astype(np.float32) * alpha + 128 * (1-alpha)).astype(np.uint8)
                            for i in range(right_pad):
                                alpha = i / max(1, right_pad)
                                new_crop[:, CROP_SIZE - right_pad + i] = (resized[:, -1].astype(np.float32) * alpha + 128 * (1-alpha)).astype(np.uint8)

                        crop_x = crop_x * scale + left_pad
                    else:  # Stretched - crop to fit
                        overflow = new_width - CROP_SIZE
                        left_crop = overflow // 2
                        new_crop = resized[:, left_crop:left_crop+CROP_SIZE]
                        crop_x = crop_x * scale - left_crop

                else:  # vertical axis
                    new_height = int(CROP_SIZE * scale)
                    resized = cv2.resize(crop, (CROP_SIZE, new_height), interpolation=cv2.INTER_LINEAR)
                    new_crop = np.zeros_like(crop, dtype=np.uint8)

                    if scale < 1.0:  # Shrunk - fill top/bottom
                        top_pad = (CROP_SIZE - new_height) // 2
                        bottom_pad = CROP_SIZE - new_height - top_pad
                        new_crop[top_pad:top_pad+new_height] = resized

                        # Fill gaps: copy edges or interpolate
                        if random.random() < 0.7:
                            for i in range(top_pad):
                                new_crop[i] = resized[0]
                            for i in range(bottom_pad):
                                new_crop[CROP_SIZE - bottom_pad + i] = resized[-1]
                        else:
                            for i in range(top_pad):
                                alpha = (top_pad - i) / max(1, top_pad)
                                new_crop[i] = (resized[0].astype(np.float32) * alpha + 128 * (1-alpha)).astype(np.uint8)
                            for i in range(bottom_pad):
                                alpha = i / max(1, bottom_pad)
                                new_crop[CROP_SIZE - bottom_pad + i] = (resized[-1].astype(np.float32) * alpha + 128 * (1-alpha)).astype(np.uint8)

                        crop_y = crop_y * scale + top_pad
                    else:  # Stretched - crop to fit
                        overflow = new_height - CROP_SIZE
                        top_crop = overflow // 2
                        new_crop = resized[top_crop:top_crop+CROP_SIZE]
                        crop_y = crop_y * scale - top_crop

                crop = new_crop

            # 2b. Aggressive vertical shift (p=0.2, expand height distribution 0→1)
            # Moves player from top to bottom of frame to increase height diversity
            if not occlusion and random.random() < 0.2:
                PLAYER_BUFFER = 70        # Safe distance from player (±70px, don't copy within this)
                PLAYER_EDGE_MARGIN = 50   # Keep player at least 50px from crop edges

                # Calculate shift range to move player across full frame height
                max_shift_up = -(crop_y - PLAYER_EDGE_MARGIN)
                max_shift_down = (CROP_SIZE - PLAYER_EDGE_MARGIN) - crop_y

                shift_y = int(random.uniform(max_shift_up, max_shift_down))

                if abs(shift_y) >= 10:  # Only apply if shift is meaningful (≥10px)
                    # Create new crop with shifted content
                    new_crop = np.zeros_like(crop, dtype=np.uint8)

                    if shift_y > 0:  # Shift down, need to fill top
                        new_crop[shift_y:] = crop[:-shift_y]
                        blank_start, blank_end = 0, shift_y
                    else:  # Shift up, need to fill bottom
                        new_crop[:CROP_SIZE + shift_y] = crop[-shift_y:]
                        blank_start, blank_end = CROP_SIZE + shift_y, CROP_SIZE

                    # Fill blank region using one of two methods
                    if random.random() < 0.7:  # 70% copy padding, 30% edge interpolation
                        # Method 1: Copy from safe rows (>70px away from player)
                        safe_rows = [i for i in range(CROP_SIZE)
                                   if abs(i - crop_y) > PLAYER_BUFFER]

                        if safe_rows:
                            # Pick source row from appropriate end
                            if shift_y > 0:
                                # Filling top - prefer rows from upper safe region
                                upper_safe = [r for r in safe_rows if r < crop_y - PLAYER_BUFFER]
                                source_row = upper_safe[-1] if upper_safe else safe_rows[0]
                            else:
                                # Filling bottom - prefer rows from lower safe region
                                lower_safe = [r for r in safe_rows if r > crop_y + PLAYER_BUFFER]
                                source_row = lower_safe[0] if lower_safe else safe_rows[-1]

                            # Tile the safe row to fill blank area (for all 4 channels)
                            for i in range(blank_start, blank_end):
                                new_crop[i] = crop[source_row]
                        else:
                            # No safe rows available (shouldn't happen unless player is huge)
                            # Fallback: extend from nearest content edge
                            if shift_y > 0:
                                edge_row = new_crop[shift_y]
                            else:
                                edge_row = new_crop[CROP_SIZE + shift_y - 1]
                            for i in range(blank_start, blank_end):
                                new_crop[i] = edge_row
                    else:
                        # Method 2: Edge interpolation with fade (more varied/synthetic)
                        if shift_y > 0:
                            # Fill top by fading from content edge
                            edge = new_crop[shift_y].astype(np.float32)
                            for i in range(blank_start, blank_end):
                                # Fade toward neutral gray (128) as we go further from edge
                                alpha = (i - blank_start) / max(1, blank_end - blank_start)
                                new_crop[i] = (edge * (1 - alpha * 0.15) + 128 * alpha * 0.15).astype(np.uint8)
                        else:
                            # Fill bottom by fading from content edge
                            edge = new_crop[CROP_SIZE + shift_y - 1].astype(np.float32)
                            for i in range(blank_start, blank_end):
                                alpha = (i - blank_start) / max(1, blank_end - blank_start)
                                new_crop[i] = (edge * (1 - alpha * 0.15) + 128 * alpha * 0.15).astype(np.uint8)

                    crop = new_crop
                    crop_y = crop_y + shift_y  # Update label position

            # 3. Largest-rectangle crop → square sample (p=0.1, skip if already occluded)
            if not occlusion and random.random() < 0.25:
                HALF = 55
                cy_i, cx_i = int(crop_y), int(crop_x)
                rects = [
                    (0,                           max(0, cy_i - HALF),               0,                            CROP_SIZE),
                    (min(CROP_SIZE, cy_i + HALF), CROP_SIZE,                         0,                            CROP_SIZE),
                    (0,                           CROP_SIZE,                          0,                            max(0, cx_i - HALF)),
                    (0,                           CROP_SIZE,                          min(CROP_SIZE, cx_i + HALF),  CROP_SIZE),
                ]
                best = max(rects, key=lambda r: (r[1]-r[0]) * (r[3]-r[2]))
                ry0, ry1, rx0, rx1 = best
                rh, rw = ry1 - ry0, rx1 - rx0
                # Only apply if rectangle is large enough for meaningful crop (min 50×50)
                if rh > 50 and rw > 50:
                    side = min(rh, rw)
                    y_off = random.randint(0, max(0, rh - side))
                    x_off = random.randint(0, max(0, rw - side))
                    sq = crop[ry0+y_off : ry0+y_off+side, rx0+x_off : rx0+x_off+side]
                    if sq.shape[0] > 0 and sq.shape[1] > 0:  # Safety check
                        crop = cv2.resize(sq, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                        occlusion = 1.0

            # 4. Downscale-upscale (p=0.1)
            if random.random() < 0.25:
                k = random.randint(310, 500)
                small = cv2.resize(crop, (k, k), interpolation=cv2.INTER_LINEAR)
                crop = cv2.resize(small, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)

            # 5. Synthetic player occlusion patch (p=0.2, skip if already occluded)
            if not occlusion and random.random() < 0.2:
                patch_size = 150
                cy_int, cx_int = int(crop_y), int(crop_x)

                # Find a valid source patch that doesn't overlap the player
                max_tries = 10
                for _ in range(max_tries):
                    src_y = random.randint(0, CROP_SIZE - patch_size)
                    src_x = random.randint(0, CROP_SIZE - patch_size)

                    # Avoid source overlapping player (±patch_size/2 around player)
                    if (abs(src_y + patch_size // 2 - cy_int) > patch_size and
                        abs(src_x + patch_size // 2 - cx_int) > patch_size):
                        break

                # Copy patch to cover player (all 4 channels)
                dst_y = max(0, min(cy_int - patch_size // 2, CROP_SIZE - patch_size))
                dst_x = max(0, min(cx_int - patch_size // 2, CROP_SIZE - patch_size))
                crop[dst_y:dst_y+patch_size, dst_x:dst_x+patch_size] = \
                    crop[src_y:src_y+patch_size, src_x:src_x+patch_size].copy()
                occlusion = 1.0

            # 6. Horizontal motion blur (p=0.2)
            if random.random() < 0.2:
                kernel_size = random.choice([3, 5])
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
                for c in range(4):
                    crop[:, :, c] = cv2.filter2D(crop[:, :, c], -1, kernel)

            # 7. Gaussian blur mild (p=0.3): σ 0.5–1.5, kernel 3×3
            if random.random() < 0.3:
                sigma = random.uniform(0.5, 1.5)
                for c in range(4):
                    crop[:, :, c] = cv2.GaussianBlur(crop[:, :, c], (3, 3), sigma)

            # 8. Gaussian blur strong (p=0.15): σ 1.5–3.0, kernel 5×5
            if random.random() < 0.15:
                sigma = random.uniform(1.5, 3.0)
                for c in range(4):
                    crop[:, :, c] = cv2.GaussianBlur(crop[:, :, c], (5, 5), sigma)

        # ── Convert to float [0, 1] ───────────────────────────────────────
        bgr = crop[:, :, :3].astype(np.float32) / 255.0
        rgb = bgr[:, :, ::-1].astype(np.float32)                 # BGR → RGB
        diff = crop[:, :, 3].astype(np.float32) / 255.0

        # ── Robust augmentations (float32, before normalization) ───────────
        if self.augment:
            # 9. Brightness scaling (p=0.4): α ∈ [0.7, 1.3]
            if random.random() < 0.4:
                alpha = random.uniform(0.7, 1.3)
                rgb = np.clip(rgb * alpha, 0.0, 1.0).astype(np.float32)
                diff = np.clip(diff * alpha, 0.0, 1.0).astype(np.float32)

            # 10. Vignette gradient (p=0.25): radial G = 1 - strength * r_norm
            if random.random() < 0.25:
                strength = random.uniform(0.0, 0.3)
                yy, xx = np.ogrid[:CROP_SIZE, :CROP_SIZE]
                yy = yy.astype(np.float32)
                xx = xx.astype(np.float32)
                cy, cx = np.float32(CROP_SIZE / 2.0), np.float32(CROP_SIZE / 2.0)
                r_sq = ((yy - cy)**2 + (xx - cx)**2)
                r_norm = np.sqrt(r_sq) / np.sqrt((cy**2 + cx**2))
                G = np.float32(1.0) - strength * r_norm
                G = np.clip(G, 0.0, 1.0).astype(np.float32)
                rgb = (rgb * G[:, :, np.newaxis]).astype(np.float32)
                diff = (diff * G).astype(np.float32)

            # 11. Hue rotation (p=0.3): ±15° in HSV H channel
            if random.random() < 0.3:
                hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                h_shift = random.uniform(-8, 8)  # OpenCV hue is 0-180, so ±8 ≈ ±15°
                hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

            # 12. Gaussian noise (p=0.3): RGB only
            if random.random() < 0.3:
                noise_std = random.uniform(0.01, 0.04)
                noise = np.random.normal(0, noise_std, rgb.shape).astype(np.float32)
                rgb = np.clip(rgb + noise, 0.0, 1.0).astype(np.float32)

        # ── ImageNet normalization ──────────────────────────────────────────
        rgb = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        diff = ((diff - DIFF_MEAN) / DIFF_STD).astype(np.float32)

        # Stack to (4, H, W)
        tensor = np.concatenate([
            rgb.transpose(2, 0, 1),    # (3, H, W)
            diff[np.newaxis],          # (1, H, W)
        ], axis=0)
        tensor = torch.from_numpy(tensor.copy())

        # Normalised height in [0, 1] relative to 620-px image height
        height_norm = float(np.clip(crop_y / CROP_SIZE, 0.0, 1.0))

        # Mode: 0 = left side (regular), 1 = right side (mirrored)
        # Based on which half of the frame the x label falls in
        mode = float(1.0 if crop_x >= CROP_SIZE / 2.0 else 0.0)

        targets = torch.tensor([height_norm, float(occlusion), mode], dtype=torch.float32)

        return tensor, targets


# ── Model ──────────────────────────────────────────────────────────────────
class GDPlayerModel(nn.Module):
    """
    YOLOv8n detection backbone adapted for 4-channel input and regression output.

    Uses YOLOv8n-det backbone + neck (richer features than classification) with custom
    MLP head. The detection backbone outputs multi-scale feature maps which are globally
    pooled before the regression head.

    4th-channel weight initialisation:
      The first Conv2d is extended from 3→4 input channels.
      Channels 0-2 keep their ImageNet-pretrained weights unchanged.
      Channel 3 (diff) is initialised as the mean of channels 0-2, which
      preserves activation scale and gives network a reasonable starting point for motion.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        from ultralytics import YOLO
        import os

        # Try to load YOLOv8n detection model
        # If pretrained weights don't exist, use classification backbone as fallback
        yolo_path = "yolov8n-det.pt"
        model_det = None

        try:
            if pretrained and os.path.exists(yolo_path):
                yolo = YOLO(yolo_path)
                model_det = yolo.model
        except Exception:
            pass

        # Fallback: Use classification model if detection model unavailable
        if model_det is None:
            yolo = YOLO("yolov8n-cls.pt")
            model_det = yolo.model

        # If not using pretrained weights, reinitialize backbone weights
        if not pretrained:
            for m in model_det.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

        # Unfreeze all backbone parameters — ultralytics loads with requires_grad=False
        for p in model_det.parameters():
            p.requires_grad_(True)

        # ── 1. Extend first conv: 3 → 4 input channels ───────────────────
        first_conv = model_det.model[0].conv   # Conv2d(3, 16, 3, 2, 1, bias=False)
        out_c = first_conv.out_channels

        new_conv = nn.Conv2d(
            4, out_c,
            first_conv.kernel_size,
            first_conv.stride,
            first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight.data            # copy RGB weights
            new_conv.weight[:, 3]  = first_conv.weight.data.mean(dim=1) # 4th = mean(RGB)
            if first_conv.bias is not None:
                new_conv.bias.data.copy_(first_conv.bias.data)

        model_det.model[0].conv = new_conv
        self.first_conv = new_conv  # Store reference for optimizer

        # ── 2. Keep backbone only (skip neck which expects multi-scale input) ────────────────
        # YOLOv8n-det structure: backbone (0-9) outputs single feature → neck (10-12) expects list of features → head (13+)
        # We use only the backbone since the neck expects multi-scale features from intermediate layers
        self.backbone_neck = nn.ModuleList()
        self.feature_dim = None

        # Keep backbone layers only (0-9), not neck or head
        for layer in model_det.model[:10]:
            self.backbone_neck.append(layer)

        # Test forward pass to get feature dimension after pooling
        with torch.no_grad():
            test_x = torch.randn(1, 4, 640, 640)
            test_features = self._get_features(test_x)
            # Global average pooling on multi-scale features
            if isinstance(test_features, (list, tuple)):
                # Detection backbone outputs list of multi-scale features
                # Pool each and concatenate
                pooled = []
                for feat in test_features:
                    pool = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                    pooled.append(pool)
                feature_vector = torch.cat(pooled, dim=1)
            else:
                # Single feature map
                feature_vector = F.adaptive_avg_pool2d(test_features, 1).flatten(1)
            self.feature_dim = feature_vector.shape[1]

        # ── 3. Create MLP regression head ──────────────────────────────────
        hidden_dim = 128
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # height, occlusion, mode
        )

        # Initialize weights
        nn.init.xavier_normal_(self.head[0].weight)
        nn.init.xavier_normal_(self.head[2].weight)
        with torch.no_grad():
            # height: sigmoid(0) = 0.5 → mid-image
            # occlusion: sigmoid(-2) ≈ 0.12 → usually visible
            # mode: sigmoid(0) = 0.5 → no bias toward left or right
            self.head[2].bias[0] = 0.0    # height
            self.head[2].bias[1] = -2.0   # occlusion
            self.head[2].bias[2] = 0.0    # mode

    def _get_features(self, x):
        """Extract features from backbone+neck without detection head."""
        for layer in self.backbone_neck:
            x = layer(x)
        return x

    def forward(self, x):
        """Forward pass: backbone+neck → pool → MLP → [height, occlusion, mode] logits."""
        features = self._get_features(x)

        # Handle multi-scale features from detection backbone
        if isinstance(features, (list, tuple)):
            pooled = []
            for feat in features:
                pool = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                pooled.append(pool)
            feature_vector = torch.cat(pooled, dim=1)
        else:
            # Single feature map
            feature_vector = F.adaptive_avg_pool2d(features, 1).flatten(1)

        # Pass through MLP head
        logits = self.head(feature_vector)
        return logits  # (B, 3) raw logits: [height, occlusion, mode]


# ── Loss ───────────────────────────────────────────────────────────────────
def compute_loss(preds, targets):
    """
    preds:   (B, 3) raw logits  [height_logit, occlusion_logit, mode_logit]
    targets: (B, 3)             [height_norm,  occlusion (0/1), mode (0/1)]

    Height loss (smooth L1) is computed only on in-frame samples.
    Occlusion loss is BCE over all samples.
    Mode loss is BCE over all samples (which side of frame player is on).
    """
    height_logit = preds[:, 0]
    occ_logit    = preds[:, 1]
    mode_logit   = preds[:, 2]
    height_tgt   = targets[:, 0]
    occ_tgt      = targets[:, 1]
    mode_tgt     = targets[:, 2]

    # Occlusion: binary cross-entropy
    loss_occ = F.binary_cross_entropy_with_logits(occ_logit, occ_tgt)

    # Mode: binary cross-entropy (left vs right side of frame)
    loss_mode = F.binary_cross_entropy_with_logits(mode_logit, mode_tgt)

    # Height: smooth L1, masked to in-frame samples (occ == 0)
    height_pred = torch.sigmoid(height_logit)
    in_frame    = (1.0 - occ_tgt)
    n_in        = in_frame.sum().clamp(min=1.0)
    loss_height = (F.smooth_l1_loss(height_pred, height_tgt, reduction="none", beta=0.25)
                   * in_frame).sum() / n_in

    return loss_height + loss_occ + loss_mode, loss_height, loss_occ, loss_mode


# ── Data preparation ───────────────────────────────────────────────────────
def prepare_data(dataset_dir: Path, labels_file: Path,
                 val_frac: float = 0.1, seed: int = 42):
    """
    Load metadata + labels, build list of usable samples, split train/val.
    Excludes samples marked as removed or that have no label.
    Returns (train_samples, val_samples).
    """
    with open(dataset_dir / "metadata.json") as f:
        metadata = json.load(f)

    with open(labels_file) as f:
        raw_labels = json.load(f)

    meta_by_id = {str(m["sample_id"]): m for m in metadata}

    samples = []
    n_removed = n_unlabeled = 0

    for sid, lbl in raw_labels.items():
        if lbl.get("removed", False):
            n_removed += 1
            continue

        not_in_frame = lbl.get("not_in_frame", False)
        has_height   = "height_y" in lbl

        if not has_height and not not_in_frame:
            n_unlabeled += 1
            continue

        meta = meta_by_id.get(sid)
        if meta is None:
            continue
        npy_path = dataset_dir / meta["filename"]
        if not npy_path.exists():
            continue

        samples.append({
            "npy_path":  npy_path,
            "video_path": meta.get("video_path", "unknown"),
            "label_y":   float(lbl.get("height_y", DATASET_SIZE / 2)),
            "label_x":   float(lbl.get("height_x", DATASET_SIZE / 2)),
            "occlusion": 1.0 if not_in_frame else 0.0,
        })

    print(f"  {len(samples)} usable samples  "
          f"({n_removed} removed, {n_unlabeled} unlabeled skipped)")

    # Video-level split: group samples by video, then split videos (not frames)
    # This ensures validation set has truly unseen videos
    rng = random.Random(seed)

    # Group samples by video using video_path from metadata
    videos = {}
    for sample in samples:
        video_path = sample["video_path"]
        if video_path not in videos:
            videos[video_path] = []
        videos[video_path].append(sample)

    # Split videos randomly
    video_ids = list(videos.keys())
    rng.shuffle(video_ids)
    n_val_videos = max(1, int(len(video_ids) * val_frac))

    val_videos = set(video_ids[:n_val_videos])
    train_samples = [s for v_id, v_samples in videos.items() if v_id not in val_videos for s in v_samples]
    val_samples = [s for v_id, v_samples in videos.items() if v_id in val_videos for s in v_samples]

    print(f"  Video-level split: {len(video_ids)} videos → "
          f"{len(video_ids) - n_val_videos} train videos, {n_val_videos} val videos")
    print(f"  Samples: {len(train_samples)} train, {len(val_samples)} val")

    return train_samples, val_samples


# ── Train / eval loops ─────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    sum_loss = sum_h = sum_occ = 0.0
    n_in_frame = n_in_frame_correct = 0  # for height MAE
    occ_correct = occ_true_positives = occ_positives = 0  # for occlusion metrics
    mode_correct = 0  # for mode accuracy

    with torch.set_grad_enabled(train):
        for imgs, targets in loader:
            imgs    = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss, lh, lo, lm = compute_loss(preds, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            sum_loss += loss.item()
            sum_h    += lh.item()
            sum_occ  += lo.item()

            # Height MAE (in pixels) for in-frame samples
            with torch.no_grad():
                in_frame_mask = (targets[:, 1] == 0)
                if in_frame_mask.any():
                    pred_h  = torch.sigmoid(preds[:, 0])[in_frame_mask]
                    true_h  = targets[:, 0][in_frame_mask]
                    n_in_frame += in_frame_mask.sum().item()
                    n_in_frame_correct += (torch.abs(pred_h - true_h) * CROP_SIZE).sum().item()

                # Occlusion metrics at threshold 0.5
                occ_pred = (torch.sigmoid(preds[:, 1]) > 0.5).float()
                occ_true = targets[:, 1]
                occ_correct += (occ_pred == occ_true).sum().item()
                occ_true_positives += ((occ_pred == 1) & (occ_true == 1)).sum().item()
                occ_positives += (occ_true == 1).sum().item()

                # Mode accuracy at threshold 0.5 (left/right side classification)
                mode_pred = (torch.sigmoid(preds[:, 2]) > 0.5).float()
                mode_true = targets[:, 2]
                mode_correct += (mode_pred == mode_true).sum().item()

    n_batches = len(loader.dataset)
    mae_px = (n_in_frame_correct / n_in_frame) if n_in_frame else float("nan")
    occ_acc = occ_correct / n_batches if n_batches else float("nan")
    occ_recall = occ_true_positives / occ_positives if occ_positives else float("nan")
    mode_acc = mode_correct / n_batches if n_batches else float("nan")
    return sum_loss / len(loader), sum_h / len(loader), sum_occ / len(loader), mae_px, occ_acc, occ_recall, mode_acc


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",   required=True)
    parser.add_argument("--labels-file",   required=True)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--val-frac",      type=float, default=0.1)
    parser.add_argument("--workers",       type=int,   default=0,
                        help="DataLoader workers (0 = main process; safer on macOS/MPS)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    dataset_dir = Path(args.dataset_dir)
    labels_file = Path(args.labels_file)
    print("Preparing data...")
    train_samples, val_samples = prepare_data(
        dataset_dir, labels_file, args.val_frac, args.seed
    )
    print(f"  Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = GDDataset(train_samples, augment=True)
    val_ds   = GDDataset(val_samples,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=False)

    # Model
    print("Building model...")
    model = GDPlayerModel(pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Optimizer — higher LR for newly initialised layers
    head_params   = list(model.head.parameters())
    first_conv_params = list(model.first_conv.parameters())
    head_ids      = {id(p) for p in head_params + first_conv_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = AdamW([
        {"params": backbone_params,  "lr": args.lr},
        {"params": first_conv_params, "lr": args.lr * 5},   # 4th-channel tuning
        {"params": head_params,       "lr": args.lr * 10},  # new regression head
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Output directory
    run_dir = Path("runs") / datetime.now().strftime("train_robust_occlusions_det_%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}/")

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    print(f"\n{'Epoch':>6}  {'Tr-Loss':>8}  {'Tr-H':>7}  {'Tr-Occ':>7}  "
          f"{'Va-Loss':>8}  {'Va-H':>7}  {'Va-Occ':>7}  {'Va-H-MAE':>8}  {'Va-Occ-Acc':>10}  {'Va-Occ-Rec':>10}  {'Va-Mode-Acc':>11}")
    print("─" * 115)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_h, tr_occ, _, _, _, _       = run_epoch(
            model, train_loader, optimizer, device, train=True)
        va_loss, va_h, va_occ, va_mae, va_occ_acc, va_occ_rec, va_mode_acc  = run_epoch(
            model, val_loader, optimizer, device, train=False)
        scheduler.step()

        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_h:>7.4f}  {tr_occ:>7.4f}  "
              f"{va_loss:>8.4f}  {va_h:>7.4f}  {va_occ:>7.4f}  {va_mae:>8.1f}  {va_occ_acc:>10.1%}  {va_occ_rec:>10.1%}  {va_mode_acc:>11.1%}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": va_loss,
        }

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(checkpoint, run_dir / "best.pt")

        torch.save(checkpoint, run_dir / "last.pt")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {run_dir}/best.pt  and  {run_dir}/last.pt")


if __name__ == "__main__":
    main()
