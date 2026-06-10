#!/usr/bin/env python3
"""
ROBUST training variant: Train a YOLO-based model with aggressive augmentations.

Compared to train.py, adds: vertical flip, synthetic occlusion, vignette,
two-tier blur, motion blur, brightness scaling, hue rotation.

Architecture: YOLOv8n-cls backbone, modified for:
  - 4-channel input (BGR channels + greyscale frame diff as channel 4)
  - Regression head: [height_logit, occlusion_logit] → sigmoid → [0, 1]

Outputs:
  player_height   [0, 1]  normalized vertical position in 620×620 image
  player_occlusion [0, 1]  probability player is not visible

Usage:
  python train_robust.py \
    --dataset-dir dataset_10000_0 \
    --labels-file dataset_10000_0/labels_2026-06-02_19-02-42.json

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
            if random.random() < 0.3:
                crop  = crop[:, ::-1, :].copy()
                crop_x = CROP_SIZE - crop_x - 1

            # 3. Synthetic player occlusion patch (p=0.2, skip if already occluded)
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

            # 4. Horizontal motion blur (p=0.2)
            if random.random() < 0.2:
                kernel_size = random.choice([3, 5])
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
                for c in range(4):
                    crop[:, :, c] = cv2.filter2D(crop[:, :, c], -1, kernel)

            # 5. Gaussian blur mild (p=0.3): σ 0.5–1.5, kernel 3×3
            if random.random() < 0.3:
                sigma = random.uniform(0.5, 1.5)
                for c in range(4):
                    crop[:, :, c] = cv2.GaussianBlur(crop[:, :, c], (3, 3), sigma)

            # 6. Gaussian blur strong (p=0.15): σ 1.5–3.0, kernel 5×5
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
            # 7. Brightness scaling (p=0.4): α ∈ [0.7, 1.3]
            if random.random() < 0.4:
                alpha = random.uniform(0.7, 1.3)
                rgb = np.clip(rgb * alpha, 0.0, 1.0).astype(np.float32)
                diff = np.clip(diff * alpha, 0.0, 1.0).astype(np.float32)

            # 8. Vignette gradient (p=0.25): radial G = 1 - strength * r_norm
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

            # 9. Hue rotation (p=0.3): ±15° in HSV H channel
            if random.random() < 0.3:
                hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                h_shift = random.uniform(-8, 8)  # OpenCV hue is 0-180, so ±8 ≈ ±15°
                hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

            # 10. Gaussian noise (p=0.3): RGB only
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
        targets = torch.tensor([height_norm, float(occlusion)], dtype=torch.float32)

        return tensor, targets


# ── Model ──────────────────────────────────────────────────────────────────
def _patch_classify_head(classify_layer):
    """
    Patch a YOLO Classify module in-place for regression:
      - Replace the final Linear(1000) → Linear(2)
      - Override forward() so it never applies softmax

    We patch in-place (rather than replacing the module) so that all YOLO
    graph-routing attributes (.f, .i, .type, etc.) on the original object
    are preserved for _predict_once().
    """
    in_features = classify_layer.linear.in_features  # 1280

    new_linear = nn.Linear(in_features, 2)
    nn.init.xavier_normal_(new_linear.weight)
    with torch.no_grad():
        # height: sigmoid(0) = 0.5 → mid-image
        # occlusion: sigmoid(-2) ≈ 0.12 → usually visible
        new_linear.bias[0] =  0.0
        new_linear.bias[1] = -2.0
    classify_layer.linear = new_linear

    # Override forward to remove the softmax that Classify applies in eval mode
    def _forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

    import types
    classify_layer.forward = types.MethodType(_forward, classify_layer)


class GDPlayerModel(nn.Module):
    """
    YOLOv8n-cls backbone adapted for 4-channel input and regression output.

    4th-channel weight initialisation:
      The first Conv2d is extended from 3→4 input channels.
      Channels 0-2 keep their ImageNet-pretrained weights unchanged.
      Channel 3 (diff) is initialised as the mean of channels 0-2, which
      preserves the expected activation scale while giving the network a
      reasonable starting point for motion features.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO("yolov8n-cls.pt" if pretrained else "yolov8n-cls.yaml")
        self.backbone = yolo.model  # ClassificationModel

        # Unfreeze all backbone parameters — ultralytics loads with requires_grad=False
        for p in self.backbone.parameters():
            p.requires_grad_(True)

        # ── 1. Extend first conv: 3 → 4 input channels ───────────────────
        first_conv = self.backbone.model[0].conv   # Conv2d(3, 16, 3, 2, 1, bias=False)
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

        self.backbone.model[0].conv = new_conv

        # ── 2. Patch classification head for regression ───────────────────
        _patch_classify_head(self.backbone.model[-1])

    def forward(self, x):
        return self.backbone(x)   # (B, 2) raw logits


# ── Loss ───────────────────────────────────────────────────────────────────
def compute_loss(preds, targets):
    """
    preds:   (B, 2) raw logits  [height_logit, occlusion_logit]
    targets: (B, 2)             [height_norm,  occlusion (0/1)]

    Height loss (smooth L1) is computed only on in-frame samples.
    Occlusion loss is BCE over all samples.
    """
    height_logit = preds[:, 0]
    occ_logit    = preds[:, 1]
    height_tgt   = targets[:, 0]
    occ_tgt      = targets[:, 1]

    # Occlusion: binary cross-entropy
    loss_occ = F.binary_cross_entropy_with_logits(occ_logit, occ_tgt)

    # Height: smooth L1, masked to in-frame samples (occ == 0)
    height_pred = torch.sigmoid(height_logit)
    in_frame    = (1.0 - occ_tgt)
    n_in        = in_frame.sum().clamp(min=1.0)
    loss_height = (F.smooth_l1_loss(height_pred, height_tgt, reduction="none", beta=0.02)
                   * in_frame).sum() / n_in

    return loss_height + loss_occ, loss_height, loss_occ


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
            "label_y":   float(lbl.get("height_y", DATASET_SIZE / 2)),
            "label_x":   float(lbl.get("height_x", DATASET_SIZE / 2)),
            "occlusion": 1.0 if not_in_frame else 0.0,
        })

    print(f"  {len(samples)} usable samples  "
          f"({n_removed} removed, {n_unlabeled} unlabeled skipped)")

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * val_frac))
    return samples[n_val:], samples[:n_val]


# ── Train / eval loops ─────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    sum_loss = sum_h = sum_occ = 0.0
    n_in_frame = n_in_frame_correct = 0  # for height MAE

    with torch.set_grad_enabled(train):
        for imgs, targets in loader:
            imgs    = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss, lh, lo = compute_loss(preds, targets)

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

    n = len(loader)
    mae_px = (n_in_frame_correct / n_in_frame) if n_in_frame else float("nan")
    return sum_loss / n, sum_h / n, sum_occ / n, mae_px


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
    head_params   = list(model.backbone.model[-1].parameters())
    first_conv_params = list(model.backbone.model[0].conv.parameters())
    head_ids      = {id(p) for p in head_params + first_conv_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = AdamW([
        {"params": backbone_params,  "lr": args.lr},
        {"params": first_conv_params, "lr": args.lr * 5},   # 4th-channel tuning
        {"params": head_params,       "lr": args.lr * 10},  # new regression head
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Output directory
    run_dir = Path("runs") / datetime.now().strftime("train_robust_%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}/")

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    print(f"\n{'Epoch':>6}  {'Tr-Loss':>8}  {'Tr-H':>7}  {'Tr-Occ':>7}  "
          f"{'Va-Loss':>8}  {'Va-H':>7}  {'Va-Occ':>7}  {'Va-MAE(px)':>10}")
    print("─" * 78)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_h, tr_occ, _       = run_epoch(
            model, train_loader, optimizer, device, train=True)
        va_loss, va_h, va_occ, va_mae  = run_epoch(
            model, val_loader, optimizer, device, train=False)
        scheduler.step()

        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_h:>7.4f}  {tr_occ:>7.4f}  "
              f"{va_loss:>8.4f}  {va_h:>7.4f}  {va_occ:>7.4f}  {va_mae:>10.1f}")

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
