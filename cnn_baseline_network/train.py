"""Train a ResNet-based CNN to predict Geometry Dash level difficulty.

Example usage:
    python cnn_baseline/train.py --videos-dir videos/

    # Custom ResNet and FCN:
    python cnn_baseline/train.py --videos-dir videos/ \\
        --resnet resnet18 --hidden-dims 256 128 --dropout 0.3

    # Tighter center crop (use 80% of the shortest edge):
    python cnn_baseline/train.py --videos-dir videos/ --crop-fraction 0.8
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import GeometryDashDataset, collate_fn
from model import GeometryDashCNN, RESNET_OUT_DIMS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Geometry Dash difficulty classifier")

    # --- Data ---
    p.add_argument("--videos-dir", type=Path, required=True,
                   help="Root directory with {n}stars/ subdirs of videos")
    p.add_argument("--frame-interval", type=float, default=4.0,
                   help="Seconds between sampled frames (default: 4)")
    p.add_argument("--crop-fraction", type=float, default=1.0,
                   help="Fraction of min(H,W) for center square crop (default: 1.0 = full)")
    p.add_argument("--image-size", type=int, default=224,
                   help="Resize cropped frame to this size before ResNet (default: 224)")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of dataset to use for validation (default: 0.1)")

    # --- Model ---
    p.add_argument("--resnet", type=str, default="resnet50",
                   choices=list(RESNET_OUT_DIMS),
                   help="ResNet backbone variant (default: resnet50)")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256],
                   help="Hidden layer sizes in the FC head (default: 512 256)")
    p.add_argument("--dropout", type=float, default=0.5,
                   help="Dropout probability in FC head (default: 0.5)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train ResNet backbone from scratch instead of using ImageNet weights")

    # --- Training ---
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                   help="Directory to save best checkpoint (default: checkpoints/)")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for frames_list, labels in loader:
        frames_list = [f.to(device) for f in frames_list]
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames_list)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for frames_list, labels in loader:
        frames_list = [f.to(device) for f in frames_list]
        labels = labels.to(device)

        logits = model(frames_list)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Dataset
    dataset = GeometryDashDataset(
        videos_dir=args.videos_dir,
        frame_interval_sec=args.frame_interval,
        crop_fraction=args.crop_fraction,
        image_size=args.image_size,
    )
    print(f"Dataset: {len(dataset)} samples")
    if len(dataset) == 0:
        raise RuntimeError(f"No videos found under {args.videos_dir}")

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # Model
    model = GeometryDashCNN(
        num_classes=10,
        resnet_version=args.resnet,
        hidden_dims=tuple(args.hidden_dims),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    print(f"Model: {args.resnet}, hidden={args.hidden_dims}, "
          f"pretrained={not args.no_pretrained}, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train loss {train_loss:.4f}  acc {train_acc:.3f}  |  "
            f"val loss {val_loss:.4f}  acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = args.checkpoint_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> saved best checkpoint ({ckpt_path})")


if __name__ == "__main__":
    main()
