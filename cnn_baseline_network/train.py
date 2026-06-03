"""Train a ResNet-based CNN to predict Geometry Dash level difficulty.

Uses cross-entropy classification with class weights to handle label imbalance.

Run extract_frames.py first to pre-extract frames from videos.

Example usage:
    python cnn_baseline_network/train.py --frames-dir frames/
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from dataset import GeometryDashDataset, collate_fn
from model import GeometryDashCNN, RESNET_OUT_DIMS

NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Geometry Dash difficulty classifier")

    # --- Data ---
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="Root directory with pre-extracted frames (output of extract_frames.py)")
    p.add_argument("--image-size", type=int, default=64,
                   help="Resize frames to this size at load time; 256 = no resize (default: 64)")
    p.add_argument("--max-frames", type=int, default=60,
                   help="Max frames to use per video (default: 60)")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of dataset to use for validation (default: 0.1)")

    # --- Model ---
    p.add_argument("--resnet", type=str, default="resnet18",
                   choices=list(RESNET_OUT_DIMS),
                   help="ResNet backbone variant (default: resnet18)")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128],
                   help="Hidden layer sizes in the FC head (default: 256 128)")
    p.add_argument("--dropout", type=float, default=0.5,
                   help="Dropout probability in FC head (default: 0.5)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train ResNet backbone from scratch instead of using ImageNet weights")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze ResNet backbone weights; only train the FC head")

    # --- Training ---
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
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
    total_loss, correct, mae_sum, total = 0.0, 0, 0.0, 0
    n_batches = len(loader)

    for i, (frames_list, labels) in enumerate(loader):
        frames_list = [f.to(device) for f in frames_list]
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames_list)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        mae_sum += (preds.float() - labels.float()).abs().sum().item()
        total += labels.size(0)

        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            print(
                f"  batch {i+1}/{n_batches}  loss {total_loss/total:.4f}"
                f"  acc {correct/total:.3f}  mae {mae_sum/total:.2f}",
                flush=True,
            )

    return total_loss / total, correct / total, mae_sum / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, mae_sum, total = 0.0, 0, 0.0, 0

    for frames_list, labels in loader:
        frames_list = [f.to(device) for f in frames_list]
        labels = labels.to(device)

        logits = model(frames_list)
        loss = criterion(logits, labels)

        preds = logits.argmax(1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        mae_sum += (preds.float() - labels.float()).abs().sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total, mae_sum / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = GeometryDashDataset(
        frames_dir=args.frames_dir,
        image_size=args.image_size,
        max_frames=args.max_frames,
    )
    print(f"Dataset: {len(dataset)} samples", flush=True)
    if len(dataset) == 0:
        raise RuntimeError(f"No pre-extracted frames found under {args.frames_dir}. "
                           "Run extract_frames.py first.")

    class_counts = [0] * NUM_CLASSES
    for _, label in dataset.samples:
        class_counts[label] += 1
    print(f"Samples per class: {class_counts}", flush=True)
    class_weights = torch.tensor(
        [1.0 / c if c > 0 else 0.0 for c in class_counts], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_indices, val_indices = random_split(
        range(len(dataset)), [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {n_train}  Val: {n_val}", flush=True)

    train_dataset = GeometryDashDataset(
        frames_dir=args.frames_dir, image_size=args.image_size,
        max_frames=args.max_frames, augment=True,
    )
    val_dataset = GeometryDashDataset(
        frames_dir=args.frames_dir, image_size=args.image_size,
        max_frames=args.max_frames, augment=False,
    )
    train_set = Subset(train_dataset, list(train_indices))
    val_set   = Subset(val_dataset,   list(val_indices))

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = GeometryDashCNN(
        num_classes=NUM_CLASSES,
        resnet_version=args.resnet,
        hidden_dims=tuple(args.hidden_dims),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    print(
        f"Model: {args.resnet}, hidden={args.hidden_dims}, "
        f"pretrained={not args.no_pretrained}, freeze_backbone={args.freeze_backbone}, "
        f"device={device}",
        flush=True,
    )

    if args.freeze_backbone:
        # layer4 (backbone index 7) trains at 10x lower LR than the head
        optimizer = torch.optim.AdamW([
            {"params": model.backbone[7].parameters(), "lr": args.lr * 0.1},
            {"params": list(model.attn.parameters()) + list(model.head.parameters()), "lr": args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_mae = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train loss {train_loss:.4f}  acc {train_acc:.3f}  mae {train_mae:.2f}  |  "
            f"val loss {val_loss:.4f}  acc {val_acc:.3f}  mae {val_mae:.2f}",
            flush=True,
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
            print(f"  -> saved best checkpoint ({ckpt_path})", flush=True)


if __name__ == "__main__":
    main()
