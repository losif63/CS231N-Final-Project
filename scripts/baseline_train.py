"""Train a baseline model on cached ResNet-18 features.

Loads pre-extracted features from processed_resnet18/*.npy files.
Trains an attention-pooling + MLP head for difficulty classification (1-10 stars).
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader


class ResNet18FeatureDataset(Dataset):
    """Load pre-extracted ResNet-18 features from .npy files.

    Each file is (T, 512) shaped features for a single video.
    Label is 0-indexed (1stars -> 0, 10stars -> 9).
    """

    def __init__(self, features_dir: str | Path):
        """Scan features_dir/{n}stars/*.npy and build record list."""
        self.features_dir = Path(features_dir)
        self.records: List[Tuple[Path, int]] = []  # (feature_path, label)

        for label in range(10):
            label_dir = self.features_dir / f"{label + 1}stars"
            if label_dir.exists():
                for npy_path in sorted(label_dir.glob("*.npy")):
                    self.records.append((npy_path, label))

        print(f"Loaded {len(self.records)} feature files")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load feature array and return (features, label).

        Returns:
            features: (T, 512) float tensor
            label: 0-indexed difficulty
        """
        npy_path, label = self.records[idx]
        features = np.load(npy_path).astype(np.float32)  # (T, 512)
        return torch.from_numpy(features), label


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """Collate batch with variable-length sequences.

    Pads all sequences to max length in the batch with zeros.

    Returns:
        features: (B, T_max, 512)
        labels: (B,)
        lengths: (B,) — actual length of each sequence before padding
    """
    features_list, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)

    # Pad to max length
    T_max = max(f.shape[0] for f in features_list)
    padded = []
    lengths = []
    for f in features_list:
        T = f.shape[0]
        lengths.append(T)
        if T < T_max:
            pad = torch.zeros(T_max - T, f.shape[1], dtype=f.dtype)
            f = torch.cat([f, pad], dim=0)
        padded.append(f)

    features = torch.stack(padded, dim=0)  # (B, T_max, 512)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return features, labels, lengths


class BaselineModel(nn.Module):
    """Attention pooling + MLP head for difficulty classification."""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.5,
        num_classes: int = 10,
    ):
        super().__init__()

        # Attention pooling: scalar score per frame
        self.attn = nn.Linear(feature_dim, 1)

        # MLP head
        layers: List[nn.Module] = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, lengths: torch.LongTensor
    ) -> torch.Tensor:
        """Forward pass with masked attention.

        Args:
            features: (B, T_max, 512)
            lengths: (B,) — actual length before padding

        Returns:
            logits: (B, num_classes)
        """
        B, T_max = features.shape[:2]
        device = features.device

        # Attention scores
        scores = self.attn(features).squeeze(-1)  # (B, T_max)

        # Mask padding positions with -inf before softmax
        mask = torch.arange(T_max, device=device)[None, :] >= lengths[:, None]  # (B, T_max)
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax weights
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T_max, 1)

        # Weighted sum of features
        pooled = (weights * features).sum(dim=1)  # (B, 512)

        # MLP head
        return self.head(pooled)  # (B, num_classes)


def stratified_train_val_test_split(
    dataset: ResNet18FeatureDataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified split by label.

    Returns:
        train_indices, val_indices, test_indices
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group indices by label
    indices_by_label = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset.records):
        indices_by_label[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for label in range(10):
        indices = indices_by_label[label]
        random.shuffle(indices)

        n = len(indices)
        if n == 0:
            continue
        # For very small datasets, ensure at least 1 sample per split (if possible)
        if n < 3:
            train_count = n
            val_count = 0
            test_count = 0
        else:
            train_count = max(1, int(n * train_frac))
            val_count = max(1, int(n * val_frac))
            test_count = max(0, n - train_count - val_count)

        train_indices.extend(indices[:train_count])
        val_indices.extend(indices[train_count : train_count + val_count])
        test_indices.extend(indices[train_count + val_count : train_count + val_count + test_count])

    return train_indices, val_indices, test_indices


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, float, float]:
    """Compute accuracy, MAE, and off1 (within 1 of correct label).

    Returns:
        accuracy: fraction of correct predictions
        mae: mean absolute error in stars (0-9 scale)
        off1: fraction of predictions within 1 of correct label
    """
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()
    mae = (preds.float() - labels.float()).abs().mean().item()
    off1 = ((preds.float() - labels.float()).abs() <= 1).float().mean().item()
    return accuracy, mae, off1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        default="processed_resnet18",
        help="Path to processed_resnet18/ root",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="processed_resnet18", help="Output directory for checkpoint")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading features from {args.features_dir}...")
    dataset = ResNet18FeatureDataset(args.features_dir)

    # Stratified split
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=args.seed
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=0,
    )

    # Build model
    model = BaselineModel(feature_dim=512, hidden_dims=(512, 256), dropout=args.dropout, num_classes=10)
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_mae = float("inf")
    best_checkpoint_path = Path(args.output_dir) / "baseline_best.pt"

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for features, labels, lengths in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(features, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.shape[0]

        train_loss /= len(train_dataset)
        scheduler.step()

        # Eval (skip if val_dataset is empty)
        if len(val_dataset) > 0:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_mae = 0.0
            val_off1 = 0.0
            with torch.no_grad():
                for features, labels, lengths in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    lengths = lengths.to(device)

                    logits = model(features, lengths)
                    loss = criterion(logits, labels)
                    acc, mae, off1 = compute_metrics(logits, labels)

                    val_loss += loss.item() * labels.shape[0]
                    val_acc += acc * labels.shape[0]
                    val_mae += mae * labels.shape[0]
                    val_off1 += off1 * labels.shape[0]

            val_loss /= len(val_dataset)
            val_acc /= len(val_dataset)
            val_mae /= len(val_dataset)
            val_off1 /= len(val_dataset)

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"Val Off1: {val_off1:.4f}"
            )

            # Checkpoint
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_mae": best_val_mae,
                    },
                    str(best_checkpoint_path),
                )
                print(f"  -> Saved best checkpoint to {best_checkpoint_path}")
        else:
            print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | (no validation set)")

    # Test
    if len(test_dataset) > 0:
        print("\nEvaluating on test set...")
        checkpoint = torch.load(str(best_checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        test_acc = 0.0
        test_mae = 0.0
        test_off1 = 0.0
        with torch.no_grad():
            for features, labels, lengths in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                logits = model(features, lengths)
                acc, mae, off1 = compute_metrics(logits, labels)

                test_acc += acc * labels.shape[0]
                test_mae += mae * labels.shape[0]
                test_off1 += off1 * labels.shape[0]

        test_acc /= len(test_dataset)
        test_mae /= len(test_dataset)
        test_off1 /= len(test_dataset)

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test MAE: {test_mae:.4f} (star difference)")
        print(f"Test Off1: {test_off1:.4f}")
    else:
        print("\nNo test set available (dataset too small)")


if __name__ == "__main__":
    main()
