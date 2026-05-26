"""Phase-2 smoke test: 1-epoch X3D-M training on a stratified subset.

Purpose: confirm the full pipeline runs end-to-end and that the loss carries
training signal. Not meant to produce a real model.

Reports:
- Train loss: first vs last batch + first/last 10-batch rolling averages.
- Val: accuracy, MAE (over softmax-expected-value), off-by-one accuracy.

Run:
    conda run -n cs231n python scripts/sanity_train_x3d.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import GeometryDashDataset, VideoRecord  # noqa: E402
from src.data.splits import make_or_load_splits  # noqa: E402
from src.models.difficulty_model import DifficultyModel  # noqa: E402
from src.models.x3d import X3DBackbone  # noqa: E402


def stratified_subset(records: List[VideoRecord], n_total: int, seed: int) -> List[VideoRecord]:
    """Pick ~n_total records keeping per-class balance (10-per-class for n=100)."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    per_class = max(1, n_total // 10)
    out: List[VideoRecord] = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        out.extend(items[:per_class])
    rng.shuffle(out)
    return out


def worker_init_fn(worker_id: int) -> None:
    """Seed RNGs per worker so train-mode segment sampling is reproducible."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-root", default="videos")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--clips-train", type=int, default=4,
                    help="K at train (smaller than the eventual 8 to keep smoke fast).")
    ap.add_argument("--clips-eval", type=int, default=8,
                    help="K at eval (smaller than n-segments to keep smoke fast).")
    ap.add_argument("--n-segments", type=int, default=24)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--aggregation", default="mean",
                    choices=["mean", "max", "topk_mean"])
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable mixed precision (default: AMP on if CUDA).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("!! CPU mode — this will be slow. Consider running on the GPU box.")
    use_amp = (not args.no_amp) and device.type == "cuda"

    splits = make_or_load_splits(args.videos_root, args.splits, seed=args.seed)
    train_recs = stratified_subset(splits.train, args.n_train, seed=args.seed)
    val_recs = stratified_subset(splits.val, args.n_val, seed=args.seed)
    print(f"Subset: train={len(train_recs)}  val={len(val_recs)}")

    ds_train = GeometryDashDataset(
        train_recs, mode="train",
        n_segments=args.n_segments,
        clips_per_video_train=args.clips_train,
    )
    ds_val = GeometryDashDataset(
        val_recs, mode="eval",
        n_segments=args.n_segments,
        clips_per_video_eval=args.clips_eval,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    print("Loading X3D-M from torch.hub (cached after first run)...")
    backbone = X3DBackbone(pretrained=True)
    model = DifficultyModel(backbone, aggregation=args.aggregation).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"feature_dim={backbone.feature_dim}  params: total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(),     "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    # AMP plumbing. torch.amp.GradScaler was unified in torch 2.3+; on 2.1.2
    # (what requirements.txt pins) we use the torch.cuda.amp path.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---------------- Train one epoch ----------------
    model.train()
    losses: List[float] = []
    t0 = time.time()
    pbar = tqdm(dl_train, desc="train")
    for batch_clips, batch_labels in pbar:
        batch_clips = batch_clips.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch_clips)
            loss = F.cross_entropy(logits, batch_labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        rolling = sum(losses[-20:]) / min(20, len(losses))
        pbar.set_postfix(loss=f"{loss.item():.3f}", roll20=f"{rolling:.3f}")

    train_dt = time.time() - t0
    n10 = min(10, len(losses))
    first_avg = sum(losses[:n10]) / n10
    last_avg = sum(losses[-n10:]) / n10
    print(
        f"\nTrain: {len(losses)} batches in {train_dt:.1f}s  "
        f"first={losses[0]:.3f} last={losses[-1]:.3f}  "
        f"first{n10}_avg={first_avg:.3f}  last{n10}_avg={last_avg:.3f}  "
        f"Δ={first_avg - last_avg:+.3f}"
    )
    print("  (Loss should drop; on 100 samples in 1 epoch don't expect much.)")

    # ---------------- Val ----------------
    model.eval()
    all_pred: List[int] = []
    all_true: List[int] = []
    all_ev: List[float] = []
    ranks = torch.arange(1, 11, device=device, dtype=torch.float32)
    with torch.no_grad():
        for batch_clips, batch_labels in tqdm(dl_val, desc="val  "):
            batch_clips = batch_clips.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch_clips)
            probs = F.softmax(logits.float(), dim=-1)
            ev = (probs * ranks).sum(dim=-1)        # in [1, 10]
            pred = logits.argmax(dim=-1)             # 0..9
            all_pred.extend(pred.cpu().tolist())
            all_true.extend(batch_labels.tolist())
            all_ev.extend(ev.cpu().tolist())

    n = len(all_true)
    acc = sum(p == t for p, t in zip(all_pred, all_true)) / n
    mae = sum(abs(ev_i - (t + 1)) for ev_i, t in zip(all_ev, all_true)) / n
    off_by_one = sum(abs(p - t) <= 1 for p, t in zip(all_pred, all_true)) / n
    print(
        f"\nVal: n={n}  acc={acc:.4f}  MAE={mae:.4f}  off_by_one_acc={off_by_one:.4f}"
    )
    print(
        "  References: uniform-random acc ≈ 0.10, MAE ≈ 3.3. "
        "Off-by-one on a random predictor ≈ 0.28."
    )


if __name__ == "__main__":
    main()
