"""Full training loop for DifficultyModel.

Cosine LR with linear warmup (default 5%), gradient accumulation, TensorBoard
logging, best-by-MAE checkpointing, and resume support.

Examples:
    python src/train.py --backbone x3d --epochs 30 --batch-size 4 \
        --out-dir runs/x3d
    python src/train.py --backbone x3d --resume runs/x3d/latest.pt \
        --out-dir runs/x3d

Checkpoints written to <out-dir>:
    best.pt    — best val-MAE so far
    latest.pt  — last completed epoch (used by --resume)

TensorBoard logs under <out-dir>/tb/. Per-step train loss + LRs; per-epoch val
loss / acc / MAE / off-by-one.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import GeometryDashDataset  # noqa: E402
from src.data.splits import make_or_load_splits  # noqa: E402
from src.models.difficulty_model import DifficultyModel  # noqa: E402
from src.models.factory import BACKBONES, build_backbone  # noqa: E402


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def cosine_warmup_lambda(total_steps: int, warmup_frac: float):
    """Linear warmup 0→1 over `warmup_frac` of training, cosine decay 1→0 after."""
    warmup_steps = max(1, int(round(total_steps * warmup_frac)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return lr_lambda


@torch.no_grad()
def evaluate(model, dl, device, use_amp) -> Dict[str, float]:
    model.eval()
    ranks = torch.arange(1, 11, device=device, dtype=torch.float32)
    preds: List[int] = []
    trues: List[int] = []
    evs: List[float] = []
    total_loss = 0.0
    n = 0
    for clips, labels in tqdm(dl, desc="val  ", leave=False):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(clips)
            loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits.float(), dim=-1)
        ev = (probs * ranks).sum(dim=-1)
        pred = logits.argmax(dim=-1)
        preds.extend(pred.cpu().tolist())
        trues.extend(labels.cpu().tolist())
        evs.extend(ev.cpu().tolist())
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)

    acc = sum(p == t for p, t in zip(preds, trues)) / n
    mae = sum(abs(ev_i - (t + 1)) for ev_i, t in zip(evs, trues)) / n
    off1 = sum(abs(p - t) <= 1 for p, t in zip(preds, trues)) / n
    return {"loss": total_loss / n, "acc": acc, "mae": mae, "off1": off1}


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    *,
    epoch: int,
    global_step: int,
    best_val_mae: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_mae": best_val_mae,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--videos-root", default="videos")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--out-dir", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="Micro-batches per optimizer step.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="Batches each worker prefetches. Ignored if num_workers=0.")
    ap.add_argument("--no-persistent-workers", action="store_true",
                    help="Disable persistent dataloader workers between epochs.")
    ap.add_argument("--clips-train", type=int, default=8)
    ap.add_argument("--clips-eval", type=int, default=24)
    ap.add_argument("--n-segments", type=int, default=24)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="Max grad norm. <=0 disables.")
    ap.add_argument("--aggregation", default="max",
                    choices=["mean", "max", "topk_mean"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=10,
                    help="Optimizer steps between TB scalar writes.")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint path to resume from (e.g. runs/x3d/latest.pt).")
    ap.add_argument("--wandb-project", default=None,
                    help="If set, log to Weights & Biases under this project.")
    ap.add_argument("--wandb-entity", default=None,
                    help="W&B entity (user or team). Defaults to wandb's own default.")
    ap.add_argument("--wandb-run-name", default=None,
                    help="Optional W&B run name (defaults to wandb-generated).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device: {device}  |  backbone: {args.backbone}  |  AMP: {use_amp}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    splits = make_or_load_splits(args.videos_root, args.splits, seed=args.seed)
    print(
        f"Splits: train={len(splits.train)}  "
        f"val={len(splits.val)}  test={len(splits.test)}"
    )

    ds_train = GeometryDashDataset(
        splits.train, mode="train",
        n_segments=args.n_segments,
        clips_per_video_train=args.clips_train,
    )
    ds_val = GeometryDashDataset(
        splits.val, mode="eval",
        n_segments=args.n_segments,
        clips_per_video_eval=args.clips_eval,
    )

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    dl_val = DataLoader(
        ds_val, batch_size=1, shuffle=False, **loader_kwargs,
    )

    print(f"Building {args.backbone} (downloads weights on first run)...")
    backbone = build_backbone(args.backbone, pretrained=True)
    model = DifficultyModel(backbone, aggregation=args.aggregation).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"feature_dim={backbone.feature_dim}  "
        f"params: total={n_total/1e6:.2f}M  trainable={n_trainp/1e6:.2f}M"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(),     "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    micro_per_epoch = len(dl_train)
    steps_per_epoch = math.ceil(micro_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_warmup_lambda(total_steps, args.warmup_frac)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_val_mae = float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if use_amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_val_mae = float(ckpt.get("best_val_mae", float("inf")))
        print(
            f"  resumed at epoch={start_epoch}  step={global_step}  "
            f"best_val_mae={best_val_mae:.4f}"
        )

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    wandb = None
    if args.wandb_project:
        import wandb as _wandb  # local import so wandb is optional
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            dir=str(out_dir),
            resume="allow",
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: List[float] = []
        t0 = time.time()

        pbar = tqdm(
            enumerate(dl_train),
            total=micro_per_epoch,
            desc=f"ep{epoch:03d}",
        )
        for mb_idx, (clips, labels) in pbar:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(clips)
                loss = F.cross_entropy(logits, labels)
                loss_scaled = loss / args.grad_accum

            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            epoch_losses.append(loss.item())

            is_last = (mb_idx + 1) == micro_per_epoch
            do_step = ((mb_idx + 1) % args.grad_accum == 0) or is_last
            if do_step:
                if args.grad_clip and args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_interval == 0:
                    rolling = sum(epoch_losses[-20:]) / min(20, len(epoch_losses))
                    lr_backbone = optimizer.param_groups[0]["lr"]
                    lr_head = optimizer.param_groups[1]["lr"]
                    writer.add_scalar("train/loss", rolling, global_step)
                    writer.add_scalar("train/lr_backbone", lr_backbone, global_step)
                    writer.add_scalar("train/lr_head", lr_head, global_step)
                    if wandb is not None:
                        wandb.log(
                            {
                                "train/loss": rolling,
                                "train/lr_backbone": lr_backbone,
                                "train/lr_head": lr_head,
                            },
                            step=global_step,
                        )

            rolling = sum(epoch_losses[-20:]) / min(20, len(epoch_losses))
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                roll20=f"{rolling:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_dt = time.time() - t0
        train_loss_avg = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"  train: avg_loss={train_loss_avg:.4f}  time={train_dt:.1f}s")

        val_metrics = evaluate(model, dl_val, device, use_amp)
        print(
            f"  val:   loss={val_metrics['loss']:.4f}  "
            f"acc={val_metrics['acc']:.4f}  "
            f"MAE={val_metrics['mae']:.4f}  "
            f"off1={val_metrics['off1']:.4f}"
        )

        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        writer.add_scalar("val/mae", val_metrics["mae"], epoch)
        writer.add_scalar("val/off1", val_metrics["off1"], epoch)
        writer.add_scalar("epoch/train_loss", train_loss_avg, epoch)
        if wandb is not None:
            wandb.log(
                {
                    "val/loss": val_metrics["loss"],
                    "val/acc": val_metrics["acc"],
                    "val/mae": val_metrics["mae"],
                    "val/off1": val_metrics["off1"],
                    "epoch/train_loss": train_loss_avg,
                    "epoch": epoch,
                },
                step=global_step,
            )

        save_checkpoint(
            out_dir / "latest.pt", model, optimizer, scheduler, scaler,
            epoch=epoch + 1, global_step=global_step,
            best_val_mae=best_val_mae, args=args,
        )
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            save_checkpoint(
                out_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch=epoch + 1, global_step=global_step,
                best_val_mae=best_val_mae, args=args,
            )
            print(f"  >>> new best val MAE = {best_val_mae:.4f}")

    writer.close()
    if wandb is not None:
        wandb.run.summary["best_val_mae"] = best_val_mae
        wandb.finish()
    print(f"\nDone. Best val MAE = {best_val_mae:.4f}. Checkpoints in {out_dir}/")


if __name__ == "__main__":
    main()
