"""Evaluate a trained DifficultyModel checkpoint.

Loads a checkpoint and runs on the requested split (default: test). Reports
overall accuracy / MAE / off-by-one, per-class precision/recall/F1, saves a
confusion-matrix PNG and a metrics.json, and prints baselines:

  - uniform-random predictor      (analytic, with off-by-one accuracy)
  - optimal constant by accuracy  (predict the modal class)
  - optimal constant by MAE       (the class that minimises mean |c - y|)

Example:
    python src/eval.py --ckpt runs/x3d/best.pt --split test
    python src/eval.py --ckpt runs/x3d/best.pt --split val --out-dir eval_out/x3d
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import GeometryDashDataset  # noqa: E402
from src.data.splits import make_or_load_splits  # noqa: E402
from src.models.difficulty_model import (  # noqa: E402
    DifficultyModel,
    decode_logits,
)
from src.models.factory import BACKBONES, build_backbone  # noqa: E402

_NUM_CLASSES = 10
_HEAD_KINDS = ("softmax", "ordinal")


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def run_inference(
    model, dl, device, use_amp, head_kind: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (preds, trues, expected_values), all length-N arrays.

    `pred` is 0..K-1; `ev` is on the stars scale 1..K. The split between
    softmax and ordinal CORN heads is handled inside `decode_logits`.
    """
    model.eval()
    all_pred: List[int] = []
    all_true: List[int] = []
    all_ev: List[float] = []
    for clips, labels in tqdm(dl, desc="eval"):
        clips = clips.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(clips)
        pred, ev = decode_logits(logits, head_kind)
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(labels.tolist())
        all_ev.extend(ev.cpu().tolist())
    return (
        np.array(all_pred, dtype=np.int64),
        np.array(all_true, dtype=np.int64),
        np.array(all_ev, dtype=np.float64),
    )


def compute_baselines(trues: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Baselines evaluated directly on the eval-set true labels (oracle)."""
    n = len(trues)
    classes = np.arange(10)
    # Uniform-random: every prediction is uniform over 0..9, independent of y.
    # Use the analytic expectation conditioned on each true label, then average.
    abs_diff = np.abs(classes[None, :] - trues[:, None])  # (n, 10)
    rand_mae = float(np.mean(np.mean(abs_diff, axis=1)))
    rand_off1 = float(np.mean(np.mean(abs_diff <= 1, axis=1)))

    counts = np.bincount(trues, minlength=10)
    best_acc_class = int(np.argmax(counts))
    best_acc = float(counts[best_acc_class] / n)
    best_acc_mae = float(np.mean(np.abs(trues - best_acc_class)))

    mae_per_c = np.array([np.mean(np.abs(trues - c)) for c in classes])
    best_mae_class = int(np.argmin(mae_per_c))
    best_mae = float(mae_per_c[best_mae_class])
    best_mae_acc = float(np.mean(trues == best_mae_class))

    return {
        "uniform_random": {
            "acc": 0.1,
            "mae": rand_mae,
            "off1": rand_off1,
        },
        "best_const_by_acc": {
            "class_stars": best_acc_class + 1,
            "acc": best_acc,
            "mae": best_acc_mae,
        },
        "best_const_by_mae": {
            "class_stars": best_mae_class + 1,
            "acc": best_mae_acc,
            "mae": best_mae,
        },
    }


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path, title: str) -> None:
    classes = list(range(1, 11))
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(10), classes)
    ax.set_yticks(range(10), classes)
    ax.set_xlabel("Predicted stars")
    ax.set_ylabel("True stars")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            v = int(cm[i, j])
            if v == 0:
                continue
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--videos-root", default="videos")
    ap.add_argument("--splits-file", default="splits.json")
    ap.add_argument("--out-dir", default=None,
                    help="Defaults to <ckpt parent>/eval_<split>/.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--clips-eval", type=int, default=24)
    ap.add_argument("--n-segments", type=int, default=24)
    ap.add_argument("--backbone", default=None,
                    help="Override backbone (default: read from ckpt args).")
    ap.add_argument("--aggregation", default=None,
                    help="Override aggregation (default: read from ckpt args).")
    ap.add_argument("--head-kind", default=None, choices=list(_HEAD_KINDS),
                    help="Override head kind (default: read from ckpt args).")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {}) or {}
    backbone_name = args.backbone or saved_args.get("backbone")
    if backbone_name not in BACKBONES:
        raise ValueError(
            f"Backbone not found in checkpoint and not provided via --backbone "
            f"(got {backbone_name!r}; options: {sorted(BACKBONES)})"
        )
    aggregation = args.aggregation or saved_args.get("aggregation", "max")
    head_kind = args.head_kind or saved_args.get("head_kind", "softmax")
    if head_kind not in _HEAD_KINDS:
        raise ValueError(
            f"Unknown head_kind {head_kind!r}; expected one of {list(_HEAD_KINDS)}."
        )
    print(f"Checkpoint: {ckpt_path}")
    print(f"  backbone={backbone_name}  aggregation={aggregation}  "
          f"head_kind={head_kind}  "
          f"epoch={ckpt.get('epoch', '?')}  "
          f"best_val_mae={ckpt.get('best_val_mae', float('nan')):.4f}")

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else ckpt_path.parent / f"eval_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = make_or_load_splits(args.videos_root, args.splits_file, seed=args.seed)
    records = getattr(splits, args.split)
    print(f"{args.split} records: {len(records)}")

    ds = GeometryDashDataset(
        records, mode="eval",
        n_segments=args.n_segments,
        clips_per_video_eval=args.clips_eval,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    print(f"Building {backbone_name}...")
    backbone = build_backbone(backbone_name, pretrained=False)
    model = DifficultyModel(
        backbone,
        aggregation=aggregation,
        head_kind=head_kind,
        num_classes=_NUM_CLASSES,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    preds, trues, evs = run_inference(model, dl, device, use_amp, head_kind)
    n = len(trues)

    acc = float(np.mean(preds == trues))
    mae_argmax = float(np.mean(np.abs(preds - trues)))
    mae_ev = float(np.mean(np.abs(evs - (trues + 1))))
    off1 = float(np.mean(np.abs(preds - trues) <= 1))
    print(f"\nOverall (n={n}):")
    print(f"  acc                  = {acc:.4f}")
    print(f"  MAE (argmax)         = {mae_argmax:.4f}")
    print(f"  MAE (expected value) = {mae_ev:.4f}")
    print(f"  off-by-one acc       = {off1:.4f}")

    label_ids = list(range(10))
    prec, rec, f1, support = precision_recall_fscore_support(
        trues, preds, labels=label_ids, zero_division=0
    )
    print("\nPer-class (stars 1..10):")
    print(f"  {'star':>4}  {'support':>7}  {'prec':>6}  {'recall':>6}  {'f1':>6}")
    for c in label_ids:
        print(
            f"  {c+1:>4}  {int(support[c]):>7d}  "
            f"{prec[c]:>6.3f}  {rec[c]:>6.3f}  {f1[c]:>6.3f}"
        )

    cm = confusion_matrix(trues, preds, labels=label_ids)
    save_confusion_matrix_png(
        cm, out_dir / "confusion_matrix.png",
        title=f"{backbone_name} — {args.split} (n={n})",
    )

    base = compute_baselines(trues)
    print("\nBaselines (on this split):")
    ub = base["uniform_random"]
    print(
        f"  uniform-random       acc={ub['acc']:.4f}  "
        f"MAE={ub['mae']:.4f}  off1={ub['off1']:.4f}"
    )
    ba = base["best_const_by_acc"]
    print(
        f"  best-const-by-acc    class={ba['class_stars']}*  "
        f"acc={ba['acc']:.4f}  MAE={ba['mae']:.4f}"
    )
    bm = base["best_const_by_mae"]
    print(
        f"  best-const-by-MAE    class={bm['class_stars']}*  "
        f"acc={bm['acc']:.4f}  MAE={bm['mae']:.4f}"
    )

    summary = {
        "ckpt": str(ckpt_path),
        "split": args.split,
        "backbone": backbone_name,
        "aggregation": aggregation,
        "head_kind": head_kind,
        "n": int(n),
        "acc": acc,
        "mae_argmax": mae_argmax,
        "mae_ev": mae_ev,
        "off1": off1,
        "per_class": {
            (c + 1): {
                "support": int(support[c]),
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "f1": float(f1[c]),
            }
            for c in label_ids
        },
        "baselines": base,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    np.save(out_dir / "confusion_matrix.npy", cm)
    print(
        f"\nWrote: {out_dir}/metrics.json, "
        f"{out_dir}/confusion_matrix.png, {out_dir}/confusion_matrix.npy"
    )


if __name__ == "__main__":
    main()
