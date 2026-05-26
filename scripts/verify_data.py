"""Sanity-check the dataset before wiring up any model.

What it does
------------
1. Builds (or loads) the stratified splits.
2. Prints per-split class distributions.
3. Loads a few samples in train mode and eval mode and prints tensor shapes,
   dtypes, and value ranges.
4. Saves a frame grid PNG per sample under `verify_out/` so you can eyeball:
   - K rows, 4 evenly-spaced frames per clip.
   - One PNG per probed sample.
5. Pulls one batch through a DataLoader to confirm collation produces the
   expected `(B, K, T, C, H, W)` shape.

Run (per your conda env note):
    conda run -n cs231n python scripts/verify_data.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make `src.*` importable when running this script from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import GeometryDashDataset  # noqa: E402
from src.data.splits import make_or_load_splits  # noqa: E402


def save_frame_grid(clips: torch.Tensor, stars_1_10: int, out_path: Path, title: str) -> None:
    """`clips` is (K, T, C, H, W) float in [0,1]. Save K rows × 4 columns."""
    K, T, C, H, W = clips.shape
    cols = 4
    frame_idxs = np.linspace(0, T - 1, cols).round().astype(int)

    fig, axes = plt.subplots(K, cols, figsize=(cols * 3.0, K * 1.7))
    if K == 1:
        axes = axes[None, :]
    for k in range(K):
        for j, t in enumerate(frame_idxs):
            ax = axes[k, j]
            img = clips[k, t].permute(1, 2, 0).cpu().numpy()
            ax.imshow(np.clip(img, 0.0, 1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"clip {k}", fontsize=8)
            if k == 0:
                ax.set_title(f"t={t}/{T - 1}", fontsize=8)
    fig.suptitle(f"{title}  |  label = {stars_1_10}★", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def class_dist(records) -> dict:
    counter = Counter(r.label for r in records)
    return {k + 1: counter.get(k, 0) for k in range(10)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-root", default="videos")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--n-samples", type=int, default=3,
                    help="How many train and eval samples to probe.")
    ap.add_argument("--n-segments", type=int, default=24)
    ap.add_argument("--clips-train", type=int, default=8)
    ap.add_argument("--clips-eval", type=int, default=None,
                    help="Defaults to n-segments. Override to keep grid PNGs small.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random as pyrand

    pyrand.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits = make_or_load_splits(args.videos_root, args.splits, seed=args.seed)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    print(f"Total records : {total}")
    print(f"  train = {len(splits.train):>5}   per-class {class_dist(splits.train)}")
    print(f"  val   = {len(splits.val):>5}   per-class {class_dist(splits.val)}")
    print(f"  test  = {len(splits.test):>5}   per-class {class_dist(splits.test)}")

    ds_train = GeometryDashDataset(
        splits.train,
        mode="train",
        n_segments=args.n_segments,
        clips_per_video_train=args.clips_train,
    )
    ds_eval = GeometryDashDataset(
        splits.val,
        mode="eval",
        n_segments=args.n_segments,
        clips_per_video_eval=args.clips_eval,
    )

    out = Path(args.out)
    print(f"\nProbing {args.n_samples} train + {args.n_samples} eval samples → {out}/")

    print("\n[train mode]")
    for i in range(min(args.n_samples, len(ds_train))):
        clips, label = ds_train[i]
        rec = ds_train.records[i]
        print(
            f"  [{i}] {Path(rec.path).name:<30}  "
            f"clips={tuple(clips.shape)}  label={label + 1}★ (0-idx {label})  "
            f"dtype={clips.dtype}  min={clips.min():.3f} max={clips.max():.3f}"
        )
        save_frame_grid(clips, label + 1, out / f"train_{i:03d}.png", Path(rec.path).name)

    print("\n[eval mode]")
    for i in range(min(args.n_samples, len(ds_eval))):
        clips, label = ds_eval[i]
        rec = ds_eval.records[i]
        print(
            f"  [{i}] {Path(rec.path).name:<30}  "
            f"clips={tuple(clips.shape)}  label={label + 1}★ (0-idx {label})  "
            f"dtype={clips.dtype}  min={clips.min():.3f} max={clips.max():.3f}"
        )
        save_frame_grid(clips, label + 1, out / f"eval_{i:03d}.png", Path(rec.path).name)

    print("\n[dataloader]")
    from torch.utils.data import DataLoader

    dl = DataLoader(ds_train, batch_size=2, shuffle=False, num_workers=0)
    batch_clips, batch_labels = next(iter(dl))
    print(
        f"  batch_clips.shape = {tuple(batch_clips.shape)}  "
        f"batch_labels = {batch_labels.tolist()}  "
        f"dtype={batch_clips.dtype}"
    )

    print(f"\nDone. Open `{out}/` to eyeball the frame grids.")


if __name__ == "__main__":
    main()
