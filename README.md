# CS231N Final Project — Geometry Dash Difficulty Prediction

Predict the community-rated star difficulty (1–10) of a Geometry Dash level
from a YouTube playthrough video. The pipeline fine-tunes Kinetics-pretrained
video backbones on ~5,500 labeled gameplay videos.

---

## Pipeline at a glance

```
gd.js (Node)              levels/{n}stars/metadata/*.json
   └── collect.js  ───►   (level metadata, grouped by star rating)
                                │
yt_dlp (Python)                 ▼
   └── download_all_videos.py ► videos/{n}stars/*.mp4
                                │
                                ▼  (label = parent dir name)
   GeometryDashDataset  ────►  K-clip tensors  ─►  X3D / R(2+1)D / SlowFast / I3D
                                                     │
                                                     ▼
                                            mean / max / topk_mean
                                                     │
                                                     ▼
                                          MLP head → 10-class logits
```

---

## Directory layout

```
src/
  data/
    dataset.py          GeometryDashDataset, VideoRecord, scan_video_dir
    splits.py           Stratified 80/10/10 splits persisted to splits.json
    transforms.py       Raw-clip transform (short-side resize + canonical width + color jitter)
  models/
    _video_ops.py       subsample_temporal, resize_spatial (shared helpers)
    x3d.py              X3DBackbone        (PyTorchVideo x3d_m,        feature_dim 2048)
    r2plus1d.py         R2Plus1DBackbone   (torchvision r2plus1d_18,   feature_dim 512)
    slowfast.py         SlowFastBackbone   (PyTorchVideo slowfast_r50, feature_dim 2304)
    i3d.py              I3DBackbone        (PyTorchVideo i3d_r50,      feature_dim 2048)
    factory.py          build_backbone(name)
    difficulty_model.py DifficultyModel: backbone → K-aggregation → MLP head
scripts/
  verify_data.py        Eyeball dataset: prints split stats, saves frame grids
  check_videos.py       One-shot decode scan to find broken/truncated videos
  sanity_train.py       1-epoch smoke training on a stratified subset (any backbone)
  sanity_train_x3d.py   Older X3D-only smoke script (superseded by sanity_train.py)
data_collection/
  get_top_videos.py     yt_dlp wrapper for a single query
  download_all_videos.py  Iterates levels/{n}stars/metadata/*.json and downloads
geometry_dash/
  collect.js            Pulls level metadata via gd.js into levels/{n}stars/metadata/
  migrate.js
videos/                 Downloaded gameplay videos (gitignored)
levels/                 Per-level metadata JSON (gitignored)
splits.json             Created on first dataset use; stratified 80/10/10
```

---

## Environment setup

Python side (training pipeline):

```bash
conda create -n cs231n python=3.10
conda activate cs231n
pip install -r requirements.txt
```

The `requirements.txt` pins:

- `torch==2.1.2`, `torchvision==0.16.2`
- `pytorchvideo==0.1.5`  (last release; works against torch 2.1 despite metadata warnings)
- `decord==0.6.0`        (frame decoding)
- `numpy<2.0`            (fvcore breaks on numpy 2.x)
- `pyyaml`, `tensorboard`, `matplotlib`, `scikit-learn`, `tqdm`

Node side (level-metadata collection only, runs once):

```bash
npm init -y
npm install gd.js
```

---

## Data collection

### 1. Level metadata (Node, one-time)

`geometry_dash/collect.js` queries `gd.js` for levels and writes metadata to
`levels/{n}stars/metadata/{level_id}.json`. Star ratings (1–10) come from the
GD API and become directory names.

```bash
node geometry_dash/collect.js
```

The difficulty list inside the script is the bucket of difficulties to pull
in this run — edit the `difficulties` array to widen/narrow.

### 2. YouTube videos (Python)

`data_collection/download_all_videos.py` walks the metadata files and queries
YouTube for `"Geometry Dash <name> by <author> <id>"`, downloading the top
result(s) into `videos/{n}stars/{id}_{k}.mp4`.

```bash
python data_collection/download_all_videos.py                 # 1 result per level, all classes
python data_collection/download_all_videos.py --num-results 3 # top 3 results per level
python data_collection/download_all_videos.py --stars 7 8 9   # only specific classes
```

Note: yt_dlp may need cookies for some videos; the repo expects a `cookies.txt`
at the root (gitignored).

### 3. Cleaning up broken downloads

YouTube downloads can be truncated mid-stream — decord will then fail with
`cannot find video stream`. Scan all files once and clean up:

```bash
python scripts/check_videos.py --workers 8                 # writes broken_videos.txt
python scripts/check_videos.py --workers 8 --delete        # asks before rm
rm splits.json                                             # so splits regenerate without removed files
```

Roughly 1% of YouTube-scraped files tend to be unreadable.

---

## Verifying the dataset (Phase 1)

Before training anything, sanity-check that K-clip extraction, label
inference, and resizing all look right:

```bash
python scripts/verify_data.py
```

This:

- Builds (or loads) the stratified 80/10/10 split → `splits.json`.
- Prints per-split per-class counts.
- Loads a few samples in `train` and `eval` mode; prints tensor shape, dtype,
  value range.
- Saves frame grids under `verify_out/` — one PNG per probed sample, K rows
  × 4 evenly-spaced frames, with the inferred label in the title.
- Pulls one DataLoader batch to confirm collation gives `(B, K, T, C, H, W)`.

What to eyeball in `verify_out/`:

1. Frames are 224 tall × ~398 wide (no center-square crop).
2. Within a row, frames advance in time (clip is animated).
3. Across rows, clips come from different parts of the video.
4. The label in the title matches the parent directory.

---

## Backbone smoke training (Phases 2 & 3)

`scripts/sanity_train.py` runs 1 epoch on a stratified 10-per-class subset
(~100 train, ~20 val by default) just to confirm the pipeline runs and the
loss carries training signal. It works with any of the four backbones:

```bash
python scripts/sanity_train.py --backbone x3d
python scripts/sanity_train.py --backbone r2plus1d
python scripts/sanity_train.py --backbone slowfast --clips-train 2 --clips-eval 4
python scripts/sanity_train.py --backbone i3d
```

What gets printed:

- Device (CUDA if available else CPU; AMP auto-enables on CUDA).
- Subset sizes per split.
- `feature_dim`, total / trainable param count.
- A tqdm bar with per-batch loss + 20-batch rolling average.
- First/last 10-batch loss averages so you can see whether the loss is moving.
- Val: accuracy, MAE (over softmax expected value), off-by-one accuracy.

References for the val numbers: a uniform-random predictor gets accuracy ≈ 0.10,
MAE ≈ 3.3, off-by-one ≈ 0.28. After 1 epoch on 100 samples, expect val to be
near-random; the point is just that everything **computes** and the **loss is
trending down**.

### First-run weight downloads

On first invocation of each backbone, `torch.hub` / torchvision fetches
pretrained Kinetics-400 weights into `~/.cache/torch/hub/`:

| Backbone   | Source          | Native input        | Approx download |
| ---------- | --------------- | ------------------- | --------------- |
| `x3d`      | PyTorchVideo    | `(3, 16, 224, 224)` | ~30 MB          |
| `r2plus1d` | torchvision     | `(3, 16, 112, 112)` | ~64 MB          |
| `slowfast` | PyTorchVideo    | slow + fast list    | ~250 MB         |
| `i3d`      | PyTorchVideo    | `(3, 32, 224, 224)` | ~120 MB         |

---

## Design notes

### Each video is one data point

The dataset emits **K clips per video**, not K separate samples. K is
configurable (train default 8, eval default = n_segments = 24). Each clip is
32 frames at sampling stride 2 from 30 FPS source (≈2 sec of gameplay).
`__getitem__` returns `(clips, label)` of shape `(K, T, C, H, W)`. The
DataLoader stacks into `(B, K, T, C, H, W)`; the model reshapes to `(B*K, ...)`,
runs the backbone, then aggregates over K before the head. **Aggregation is
learned end-to-end, not done by averaging predictions.**

### Train vs. eval temporal sampling

The video is divided into `n_segments=24` equal segments and we take one clip
per chosen segment.

- **Train**: pick a random subset of 8 segments per epoch; random start within
  each segment (mild temporal augmentation).
- **Eval**: all 24 segments, deterministically centered.

This is why a random predictor's baseline MAE drops with `clips_per_video_eval`
(more temporal context → less noisy expected value).

### Per-backbone preprocessing

The dataset emits one canonical "raw" clip per video: aspect-preserving
resize to **short side = 224**, horizontal context preserved at canonical
width **W = 398** (center-crop if wider, symmetric zero-pad if narrower).
Float in `[0, 1]`. **Each backbone then owns its own** subsample / resize /
normalization (and pathway construction for SlowFast). Backbones share a
common signature: `(B, C, T, H, W) → (B, feature_dim)`.

### Aggregation strategies

`DifficultyModel(backbone, aggregation=...)` supports:

- `mean` — average over K.
- `max` — feature-wise max over K.
- `topk_mean` — mean of the top-k features along the K dimension.

All three are differentiable and learned end-to-end with the head.

### Robustness

- **Decode errors are recoverable**: `__getitem__` retries forward through
  the dataset on `DECORDError` / `OSError` and marks broken indices in a
  per-worker set so they're skipped in future calls. The real fix is still
  to run `scripts/check_videos.py --delete` once.
- **Labels are 0-indexed internally** (CE-friendly); displayed as 1–10 stars
  in user-facing output.
- **AMP**: `torch.cuda.amp.GradScaler` + `autocast`, auto-enabled on CUDA and
  no-op on CPU.

---

## What is not yet implemented

Phase 4+ deliverables, in order:

1. Full training script (`src/train.py`): cosine LR with 5% warmup, gradient
   accumulation, TensorBoard logging, best-by-MAE checkpointing, resume.
2. Evaluation script (`src/eval.py`): per-class precision/recall, confusion
   matrix PNG, baselines (uniform-random + optimal constant predictor by
   MAE and by accuracy).
3. YAML configs in `configs/` (`base.yaml`, `x3d.yaml`, `r2plus1d.yaml`,
   `slowfast.yaml`, `i3d.yaml`), loaded into a dataclass — no Hydra.

---

## Known gotchas

- **`torch.hub.load`** first call needs internet; subsequent calls use the
  cache. If you ever need a fully-offline run, pre-warm with one connected run.
- **CPU runs are slow.** For local pipeline-level smoke testing, shrink with
  `--n-train 20 --n-val 8 --num-workers 0 --clips-train 2 --clips-eval 2`.
- **`splits.json` is sticky**: once written it's reused as-is. If you delete
  or add videos, delete `splits.json` before the next run so it regenerates.
- **PyTorchVideo head shape**: each backbone wrapper asserts the
  expected `head.proj` is a `Linear` before stripping it; if PyTorchVideo
  changes its head structure in a future version, the error message names
  the offending backbone.

---

## License & data

Videos and level metadata are scraped from third-party sources for academic
use within this CS231N project. They are gitignored.
