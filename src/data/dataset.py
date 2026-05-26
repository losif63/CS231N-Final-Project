"""GeometryDashDataset — one data point per video, K clips per data point.

Reads videos at a target FPS (default 30) regardless of source FPS by mapping
target-fps frame indices into source-fps indices on the fly. Each video is
divided into `n_segments` equal segments; we extract one clip per chosen
segment:

  - train mode: random subset of `clips_per_video_train` segments per epoch,
    random start within each segment (temporal augmentation).
  - eval mode: all `n_segments` segments (or `clips_per_video_eval` if set),
    deterministically centered.

Returns `(clips, label)` where clips has shape `(K, T, C, H, W)` and label is
0-indexed (0..9) corresponding to 1..10 stars. K is constant within a split.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import decord
import numpy as np
import torch
from decord._ffi.base import DECORDError
from torch.utils.data import Dataset

from .transforms import RawClipTransform

decord.bridge.set_bridge("torch")

_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}


@dataclass(frozen=True)
class VideoRecord:
    path: str
    label: int  # 0..9, where label = stars - 1


def scan_video_dir(root: str | os.PathLike) -> List[VideoRecord]:
    """Walk `<root>/{n}stars/*.mp4` and return records sorted by path.

    Labels are 0-indexed: `1stars` → 0, `10stars` → 9. Directories whose names
    don't match `{int}stars` are ignored. Star counts outside 1..10 are ignored
    (e.g. `0stars` from the unrated levels bucket is dropped).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Videos root not found: {root}")

    records: List[VideoRecord] = []
    for star_dir in sorted(root.iterdir()):
        if not star_dir.is_dir():
            continue
        name = star_dir.name
        if not name.endswith("stars"):
            continue
        try:
            stars = int(name[: -len("stars")])
        except ValueError:
            continue
        if not (1 <= stars <= 10):
            continue
        label = stars - 1
        for vid in sorted(star_dir.iterdir()):
            if vid.is_file() and vid.suffix.lower() in _VIDEO_EXTS:
                records.append(VideoRecord(path=str(vid), label=label))
    return records


class GeometryDashDataset(Dataset):
    """See module docstring."""

    def __init__(
        self,
        records: List[VideoRecord],
        *,
        mode: str,
        n_segments: int = 24,
        clips_per_video_train: int = 8,
        clips_per_video_eval: Optional[int] = None,
        T: int = 32,
        sampling_stride: int = 2,
        target_fps: int = 30,
        short_side: int = 224,
        canon_width: int = 398,
        color_jitter: float = 0.2,
    ) -> None:
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        if clips_per_video_train > n_segments:
            raise ValueError(
                f"clips_per_video_train ({clips_per_video_train}) > n_segments ({n_segments})"
            )
        eval_k = clips_per_video_eval if clips_per_video_eval is not None else n_segments
        if eval_k > n_segments:
            raise ValueError(
                f"clips_per_video_eval ({eval_k}) > n_segments ({n_segments})"
            )

        self.records = list(records)
        self.mode = mode
        self.n_segments = n_segments
        self.clips_per_video_train = clips_per_video_train
        self.clips_per_video_eval = eval_k
        self.T = T
        self.sampling_stride = sampling_stride
        self.target_fps = target_fps
        self.clip_len_30 = T * sampling_stride  # span of one clip in target-fps frames

        self.transform = RawClipTransform(
            short_side=short_side,
            canon_width=canon_width,
            color_jitter=color_jitter if mode == "train" else 0.0,
        )

        # Per-instance (per-worker, in DataLoader) cache of indices we've
        # found to be unreadable; we skip them on retry.
        self._broken: set[int] = set()

    @property
    def K(self) -> int:
        return (
            self.clips_per_video_train
            if self.mode == "train"
            else self.clips_per_video_eval
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load `idx`, skipping forward through broken videos on decode errors.

        A single corrupt file would otherwise kill the DataLoader. We mark
        failing indices in `self._broken` so the same worker won't retry them.
        """
        n = len(self.records)
        cur = idx
        tried = 0
        while tried < n:
            if cur in self._broken:
                cur = (cur + 1) % n
                tried += 1
                continue
            try:
                return self._load(cur)
            except (DECORDError, OSError, RuntimeError) as e:
                warnings.warn(
                    f"Skipping unreadable video at idx={cur} "
                    f"({self.records[cur].path}): {type(e).__name__}: {e}",
                    stacklevel=2,
                )
                self._broken.add(cur)
                cur = (cur + 1) % n
                tried += 1
        raise RuntimeError(
            f"All {n} videos failed to load (broken indices: {sorted(self._broken)[:10]}...)"
        )

    def _load(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        vr = decord.VideoReader(rec.path, num_threads=1)
        src_fps = float(vr.get_avg_fps()) or float(self.target_fps)
        n_src = len(vr)
        if n_src == 0:
            raise RuntimeError(f"Video has zero frames: {rec.path}")

        # Video length in target-fps frames.
        n_30 = max(1, int(round(n_src * self.target_fps / src_fps)))
        if n_30 < self.clip_len_30:
            warnings.warn(
                f"{rec.path}: only {n_30} frames @ {self.target_fps} fps "
                f"(< clip_len {self.clip_len_30}); looping frames to fill clips.",
                stacklevel=2,
            )

        # Choose segments.
        if self.mode == "train":
            chosen = sorted(random.sample(range(self.n_segments), k=self.K))
        else:
            # Evenly spaced segments when eval K < n_segments; all otherwise.
            if self.K == self.n_segments:
                chosen = list(range(self.n_segments))
            else:
                # Linear sampling, deterministic.
                step = self.n_segments / self.K
                chosen = [min(self.n_segments - 1, int(round(i * step))) for i in range(self.K)]

        # Compute target-fps start index for each clip.
        seg_len = n_30 / self.n_segments
        clip_starts_30: List[int] = []
        for s in chosen:
            seg_start = s * seg_len
            seg_end = (s + 1) * seg_len
            if self.mode == "train":
                hi = max(seg_start, seg_end - self.clip_len_30)
                start = random.uniform(seg_start, hi) if hi > seg_start else seg_start
            else:
                start = (seg_start + seg_end) / 2.0 - self.clip_len_30 / 2.0
            # Clamp into a valid range for finite-length videos.
            if n_30 > self.clip_len_30:
                start = float(np.clip(start, 0.0, n_30 - self.clip_len_30))
            else:
                start = 0.0
            clip_starts_30.append(int(round(start)))

        # Source-frame indices for every (clip, frame).
        flat_src_indices: List[int] = []
        for cs in clip_starts_30:
            for t in range(self.T):
                idx30 = cs + t * self.sampling_stride
                idx30 = idx30 % n_30  # loop for very short videos
                src_idx = int(round(idx30 * src_fps / self.target_fps))
                if src_idx < 0:
                    src_idx = 0
                elif src_idx >= n_src:
                    src_idx = n_src - 1
                flat_src_indices.append(src_idx)

        # decord with torch bridge → uint8 tensor (N, H, W, C).
        frames = vr.get_batch(flat_src_indices)
        del vr  # release file handle promptly

        # (K, T, H, W, C) → transform → (K, T, C, H, W) float in [0,1].
        frames = frames.reshape(self.K, self.T, *frames.shape[1:])
        clips = self.transform(frames)

        return clips, rec.label
