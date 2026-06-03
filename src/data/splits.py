"""Stratified train/val/test split, persisted to JSON for reproducibility."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .dataset import VideoRecord, scan_video_dir


@dataclass
class Splits:
    train: List[VideoRecord]
    val: List[VideoRecord]
    test: List[VideoRecord]


def make_or_load_splits(
    videos_root: str,
    splits_path: str,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> Splits:
    """Load splits from `splits_path` if it exists, else create + save.

    Splitting is stratified by difficulty class so each split keeps the
    per-class distribution close to the overall one. Paths missing from the
    on-disk file (e.g. a video that was deleted since the split was made) are
    silently skipped; paths missing from the current scan are dropped on load.
    """
    if not (0.0 < train_frac < 1.0 and 0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0):
        raise ValueError("Fractions must be in (0,1) / [0,1).")
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0.")

    sp = Path(splits_path)
    records = scan_video_dir(videos_root)
    by_path = {r.path: r for r in records}

    if sp.exists():
        with sp.open() as f:
            data = json.load(f)
        return Splits(
            train=[by_path[p] for p in data["train"] if p in by_path],
            val=[by_path[p] for p in data["val"] if p in by_path],
            test=[by_path[p] for p in data["test"] if p in by_path],
        )

    buckets: Dict[int, List[VideoRecord]] = defaultdict(list)
    for r in records:
        buckets[r.label].append(r)

    rng = random.Random(seed)
    train: List[VideoRecord] = []
    val: List[VideoRecord] = []
    test: List[VideoRecord] = []
    for label in sorted(buckets):
        items = buckets[label][:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Anything left (could be 1 off due to rounding) goes to test.
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("w") as f:
        json.dump(
            {
                "seed": seed,
                "train": [r.path for r in train],
                "val": [r.path for r in val],
                "test": [r.path for r in test],
            },
            f,
            indent=2,
        )
    return Splits(train=train, val=val, test=test)
