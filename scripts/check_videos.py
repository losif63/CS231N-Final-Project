"""Scan every video and report ones decord cannot read.

Use this to clean up the dataset after a download/copy that may have left
truncated files. The smoke-test DECORDError you hit is exactly what this
catches — open + small frame fetch fails when the file has no usable stream.

Run:
    python scripts/check_videos.py
    python scripts/check_videos.py --workers 8
    python scripts/check_videos.py --delete       # rm broken files (asks first)

Outputs:
- broken_videos.txt : one path per line, broken files only.
- A per-class summary table on stdout.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import scan_video_dir  # noqa: E402


def _probe(path: str) -> Tuple[str, bool, str]:
    """Return (path, ok, reason). Imports happen here so workers stay light."""
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        n = len(vr)
        if n <= 0:
            return path, False, f"zero frames (len={n})"
        # Touch the first and last frame to catch broken indexes too.
        _ = vr.get_batch([0, max(0, n - 1)])
        return path, True, ""
    except Exception as e:  # noqa: BLE001 — we want every failure mode
        return path, False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-root", default="videos")
    ap.add_argument("--out", default="broken_videos.txt")
    ap.add_argument("--workers", type=int, default=4,
                    help="Process workers. Use 1 to disable multiprocessing.")
    ap.add_argument("--delete", action="store_true",
                    help="Delete broken files after listing (prompts for confirmation).")
    args = ap.parse_args()

    records = scan_video_dir(args.videos_root)
    print(f"Scanning {len(records)} videos with {args.workers} worker(s)...")

    broken: List[Tuple[str, str, int]] = []  # (path, reason, label)
    path_to_label = {r.path: r.label for r in records}

    if args.workers <= 1:
        it = tqdm(records, desc="scan")
        for r in it:
            _, ok, reason = _probe(r.path)
            if not ok:
                broken.append((r.path, reason, r.label))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_probe, r.path): r.path for r in records}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="scan"):
                path, ok, reason = fut.result()
                if not ok:
                    broken.append((path, reason, path_to_label[path]))

    out = Path(args.out)
    out.write_text("\n".join(p for p, _, _ in broken) + ("\n" if broken else ""))
    print(f"\nBroken: {len(broken)} / {len(records)}  →  {out}")

    if broken:
        per_class: Counter = Counter(label for _, _, label in broken)
        print("\nBy class (stars):")
        for lbl in range(10):
            n = per_class.get(lbl, 0)
            if n:
                print(f"  {lbl + 1}★ : {n}")

        print("\nFirst 10 broken files (path → reason):")
        for path, reason, _ in broken[:10]:
            print(f"  {path}\n    -> {reason}")

        if args.delete:
            ans = input(f"\nDelete all {len(broken)} broken files? [y/N] ").strip().lower()
            if ans == "y":
                removed = 0
                for path, _, _ in broken:
                    try:
                        Path(path).unlink()
                        removed += 1
                    except OSError as e:
                        print(f"  failed to remove {path}: {e}")
                print(f"Removed {removed} files.")
                print(
                    "If you've already created splits.json, delete it so the "
                    "split regenerates without the removed files."
                )
            else:
                print("Skipped deletion.")


if __name__ == "__main__":
    main()
