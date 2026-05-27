"""Pre-encode all training videos to a canonical 30fps, 224x398 MP4 layout.

The dataset's per-__getitem__ decord work is dominated by random-access
decoding of full-resolution (often 1080p) source videos with variable fps.
Re-encoding once to the dataset's canonical resolution + fps with a short
GOP gives ~5–10x faster epochs and removes the resize / fps-remap work from
the hot path.

Walks `<src>/{n}stars/*.{mp4,mkv,webm,mov}` (1..10 stars only — matches
`scan_video_dir`) and writes `<dst>/{n}stars/<basename>.mp4` with:

  - 30 fps fixed
  - scale-to-fit + center-pad to 224 (H) x 398 (W). Exact for 16:9 sources
    (the typical YouTube gameplay case); 4:3 etc. get black side bars.
  - libx264, CRF 23, short GOP (keyframe every 15 frames → 0.5s @ 30fps) so
    decord's clip-sampling pattern doesn't pay a big keyframe-walk tax.
  - No audio.

Requires `ffmpeg` on PATH. Install with e.g.:
    sudo apt install ffmpeg

Atomic writes: each output is first written to `<dst>.part.mp4` and renamed
on success. Re-runs skip files that already exist (use --overwrite to redo).

After running, delete splits.json so it regenerates against the new layout,
then point training at the new dir:

    python scripts/preencode_videos.py --src videos --dst videos_processed
    rm splits.json
    python src/train.py --backbone x3d --videos-root videos_processed ...

Example with more parallelism on a beefy box:
    python scripts/preencode_videos.py --workers 12 --threads-per-job 2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}


def find_inputs(src_root: Path) -> List[Path]:
    """Mirror `scan_video_dir`: only `{1..10}stars/*.<ext>` files."""
    out: List[Path] = []
    for star_dir in sorted(src_root.iterdir()):
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
        for vid in sorted(star_dir.iterdir()):
            if vid.is_file() and vid.suffix.lower() in _VIDEO_EXTS:
                out.append(vid)
    return out


def dst_path(src: Path, src_root: Path, dst_root: Path) -> Path:
    return dst_root / src.relative_to(src_root).with_suffix(".mp4")


def encode_one(
    src: Path,
    dst: Path,
    *,
    height: int,
    width: int,
    fps: int,
    crf: int,
    gop: int,
    threads: int,
    timeout_s: int,
    overwrite: bool,
) -> Tuple[Path, str, str]:
    """Returns (src, status, msg). status ∈ {"ok", "skip", "fail"}."""
    if dst.exists() and not overwrite:
        return (src, "skip", "exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    if tmp.exists():
        tmp.unlink()

    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(src),
        "-vf", vf,
        "-r", str(fps),
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-g", str(gop),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", str(threads),
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink()
        return (src, "fail", f"timeout after {timeout_s}s")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return (src, "fail", f"{type(e).__name__}: {e}")

    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return (src, "fail", (proc.stderr or "").strip()[-500:] or "ffmpeg failed")

    tmp.rename(dst)
    return (src, "ok", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="videos")
    ap.add_argument("--dst", default="videos_processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads-per-job", type=int, default=2,
                    help="ffmpeg -threads value. workers * threads ~ logical cores.")
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=398)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=23,
                    help="Lower=better quality+bigger. 18 visually lossless, 28 small.")
    ap.add_argument("--gop", type=int, default=15,
                    help="Keyframe interval (frames). Smaller=faster seek, bigger files.")
    ap.add_argument("--timeout", type=int, default=900,
                    help="Per-video timeout in seconds.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process this many videos (0=all). For dry runs.")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install it (e.g. `sudo apt install ffmpeg`).",
              file=sys.stderr)
        sys.exit(1)

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    if not src_root.is_dir():
        print(f"ERROR: src not found: {src_root}", file=sys.stderr)
        sys.exit(1)
    dst_root.mkdir(parents=True, exist_ok=True)

    inputs = find_inputs(src_root)
    if args.limit:
        inputs = inputs[: args.limit]
    print(f"Found {len(inputs)} input videos in {src_root}.")
    print(f"Writing to {dst_root}/")
    print(f"Workers: {args.workers}  ffmpeg threads/job: {args.threads_per_job}")
    print(
        f"Target: {args.width}x{args.height} @ {args.fps}fps  "
        f"CRF={args.crf}  GOP={args.gop}"
    )

    jobs = [(src, dst_path(src, src_root, dst_root)) for src in inputs]

    ok = 0
    skipped = 0
    failed: List[Tuple[Path, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                encode_one, src, dst,
                height=args.height, width=args.width, fps=args.fps,
                crf=args.crf, gop=args.gop,
                threads=args.threads_per_job,
                timeout_s=args.timeout,
                overwrite=args.overwrite,
            )
            for src, dst in jobs
        ]
        with tqdm(total=len(futures), desc="encode") as pbar:
            for fut in as_completed(futures):
                src, status, msg = fut.result()
                if status == "skip":
                    skipped += 1
                elif status == "ok":
                    ok += 1
                else:
                    failed.append((src, msg))
                pbar.update(1)
                pbar.set_postfix(ok=ok, skip=skipped, fail=len(failed))

    print(f"\nDone. ok={ok}  skipped={skipped}  failed={len(failed)}")
    if failed:
        print(f"First {min(10, len(failed))} failures:")
        for src, msg in failed[:10]:
            try:
                rel = src.relative_to(src_root)
            except ValueError:
                rel = src
            print(f"  {rel}: {msg}")
        log_path = dst_root.parent / "preencode_failed.txt"
        with log_path.open("w") as f:
            for src, msg in failed:
                f.write(f"{src}\t{msg}\n")
        print(f"  Full list: {log_path}")

    print("\nNext steps:")
    print(f"  rm splits.json    # regenerate against {dst_root.name}/")
    print(f"  python src/train.py --backbone <name> --videos-root {dst_root}")


if __name__ == "__main__":
    main()
