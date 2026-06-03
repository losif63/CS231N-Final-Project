"""Pre-extract frames from all videos at a fixed interval.

Output structure:
    frames_dir/{n}stars/{video_stem}/0001.jpg
                                     0002.jpg
                                     ...

Frames are center-cropped to a square and saved at --size x --size pixels.
This is a one-time step; training then loads JPEGs directly (much faster).

Run from the repo root:
    python cnn_baseline_network/extract_frames.py --videos-dir videos/ --frames-dir frames/

Test with a small subset:
    python cnn_baseline_network/extract_frames.py --videos-dir videos/ --frames-dir frames/ --limit 5
"""

import argparse
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def extract_video(video_path: Path, output_dir: Path, frame_interval: float, size: int) -> tuple[int, bool]:
    """Extract frames from one video using ffmpeg.

    Returns (n_frames_extracted, was_skipped).
    """
    existing = list(output_dir.glob("*.jpg"))
    if existing:
        return len(existing), True

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = 1.0 / frame_interval
    # crop='min(iw,ih)':'min(iw,ih)' takes the largest centered square,
    # then scale brings it to the target resolution.
    vf = f"fps={fps},crop='min(iw,ih)':'min(iw,ih)',scale={size}:{size}"

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", vf,
        "-q:v", "2",          # high-quality JPEG (1=best, 31=worst)
        str(output_dir / "%04d.jpg"),
        "-hide_banner", "-loglevel", "error",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warn] ffmpeg failed for {video_path.name}: {result.stderr[:120]}", flush=True)
        return 0, False

    n_frames = len(list(output_dir.glob("*.jpg")))
    return n_frames, False


def collect_videos(videos_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (video_path, stars_folder_name) for all valid videos."""
    entries = []
    for stars_dir in sorted(videos_dir.iterdir()):
        if not stars_dir.is_dir():
            continue
        name = stars_dir.name
        if not name.endswith("stars"):
            continue
        try:
            stars = int(name[: -len("stars")])
        except ValueError:
            continue
        if not (1 <= stars <= 10):
            continue
        for video_file in sorted(stars_dir.iterdir()):
            if video_file.suffix.lower() in VIDEO_EXTENSIONS:
                entries.append((video_file, name))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract frames from Geometry Dash videos")
    parser.add_argument("--videos-dir", type=Path, default=Path("videos/"),
                        help="Root dir with {n}stars/ subdirs of videos")
    parser.add_argument("--frames-dir", type=Path, default=Path("frames/"),
                        help="Output root dir for extracted frames")
    parser.add_argument("--frame-interval", type=float, default=4.0,
                        help="Seconds between extracted frames (default: 4)")
    parser.add_argument("--size", type=int, default=256,
                        help="Output frame resolution — square (default: 256)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many videos, 0 = no limit (useful for testing)")
    args = parser.parse_args()

    videos = collect_videos(args.videos_dir)
    if args.limit:
        videos = videos[: args.limit]

    print(f"Processing {len(videos)} videos -> {args.frames_dir}", flush=True)

    total_frames = 0
    skipped = 0
    for i, (video_path, stars_name) in enumerate(videos):
        output_dir = args.frames_dir / stars_name / video_path.stem
        n_frames, was_skipped = extract_video(video_path, output_dir, args.frame_interval, args.size)
        total_frames += n_frames
        if was_skipped:
            skipped += 1
            tag = "skip"
        else:
            tag = "done"
        print(f"[{i+1}/{len(videos)}] {tag}  {stars_name}/{video_path.name}  ({n_frames} frames)", flush=True)

    print(f"\nFinished. {len(videos) - skipped} extracted, {skipped} skipped (already existed).", flush=True)
    print(f"Total frames on disk: {total_frames}", flush=True)


if __name__ == "__main__":
    main()
