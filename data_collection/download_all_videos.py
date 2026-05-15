"""Iterate over levels/{n}stars/metadata/*.json and download YouTube playthroughs.

Output layout: videos/{n}stars/{id}_{k}.{ext}, where k is the 1-based index of the
result returned by the YouTube search query (so num_results > 1 produces _1, _2, ...).

Run from the repo root:
    python data_collection/download_all_videos.py
    python data_collection/download_all_videos.py --num-results 3
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yt_dlp


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEVELS_DIR = REPO_ROOT / "levels"
DEFAULT_VIDEOS_DIR = REPO_ROOT / "videos"
DEFAULT_COOKIES_FILE = REPO_ROOT / "cookies.txt"


def build_query(level_name: str, author_name: str, level_id: int) -> str:
    return f"Geometry Dash {level_name} by {author_name} {level_id}"


def existing_indexes(target_dir: Path, level_id: int, num_results: int) -> set[int]:
    found: set[int] = set()
    for k in range(1, num_results + 1):
        if any(target_dir.glob(f"{level_id}_{k}.*")):
            found.add(k)
    return found


def download_for_level(
    metadata: dict,
    stars_bucket: str,
    videos_dir: Path,
    num_results: int,
    download_delay: int,
    cookies_file: Path | None,
) -> None:
    level_id = metadata["id"]
    level_name = metadata["name"]
    author_name = metadata["author"]["username"]

    target_dir = videos_dir / stars_bucket
    target_dir.mkdir(parents=True, exist_ok=True)

    have = existing_indexes(target_dir, level_id, num_results)
    missing = [k for k in range(1, num_results + 1) if k not in have]
    if not missing:
        print(f"[skip] {stars_bucket}/{level_id}: all {num_results} video(s) present")
        return

    query = build_query(level_name, author_name, level_id)
    print(f"[get ] {stars_bucket}/{level_id}: missing k={missing} -- {query!r}")

    # %(playlist_index)d expands to the result rank from `ytsearchN:`, giving us k.
    outtmpl = str(target_dir / f"{level_id}_%(playlist_index)d.%(ext)s")

    options = {
        "format": (
            "bestvideo[height=720][ext=mp4]"
            "/bestvideo[height<=720][ext=mp4]"
            "/bestvideo[height<=720]"
        ),
        "outtmpl": outtmpl,
        "restrictfilenames": True,
        "sleep_interval": download_delay,
        "playlist_items": ",".join(str(k) for k in missing),
        "ignoreerrors": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["default", "web", "mweb", "tv"],
            }
        },
    }
    if cookies_file is not None:
        options["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([f"ytsearch{num_results}:{query}"])


def iter_metadata(levels_dir: Path):
    for stars_dir in sorted(levels_dir.iterdir()):
        if not stars_dir.is_dir():
            continue
        metadata_dir = stars_dir / "metadata"
        if not metadata_dir.is_dir():
            continue
        for json_path in sorted(metadata_dir.glob("*.json")):
            with json_path.open() as f:
                metadata = json.load(f)
            yield stars_dir.name, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download top YouTube playthroughs for every level under levels/.",
    )
    parser.add_argument("--levels-dir", type=Path, default=DEFAULT_LEVELS_DIR)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--num-results", type=int, default=5)
    parser.add_argument("--download-delay", type=int, default=2)
    parser.add_argument("--cookies-file", type=Path, default=DEFAULT_COOKIES_FILE)
    parser.add_argument(
        "--break-every",
        type=int,
        default=20,
        help="Take a long break every N successfully attempted downloads (0 disables).",
    )
    parser.add_argument(
        "--break-min-seconds",
        type=int,
        default=60,
        help="Minimum seconds to sleep when taking a long break.",
    )
    parser.add_argument(
        "--break-max-seconds",
        type=int,
        default=180,
        help="Maximum seconds to sleep when taking a long break.",
    )
    parser.add_argument(
        "--per-video-min-seconds",
        type=int,
        default=10,
        help="Minimum seconds to sleep between video downloads.",
    )
    parser.add_argument(
        "--per-video-max-seconds",
        type=int,
        default=30,
        help="Maximum seconds to sleep between video downloads.",
    )
    args = parser.parse_args()

    if args.break_min_seconds > args.break_max_seconds:
        parser.error("--break-min-seconds must be <= --break-max-seconds")
    if args.per_video_min_seconds > args.per_video_max_seconds:
        parser.error("--per-video-min-seconds must be <= --per-video-max-seconds")

    cookies_file = args.cookies_file if args.cookies_file.is_file() else None
    if cookies_file is None:
        print(f"[warn] cookies file not found at {args.cookies_file}; proceeding without cookies", file=sys.stderr)

    downloads_since_break = 0
    for stars_bucket, metadata in iter_metadata(args.levels_dir):
        level_id = metadata.get("id", "?")
        target_dir = args.videos_dir / stars_bucket
        before = existing_indexes(target_dir, level_id, args.num_results) if target_dir.is_dir() else set()
        try:
            download_for_level(
                metadata,
                stars_bucket,
                args.videos_dir,
                args.num_results,
                args.download_delay,
                cookies_file,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[err ] {stars_bucket}/{level_id}: {e}", file=sys.stderr)
        after = existing_indexes(target_dir, level_id, args.num_results) if target_dir.is_dir() else set()
        new_downloads = len(after - before)

        if new_downloads > 0:
            downloads_since_break += new_downloads
            if args.break_every > 0 and downloads_since_break >= args.break_every:
                nap = random.randint(args.break_min_seconds, args.break_max_seconds)
                print(f"[rest] sleeping {nap}s after {downloads_since_break} download(s) to avoid rate limiting")
                time.sleep(nap)
                downloads_since_break = 0
            else:
                nap = random.randint(args.per_video_min_seconds, args.per_video_max_seconds)
                print(f"[wait] sleeping {nap}s before next download")
                time.sleep(nap)


if __name__ == "__main__":
    main()
