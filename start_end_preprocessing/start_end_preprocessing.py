"""
Scan gameplay videos and record the start/end frames that should be used
for model training.

  start_frame  — first frame AFTER the last "Attempt N" overlay disappears
  end_frame    — last frame BEFORE the "Level Complete!" overlay appears

Both overlays leak information the model should not see (attempt count,
post-completion stats), so they are excluded from the training window.

Usage:
    # process all videos with default settings
    python start_end_preprocessing/start_end_preprocessing.py

    # tune thresholds or search regions manually
    python start_end_preprocessing/start_end_preprocessing.py \\
        --attempt-threshold 0.65 \\
        --level-complete-threshold 0.65 \\
        --frame-step 2 \\
        --scales 0.75 1.0 1.25 \\
        --attempt-search-region 0 0 960 200 \\
        --level-complete-search-region 0 200 960 280

Output: start_end_preprocessing/start_end_frames.json
Each entry is keyed by the video's path relative to the repo root.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = REPO_ROOT / "videos"
TEMPLATES_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = TEMPLATES_DIR / "start_end_frames.json"

ATTEMPT_TEMPLATE_PATH = TEMPLATES_DIR / "AttemptTemplate.png"
LEVEL_COMPLETE_TEMPLATE_PATH = TEMPLATES_DIR / "LevelCompleteTemplate.png"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def load_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a PNG template.

    Returns (bgr, mask) where mask is a uint8 array derived from the alpha
    channel (255 = use pixel, 0 = ignore).  Fully-transparent pixels in the
    template image are zeroed out so they don't contribute to the match score
    even if mask support is unavailable at runtime.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3].copy()
        bgr[alpha == 0] = 0  # zero-fill transparent pixels
        mask = alpha
    else:
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask = np.full(bgr.shape[:2], 255, dtype=np.uint8)

    return bgr, mask


def resize_template(
    template: np.ndarray, mask: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    if scale == 1.0:
        return template, mask
    h, w = template.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    t = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return t, m


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def best_match_score(
    frame: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    scales: tuple[float, ...],
    search_region: tuple[int, int, int, int] | None,
) -> float:
    """
    Return the best TM_CCOEFF_NORMED score for the template anywhere in the
    frame, trying each scale in `scales`.

    search_region: (x, y, w, h) to restrict the search area, or None for the
    whole frame.  Leaving this as None (default) means the entire frame is
    searched — constrain it once you know where the overlays appear.
    """
    if search_region is not None:
        x, y, w, h = search_region
        roi = frame[y : y + h, x : x + w]
    else:
        roi = frame

    best = -1.0
    for scale in scales:
        t, m = resize_template(template, mask, scale)
        th, tw = t.shape[:2]
        rh, rw = roi.shape[:2]
        if th > rh or tw > rw:
            continue  # template is larger than the ROI at this scale

        try:
            result = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED, mask=m)
        except cv2.error:
            # Older OpenCV builds don't support mask for TM_CCOEFF_NORMED;
            # fall back to ignoring the mask (transparent pixels were already
            # zeroed in load_template so the impact is minor).
            result = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best:
            best = max_val

    return best


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    attempt_template: np.ndarray,
    attempt_mask: np.ndarray,
    level_complete_template: np.ndarray,
    level_complete_mask: np.ndarray,
    attempt_threshold: float,
    level_complete_threshold: float,
    frame_step: int,
    attempt_scales: tuple[float, ...],
    level_complete_scales: tuple[float, ...],
    attempt_search_region: tuple[int, int, int, int] | None,
    level_complete_search_region: tuple[int, int, int, int] | None,
) -> dict:
    """
    Scan `video_path` and return a dict with:
      start_frame, end_frame, total_frames, fps,
      attempt_frames_detected, level_complete_frames_detected
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    attempt_hit_frames: list[int] = []
    level_complete_hit_frames: list[int] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            a_score = best_match_score(
                frame, attempt_template, attempt_mask,
                attempt_scales, attempt_search_region,
            )
            if a_score >= attempt_threshold:
                attempt_hit_frames.append(frame_idx)

            lc_score = best_match_score(
                frame, level_complete_template, level_complete_mask,
                level_complete_scales, level_complete_search_region,
            )
            if lc_score >= level_complete_threshold:
                level_complete_hit_frames.append(frame_idx)

        frame_idx += 1

    cap.release()

    # Start: first frame AFTER the last "Attempt N" text disappears.
    # With frame_step > 1 we add frame_step rather than 1 so we don't
    # accidentally include the tail of the overlay that may fall between
    # two sampled frames.
    if attempt_hit_frames:
        start_frame = attempt_hit_frames[-1] + frame_step
    else:
        start_frame = 0

    # End: last frame BEFORE "Level Complete!" appears.
    # Symmetric logic: subtract frame_step to stay safely before the
    # first confirmed detection.
    if level_complete_hit_frames:
        end_frame = level_complete_hit_frames[0] - frame_step
        end_frame = max(start_frame, end_frame)
    else:
        end_frame = total_frames - 1

    return {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "attempt_frames_detected": len(attempt_hit_frames),
        "level_complete_frames_detected": len(level_complete_hit_frames),
    }


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------

def iter_videos(videos_dir: Path):
    """Yield all video files found under `videos_dir`."""
    for bucket_dir in sorted(videos_dir.iterdir()):
        if not bucket_dir.is_dir():
            continue
        for video_file in sorted(bucket_dir.iterdir()):
            if video_file.suffix.lower() in VIDEO_EXTENSIONS:
                yield video_file


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find start/end frames in Geometry Dash gameplay videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)

    # Matching thresholds — lower = more lenient, higher = stricter.
    # 0.7 is a reasonable starting point; tune after seeing real videos.
    parser.add_argument(
        "--attempt-threshold", type=float, default=0.7,
        help="TM_CCOEFF_NORMED score required to count a frame as showing 'Attempt N'.",
    )
    parser.add_argument(
        "--level-complete-threshold", type=float, default=0.7,
        help="TM_CCOEFF_NORMED score required to count a frame as showing 'Level Complete!'.",
    )

    # Frame sampling — 1 checks every frame (accurate but slow for long
    # videos); increase to speed up at the cost of boundary precision.
    parser.add_argument(
        "--frame-step", type=int, default=1,
        help="Check every Nth frame. 1 = all frames (most accurate).",
    )

    # Scale factors — try multiple to handle resolution mismatches between
    # the template PNG and the actual video.  Default [1.0] is unconstrained.
    parser.add_argument(
        "--scales", type=float, nargs="+", default=[1.0],
        help="Template scale factors to try (e.g. 0.5 1.0 1.5).",
    )

    # Search-region options — leave as None (whole frame) until you know
    # which part of the screen each overlay occupies in your video set.
    parser.add_argument(
        "--attempt-search-region", type=int, nargs=4, default=None,
        metavar=("X", "Y", "W", "H"),
        help="Pixel rectangle to restrict 'Attempt N' search (whole frame if omitted).",
    )
    parser.add_argument(
        "--level-complete-search-region", type=int, nargs=4, default=None,
        metavar=("X", "Y", "W", "H"),
        help="Pixel rectangle to restrict 'Level Complete!' search (whole frame if omitted).",
    )

    # Incremental mode — skip videos that already have an entry in the JSON.
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process videos that already have a result in the output JSON.",
    )

    args = parser.parse_args()

    attempt_template, attempt_mask = load_template(ATTEMPT_TEMPLATE_PATH)
    level_complete_template, level_complete_mask = load_template(LEVEL_COMPLETE_TEMPLATE_PATH)

    # Load existing results for incremental updates.
    if args.output_json.exists():
        with args.output_json.open() as f:
            try:
                results: dict = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    if not args.videos_dir.is_dir():
        print(
            f"[warn] Videos directory not found: {args.videos_dir}\n"
            "[warn] No videos processed.  JSON unchanged.",
            file=sys.stderr,
        )
        return

    scales = tuple(args.scales)
    attempt_sr = tuple(args.attempt_search_region) if args.attempt_search_region else None
    lc_sr = tuple(args.level_complete_search_region) if args.level_complete_search_region else None

    video_paths = list(iter_videos(args.videos_dir))
    if not video_paths:
        print("[warn] No video files found under", args.videos_dir, file=sys.stderr)

    processed = skipped = errors = 0
    for video_path in video_paths:
        key = str(video_path.relative_to(REPO_ROOT))

        if not args.force and key in results and "error" not in results[key]:
            print(f"[skip] {key}")
            skipped += 1
            continue

        print(f"[proc] {key} ...", end=" ", flush=True)
        try:
            info = process_video(
                video_path,
                attempt_template, attempt_mask,
                level_complete_template, level_complete_mask,
                attempt_threshold=args.attempt_threshold,
                level_complete_threshold=args.level_complete_threshold,
                frame_step=args.frame_step,
                attempt_scales=scales,
                level_complete_scales=scales,
                attempt_search_region=attempt_sr,
                level_complete_search_region=lc_sr,
            )
            results[key] = info
            print(
                f"start={info['start_frame']}  end={info['end_frame']}  "
                f"(attempt_hits={info['attempt_frames_detected']}, "
                f"lc_hits={info['level_complete_frames_detected']}, "
                f"fps={info['fps']:.1f}, total={info['total_frames']})"
            )
            processed += 1
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            results[key] = {"error": str(exc)}
            errors += 1

    with args.output_json.open("w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n[done] processed={processed}  skipped={skipped}  errors={errors}"
        f"\n[done] Results written to {args.output_json}"
    )


if __name__ == "__main__":
    main()
