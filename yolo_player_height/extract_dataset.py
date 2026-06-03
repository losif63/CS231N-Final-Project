#!/usr/bin/env python3
"""
Extract 4-channel (RGB + diff) training samples from Geometry Dash gameplay videos.

Each sample is a 660x660 numpy array:
- Channels 0-2: BGR (center square crop, resized to 660x660)
- Channel 3: Greyscale difference to previous frame (~30 FPS)

Crops exclude first/last 20s of each video.
"""

import cv2
import numpy as np
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict


# ── Crop geometry ──────────────────────────────────────────────────────────
CROP_RATIO = 415 / 460      # Fraction of frame height for YOLO crop side
BUFFER_PX = 10              # Buffer pixels on each side (for augmentation)
STORAGE_SIZE = 660          # Final stored size (resized to match buffer math)
SKIP_SECONDS = 20           # Skip first/last N seconds of video


def compute_crop_coords(h: int, w: int) -> tuple:
    """
    Compute crop coordinates for 660x660 storage crop (center, with 10px buffer).
    Returns (x0, y0, x1, y1) for slicing frame[y0:y1, x0:x1].
    """
    # YOLO crop side (before buffer)
    yolo_side = int(h * CROP_RATIO)

    # Storage side with buffer
    storage_side = yolo_side + 2 * BUFFER_PX

    # Center coordinates
    cx = w // 2
    cy = h // 2
    half = storage_side // 2

    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)

    return x0, y0, x1, y1


def get_valid_frame_range(total_frames: int, fps: float) -> tuple:
    """
    Return (start_frame, end_frame) excluding first/last 20 seconds.
    """
    skip_frames = int(SKIP_SECONDS * fps)
    start = skip_frames
    end = max(start + 1, total_frames - skip_frames)
    return start, end


def extract_sample(video_path: Path, frame_idx: int, fps: float, h: int, w: int) -> np.ndarray:
    """
    Extract a single 4-channel sample from video.

    Returns array of shape (660, 660, 4) with dtype uint8.
    Channels 0-2: BGR, Channel 3: greyscale diff to previous frame.
    """
    # Determine offset to previous frame (~30 FPS)
    prev_offset = max(1, round(fps / 30.0))
    prev_idx = frame_idx - prev_offset

    if prev_idx < 0:
        raise ValueError(f"Frame {frame_idx} too early (can't get prev_offset={prev_offset})")

    # Open video and read frames
    cap = cv2.VideoCapture(str(video_path))
    try:
        # Read previous frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
        ret_prev, prev_frame = cap.read()
        if not ret_prev:
            raise ValueError(f"Could not read prev frame at {prev_idx}")

        # Read current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_curr, curr_frame = cap.read()
        if not ret_curr:
            raise ValueError(f"Could not read curr frame at {frame_idx}")
    finally:
        cap.release()

    # Compute crop coordinates
    x0, y0, x1, y1 = compute_crop_coords(h, w)

    # Crop and resize to 660x660
    prev_crop = prev_frame[y0:y1, x0:x1]
    curr_crop = curr_frame[y0:y1, x0:x1]

    prev_resized = cv2.resize(prev_crop, (STORAGE_SIZE, STORAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    curr_resized = cv2.resize(curr_crop, (STORAGE_SIZE, STORAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    # Compute greyscale difference
    prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(curr_gray - prev_gray)
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Stack into 4-channel array
    sample = np.stack([curr_resized[:, :, 0], curr_resized[:, :, 1], curr_resized[:, :, 2], diff], axis=2)

    return sample.astype(np.uint8)


def make_dataset_dir(n_samples: int) -> Path:
    """
    Create dataset folder with collision handling.
    Returns the path to the new dataset folder.
    """
    base_dir = Path(".")
    m = 0
    while True:
        folder_name = f"dataset_{n_samples}_{m}"
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Created dataset folder: {folder_name}/")
            return folder_path
        m += 1


def main():
    parser = argparse.ArgumentParser(description="Extract 4-channel training samples from gameplay videos.")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to extract")
    parser.add_argument("--data-source", default="data_source.json", help="Path to data_source.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data source
    data_source_path = Path(args.data_source)
    if not data_source_path.exists():
        print(f"ERROR: {data_source_path} not found")
        return

    with open(data_source_path) as f:
        videos = json.load(f)

    print(f"Loaded {len(videos)} videos from {data_source_path}")

    # Create dataset folder
    dataset_dir = make_dataset_dir(args.n_samples)

    # Extract samples
    metadata = []
    sample_id = 0
    failures = 0
    max_retries = 5

    print(f"\nExtracting {args.n_samples} samples...")

    while sample_id < args.n_samples:
        retries = 0
        while retries < max_retries:
            try:
                # Pick a random video
                video_info = random.choice(videos)
                video_path = Path(video_info["path"])

                if not video_path.exists():
                    print(f"  ✗ Video not found: {video_path}")
                    retries += 1
                    continue

                h = video_info["height"]
                w = video_info["width"]
                fps = video_info["fps"]

                # Open video and check frame count
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if total_frames < 2:
                    print(f"  ✗ Video too short ({total_frames} frames): {video_path.name}")
                    retries += 1
                    continue

                # Get valid frame range
                start, end = get_valid_frame_range(total_frames, fps)
                prev_offset = max(1, round(fps / 30.0))

                # Ensure we can get a previous frame
                if start + prev_offset >= end:
                    print(f"  ✗ Valid frame range too small: {start}..{end}")
                    retries += 1
                    continue

                # Pick a random valid frame
                frame_idx = random.randint(start + prev_offset, end - 1)

                # Extract sample
                sample = extract_sample(video_path, frame_idx, fps, h, w)

                # Save sample
                sample_filename = f"sample_{sample_id:04d}.npy"
                sample_path = dataset_dir / sample_filename
                np.save(sample_path, sample)

                # Record metadata
                metadata.append({
                    "sample_id": sample_id,
                    "filename": sample_filename,
                    "video_path": video_info["path"],
                    "frame_index": frame_idx,
                    "fps": video_info["fps"],
                    "stars": video_info["stars"],
                    "source_resolution": [w, h],
                    "crop_size": int(h * CROP_RATIO),
                })

                print(f"  [{sample_id+1}/{args.n_samples}] {video_info['path']} frame {frame_idx}")
                sample_id += 1
                break

            except Exception as e:
                retries += 1
                if retries < max_retries:
                    continue
                else:
                    failures += 1
                    print(f"  ✗ Failed after {max_retries} retries: {str(e)[:60]}")
                    break

    # Save metadata
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset extraction complete!")
    print(f"  Samples extracted: {sample_id}")
    print(f"  Failures: {failures}")
    print(f"  Output directory: {dataset_dir.absolute()}")
    print(f"  Metadata: {metadata_path.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
