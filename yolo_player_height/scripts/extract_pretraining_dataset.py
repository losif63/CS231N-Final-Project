#!/usr/bin/env python3
"""
Extract trajectory pretraining dataset from labeled clips and original videos.

For each labeled clip (64 frames with labels), extract 48 video frames:
- 16 frames before clip start (unlabeled, from video)
- 32 frames from clip start (labeled)
Store as HDF5 with gzip compression.

Usage:
    conda run -n cv_final_proj python scripts/extract_pretraining_dataset.py \
        --num-videos 10  # Extract from 10 videos for testing
"""

import argparse
import json
import h5py
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

# Constants from the labeling pipeline
LABEL_DIR = Path("pretraining_processed_labels")
VIDEOS_PROCESSED_DIR = Path("../videos_processed")
OUTPUT_HDF5 = Path("pretraining_dataset.hdf5")
OUTPUT_SPLIT = Path("pretraining_split.json")

FRAME_WIDTH = 224
FRAME_HEIGHT = 224
CLIP_START_OFFSET = 16  # 16 frames before labeled clip
CLIP_TOTAL_FRAMES = 48  # 16 before + 32 from clip
LABELED_FRAMES = 64


def get_video_path(video_path_str):
    """Resolve video path from label metadata."""
    # video_path_str is like "videos_processed/8stars/57484909_2.mp4"
    full_path = (Path(__file__).parent.parent.parent / video_path_str).resolve()
    return full_path


def extract_frames_for_clip(video_path, clip_start_frame, clip_end_frame):
    """
    Extract 48 frames for a clip: 16 before start + 32 from start.

    Returns:
        frames: (48, 224, 224, 3) uint8 array or None if failed
        orig_h, orig_w: Original video resolution
    """
    if not video_path.exists():
        return None, None, None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None

    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Frame indices to extract
    frame_start = clip_start_frame - CLIP_START_OFFSET
    frame_end = clip_start_frame + 32  # 16 before + 32 from clip

    frames = []
    for frame_idx in range(frame_start, frame_end):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            cap.release()
            return None, None, None

        # Resize to 224×224
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    return np.stack(frames, axis=0).astype(np.uint8), orig_h, orig_w


def extract_dataset(num_videos=None, test_mode=False):
    """
    Extract pretraining dataset from labeled clips.

    Args:
        num_videos: If set, only process this many videos (for testing)
        test_mode: If True, only extract 3 clips per video
    """

    # Scan label directory for clips
    print(f"Scanning {LABEL_DIR}...")
    label_files = sorted(LABEL_DIR.glob("*stars/*.json"))
    print(f"Found {len(label_files)} labeled video files")

    # Load all clips
    all_clips = []
    for label_file in label_files:
        star_level = label_file.parent.name
        video_name = label_file.stem

        with open(label_file) as f:
            data = json.load(f)

        video_path = data["video_path"]

        for clip_idx, clip in enumerate(data["clips"]):
            all_clips.append({
                "label_file": str(label_file),
                "star_level": star_level,
                "video_name": video_name,
                "video_path": video_path,
                "clip_start": clip["start_frame"],
                "clip_end": clip["end_frame"],
                "heights": np.array(clip["frames"], dtype=object),  # Will extract height values
                "heights_list": [f["height"] for f in clip["frames"]],
                "occlusions_list": [f["occlusion"] for f in clip["frames"]],
            })

    print(f"Total clips: {len(all_clips)}")

    # Split by video for train/val
    videos = {}
    for clip in all_clips:
        video_path = clip["video_path"]
        if video_path not in videos:
            videos[video_path] = []
        videos[video_path].append(clip)

    print(f"Total unique videos: {len(videos)}")

    # Create train/val split at video level
    video_list = list(videos.keys())
    random.seed(42)
    random.shuffle(video_list)

    split_idx = int(len(video_list) * 0.8)
    train_videos = set(video_list[:split_idx])
    val_videos = set(video_list[split_idx:])

    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    # Optionally limit to N videos
    if num_videos:
        video_list = video_list[:num_videos]
        print(f"Limiting to {num_videos} videos for testing")

    # Create HDF5 file
    print(f"\nCreating {OUTPUT_HDF5}...")

    with h5py.File(OUTPUT_HDF5, "w", libver="latest") as hf:
        # Create dataset groups
        clips_group = hf.create_group("clips")
        metadata_group = hf.create_group("metadata")

        clip_count = 0
        failed_count = 0

        # Track which videos we process (and clip count per video)
        processed_videos = defaultdict(int)

        for i, clip in enumerate(all_clips):
            video_path = clip["video_path"]

            # Skip if video not in selected list
            if num_videos and video_path not in {v for v in video_list}:
                continue

            # Limit clips per video for testing
            if test_mode and processed_videos[video_path] >= 3:
                continue

            video_path_full = get_video_path(video_path)

            # Extract frames
            frames, orig_h, orig_w = extract_frames_for_clip(
                video_path_full,
                clip["clip_start"],
                clip["clip_end"]
            )

            if frames is None:
                failed_count += 1
                if failed_count % 10 == 0:
                    print(f"  Failed: {failed_count}")
                continue

            # Create clip group
            clip_group = clips_group.create_group(f"clip_{clip_count:05d}")
            clip_group.create_dataset("frames", data=frames, compression="gzip",
                                     compression_opts=4)
            clip_group.create_dataset("heights", data=np.array(clip["heights_list"],
                                                               dtype=np.float32),
                                     compression="gzip", compression_opts=4)
            clip_group.create_dataset("occlusion", data=np.array(clip["occlusions_list"],
                                                                 dtype=np.float32),
                                     compression="gzip", compression_opts=4)

            # Store metadata
            clip_group.attrs["video_path"] = video_path
            clip_group.attrs["clip_start"] = clip["clip_start"]
            clip_group.attrs["clip_end"] = clip["clip_end"]
            clip_group.attrs["video_name"] = clip["video_name"]
            clip_group.attrs["star_level"] = clip["star_level"]
            clip_group.attrs["split"] = "train" if video_path in train_videos else "val"
            clip_group.attrs["original_height"] = orig_h
            clip_group.attrs["original_width"] = orig_w

            clip_count += 1
            processed_videos[video_path] += 1

            if clip_count % 100 == 0:
                print(f"  Extracted {clip_count} clips...")

    print(f"\nExtraction complete!")
    print(f"  Total clips: {clip_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Success rate: {100 * clip_count / max(1, clip_count + failed_count):.1f}%")

    # Save split info
    split_info = {
        "train_videos": list(train_videos),
        "val_videos": list(val_videos),
        "num_clips": clip_count,
        "num_failed": failed_count,
    }
    with open(OUTPUT_SPLIT, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"  Split saved to {OUTPUT_SPLIT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=None,
                       help="Limit to N videos (for testing)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: only extract 3 clips per video")
    args = parser.parse_args()

    extract_dataset(num_videos=args.num_videos, test_mode=args.test)
