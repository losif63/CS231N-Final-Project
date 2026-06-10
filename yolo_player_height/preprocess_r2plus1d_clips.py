#!/usr/bin/env python3
"""
Preprocess dataset_10000_0 to 320x320 5-frame clips for r2plus1d training.

For each labeled sample:
  1. Get video_path and frame_index from metadata
  2. Load 5 consecutive frames from the video
  3. Rescale labels from original frame coordinates to 320×320
  4. Save as (5, 320, 320, 3) uint8 RGB npy file

Written using Claude Code
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import argparse


CLIP_SIZE = 5
HALF_CLIP = CLIP_SIZE // 2  # 2 frames before and after
OUTPUT_RESOLUTION = 320

# From extract_dataset.py: how the original 660x660 frames were created
CROP_RATIO = 415 / 460
BUFFER_PX = 10
LABEL_RESOLUTION = 660  # Resolution that labels are in

# Scaling factor: labels in 660x660 space → 320x320 output space
SCALE_FACTOR = OUTPUT_RESOLUTION / LABEL_RESOLUTION


def compute_crop_coords(h, w):
    """Compute center crop coordinates using same formula as extract_dataset.py."""
    yolo_side = int(h * CROP_RATIO)
    storage_side = yolo_side + 2 * BUFFER_PX
    cx = w // 2
    cy = h // 2
    half = storage_side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    return x0, y0, x1, y1


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset to 5-frame 320x320 clips")
    parser.add_argument("--input-dir", default="dataset_10000_0", help="Input dataset directory")
    parser.add_argument("--output-dir", default="r2plus1d_dataset_10000_0", help="Output dataset directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load metadata and labels
    print("Loading metadata and labels...")
    with open(input_dir / "metadata.json") as f:
        metadata = json.load(f)

    with open(input_dir / "labels_4567.json") as f:
        labels = json.load(f)

    # Create metadata index by sample_id
    meta_by_id = {str(m["sample_id"]): m for m in metadata}

    # Filter to only samples that have valid labels
    print("Filtering to labeled samples...")
    labeled_samples = []
    for sample_id_str, lbl in labels.items():
        if lbl.get("removed", False):
            continue

        not_in_frame = lbl.get("not_in_frame", False)
        has_height = "height_y" in lbl

        if not has_height and not not_in_frame:
            continue

        meta = meta_by_id.get(sample_id_str)
        if meta is None:
            continue

        labeled_samples.append({
            "sample_id": int(sample_id_str),
            "video_path": meta.get("video_path", "unknown"),
            "frame_index": meta.get("frame_index", 0),
            "label_y": float(lbl.get("height_y", OUTPUT_RESOLUTION / 2)),
            "label_x": float(lbl.get("height_x", OUTPUT_RESOLUTION / 2)),
            "occlusion": 1.0 if not_in_frame else 0.0,
        })

    print(f"Found {len(labeled_samples)} labeled samples")

    # Group labeled samples by video
    print("Grouping by video...")
    samples_by_video = defaultdict(list)
    for sample in labeled_samples:
        video_path = sample["video_path"]
        samples_by_video[video_path].append(sample)

    # Process each video
    print(f"\nProcessing {len(samples_by_video)} videos...")
    output_metadata = []
    output_labels = {}
    clip_idx = 0
    failed = 0

    for video_path, video_samples in sorted(samples_by_video.items()):
        # Resolve video path relative to input_dir parent
        video_path_resolved = (input_dir.parent / video_path).resolve()

        if not video_path_resolved.exists():
            print(f"  WARNING: Video not found: {video_path}")
            failed += len(video_samples)
            continue

        print(f"  {video_path} ({len(video_samples)} labeled frames)")

        # Open video once
        cap = cv2.VideoCapture(str(video_path_resolved))
        if not cap.isOpened():
            print(f"    ERROR: Cannot open video")
            failed += len(video_samples)
            continue

        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process each labeled sample in this video
        for sample in video_samples:
            sample_id = sample["sample_id"]
            frame_index = sample["frame_index"]

            # Check if we have enough frames before and after
            if frame_index < HALF_CLIP or frame_index >= video_frame_count - HALF_CLIP:
                failed += 1
                continue

            # Load 5 consecutive frames from original video (2 before, center, 2 after)
            clip_frame_indices = list(range(frame_index - HALF_CLIP, frame_index + HALF_CLIP + 1))
            clip_frames = []
            crop_h = None
            crop_w = None

            for clip_frame_idx in clip_frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_frame_idx)
                ret, frame = cap.read()

                if not ret or frame is None:
                    failed += 1
                    break

                orig_h, orig_w = frame.shape[:2]

                # Apply same center crop as extract_dataset.py
                x0, y0, x1, y1 = compute_crop_coords(orig_h, orig_w)
                frame_cropped = frame[y0:y1, x0:x1]

                if crop_h is None:
                    crop_h, crop_w = frame_cropped.shape[:2]

                # Resize cropped frame to output resolution
                frame_resized = cv2.resize(frame_cropped, (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION),
                                          interpolation=cv2.INTER_LINEAR)

                # Convert BGR to RGB
                if frame_resized.shape[2] == 4:
                    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGRA2RGB)
                else:
                    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                clip_frames.append(frame_resized)

            if len(clip_frames) != CLIP_SIZE:
                failed += 1
                continue

            # Stack frames: (5, 320, 320, 3)
            clip_array = np.stack(clip_frames, axis=0).astype(np.uint8)

            # Labels are in 660×660 space (from label_dataset.py labeling)
            # Scale to 320×320 output space
            label_y_rescaled = sample["label_y"] * SCALE_FACTOR
            label_x_rescaled = sample["label_x"] * SCALE_FACTOR

            # Save clip
            clip_filename = f"clip_{clip_idx:05d}.npy"
            clip_path = output_dir / clip_filename
            np.save(clip_path, clip_array)

            # Add to output metadata
            output_metadata.append({
                "sample_id": clip_idx,
                "filename": clip_filename,
                "video_path": video_path,
                "center_frame_index": frame_index,
                "frame_indices": clip_frame_indices,
                "original_sample_id": sample_id,
            })

            # Add to output labels with rescaled positions
            output_labels[str(clip_idx)] = {
                "removed": False,
                "not_in_frame": sample["occlusion"] > 0.5,
                "height_y": label_y_rescaled,
                "height_x": label_x_rescaled,
            }

            clip_idx += 1

        cap.release()

    print(f"\nPreprocessing complete:")
    print(f"  Total clips created: {clip_idx}")
    print(f"  Failed/skipped: {failed}")
    print(f"  Success rate: {100 * clip_idx / max(1, clip_idx + failed):.1f}%")

    # Save metadata and labels
    print(f"\nSaving to {output_dir}...")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(output_metadata, f, indent=2)

    with open(output_dir / "labels_4567.json", "w") as f:
        json.dump(output_labels, f, indent=2)

    print(f"  Saved {len(output_metadata)} clip metadata")
    print(f"  Saved {len(output_labels)} labels")
    print(f"  Saved {clip_idx} clip arrays (.npy files)")


if __name__ == "__main__":
    main()
