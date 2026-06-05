#!/usr/bin/env python3
"""
Preprocess dataset to 8-frame clip format for efficient 3D CNN training.

Converts dataset_10000_0 into 4D_dataset_10000_0 where each sample is an 8-frame clip.

Input:  dataset_10000_0/
  - metadata.json: sample metadata with video_path, frame_index
  - labels_4567.json: labels (height_y, height_x, occlusion)

Output: 4D_dataset_10000_0/
  - clip_0000.npy: (8, 660, 660, 3) uint8 RGB frames
  - clip_0001.npy: (8, 660, 660, 3)
  - ...
  - metadata.json: clip metadata (video_path, frame_indices)
  - labels_4567.json: labels (same format, but now for clip reference frame)

Each clip uses 8 consecutive frames from the video:
  - Center frame is the original sample's frame (labeled frame)
  - Frame indices: [center-4, center-3, center-2, center-1, center, center+1, center+2, center+3]
  - Label is preserved from the center frame
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import argparse


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset to 8-frame clip format")
    parser.add_argument("--input-dir", default="dataset_10000_0", help="Input dataset directory")
    parser.add_argument("--output-dir", default="4D_dataset_10000_0", help="Output dataset directory")
    parser.add_argument("--clip-size", type=int, default=8, help="Frames per clip")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    clip_size = args.clip_size
    half_clip = clip_size // 2

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load metadata and labels
    print("Loading metadata and labels...")
    with open(input_dir / "metadata.json") as f:
        metadata = json.load(f)

    with open(input_dir / "labels_4567.json") as f:
        labels = json.load(f)

    # Group samples by video to determine frame ranges
    print("Organizing samples by video...")
    samples_by_video = defaultdict(list)
    for sample in metadata:
        video_path = sample.get("video_path", "unknown")
        samples_by_video[video_path].append(sample)

    # Sort by frame index within each video
    for video_path in samples_by_video:
        samples_by_video[video_path].sort(key=lambda s: s.get("frame_index", 0))

    # Preprocess clips
    print(f"\nPreprocessing clips (clip_size={clip_size})...")
    output_metadata = []
    output_labels = {}
    clip_idx = 0
    skipped = 0

    for video_path, video_samples in samples_by_video.items():
        print(f"  Processing {video_path}...")

        # Open video
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"    WARNING: Cannot open {video_path}")
                skipped += len(video_samples)
                continue

            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            print(f"    WARNING: Error opening {video_path}: {e}")
            skipped += len(video_samples)
            continue

        # For each sample in this video
        for sample in video_samples:
            sample_id = sample["sample_id"]
            frame_index = sample.get("frame_index", 0)
            label_str = str(sample_id)

            # Check if we have label for this sample
            if label_str not in labels:
                skipped += 1
                continue

            lbl = labels[label_str]

            # Skip if removed or has no height label and not marked as not_in_frame
            if lbl.get("removed", False):
                skipped += 1
                continue

            not_in_frame = lbl.get("not_in_frame", False)
            has_height = "height_y" in lbl

            if not has_height and not not_in_frame:
                skipped += 1
                continue

            # Check if we have enough frames before and after (for clip centering)
            if frame_index < half_clip or frame_index >= video_frame_count - half_clip:
                skipped += 1
                continue

            # Load clip frames
            clip_frame_indices = list(range(frame_index - half_clip, frame_index + half_clip))
            clip_frames = []

            for clip_frame_idx in clip_frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_frame_idx)
                ret, frame = cap.read()

                if not ret:
                    print(f"    WARNING: Cannot read frame {clip_frame_idx} from {video_path}")
                    break

                # Ensure 660x660 (resize if needed)
                if frame.shape[0] != 660 or frame.shape[1] != 660:
                    frame = cv2.resize(frame, (660, 660), interpolation=cv2.INTER_LINEAR)

                # Convert BGR to RGB (remove alpha if present)
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                clip_frames.append(frame)

            if len(clip_frames) != clip_size:
                skipped += 1
                continue

            # Stack frames: (8, 660, 660, 3)
            clip_array = np.stack(clip_frames, axis=0).astype(np.uint8)

            # Save clip
            clip_filename = f"clip_{clip_idx:04d}.npy"
            clip_path = output_dir / clip_filename
            np.save(clip_path, clip_array)

            # Add to output metadata
            output_metadata.append({
                "sample_id": clip_idx,
                "filename": clip_filename,
                "video_path": video_path,
                "center_frame_index": frame_index,  # The labeled frame (center of clip)
                "frame_indices": clip_frame_indices,  # All 8 frame indices in clip
                "original_sample_id": sample_id,
            })

            # Add to output labels (same as input, but indexed by clip_id)
            output_labels[str(clip_idx)] = lbl

            clip_idx += 1

        cap.release()

    print(f"\nPreprocessing complete:")
    print(f"  Total clips: {clip_idx}")
    print(f"  Skipped: {skipped}")
    print(f"  Success rate: {100 * clip_idx / (clip_idx + skipped):.1f}%")

    # Save metadata and labels
    print(f"\nSaving to {output_dir}...")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(output_metadata, f, indent=2)

    with open(output_dir / "labels_4567.json", "w") as f:
        json.dump(output_labels, f, indent=2)

    print(f"  Saved {len(output_metadata)} clip metadata to metadata.json")
    print(f"  Saved {len(output_labels)} labels to labels_4567.json")
    print(f"  Saved {clip_idx} clip arrays (.npy files)")


if __name__ == "__main__":
    main()
