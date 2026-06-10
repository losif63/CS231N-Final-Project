#!/usr/bin/env python3
"""
Label pretraining dataset: extract 64-frame clips from videos_processed/ and generate
height/occlusion/mode predictions using the trained r2plus1d model.

For each video, labels 8 non-overlapping 64-frame clips (skipping first 10s and last 30s).
Handles resumption by checking existing JSON files.
Uses batching for speed.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Constants
CROP_RATIO = 415 / 460
BUFFER_PX = 10
INPUT_RESOLUTION = 320
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLIP_LENGTH = 64  # Frames per clip
NUM_CLIPS_PER_VIDEO = 8
SKIP_START_SECONDS = 10
SKIP_END_SECONDS = 30
MIN_VIDEO_LENGTH_SECONDS = 40
BATCH_SIZE = 16
PROCESSED_FPS = 30

# Paths
VIDEOS_PROCESSED_DIR = Path("../videos_processed")
VIDEOS_ORIGINAL_DIR = Path("../videos")
OUTPUT_DIR = Path("pretraining_processed_labels")
ALL_CLIPS_JSON = OUTPUT_DIR / "all_clips.json"


def compute_crop_coords(h, w):
    """Compute center crop using same formula as extract_dataset.py."""
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


def preprocess_frame(frame):
    """Crop and normalize frame for model input."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = compute_crop_coords(h, w)
    frame_cropped = frame[y0:y1, x0:x1]
    frame_resized = cv2.resize(frame_cropped, (INPUT_RESOLUTION, INPUT_RESOLUTION),
                               interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_float = frame_rgb.astype(np.float32) / 255.0
    frame_norm = (frame_float - IMAGENET_MEAN) / IMAGENET_STD
    return frame_norm.transpose(2, 0, 1)


def load_model(model_path, device):
    """Load trained r2plus1d model."""
    from torchvision.models.video import r2plus1d_18

    model = r2plus1d_18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3),
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, clip_list, device):
    """
    Predict on a batch of 5-frame clips.
    clip_list: list of lists, each inner list has 5 preprocessed frames (3, 320, 320)
    Returns list of (height, occlusion, mode) predictions.
    """
    if not clip_list:
        return []

    # Stack all clips: (B, 5, 3, 320, 320)
    batch_tensor = np.stack([np.stack(clip, axis=0) for clip in clip_list], axis=0)
    batch_tensor = torch.from_numpy(batch_tensor).float().to(device)

    # Transpose to (B, 3, 5, 320, 320) as expected by model
    batch_tensor = batch_tensor.transpose(1, 2)

    # Get predictions
    logits = model(batch_tensor)  # (B, 3)

    results = []
    for i in range(logits.shape[0]):
        pred_y = logits[i, 0].item()
        pred_occ = torch.sigmoid(logits[i, 1]).item()
        pred_mode = torch.sigmoid(logits[i, 2]).item()
        results.append((pred_y, pred_occ, pred_mode))

    return results


def get_original_frame_index(processed_frame_idx, original_fps):
    """
    Convert processed video frame index (30 FPS) to original video frame index.
    """
    time_seconds = processed_frame_idx / PROCESSED_FPS
    original_frame_idx = int(np.round(time_seconds * original_fps))
    return original_frame_idx


def get_clip_positions(processed_video_fps, processed_video_frame_count, existing_clips=None):
    """
    Get valid non-overlapping clip positions (64 frames each).
    Skip first 10s and last 30s.
    Avoid overlap with existing clips if possible.
    """
    video_duration_seconds = processed_video_frame_count / processed_video_fps
    skip_start_frames = int(SKIP_START_SECONDS * processed_video_fps)
    skip_end_frames = int(SKIP_END_SECONDS * processed_video_fps)

    min_duration = MIN_VIDEO_LENGTH_SECONDS * processed_video_fps
    if processed_video_frame_count < min_duration:
        return []

    # Valid range for clip starts
    min_start = skip_start_frames
    max_start = processed_video_frame_count - skip_end_frames - CLIP_LENGTH

    if max_start <= min_start:
        return []

    # Mark existing clip regions
    marked = set()
    if existing_clips:
        for clip in existing_clips:
            for frame_idx in range(clip["start_frame"], clip["end_frame"]):
                marked.add(frame_idx)

    # Find non-overlapping clip positions
    clips = []
    current_pos = min_start
    while len(clips) < NUM_CLIPS_PER_VIDEO and current_pos + CLIP_LENGTH <= max_start + 1:
        # Check if this position overlaps with existing clips
        overlaps = any(frame_idx in marked for frame_idx in range(current_pos, current_pos + CLIP_LENGTH))

        if not overlaps:
            clips.append({"start_frame": current_pos, "end_frame": current_pos + CLIP_LENGTH})
            # Mark these frames as used
            for frame_idx in range(current_pos, current_pos + CLIP_LENGTH):
                marked.add(frame_idx)

        current_pos += CLIP_LENGTH

    # If we couldn't find enough non-overlapping clips, just use sequential positions
    if len(clips) < NUM_CLIPS_PER_VIDEO:
        clips = []
        for i in range(NUM_CLIPS_PER_VIDEO):
            start = min_start + i * CLIP_LENGTH
            end = start + CLIP_LENGTH
            if end <= max_start + 1:
                clips.append({"start_frame": start, "end_frame": end})

    return clips


def label_video(model, device, processed_video_path, original_video_path, output_json_path):
    """
    Label a single video with predictions.
    Returns True if successful, False otherwise.
    """
    print(f"\n  Processing: {processed_video_path.name}")

    # Load or create JSON
    if output_json_path.exists():
        with open(output_json_path) as f:
            video_data = json.load(f)
        existing_clips = video_data.get("clips", [])
        print(f"    Found {len(existing_clips)} existing clips")
    else:
        video_data = {
            "video_path": str(processed_video_path.relative_to(VIDEOS_PROCESSED_DIR.parent)),
            "original_video_path": str(original_video_path.relative_to(VIDEOS_ORIGINAL_DIR.parent)),
            "clips": []
        }
        existing_clips = []

    # If already has all clips, skip
    if len(existing_clips) >= NUM_CLIPS_PER_VIDEO:
        print(f"    Already has {len(existing_clips)} clips, skipping")
        return True

    # Open processed video to get properties
    cap_proc = cv2.VideoCapture(str(processed_video_path))
    if not cap_proc.isOpened():
        print(f"    ERROR: Cannot open processed video")
        return False

    processed_fps = cap_proc.get(cv2.CAP_PROP_FPS)
    processed_frame_count = int(cap_proc.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_proc.release()

    # Open original video to get FPS
    cap_orig = cv2.VideoCapture(str(original_video_path))
    if not cap_orig.isOpened():
        print(f"    ERROR: Cannot open original video")
        return False

    original_fps = cap_orig.get(cv2.CAP_PROP_FPS)
    cap_orig.release()

    # Get clip positions to label
    clips_to_label = get_clip_positions(processed_fps, processed_frame_count, existing_clips)
    if not clips_to_label:
        print(f"    No valid clips to label (video too short or already fully labeled)")
        return True

    print(f"    Will label {len(clips_to_label)} new clips")

    # Label each clip
    for clip_idx, clip_info in enumerate(clips_to_label):
        start_frame = clip_info["start_frame"]
        end_frame = clip_info["end_frame"]

        print(f"      Clip {clip_idx + 1}/{len(clips_to_label)}: frames {start_frame}-{end_frame}")

        # Load frames with 2-frame context on each side for 5-frame windows
        # Load [start_frame - 2, end_frame + 2) to get context for all 64 target frames
        load_start_frame = start_frame - 2
        load_end_frame = end_frame + 2

        cap_orig = cv2.VideoCapture(str(original_video_path))

        # Read and preprocess all frames
        preprocessed_frames = []
        for processed_frame_idx in range(load_start_frame, load_end_frame):
            original_frame_idx = get_original_frame_index(processed_frame_idx, original_fps)
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = cap_orig.read()

            if not ret:
                print(f"        ERROR: Cannot read frame {original_frame_idx} from original video")
                cap_orig.release()
                return False

            preprocessed_frames.append(preprocess_frame(frame))

        cap_orig.release()

        # Generate 5-frame clips from the loaded frames
        predictions = []
        clip_batch = []
        prediction_count = 0

        for processed_frame_idx in range(start_frame, end_frame):
            # Index in preprocessed_frames for this target frame
            frame_idx_in_loaded = processed_frame_idx - load_start_frame

            # Create 5-frame window: [frame_idx - 2, frame_idx - 1, frame_idx, frame_idx + 1, frame_idx + 2]
            window_indices = [
                frame_idx_in_loaded - 2,
                frame_idx_in_loaded - 1,
                frame_idx_in_loaded,
                frame_idx_in_loaded + 1,
                frame_idx_in_loaded + 2,
            ]

            # Get frames for this window
            window_frames = [preprocessed_frames[idx] for idx in window_indices]
            clip_batch.append(window_frames)

            # Predict when batch is full or at end of clip
            if len(clip_batch) >= BATCH_SIZE or processed_frame_idx == end_frame - 1:
                batch_predictions = predict_batch(model, clip_batch, device)
                for pred in batch_predictions:
                    predictions.append({
                        "processed_video_frame_index": start_frame + prediction_count,
                        "height": pred[0],
                        "occlusion": pred[1],
                        "mode": pred[2]
                    })
                    prediction_count += 1
                clip_batch = []

        # Add to video data
        video_data["clips"].append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": predictions
        })

    # Save updated JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(video_data, f, indent=2)

    print(f"    Saved {len(clips_to_label)} clips to {output_json_path}")
    return True


def update_all_clips_json():
    """Update the main all_clips.json with all labeled clips."""
    all_clips = {"clips": []}

    for star_dir in OUTPUT_DIR.glob("*stars"):
        if not star_dir.is_dir():
            continue

        for video_json in star_dir.glob("*.json"):
            with open(video_json) as f:
                video_data = json.load(f)

            for clip in video_data.get("clips", []):
                all_clips["clips"].append({
                    "video_path": video_data["video_path"],
                    "processed_video_path": video_data["video_path"],
                    "star_level": star_dir.name,
                    "start_frame": clip["start_frame"],
                    "end_frame": clip["end_frame"]
                })

    with open(ALL_CLIPS_JSON, "w") as f:
        json.dump(all_clips, f, indent=2)

    print(f"\nUpdated all_clips.json with {len(all_clips['clips'])} labeled clips")


def main():
    parser = argparse.ArgumentParser(description="Label pretraining dataset with r2plus1d predictions")
    parser.add_argument("--model", default="best_model_r2plus1d.pt",
                       help="Path to trained model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Batch size for inference")
    parser.add_argument("--num-videos", type=int, default=None,
                       help="Limit to N videos (default: no limit)")

    args = parser.parse_args()

    # Prefer CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all processed videos
    print(f"\nScanning {VIDEOS_PROCESSED_DIR}...")
    videos_by_star = defaultdict(list)

    for star_dir in VIDEOS_PROCESSED_DIR.glob("*stars"):
        if not star_dir.is_dir():
            continue

        # Create output star directory
        output_star_dir = OUTPUT_DIR / star_dir.name
        output_star_dir.mkdir(parents=True, exist_ok=True)

        # Find all MP4s in this star directory
        for video_file in star_dir.glob("*.mp4"):
            videos_by_star[star_dir.name].append(video_file)

    total_videos = sum(len(v) for v in videos_by_star.values())
    print(f"Found {total_videos} videos across {len(videos_by_star)} star levels")

    # Process videos in random order
    all_videos = []
    for star_level in sorted(videos_by_star.keys()):
        for video_path in videos_by_star[star_level]:
            all_videos.append((star_level, video_path))

    random.shuffle(all_videos)

    # Label videos
    labeled_count = 0
    skipped_count = 0
    videos_to_process = all_videos if args.num_videos is None else all_videos[:args.num_videos]

    print(f"Processing {len(videos_to_process)} video(s)...")

    for star_level, processed_video_path in videos_to_process:
        # Find corresponding original video
        relative_path = processed_video_path.relative_to(VIDEOS_PROCESSED_DIR)
        original_video_path = VIDEOS_ORIGINAL_DIR / relative_path

        if not original_video_path.exists():
            print(f"WARNING: Original video not found for {processed_video_path.name}")
            skipped_count += 1
            continue

        # Label this video
        output_json_path = OUTPUT_DIR / star_level / f"{processed_video_path.stem}.json"

        success = label_video(model, device, processed_video_path, original_video_path, output_json_path)
        if success:
            labeled_count += 1
        else:
            skipped_count += 1

    print(f"\n{'='*60}")
    print(f"Labeling complete!")
    print(f"  Labeled: {labeled_count}")
    print(f"  Skipped: {skipped_count}")

    # Update all_clips.json
    update_all_clips_json()


if __name__ == "__main__":
    main()
