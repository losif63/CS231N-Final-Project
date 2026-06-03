#!/usr/bin/env python3
"""
Create a data_source.json file with video metadata for training.
Filters for videos with ~30 FPS or ~60 FPS and samples 1000 of them.
"""

import cv2
import json
import random
from pathlib import Path
from collections import defaultdict

def get_video_metadata(video_path):
    """Extract FPS and resolution from a video using OpenCV."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if fps > 0 and width > 0 and height > 0:
                return fps, width, height
    except Exception:
        pass
    return None, None, None

def get_star_rating(video_path):
    """Extract star rating from the folder structure."""
    parts = video_path.parts
    for part in parts:
        if part.endswith('stars'):
            return int(part[:-5])  # Remove 'stars' suffix
    return None

def main():
    videos_dir = Path("../videos")

    if not videos_dir.exists():
        print(f"Error: {videos_dir} does not exist")
        return

    # Find all video files excluding 0stars
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    all_videos = []

    print("Scanning for video files...")
    for video_path in videos_dir.rglob("*"):
        if video_path.suffix.lower() in video_extensions:
            # Skip 0stars folder
            if "0stars" not in str(video_path):
                all_videos.append(video_path)

    print(f"Found {len(all_videos)} total videos (excluding 0stars)")

    # Extract metadata and filter for ~30 FPS or ~60 FPS
    print("\nExtracting framerates, resolution, and filtering...")
    videos_30fps = []
    videos_60fps = []
    skipped = 0

    for i, video_path in enumerate(all_videos):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(all_videos)}")

        fps, width, height = get_video_metadata(video_path)
        star_rating = get_star_rating(video_path)

        if fps is None or star_rating is None:
            skipped += 1
            continue

        # Filter for ~30 FPS (25-35) or ~60 FPS (55-65)
        if 25 <= fps <= 35:
            videos_30fps.append({
                'path': str(video_path),
                'fps': fps,
                'stars': star_rating,
                'width': width,
                'height': height
            })
        elif 55 <= fps <= 65:
            videos_60fps.append({
                'path': str(video_path),
                'fps': fps,
                'stars': star_rating,
                'width': width,
                'height': height
            })
        else:
            skipped += 1

    print(f"Found {len(videos_30fps)} videos at ~30 FPS")
    print(f"Found {len(videos_60fps)} videos at ~60 FPS")
    print(f"Skipped {skipped} videos (no metadata or out of range)")

    # Combine and sample
    all_valid = videos_30fps + videos_60fps
    target_count = 1000

    if len(all_valid) < target_count:
        print(f"Warning: Only {len(all_valid)} valid videos available, using all of them")
        sampled = all_valid
    else:
        sampled = random.sample(all_valid, target_count)
        print(f"Sampled {target_count} videos")

    # Create output with indices
    data_source = []
    for idx, video_info in enumerate(sampled):
        data_source.append({
            'index': idx,
            'path': video_info['path'],
            'fps': round(video_info['fps'], 2),
            'stars': video_info['stars'],
            'width': video_info['width'],
            'height': video_info['height']
        })

    # Save to JSON
    output_path = Path("data_source.json")
    with open(output_path, 'w') as f:
        json.dump(data_source, f, indent=2)

    print(f"\nSaved {len(data_source)} videos to {output_path}")

    # Print summary statistics
    fps_values = [v['fps'] for v in data_source]
    star_counts = defaultdict(int)
    resolution_counts = defaultdict(int)

    for v in data_source:
        star_counts[v['stars']] += 1
        resolution = f"{v['width']}x{v['height']}"
        resolution_counts[resolution] += 1

    print(f"\nSummary:")
    print(f"  Average FPS: {sum(fps_values) / len(fps_values):.1f}")
    print(f"\n  Videos by star rating:")
    for stars in sorted(star_counts.keys()):
        count = star_counts[stars]
        percentage = (count / len(data_source)) * 100
        print(f"    {stars} stars: {count} videos ({percentage:.1f}%)")

    print(f"\n  Videos by resolution:")
    for resolution in sorted(resolution_counts.keys(), key=lambda x: resolution_counts[x], reverse=True):
        count = resolution_counts[resolution]
        percentage = (count / len(data_source)) * 100
        print(f"    {resolution}: {count} videos ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
