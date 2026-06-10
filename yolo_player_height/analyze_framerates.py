#!/usr/bin/env python3
"""
Analyze the distribution of framerates across video files.
Samples videos if there are too many to process quickly.
"""

import cv2
import random
from pathlib import Path
from collections import Counter
import statistics

def get_fps_opencv(video_path):
    """Extract FPS from a video using OpenCV."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                return fps
    except Exception as e:
        pass

    return None


def main():
    videos_dir = Path("../videos")

    if not videos_dir.exists():
        print(f"Error: {videos_dir} does not exist")
        return

    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    all_videos = []

    print("Scanning for video files...")
    for video_path in videos_dir.rglob("*"):
        if video_path.suffix.lower() in video_extensions:
            all_videos.append(video_path)

    print(f"Found {len(all_videos)} total videos")

    # Sample if needed
    sample_size = 500
    if len(all_videos) > sample_size:
        videos_to_process = random.sample(all_videos, sample_size)
        print(f"Sampling {sample_size} videos for analysis")
    else:
        videos_to_process = all_videos
        print(f"Processing all {len(videos_to_process)} videos")

    # Extract framerates
    framerates = []
    print("\nExtracting framerates...")

    for i, video_path in enumerate(videos_to_process):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(videos_to_process)}")

        fps = get_fps_opencv(video_path)
        if fps is not None:
            framerates.append(fps)

    if not framerates:
        print("No framerates extracted!")
        return

    # Analyze distribution
    print(f"\n{'='*60}")
    print(f"Framerate Analysis ({len(framerates)} videos analyzed)")
    print(f"{'='*60}")
    print(f"Mean:   {statistics.mean(framerates):.2f} fps")
    print(f"Median: {statistics.median(framerates):.2f} fps")
    print(f"StdDev: {statistics.stdev(framerates):.2f} fps")
    print(f"Min:    {min(framerates):.2f} fps")
    print(f"Max:    {max(framerates):.2f} fps")

    # Round to nearest integer for cleaner distribution
    print(f"\n{'='*60}")
    print("Distribution by integer FPS:")
    print(f"{'='*60}")
    fps_rounded = [round(f) for f in framerates]
    fps_counts = Counter(fps_rounded)

    for fps in sorted(fps_counts.keys()):
        count = fps_counts[fps]
        percentage = (count / len(framerates)) * 100
        bar = "█" * int(percentage / 2)
        print(f"{fps:3d} fps: {count:4d} videos ({percentage:5.1f}%) {bar}")

    # Show exact values if there are interesting variations
    print(f"\n{'='*60}")
    print("Unique framerates (up to 20 most common):")
    print(f"{'='*60}")
    fps_counts_exact = Counter(framerates)
    for fps, count in fps_counts_exact.most_common(20):
        percentage = (count / len(framerates)) * 100
        print(f"{fps:7.3f} fps: {count:4d} videos ({percentage:5.1f}%)")


if __name__ == "__main__":
    main()
