"""Extract ResNet-18 features from videos in videos_processed/ and save to processed_resnet18/.

Videos are sampled at 1 frame every 4 seconds (120 frames at 30fps).
ResNet-18 is frozen and runs in no_grad mode on MPS.

Processing order is interleaved across classes to ensure balanced resumption.
Uses a prefetch thread to overlap I/O with GPU computation.

Written using Claude Code
"""

import argparse
import os
import queue
import threading
import time
import psutil
from pathlib import Path
from typing import Generator, Optional, Tuple

import decord
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ResNet18_Weights

decord.bridge.set_bridge("torch")

# ImageNet normalization stats
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ProgressTracker:
    """Track extraction progress with ETA and memory usage."""

    def __init__(self, total_videos: int):
        self.total_videos = total_videos
        self.start_time = time.time()
        self.processed = 0
        self.skipped = 0
        self.errors = 0
        self.last_log_time = self.start_time

    def update(self, video_path: Path, skipped: bool = False, error: bool = False):
        """Update progress after processing a video."""
        self.processed += 1
        if skipped:
            self.skipped += 1
        if error:
            self.errors += 1

        current_time = time.time()
        # Log every 10 videos or every 30 seconds
        if self.processed % 10 == 0 or (current_time - self.last_log_time) > 30:
            self._log_progress()
            self.last_log_time = current_time

    def _log_progress(self):
        """Print progress information."""
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = (self.total_videos - self.processed) / rate if rate > 0 else 0

        # Memory usage
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        mem_percent = process.memory_percent()

        new_videos = self.processed - self.skipped - self.errors
        percent = 100 * self.processed / self.total_videos

        print(
            f"[Progress] {self.processed:4d}/{self.total_videos} ({percent:5.1f}%) | "
            f"New: {new_videos:4d} | Skipped: {self.skipped:4d} | Errors: {self.errors:2d} | "
            f"Rate: {rate:5.1f} vid/s | ETA: {remaining/60:6.1f}m | "
            f"Memory: {mem_mb:7.0f}MB ({mem_percent:5.1f}%)"
        )

    def finish(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        new_videos = self.processed - self.skipped - self.errors

        print(f"\n{'='*90}")
        print(f"Extraction Complete!")
        print(f"{'='*90}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Videos processed: {new_videos} new + {self.skipped} skipped + {self.errors} errors = {self.processed} total")
        print(f"Average rate: {self.processed/elapsed:.1f} videos/second")

        # Final memory
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        print(f"Final memory usage: {mem_mb:.0f}MB")
        print(f"{'='*90}\n")


def build_resnet18(device: torch.device) -> nn.Module:
    """Load pretrained ResNet-18, strip final FC, freeze all params."""
    backbone = tvm.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Strip final FC layer; keep everything up to (and including) avgpool
    model = nn.Sequential(*list(backbone.children())[:-1])
    model.to(device)
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    return model


def scan_videos_by_class(videos_root: str | Path) -> dict[int, list[Path]]:
    """Scan videos_processed/{n}stars/ and return dict[label] -> list of .mp4 paths.

    Label is 0-indexed (1stars -> 0, 10stars -> 9).
    """
    videos_root = Path(videos_root)
    videos_by_class = {i: [] for i in range(10)}

    for star_dir in sorted(videos_root.iterdir()):
        if not star_dir.is_dir():
            continue
        name = star_dir.name
        if not name.endswith("stars"):
            continue
        try:
            stars = int(name[: -len("stars")])
        except ValueError:
            continue
        if not (1 <= stars <= 10):
            continue
        label = stars - 1

        for vid_path in sorted(star_dir.iterdir()):
            if vid_path.is_file() and vid_path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                videos_by_class[label].append(vid_path)

    return videos_by_class


def interleave_videos(videos_by_class: dict[int, list[Path]]) -> Generator[Tuple[int, Path], None, None]:
    """Yield (label, video_path) in round-robin order across classes.

    Ensures balanced processing: if interrupted, each class has roughly equal counts.
    """
    max_per_class = max(len(paths) for paths in videos_by_class.values())
    for round_idx in range(max_per_class):
        for label in range(10):
            if round_idx < len(videos_by_class[label]):
                yield label, videos_by_class[label][round_idx]


def extract_frames(video_path: Path, stride: int = 120) -> Optional[torch.Tensor]:
    """Load video and extract frames at the given stride (1 frame per `stride` frames).

    Returns:
        Tensor of shape (T, C, H, W) float in [0, 1], or None on error.
    """
    try:
        vr = decord.VideoReader(str(video_path), num_threads=1)
        n_frames = len(vr)
        if n_frames == 0:
            return None

        # Sample frames: [0, stride, 2*stride, ...]
        frame_indices = list(range(0, n_frames, stride))
        if not frame_indices:
            frame_indices = [0]

        # decord returns (N, H, W, C) — with torch bridge, already a tensor
        frames = vr.get_batch(frame_indices)  # (T, H, W, C)
        # Handle all possible return types
        if isinstance(frames, torch.Tensor):
            # Already a tensor, convert to float
            frames = frames.float() / 255.0
        elif hasattr(frames, 'asnumpy'):
            # decord NDArray type, convert via numpy
            frames = torch.from_numpy(frames.asnumpy()).float() / 255.0
        else:
            # numpy array or other
            frames = torch.from_numpy(frames).float() / 255.0
        # Permute to (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W) in [0, 1]
        del vr
        return frames
    except Exception as e:
        print(f"  Error reading {video_path}: {e}")
        return None


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """Center-crop to 224x224 and normalize with ImageNet stats.

    Input: (T, C, H, W) in [0, 1], H=224, W=398.
    Output: (T, C, 224, 224) normalized.
    """
    T, C, H, W = frames.shape
    # Center crop width: W=398 -> 224; start = (398 - 224) // 2 = 87
    start_w = (W - 224) // 2
    frames = frames[:, :, :, start_w : start_w + 224]  # (T, C, 224, 224)

    # Normalize with ImageNet stats
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return frames


def prefetch_worker(
    video_iter: Generator[Tuple[int, Path], None, None],
    out_queue: queue.Queue,
    limit: Optional[int] = None,
) -> None:
    """Producer thread: load and preprocess frames, put into queue.

    Puts (label, video_path, frames) tuples, or None to signal end.
    """
    count = 0
    try:
        for label, video_path in video_iter:
            if limit is not None and count >= limit:
                break

            frames = extract_frames(video_path)
            if frames is None:
                continue

            frames = preprocess_frames(frames)
            out_queue.put((label, video_path, frames))
            count += 1
    finally:
        out_queue.put(None)  # Sentinel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos-dir",
        default="videos_processed",
        help="Path to videos_processed/ root",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_resnet18",
        help="Path to output processed_resnet18/ root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process (for testing)",
    )
    args = parser.parse_args()

    videos_root = Path(args.videos_dir)
    output_root = Path(args.output_dir)

    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*90}")
    print(f"ResNet-18 Feature Extraction")
    print(f"{'='*90}")
    print(f"Device: {device}")
    print(f"Videos directory: {videos_root}")
    print(f"Output directory: {output_root}")
    if args.limit:
        print(f"Limit: {args.limit} videos")
    print(f"{'='*90}\n")

    # Build model
    print("Loading ResNet-18 model...")
    model = build_resnet18(device)
    model.eval()
    print("Model loaded (frozen, no_grad enabled)")

    # Create output directories
    for label in range(10):
        label_dir = output_root / f"{label + 1}stars"
        label_dir.mkdir(parents=True, exist_ok=True)

    # Scan videos and organize by class
    print("\nScanning videos by class...")
    videos_by_class = scan_videos_by_class(videos_root)
    total_videos = sum(len(paths) for paths in videos_by_class.values())
    print(f"Found {total_videos} total videos")
    for label in range(10):
        count = len(videos_by_class[label])
        if count > 0:
            print(f"  {label + 1}stars: {count} videos")
    print()

    # Interleave videos across classes
    video_iter = interleave_videos(videos_by_class)

    # Start prefetch thread (maxsize=1 reduces memory overhead)
    prefetch_queue: queue.Queue = queue.Queue(maxsize=1)
    prefetch_thread = threading.Thread(
        target=prefetch_worker,
        args=(video_iter, prefetch_queue, args.limit),
        daemon=False,
    )
    prefetch_thread.start()

    # Main processing loop
    progress = ProgressTracker(total_videos)

    with torch.no_grad():
        while True:
            item = prefetch_queue.get()
            if item is None:
                break

            label, video_path, frames = item

            # Check if already processed
            output_path = output_root / f"{label + 1}stars" / f"{video_path.stem}.npy"
            if output_path.exists():
                progress.update(video_path, skipped=True)
                continue

            try:
                # Run inference
                frames = frames.to(device)
                features = model(frames)  # (T, 512, 1, 1)
                features = features.reshape(features.shape[0], -1)  # (T, 512)

                # Save to numpy
                features_np = features.cpu().numpy().astype(np.float32)
                np.save(str(output_path), features_np)

                # Clear GPU memory after processing each video
                del frames, features, features_np
                if device.type == 'mps':
                    torch.mps.empty_cache()

                progress.update(video_path, skipped=False)
            except Exception as e:
                print(f"  Error processing {video_path}: {e}")
                # Clear memory on error too
                if device.type == 'mps':
                    torch.mps.empty_cache()
                progress.update(video_path, error=True)

    prefetch_thread.join()
    progress.finish()


if __name__ == "__main__":
    main()
