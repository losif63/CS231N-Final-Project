"""Label videos with frame-by-frame height/occlusion predictions.

Extracts frames from source videos at 30fps using ffmpeg (piped directly to model,
no intermediate files) and predicts player height [0,1] and occlusion [0,1] for each.

The model receives 4-channel input (RGB + frame difference) at 620x620 resolution.

Outputs:
  <label-dir>/{n}stars/<basename>_labels.json  — frame-by-frame predictions

Example:
    python scripts/height_label_videos.py \
        --src videos \
        --label-dir videos_processed \
        --model-checkpoint yolo_player_height/runs/train_robust_occlusions.../best.pt

The script uses ffmpeg to extract frames at 30fps piped directly to the model
(no intermediate frame files). For 5500 1-2min videos, expect ~1-2 hours total.

Output JSON structure:
    {
        "video_path": "path/to/video.mp4",
        "fps": 30,
        "num_frames": 180,
        "frames": [
            {"frame_idx": 0, "height": 0.45, "occlusion": 0.1},
            ...
        ]
    }

Written using Claude Code
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, BinaryIO

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}

# Constants matching train_robust_occlusions.py
CROP_SIZE = 620
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIFF_MEAN = 0.05
DIFF_STD = 0.15


class GDPlayerModel(nn.Module):
    """YOLOv8n-cls backbone adapted for 4-channel input and MLP head."""

    def __init__(self, model_path: Path):
        super().__init__()
        from ultralytics import YOLO

        checkpoint = torch.load(model_path, map_location="cpu")
        yolo = YOLO("yolov8n-cls.pt")
        self.backbone = yolo.model

        # ── Extend first conv: 3 → 4 input channels ───────────────────
        first_conv = self.backbone.model[0].conv
        out_c = first_conv.out_channels
        new_conv = nn.Conv2d(
            4, out_c,
            first_conv.kernel_size,
            first_conv.stride,
            first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight.data
            new_conv.weight[:, 3] = first_conv.weight.data.mean(dim=1)
            if first_conv.bias is not None:
                new_conv.bias.data.copy_(first_conv.bias.data)
        self.backbone.model[0].conv = new_conv

        # ── Patch classification head for MLP regression ───────────────────
        classify_layer = self.backbone.model[-1]
        in_features = classify_layer.linear.in_features  # 1280
        hidden_features = 128

        # Build MLP: 1280 → 128 → 2
        mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, 2),
        )
        classify_layer.linear = mlp

        # Override forward to use MLP
        def _forward(self, x):
            if isinstance(x, list):
                x = torch.cat(x, 1)
            return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

        import types
        classify_layer.forward = types.MethodType(_forward, classify_layer)

        # Load state dict, handling both direct and wrapped checkpoints
        state_dict = checkpoint["model_state_dict"]

        # If checkpoint has "backbone." prefix, strip it
        if all(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        return self.backbone(x)


def find_inputs(src_root: Path) -> List[Path]:
    """Find all videos matching {1..10}stars/*.<ext>."""
    out: List[Path] = []
    for star_dir in sorted(src_root.iterdir()):
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
        for vid in sorted(star_dir.iterdir()):
            if vid.is_file() and vid.suffix.lower() in _VIDEO_EXTS:
                out.append(vid)
    return out


def dst_label_path(src: Path, src_root: Path, label_root: Path) -> Path:
    """Map source video path to label JSON path."""
    rel = src.relative_to(src_root)
    return label_root / rel.parent / (rel.stem + "_labels.json")


def read_ppm_frame(pipe: BinaryIO, width: int, height: int) -> np.ndarray | None:
    """Read a single PPM frame from ffmpeg pipe.

    Returns (H, W, 3) uint8 BGR array, or None on EOF.
    """
    # Read PPM header: "P6\n<width> <height>\n255\n"
    header = b""
    while header.count(b"\n") < 3:
        chunk = pipe.read(1)
        if not chunk:
            return None
        header += chunk

    # Validate header
    if not header.startswith(b"P6"):
        raise ValueError(f"Invalid PPM header: {header[:10]}")

    # Read RGB data: width * height * 3 bytes
    n_bytes = width * height * 3
    data = pipe.read(n_bytes)
    if len(data) != n_bytes:
        return None

    # Reshape to (H, W, 3) RGB
    frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    # Convert RGB → BGR for consistency with decord
    return frame[:, :, ::-1]


def label_video(
    src: Path,
    label_path: Path,
    model: GDPlayerModel,
    device: torch.device,
    fps: int = 30,
    overwrite: bool = False,
) -> Tuple[Path, str, str]:
    """
    Extract frames from video via ffmpeg pipe, run through model, save predictions.

    Returns (src, status, msg). status ∈ {"ok", "skip", "fail"}.
    """
    if label_path.exists() and not overwrite:
        return (src, "skip", "labels_exist")

    label_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Use ffmpeg to extract frames at 30fps, pipe as PPM (fast, no re-encoding)
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(src),
            "-vf", f"fps={fps}",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "ppm",
            "-",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frames_data = []
        prev_frame = None
        model.eval()
        frame_idx = 0

        with torch.no_grad():
            while True:
                # Read next PPM frame from pipe
                frame = read_ppm_frame(proc.stdout, CROP_SIZE, CROP_SIZE)
                if frame is None:
                    break

                # Compute frame difference
                if prev_frame is not None:
                    diff = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))
                    diff = diff.mean(axis=2)  # Average RGB channels
                else:
                    diff = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)

                prev_frame = frame.copy()

                # Convert BGR → RGB and normalize
                bgr_float = frame.astype(np.float32) / 255.0
                rgb = bgr_float[:, :, ::-1]
                rgb = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)

                diff_norm = ((diff / 255.0 - DIFF_MEAN) / DIFF_STD).astype(np.float32)

                # Stack to (4, H, W)
                tensor = np.concatenate([
                    rgb.transpose(2, 0, 1),      # (3, H, W)
                    diff_norm[np.newaxis],       # (1, H, W)
                ], axis=0)
                tensor = torch.from_numpy(tensor.copy()).unsqueeze(0).to(device)

                # Model prediction
                with torch.no_grad():
                    logits = model(tensor)  # (1, 2)
                    height_pred = torch.sigmoid(logits[0, 0]).item()
                    occlusion_pred = torch.sigmoid(logits[0, 1]).item()

                frames_data.append({
                    "frame_idx": frame_idx,
                    "height": float(height_pred),
                    "occlusion": float(occlusion_pred),
                })
                frame_idx += 1

        # Wait for process to finish
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            return (src, "fail", f"ffmpeg error: {stderr[:200]}")

        if not frames_data:
            return (src, "fail", "no frames extracted")

        # Write labels to JSON
        output = {
            "video_path": str(src),
            "fps": fps,
            "num_frames": len(frames_data),
            "frames": frames_data,
        }

        with open(label_path, "w") as f:
            json.dump(output, f, indent=2)

        return (src, "ok", "")

    except Exception as e:
        return (src, "fail", f"{type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="videos",
                    help="Source videos directory (original, full-resolution).")
    ap.add_argument("--label-dir", default="videos_processed",
                    help="Directory where label JSONs will be written.")
    ap.add_argument("--model-checkpoint", default="yolo_player_height/yolo_mlp_label_model.pt",
                    help="Path to trained model checkpoint (.pt file).")
    ap.add_argument("--fps", type=int, default=30,
                    help="Frame rate to extract at.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run model on (cuda or cpu).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing label files.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process this many videos (0=all). For testing.")
    args = ap.parse_args()

    # Check paths
    src_root = Path(args.src).resolve()
    label_root = Path(args.label_dir).resolve()
    model_path = Path(args.model_checkpoint).resolve()

    if not src_root.is_dir():
        print(f"ERROR: src not found: {src_root}", file=sys.stderr)
        sys.exit(1)

    if not model_path.exists():
        print(f"ERROR: model checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    label_root.mkdir(parents=True, exist_ok=True)

    # Find videos
    inputs = find_inputs(src_root)
    if args.limit:
        inputs = inputs[: args.limit]

    print(f"Found {len(inputs)} videos in {src_root}")
    print(f"Labels will be written to {label_root}/")
    print(f"Target: {args.fps}fps extraction, model input 620×620")
    print(f"Model checkpoint: {model_path}")

    # Load model (main process)
    print("Loading model...")
    device = torch.device(args.device)
    model = GDPlayerModel(model_path).to(device)
    model.eval()

    # Process videos sequentially (GPU memory, streaming ffmpeg pipes don't parallelize well)
    ok = 0
    skipped = 0
    failed: List[Tuple[Path, str]] = []

    print(f"\nProcessing {len(inputs)} videos...")
    with tqdm(total=len(inputs), desc="label") as pbar:
        for src in inputs:
            label_path = dst_label_path(src, src_root, label_root)
            src_rel, status, msg = label_video(src, label_path, model, device, args.fps, args.overwrite)

            if status == "skip":
                skipped += 1
            elif status == "ok":
                ok += 1
            else:
                failed.append((src, msg))

            pbar.update(1)
            pbar.set_postfix(ok=ok, skip=skipped, fail=len(failed))

    # Summary
    print(f"\n{'='*60}")
    print(f"Done. ok={ok}  skipped={skipped}  failed={len(failed)}")

    if failed:
        print(f"\nFirst {min(5, len(failed))} failures:")
        for src, msg in failed[:5]:
            try:
                rel = src.relative_to(src_root)
            except ValueError:
                rel = src
            print(f"  {rel}: {msg}")


if __name__ == "__main__":
    main()
