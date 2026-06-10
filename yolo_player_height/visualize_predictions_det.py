#!/usr/bin/env python3
"""Visualize height/occlusion predictions on a video using detection backbone model.

Loads a trained detection backbone model and runs inference on every frame of a video, visualizing:
  - Height [0,1] as a thin vertical red line
  - Occlusion (> 0.5 threshold) as a red X (two diagonal lines)
  - Raw predictions as text overlay

Output is full-resolution at 30 FPS.

Usage:
    python visualize_predictions_det.py \
        --video ../videos/7stars/49295819_1.mp4 \
        --model yolo_mlp_det_label_model.pt \
        --output output_video.mp4

    # Or with defaults:
    python visualize_predictions_det.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Constants
CROP_SIZE = 620
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIFF_MEAN = 0.05
DIFF_STD = 0.15

# Visualization constants
RED = (0, 0, 255)  # BGR
LINE_THICKNESS = 2
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2
OCCLUSION_THRESHOLD = 0.5


class GDPlayerModel(nn.Module):
    """YOLOv8n detection backbone adapted for 4-channel input and MLP head."""

    def __init__(self, model_path: Path, pretrained=True):
        super().__init__()
        from ultralytics import YOLO
        import os

        # Load YOLOv8n detection model
        yolo_path = "yolov8n.pt"
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"Detection model not found: {yolo_path}")

        yolo = YOLO(yolo_path)
        model_det = yolo.model
        self.is_detection = True

        # Unfreeze all backbone parameters — ultralytics loads with requires_grad=False
        for p in model_det.parameters():
            p.requires_grad_(True)

        # ── 1. Extend first conv: 3 → 4 input channels ───────────────────
        first_conv = model_det.model[0].conv   # Conv2d(3, 16, 3, 2, 1, bias=False)
        out_c = first_conv.out_channels

        new_conv = nn.Conv2d(
            4, out_c,
            first_conv.kernel_size,
            first_conv.stride,
            first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight.data            # copy RGB weights
            new_conv.weight[:, 3]  = first_conv.weight.data.mean(dim=1) # 4th = mean(RGB)
            if first_conv.bias is not None:
                new_conv.bias.data.copy_(first_conv.bias.data)

        model_det.model[0].conv = new_conv
        self.first_conv = new_conv  # Store reference for optimizer

        # ── 2. Keep backbone + neck, remove detection head ────────────────
        # The detection head (Detect) is typically the last layer.
        # We'll find it and remove it, then add our own MLP head.
        # In YOLOv8n-det, the structure is: backbone (0-9) + neck (10-12) + head (13)
        self.backbone_neck = nn.ModuleList()
        self.feature_dim = None

        # Keep all layers except the final Detect head
        for layer in model_det.model[:-1]:
            self.backbone_neck.append(layer)

        # Test forward pass to get feature dimension after pooling
        with torch.no_grad():
            test_x = torch.randn(1, 4, 640, 640)
            test_features = self._get_features(test_x)
            # Global average pooling on multi-scale features
            if isinstance(test_features, (list, tuple)):
                # Detection backbone outputs list of multi-scale features
                # Pool each and concatenate
                pooled = []
                for feat in test_features:
                    pool = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                    pooled.append(pool)
                feature_vector = torch.cat(pooled, dim=1)
            else:
                # Single feature map
                feature_vector = F.adaptive_avg_pool2d(test_features, 1).flatten(1)
            self.feature_dim = feature_vector.shape[1]

        # ── 3. Create MLP regression head ──────────────────────────────────
        hidden_dim = 128
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        # Initialize weights
        nn.init.xavier_normal_(self.head[0].weight)
        nn.init.xavier_normal_(self.head[2].weight)
        with torch.no_grad():
            # height: sigmoid(0) = 0.5 → mid-image
            # occlusion: sigmoid(-2) ≈ 0.12 → usually visible
            self.head[2].bias[0] = 0.0
            self.head[2].bias[1] = -2.0

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def _get_features(self, x):
        """Extract features from backbone+neck (detection model only)."""
        for layer in self.backbone_neck:
            x = layer(x)
        return x

    def forward(self, x):
        if self.is_detection:
            features = self._get_features(x)
            # Handle multi-scale features
            if isinstance(features, (list, tuple)):
                pooled = []
                for feat in features:
                    pool = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                    pooled.append(pool)
                feature_vector = torch.cat(pooled, dim=1)
            else:
                feature_vector = F.adaptive_avg_pool2d(features, 1).flatten(1)
            return self.head(feature_vector)
        else:
            return self.backbone(x)


def process_video(video_path: Path, model_path: Path, output_path: Path, device: torch.device):
    """Process video, run inference, save visualization."""

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"  Resolution: {width}×{height}")
    print(f"  Frames: {frame_count}")

    # Setup video writer (30 FPS, full resolution)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

    # Load model
    print(f"Loading model: {model_path}")
    model = GDPlayerModel(model_path).to(device)
    model.eval()

    prev_frame = None
    frame_idx = 0

    print("Processing frames...")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to crop size for inference
            frame_resized = cv2.resize(frame, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)

            # Compute frame difference
            if prev_frame is not None:
                diff = np.abs(frame_resized.astype(np.float32) - prev_frame.astype(np.float32))
                diff = diff.mean(axis=2)
            else:
                diff = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)

            prev_frame = frame_resized.copy()

            # Prepare input: BGR → RGB, normalize
            bgr_float = frame_resized.astype(np.float32) / 255.0
            rgb = bgr_float[:, :, ::-1]
            rgb = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
            diff_norm = ((diff / 255.0 - DIFF_MEAN) / DIFF_STD).astype(np.float32)

            # Stack to (4, H, W)
            tensor = np.concatenate([
                rgb.transpose(2, 0, 1),
                diff_norm[np.newaxis],
            ], axis=0)
            tensor = torch.from_numpy(tensor.copy()).unsqueeze(0).to(device)

            # Model prediction
            logits = model(tensor)  # (1, 2)
            height_pred = torch.sigmoid(logits[0, 0]).item()
            occlusion_pred = torch.sigmoid(logits[0, 1]).item()
            is_occluded = occlusion_pred > OCCLUSION_THRESHOLD

            # Draw visualizations on original frame
            frame_vis = frame.copy()

            # Draw height line (scale from 620px to actual frame height)
            height_px = int(height_pred * height)
            cv2.line(frame_vis, (0, height_px), (width, height_px), RED, LINE_THICKNESS)

            # Draw occlusion X (if occluded)
            if is_occluded:
                # Diagonal from top-left to bottom-right
                cv2.line(frame_vis, (0, 0), (width, height), RED, LINE_THICKNESS)
                # Diagonal from top-right to bottom-left
                cv2.line(frame_vis, (width, 0), (0, height), RED, LINE_THICKNESS)

            # Draw text overlay (top-left corner)
            text = f"H: {height_pred:.2f}  O: {occlusion_pred:.2f}"
            cv2.putText(frame_vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       TEXT_SCALE, RED, TEXT_THICKNESS)

            # Write frame to output video
            out.write(frame_vis)

            frame_idx += 1
            if frame_idx % max(1, frame_count // 10) == 0:
                print(f"  {frame_idx}/{frame_count} frames processed")

    cap.release()
    out.release()

    print(f"✓ Output saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize height/occlusion predictions on video using detection backbone model."
    )
    parser.add_argument("--video", default="../videos/7stars/49295819_1.mp4",
                       help="Input video path.")
    parser.add_argument("--model", default="yolo_mlp_det_label_model.pt",
                       help="Model checkpoint path.")
    parser.add_argument("--output", default="det_output_predictions.mp4",
                       help="Output video path.")
    parser.add_argument("--device", default="cpu",
                       help="Device (cpu, cuda, mps).")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    device = torch.device(args.device)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    process_video(video_path, model_path, output_path, device)


if __name__ == "__main__":
    main()
