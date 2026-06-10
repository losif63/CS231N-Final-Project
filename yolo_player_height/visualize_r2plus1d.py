#!/usr/bin/env python3
"""
Visualize r2plus1d predictions on video.

Reads a video, extracts 5-frame clips, and visualizes:
- Player height (horizontal line)
- Occlusion probability (red X if > 0.5)
- Game mode (greyscale overlay for right/mirror side)

Output video shows predictions on center frame of each clip.

Written using Claude Code
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from pathlib import Path

# Constants from preprocessing
CROP_RATIO = 415 / 460
BUFFER_PX = 10
INPUT_RESOLUTION = 320
OUTPUT_RESOLUTION = 320  # Display resolution
CLIP_SIZE = 5
HALF_CLIP = 2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def preprocess_frame(frame):
    """Crop and resize frame to model input size."""
    h, w = frame.shape[:2]

    # Apply same crop as preprocessing
    x0, y0, x1, y1 = compute_crop_coords(h, w)
    frame_cropped = frame[y0:y1, x0:x1]

    # Resize to 320x320
    frame_resized = cv2.resize(frame_cropped, (INPUT_RESOLUTION, INPUT_RESOLUTION),
                               interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB and normalize
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_float = frame_rgb.astype(np.float32) / 255.0
    frame_norm = (frame_float - IMAGENET_MEAN) / IMAGENET_STD
    frame_tensor = frame_norm.transpose(2, 0, 1)

    return frame_tensor


@torch.no_grad()
def predict(model, clip_frames, device):
    """
    Get predictions for a 5-frame clip.

    Returns:
        height: float in [0, 1]
        occlusion: float in [0, 1]
        mode: float in [0, 1] (0=left/regular, 1=right/mirror)
    """
    # Stack frames and normalize
    frames_tensor = np.stack(clip_frames, axis=0)  # (5, 3, 320, 320)

    # Convert to torch and move to device
    clip_tensor = torch.from_numpy(frames_tensor).float().to(device)

    # Model expects (B, C, T, H, W)
    clip_tensor = clip_tensor.unsqueeze(0)  # (1, 5, 3, 320, 320)
    clip_tensor = clip_tensor.transpose(1, 2)  # (1, 3, 5, 320, 320)

    # Get predictions
    logits = model(clip_tensor)  # (1, 3)

    pred_y = logits[0, 0].item()  # Height in [0, 1]
    pred_occ = torch.sigmoid(logits[0, 1]).item()  # Occlusion probability
    pred_mode = torch.sigmoid(logits[0, 2]).item()  # Mode probability

    return pred_y, pred_occ, pred_mode


def visualize_predictions(video_path, model_path, output_path="r2plus1d_output.mp4",
                         num_frames=200):
    """
    Run predictions on video and save visualization.

    Args:
        video_path: Path to input video
        model_path: Path to trained model
        output_path: Path to output video
        num_frames: Number of frames to process (default 200)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)

    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Video: {orig_h}×{orig_w} @ {fps} fps, {total_frames} total frames")
    print(f"Processing {min(num_frames, total_frames)} frames...")

    # Compute crop to know dimensions
    x0, y0, x1, y1 = compute_crop_coords(orig_h, orig_w)
    crop_h, crop_w = y1 - y0, x1 - x0

    # Setup output video writer (output at full resolution)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    frame_buffer = []
    predictions = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= num_frames:
            break

        # Add to buffer
        frame_buffer.append(frame.copy())

        # Once we have 5 frames, make a prediction on the center frame
        if len(frame_buffer) == CLIP_SIZE:
            # Preprocess all 5 frames
            clip_frames = [preprocess_frame(f) for f in frame_buffer]

            # Get prediction
            pred_y, pred_occ, pred_mode = predict(model, clip_frames, device)
            predictions.append((pred_y, pred_occ, pred_mode))

            # Draw on center frame (frame_buffer[HALF_CLIP])
            center_frame = frame_buffer[HALF_CLIP].copy()

            # Convert predictions to pixel coordinates
            pred_y_px = pred_y * INPUT_RESOLUTION
            pred_mode_px = pred_mode * INPUT_RESOLUTION

            # Scale back to original resolution
            # Predictions are in 320×320 space, need to scale to crop space, then to original
            pred_y_crop = pred_y_px * (crop_h / INPUT_RESOLUTION)
            pred_y_orig = pred_y_crop + y0

            # Draw height line (horizontal across full width)
            y_line = int(pred_y_orig)
            if 0 <= y_line < orig_h:
                cv2.line(center_frame, (0, y_line), (orig_w, y_line), (0, 255, 0), 2)

            # Draw occlusion X if probability > 0.5
            if pred_occ > 0.5:
                # Draw X in center
                center_x, center_y = orig_w // 2, orig_h // 2
                radius = 50
                cv2.line(center_frame, (center_x - radius, center_y - radius),
                        (center_x + radius, center_y + radius), (0, 0, 255), 3)
                cv2.line(center_frame, (center_x + radius, center_y - radius),
                        (center_x - radius, center_y + radius), (0, 0, 255), 3)

            # Apply greyscale overlay for mode (right side = mirror)
            if pred_mode > 0.5:
                # Greyscale the right half
                gray = cv2.cvtColor(center_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                right_half = center_frame[:, orig_w // 2:]
                gray_right = gray[:, orig_w // 2:]
                # Blend: mostly greyscale with some color
                center_frame[:, orig_w // 2:] = (gray_right * 0.7 + right_half * 0.3).astype(np.uint8)

            # Add text with predictions
            text = f"H:{pred_y:.4f} O:{pred_occ:.4f} M:{pred_mode:.4f}"
            cv2.putText(center_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)

            out.write(center_frame)
            processed += 1

            if processed % 50 == 0:
                print(f"  Processed {processed} frames...")

            # Slide buffer
            frame_buffer.pop(0)

        frame_idx += 1

    cap.release()
    out.release()

    print(f"\nVisualization complete!")
    print(f"  Processed: {processed} frames")
    print(f"  Saved to: {output_path}")
    print(f"  Predictions shape: {len(predictions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize r2plus1d predictions")
    parser.add_argument("--model", default="best_model_r2plus1d.pt",
                       help="Path to trained model (default: best_model_r2plus1d.pt)")
    parser.add_argument("--video",
                       default="../videos/8stars/57484909_2.mp4",
                       help="Path to input video (default: ../videos/8stars/57484909_2.mp4)")
    parser.add_argument("--output", default="r2plus1d_output.mp4",
                       help="Path to output video (default: r2plus1d_output.mp4)")
    parser.add_argument("--frames", type=int, default=200,
                       help="Number of frames to process (default: 200)")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found at {video_path}")
        exit(1)

    visualize_predictions(video_path, model_path, args.output, args.frames)
