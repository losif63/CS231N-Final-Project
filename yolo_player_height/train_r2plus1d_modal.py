"""
Train r2plus1d on Modal with A100 GPU.

Usage:
  modal run train_r2plus1d_modal.py

This will:
  1. Upload the dataset zip to Modal volume
  2. Uncompress it
  3. Train the model on A100 GPU
  4. Download results
"""

import modal
import subprocess
import zipfile
from pathlib import Path

# Create Modal app and volume
app = modal.App(name="r2plus1d-training")
volume = modal.Volume.from_name("r2plus1d-data", create_if_missing=True)

# Define the training image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    .pip_install(
        "torch==2.1.1",
        "torchvision==0.16.1",
        "numpy<2",  # Pin NumPy to <2 (torch 2.1.1 compiled against 1.x)
        "opencv-python-headless",  # Headless version for servers
        "tqdm",
        "tensorboard",
    )
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": volume},
    timeout=60*60*5,
    memory=50 * 1024,  # 50GB RAM
)
def train_r2plus1d():
    """Run r2plus1d training on A100."""
    import os

    data_dir = Path("/data")
    dataset_zip = data_dir / "r2plus1d_dataset_10000_0.zip"
    dataset_dir = data_dir / "r2plus1d_dataset_10000_0"

    print(f"Working directory: {os.getcwd()}")
    print(f"Data directory: {data_dir}")
    print(f"Available: {list(data_dir.iterdir())}")

    # Check if already extracted
    if dataset_dir.exists():
        print("Dataset already extracted")
    else:
        print(f"Extracting {dataset_zip}...")
        if dataset_zip.exists():
            with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete")
        else:
            print(f"ERROR: {dataset_zip} not found")
            return

    # Create training script
    train_script = """
import torch
import torch.nn as nn
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.video import r2plus1d_18

CLIP_SIZE = 5
INPUT_RESOLUTION = 320
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class R2Plus1DClipDataset(Dataset):
    def __init__(self, samples, dataset_dir, augment=False):
        self.samples = samples
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        npy_path = self.dataset_dir / s["filename"]
        clip_array = np.load(npy_path)

        frames = [clip_array[i].copy() for i in range(CLIP_SIZE)]
        label_y = s["label_y"]
        label_x = s["label_x"]
        occlusion = s["occlusion"]

        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                frames = [cv2.flip(f, 1) for f in frames]
                label_x = INPUT_RESOLUTION - label_x
            # Vertical flip
            if random.random() < 0.5:
                frames = [cv2.flip(f, 0) for f in frames]
                label_y = INPUT_RESOLUTION - label_y

        # Convert to float and normalize
        frames_tensor = []
        for f in frames:
            f = f.astype(np.float32) / 255.0
            f = (f - IMAGENET_MEAN) / IMAGENET_STD
            f = f.transpose(2, 0, 1)
            frames_tensor.append(f)

        clip_tensor = np.stack(frames_tensor, axis=0)

        return {
            "clip": torch.from_numpy(clip_tensor).float(),
            "label_y": torch.tensor(label_y / INPUT_RESOLUTION, dtype=torch.float32),
            "label_x": torch.tensor(label_x / INPUT_RESOLUTION, dtype=torch.float32),
            "occlusion": torch.tensor(occlusion, dtype=torch.float32),
        }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load dataset
    dataset_dir = Path("/data/r2plus1d_dataset_10000_0")
    with open(dataset_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(dataset_dir / "labels_4567.json") as f:
        labels = json.load(f)

    # Prepare labels
    labeled_samples = []
    for item in metadata:
        sample_id = str(item["sample_id"])
        if sample_id not in labels:
            continue
        lbl = labels[sample_id]
        if lbl.get("removed", False):
            continue

        labeled_samples.append({
            "filename": item["filename"],
            "label_y": lbl.get("height_y", INPUT_RESOLUTION / 2),
            "label_x": lbl.get("height_x", INPUT_RESOLUTION / 2),
            "occlusion": 1.0 if lbl.get("not_in_frame", False) else 0.0,
        })

    print(f"Total labeled samples: {len(labeled_samples)}")

    # Video-level split (all from same video go together)
    random.seed(42)
    random.shuffle(labeled_samples)
    split_idx = int(0.8 * len(labeled_samples))
    train_samples = labeled_samples[:split_idx]
    val_samples = labeled_samples[split_idx:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_dataset = R2Plus1DClipDataset(train_samples, dataset_dir, augment=True)
    val_dataset = R2Plus1DClipDataset(val_samples, dataset_dir, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Model
    model = r2plus1d_18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3),
    )
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Loss functions
    height_loss_fn = nn.SmoothL1Loss(beta=0.25)
    occlusion_loss_fn = nn.BCEWithLogitsLoss()
    mode_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            clip = batch["clip"].to(device)
            label_y = batch["label_y"].to(device)
            occlusion = batch["occlusion"].to(device)

            # Model expects (B, C, T, H, W)
            clip = clip.transpose(1, 2)

            logits = model(clip)
            pred_y = logits[:, 0]
            pred_occ = logits[:, 1]
            pred_mode = logits[:, 2]

            loss_h = height_loss_fn(pred_y, label_y)
            loss_occ = occlusion_loss_fn(pred_occ, occlusion)
            loss_mode = mode_loss_fn(pred_mode, torch.zeros_like(pred_mode))

            loss = loss_h + loss_occ + loss_mode

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                clip = batch["clip"].to(device)
                label_y = batch["label_y"].to(device)
                occlusion = batch["occlusion"].to(device)

                clip = clip.transpose(1, 2)

                logits = model(clip)
                pred_y = logits[:, 0]
                pred_occ = logits[:, 1]
                pred_mode = logits[:, 2]

                loss_h = height_loss_fn(pred_y, label_y)
                loss_occ = occlusion_loss_fn(pred_occ, occlusion)
                loss_mode = mode_loss_fn(pred_mode, torch.zeros_like(pred_mode))

                loss = loss_h + loss_occ + loss_mode
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "/data/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


import cv2
import torch

if __name__ == "__main__":
    train()
"""

    # Write and run training script
    train_file = Path("/data/train.py")
    train_file.write_text(train_script)

    print("Running training...")
    result = subprocess.run(
        ["python", str(train_file)],
        cwd="/data",
        capture_output=False,
    )

    if result.returncode == 0:
        print("Training completed successfully!")
        # List output files
        print("Output files:")
        for f in data_dir.glob("*.pt"):
            print(f"  {f.name} ({f.stat().st_size / 1e9:.2f}GB)")
    else:
        print(f"Training failed with return code {result.returncode}")


@app.local_entrypoint()
def main():
    """Local entrypoint to run training."""
    local_zip = Path("r2plus1d_dataset_10000_0.zip")
    if not local_zip.exists():
        print(f"ERROR: {local_zip} not found")
        return

    # Try to upload, but skip if already exists
    try:
        print(f"Uploading {local_zip.name} ({local_zip.stat().st_size / 1e9:.2f}GB)...")
        with volume.batch_upload() as batch:
            batch.put_file(local_zip, f"/r2plus1d_dataset_10000_0.zip")
        print("Upload complete")
    except FileExistsError:
        print("Dataset already in volume, skipping upload")

    print("Starting training on A100...")
    train_r2plus1d.remote()
    print("Done!")
