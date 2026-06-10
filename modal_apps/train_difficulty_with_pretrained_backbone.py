"""
Train difficulty model with pretrained trajectory backbone on Modal (Modal 1.0).

Usage:
  modal run modal_apps/train_difficulty_with_pretrained_backbone.py \
    --gpu h100 --epochs 10 --batch-size 4 \
    --wandb-project gd-difficulty --wandb-run-name slowfast-pretrained-v1
"""

import modal
import subprocess
import sys
from pathlib import Path

app = modal.App(name="difficulty-training-pretrained")
data_vol = modal.Volume.from_name("pretrain-data", create_if_missing=True)

# Define image with local src code included (Modal 1.0 style)
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install(
        "torch",
        "torchvision",
        "pytorchvideo",
        "fvcore",
        "h5py",
        "numpy",
        "decord",
        "tensorboard",
        "wandb",
    )
    .add_local_python_source("src")  # Include src package - Modal 1.0 style
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": data_vol},
    timeout=60*60*12,
    memory=30 * 1024 * 2 * 2 * 2,
    secrets=[modal.Secret.from_name("wandb-secret")],
    cpu=20.0
)
def run_difficulty_training_h100(
    epochs: int = 50,
    batch_size: int = 32,
    clips_train: int = 8,
    clips_eval: int = 24,
    num_workers: int = 4,
    wandb_project: str = "gd-difficulty",
    wandb_entity: str = None,
    wandb_run_name: str = "slowfast-pretrained",
):
    """Run difficulty training on H100."""
    return _run_training_impl(epochs, batch_size, clips_train, clips_eval, num_workers, wandb_project, wandb_entity, wandb_run_name)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol},
    timeout=3600,
    memory=30 * 1024,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_difficulty_training_a100(
    epochs: int = 50,
    batch_size: int = 32,
    clips_train: int = 8,
    clips_eval: int = 24,
    num_workers: int = 4,
    wandb_project: str = "gd-difficulty",
    wandb_entity: str = None,
    wandb_run_name: str = "slowfast-pretrained",
):
    """Run difficulty training on A100."""
    return _run_training_impl(epochs, batch_size, clips_train, clips_eval, num_workers, wandb_project, wandb_entity, wandb_run_name)


def _run_training_impl(
    epochs: int,
    batch_size: int,
    clips_train: int,
    clips_eval: int,
    num_workers: int,
    wandb_project: str,
    wandb_entity: str,
    wandb_run_name: str,
):
    """Run training implementation."""
    import json
    import torch

    # Debug checkpoint structure
    print("\n=== CHECKPOINT DEBUG ===")
    backbone_ckpt = Path("/data/checkpoints/pretrain/best_model.pt")
    ckpt_data = torch.load(backbone_ckpt, map_location="cpu")
    print(f"Top-level checkpoint keys: {list(ckpt_data.keys())}")

    if "model_state" in ckpt_data:
        model_state = ckpt_data["model_state"]
        print(f"\nmodel_state has {len(model_state)} keys")
        print(f"First 10 keys: {list(model_state.keys())[:10]}")
    elif "model" in ckpt_data:
        model_state = ckpt_data["model"]
        print(f"\nmodel has {len(model_state)} keys")
        print(f"First 10 keys: {list(model_state.keys())[:10]}")
    print("=== END DEBUG ===\n")

    # Fix splits.json paths
    print("Fixing splits.json paths...")
    splits_path = Path("/data/splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    fixed_splits = {"seed": splits.get("seed", 0)}
    for split_name in ["train", "val", "test"]:
        if split_name in splits:
            fixed_paths = []
            for p in splits[split_name]:
                if p.startswith("/data/videos_processed/"):
                    abs_path = p
                elif p.startswith("videos_processed/"):
                    abs_path = "/data/" + p
                elif p.startswith("/data/"):
                    if "/videos_processed/" not in p:
                        abs_path = p.replace("/data/", "/data/videos_processed/")
                    else:
                        abs_path = p
                else:
                    abs_path = "/data/videos_processed/" + p
                fixed_paths.append(abs_path)
            fixed_splits[split_name] = fixed_paths

    with open(splits_path, "w") as f:
        json.dump(fixed_splits, f)

    print(f"✓ Fixed splits: train={len(fixed_splits.get('train', []))}, val={len(fixed_splits.get('val', []))}, test={len(fixed_splits.get('test', []))}\n")

    # Build training command - src code is now available via Image.add_local_python_source
    cmd = [
        sys.executable,
        "-m", "src.train",  # Run as module since src is in the path
        "--backbone", "slowfast",
        "--videos-root", "/data/videos_processed",
        "--splits", "/data/splits.json",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--clips-train", str(clips_train),
        "--clips-eval", str(clips_eval),
        "--pretrained-backbone-checkpoint", "/data/checkpoints/pretrain/best_model.pt",
        "--wandb-project", wandb_project,
        "--wandb-run-name", wandb_run_name,
        "--out-dir", "/data/runs/slowfast-pretrained",
        "--num-workers", str(num_workers),
    ]

    if wandb_entity:
        cmd.extend(["--wandb-entity", wandb_entity])

    print(f"Running: {' '.join(cmd[:4])} ...\n")
    result = subprocess.run(cmd)

    return result.returncode == 0


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    epochs: int = 50,
    batch_size: int = 32,
    clips_train: int = 8,
    clips_eval: int = 24,
    num_workers: int = 4,
    wandb_project: str = "gd-difficulty",
    wandb_entity: str = None,
    wandb_run_name: str = "slowfast-pretrained",
):
    """Main entrypoint."""
    print("=" * 60)
    print("DIFFICULTY TRAINING WITH PRETRAINED BACKBONE")
    print("=" * 60)

    gpu = gpu.lower()
    if gpu not in ["h100", "a100"]:
        print(f"✗ Unknown GPU: {gpu}")
        return

    print(f"\nStarting on {gpu.upper()}...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  W&B: {wandb_project}/{wandb_run_name}\n")

    if gpu == "h100":
        success = run_difficulty_training_h100.remote(
            epochs=epochs,
            batch_size=batch_size,
            clips_train=clips_train,
            clips_eval=clips_eval,
            num_workers=num_workers,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
        )
    else:
        success = run_difficulty_training_a100.remote(
            epochs=epochs,
            batch_size=batch_size,
            clips_train=clips_train,
            clips_eval=clips_eval,
            num_workers=num_workers,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
        )

    if success:
        print("\n✓ TRAINING COMPLETE")
    else:
        print("\n✗ TRAINING FAILED")
