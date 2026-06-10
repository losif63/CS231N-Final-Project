"""
Modal app for trajectory pretraining with SlowFast backbone.

Supports both A100 and H100 GPUs.

Usage:
    modal run modal_apps/pretrain_app.py --epochs 50 --batch-size 32 --gpu h100
    modal run modal_apps/pretrain_app.py --epochs 50 --batch-size 32 --gpu a100

Written using Claude Code
"""

import subprocess
import sys
from pathlib import Path

import modal

# Create Modal app and volume
app = modal.App(name="trajectory-pretrain")
volume = modal.Volume.from_name("pretrain-data", create_if_missing=True)

# Create image with dependencies
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
        "decord",  # For data loading
    )
)


# H100 training function
@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100",
    timeout=3600,
    memory=30 * 1024,
)
def run_training_h100(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    backbone: str = "slowfast",
):
    """Run trajectory pretraining on Modal H100 GPU."""
    return _run_training_impl(epochs, batch_size, learning_rate, backbone, "H100")


# A100 training function
@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=3600,
    memory=30 * 1024,
)
def run_training_a100(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    backbone: str = "slowfast",
):
    """Run trajectory pretraining on Modal A100 GPU."""
    return _run_training_impl(epochs, batch_size, learning_rate, backbone, "A100")


# Helper function to check if dataset exists in volume
@app.function(volumes={"/data": volume})
def check_dataset_exists():
    """Check if dataset exists in volume."""
    from pathlib import Path
    hdf5_path = Path("/data/pretraining_dataset.hdf5")
    tar_path = Path("/data/pretraining_dataset.tar.gz")
    return hdf5_path.exists(), tar_path.exists()


# Helper function to check if code exists in volume
@app.function(volumes={"/data": volume})
def check_code_exists():
    """Check if code exists in volume."""
    from pathlib import Path
    src_path = Path("/data/src")
    return src_path.exists() and bool(list(src_path.glob("**/*.py")))


def _run_training_impl(epochs, batch_size, learning_rate, backbone, gpu_name):
    """Shared training implementation."""
    import os

    print(f"\n{'=' * 60}")
    print(f"TRAINING ON {gpu_name} GPU")
    print(f"{'=' * 60}\n")

    data_dir = Path("/data")
    dataset_tar_gz = data_dir / "pretraining_dataset.tar.gz"
    dataset_hdf5 = data_dir / "pretraining_dataset.hdf5"

    print(f"Data directory: {data_dir}")
    print(f"Available files: {list(data_dir.iterdir())}")

    # Check if dataset is valid (correct size)
    EXPECTED_SIZE = 3.2e9  # 3.2GB
    MINIMUM_SIZE = 3.0e9   # At least 3.0GB

    if dataset_hdf5.exists():
        file_size = dataset_hdf5.stat().st_size
        if file_size > MINIMUM_SIZE:
            print(f"✓ Dataset already extracted and valid ({file_size / 1e9:.1f} GB)")
        else:
            print(f"✗ Dataset file corrupted ({file_size / 1e9:.1f} GB), re-extracting...")
            dataset_hdf5.unlink()  # Delete corrupted file

            if dataset_tar_gz.exists():
                print(f"✓ Extracting {dataset_tar_gz.name}...")
                import tarfile
                with tarfile.open(dataset_tar_gz, "r:gz") as tar:
                    tar.extractall(data_dir)
                print(f"✓ Extraction complete")
            else:
                print(f"✗ ERROR: {dataset_tar_gz} not found for re-extraction")
                return False
    else:
        if dataset_tar_gz.exists():
            print(f"✓ Extracting {dataset_tar_gz.name}...")
            import tarfile
            with tarfile.open(dataset_tar_gz, "r:gz") as tar:
                tar.extractall(data_dir)
            print(f"✓ Extraction complete")
        else:
            print(f"✗ ERROR: {dataset_tar_gz} not found")
            return False

    # Verify extraction
    if not dataset_hdf5.exists():
        print(f"✗ Extraction failed - HDF5 not found")
        return False

    file_size = dataset_hdf5.stat().st_size
    if file_size < MINIMUM_SIZE:
        print(f"✗ Extracted file too small ({file_size / 1e9:.1f} GB), expected ~3.2 GB")
        return False

    print(f"✓ Dataset ready: {dataset_hdf5} ({dataset_hdf5.stat().st_size / 1e9:.1f} GB)")

    # Run training
    cmd = [
        sys.executable,
        "src/pretrain.py",
        "--backbone", backbone,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--dataset-path", "/data/pretraining_dataset.hdf5",
        "--save-dir", "/data/checkpoints/pretrain",
        "--num-workers", "4",
    ]

    print(f"\nRunning: {' '.join(cmd[:2])} ...\n")

    result = subprocess.run(cmd, cwd="/data")

    if result.returncode == 0:
        print("\n✓ Training completed successfully!")
        return True
    else:
        print(f"\n✗ Training failed with exit code {result.returncode}")
        return False


@app.local_entrypoint()
def main(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    backbone: str = "slowfast",
    gpu: str = "h100",
):
    """
    Local entrypoint: uploads dataset using batch_upload() for efficiency.

    Usage:
        modal run modal_apps/pretrain_app.py --epochs 50 --batch-size 32 --gpu h100
        modal run modal_apps/pretrain_app.py --epochs 50 --batch-size 32 --gpu a100
    """
    gpu = gpu.lower()
    if gpu not in ["a100", "h100"]:
        print(f"✗ Unknown GPU: {gpu}. Choose 'a100' or 'h100'")
        return

    print("=" * 60)
    print(f"TRAJECTORY PRETRAINING ON MODAL ({gpu.upper()})")
    print("=" * 60)

    # Use compressed dataset
    local_tar_gz = Path("yolo_player_height/pretraining_dataset.tar.gz")

    if not local_tar_gz.exists():
        print(f"\n✗ ERROR: {local_tar_gz} not found")
        print(f"\nCreate it with:")
        print(f"  cd yolo_player_height")
        print(f"  tar czf pretraining_dataset.tar.gz pretraining_dataset.hdf5")
        return

    tar_size_gb = local_tar_gz.stat().st_size / 1e9

    print(f"\n✓ Found: {local_tar_gz.resolve()} ({tar_size_gb:.1f} GB)")

    # Check if dataset already exists in volume
    print(f"\n1. Checking Modal volume for existing dataset...")

    hdf5_exists, tar_exists = check_dataset_exists.remote()

    if hdf5_exists:
        print("   ✓ Extracted dataset already in volume, skipping upload")
    elif tar_exists:
        print("   ✓ Compressed dataset already in volume, skipping upload")
    else:
        print(f"   Uploading dataset to Modal volume (streaming)...")
        try:
            with volume.batch_upload() as batch:
                batch.put_file(local_tar_gz, "/pretraining_dataset.tar.gz")
            print("   ✓ Upload complete")
        except FileExistsError:
            print("   ✓ Dataset already in volume")

    # Upload code
    print(f"\n2. Checking for training code...")

    code_exists = check_code_exists.remote()

    if code_exists:
        print("   ✓ Code already in volume, skipping upload")
    else:
        print("   ✓ Uploading training code...")
        try:
            with volume.batch_upload() as batch:
                for py_file in Path("src").rglob("*.py"):
                    batch.put_file(py_file, f"/src/{py_file.relative_to('src')}")
            print("   ✓ Code uploaded")
        except FileExistsError:
            print("   ✓ Code already in volume")

    # Start training
    print(f"\n3. Starting training on Modal {gpu.upper()}...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Backbone: {backbone}")
    print()

    # Choose GPU function
    if gpu == "h100":
        success = run_training_h100.remote(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            backbone=backbone,
        )
    else:  # a100
        success = run_training_a100.remote(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            backbone=backbone,
        )

    if success:
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nCheckpoints saved to: /data/checkpoints/pretrain/")
        print("View training history:")
        print("  cat /data/checkpoints/pretrain/history.json | python -m json.tool")
    else:
        print("\n" + "=" * 60)
        print("✗ TRAINING FAILED")
        print("=" * 60)


if __name__ == "__main__":
    main()
