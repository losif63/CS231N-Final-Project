#!/usr/bin/env python3
"""
Extract SlowFast backbone from trained TrajectoryModel checkpoint.

The trajectory model trains a backbone + MLP head for height prediction.
We extract just the backbone weights to use as pretrained initialization
for difficulty prediction training.

Usage:
    python scripts/extract_trajectory_backbone.py \
        --trajectory-checkpoint path/to/best_model.pt \
        --output pretrained_slowfast_backbone.pt

Written using Claude Code
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_backbone(checkpoint_path, output_path):
    """
    Extract backbone state_dict from TrajectoryModel checkpoint.

    Args:
        checkpoint_path: Path to trajectory model checkpoint
        output_path: Path to save extracted backbone weights
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Checkpoint structure: {"model_state": state_dict, "epoch": ..., "args": ...}
    model_state = checkpoint.get("model_state") or checkpoint.get("model")

    if model_state is None:
        print(f"ERROR: Could not find model state in checkpoint")
        print(f"Available keys: {list(checkpoint.keys())}")
        return False

    print(f"Total keys in checkpoint: {len(model_state)}")

    # Extract backbone keys (prefixed with "backbone.")
    backbone_state = {}
    for key, value in model_state.items():
        if key.startswith("backbone."):
            # Remove "backbone." prefix
            new_key = key[len("backbone."):]
            backbone_state[new_key] = value

    print(f"Extracted {len(backbone_state)} backbone keys")

    if len(backbone_state) == 0:
        print("WARNING: No backbone keys found! Checkpoint structure may be different.")
        print(f"Available keys: {list(model_state.keys())[:10]}...")
        return False

    # Save extracted backbone
    torch.save(backbone_state, output_path)
    print(f"✓ Saved backbone to: {output_path}")

    # Verify by trying to load into a fresh SlowFast model
    print("\nVerifying backbone state_dict...")
    try:
        from src.models.factory import build_backbone

        backbone = build_backbone("slowfast", pretrained=False)
        backbone.load_state_dict(backbone_state)

        print(f"✓ Successfully loaded into SlowFast backbone")
        print(f"  Feature dimension: {backbone.feature_dim}")

        return True
    except Exception as e:
        print(f"✗ Error loading backbone: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SlowFast backbone from TrajectoryModel checkpoint"
    )
    parser.add_argument(
        "--trajectory-checkpoint",
        type=str,
        required=True,
        help="Path to trajectory model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pretrained_slowfast_backbone.pt",
        help="Output path for extracted backbone",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.trajectory_checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        exit(1)

    output_path = Path(args.output)

    success = extract_backbone(str(checkpoint_path), str(output_path))
    exit(0 if success else 1)
