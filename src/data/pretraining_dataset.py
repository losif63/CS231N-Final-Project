"""
Trajectory pretraining dataset loader.

Loads HDF5 file with 48-frame video clips and 64 label values per clip.
At training time, randomly samples offsets to create (16-frame input, 32-frame output) pairs.

Written using Claude Code
"""

from __future__ import annotations

import h5py
import json
import numpy as np
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    """
    Load trajectory pretraining data from HDF5.

    Each sample in the HDF5 file contains:
    - frames: (48, 224, 224, 3) uint8 video frames
    - heights: (64,) float32 height labels
    - occlusion: (64,) float32 occlusion labels

    At training time, randomly sample offset ∈ [0, 16] to create:
    - Input: 16 frames at positions [offset, offset+16)
    - Target: 32 height labels + 32 occlusion labels at [offset+16, offset+48)
    """

    # Normalization (ImageNet stats)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        dataset_path: str | Path = "yolo_player_height/pretraining_dataset.hdf5",
        split: str = "train",
        augment: bool = False,
    ):
        """
        Args:
            dataset_path: Path to HDF5 file
            split: "train" or "val"
            augment: Whether to apply data augmentation
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.augment = augment

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # Load split info to know which clips belong to this split
        split_file = self.dataset_path.parent / "pretraining_split.json"
        if split_file.exists():
            with open(split_file) as f:
                split_data = json.load(f)
                self.split_videos = set(split_data[f"{split}_videos"])
        else:
            self.split_videos = None

        # Scan HDF5 to find valid clips for this split
        with h5py.File(self.dataset_path, "r") as hf:
            clip_ids = []
            for clip_id in sorted(hf["clips"].keys()):
                clip_group = hf["clips"][clip_id]

                # Check split if split info available
                if self.split_videos is not None:
                    video_path = clip_group.attrs.get("video_path", "")
                    if video_path not in self.split_videos:
                        continue

                clip_ids.append(clip_id)

        self.clip_ids = clip_ids
        print(f"PretrainingDataset ({split}): {len(self.clip_ids)} clips")

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]

        with h5py.File(self.dataset_path, "r") as hf:
            clip_group = hf["clips"][clip_id]

            frames = clip_group["frames"][()]  # (48, 224, 224, 3) uint8
            heights = clip_group["heights"][()]  # (64,) float32
            occlusion = clip_group["occlusion"][()]  # (64,) float32

        # Sample random offset ∈ [0, 16]
        offset = random.randint(0, 16)

        # Extract input frames (16 frames)
        input_frames = frames[offset : offset + 16]  # (16, 224, 224, 3)

        # Extract target labels (32 frames)
        target_heights = heights[offset + 16 : offset + 48]  # (32,)
        target_occlusion = occlusion[offset + 16 : offset + 48]  # (32,)

        # Augmentation (if train)
        if self.augment:
            input_frames = self._augment_frames(input_frames)

        # Convert to torch tensors
        # input_frames: (16, 224, 224, 3) uint8 → (16, 3, 224, 224) float32 normalized
        input_frames = torch.from_numpy(input_frames).float()
        input_frames = input_frames.permute(0, 3, 1, 2)  # (16, 3, 224, 224)
        input_frames = input_frames / 255.0  # Normalize to [0, 1]
        input_frames = (input_frames - torch.tensor(self.MEAN).view(1, 3, 1, 1)) / \
                       torch.tensor(self.STD).view(1, 3, 1, 1)

        target_heights = torch.from_numpy(target_heights).float()  # (32,)
        target_occlusion = torch.from_numpy(target_occlusion).float()  # (32,)

        return {
            "frames": input_frames,  # (16, 3, 224, 224)
            "heights": target_heights,  # (32,)
            "occlusion": target_occlusion,  # (32,)
        }

    def _augment_frames(self, frames):
        """Apply light augmentations to frame stack."""
        # Horizontal flip
        if random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()  # Flip width dimension

        # Vertical flip
        if random.random() < 0.5:
            frames = np.flip(frames, axis=1).copy()  # Flip height dimension

        return frames
