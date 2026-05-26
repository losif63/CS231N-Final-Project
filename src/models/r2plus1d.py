"""R(2+1)D-18 backbone wrapper (torchvision, Kinetics-400 pretrained).

Input: `(B, C=3, T, H, W)` raw clip in [0, 1] from the dataset. Internally
subsamples to 16 frames, resizes to 112x112 (R(2+1)D's native input), normalizes
with the KINETICS400_V1 weights' stats, then runs the net with the classifier
`fc` stripped. Returns `(B, feature_dim)` per clip (feature_dim = 512).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from ._video_ops import resize_spatial, subsample_temporal


class R2Plus1DBackbone(nn.Module):
    NATIVE_T: int = 16
    NATIVE_HW: int = 112
    MEAN: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645)
    STD: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.net = r2plus1d_18(weights=weights)

        if not isinstance(self.net.fc, nn.Linear):
            raise RuntimeError(
                f"Unexpected r2plus1d_18 head (fc={type(self.net.fc)}); cannot strip classifier."
            )
        self.feature_dim: int = self.net.fc.in_features
        self.net.fc = nn.Identity()

        self.register_buffer(
            "_mean", torch.tensor(self.MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(self.STD).view(1, 3, 1, 1, 1), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W); got shape {tuple(x.shape)}")
        x = subsample_temporal(x, self.NATIVE_T)
        x = resize_spatial(x, self.NATIVE_HW)
        x = (x - self._mean) / self._std
        return self.net(x)
