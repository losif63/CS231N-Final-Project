"""I3D-R50 backbone wrapper (PyTorchVideo, Kinetics-400 pretrained).

Input: `(B, C=3, T, H, W)` raw clip in [0, 1] from the dataset. Uses the raw
32-frame clip directly (NATIVE_T = 32), resizes spatially to 224x224,
normalizes with I3D's Kinetics-400 stats, then runs the net with the
classifier head stripped. Returns `(B, feature_dim)` per clip.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ._video_ops import resize_spatial, subsample_temporal


class I3DBackbone(nn.Module):
    NATIVE_T: int = 32
    NATIVE_HW: int = 224
    MEAN: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    STD: Tuple[float, float, float] = (0.225, 0.225, 0.225)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.net = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "i3d_r50",
            pretrained=pretrained,
        )

        head = self.net.blocks[-1]
        if not (hasattr(head, "proj") and isinstance(head.proj, nn.Linear)):
            raise RuntimeError(
                f"Unexpected I3D head structure (proj={type(getattr(head, 'proj', None))}); "
                "cannot strip classifier."
            )
        head.proj = nn.Identity()
        if getattr(head, "activation", None) is not None:
            head.activation = nn.Identity()

        self.register_buffer(
            "_mean", torch.tensor(self.MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(self.STD).view(1, 3, 1, 1, 1), persistent=False
        )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.NATIVE_T, self.NATIVE_HW, self.NATIVE_HW)
            out = self.net((dummy - self._mean) / self._std)
        if was_training:
            self.train()
        if out.ndim != 2:
            raise RuntimeError(f"I3D stripped head returned ndim={out.ndim}, expected 2.")
        self.feature_dim: int = out.size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W); got shape {tuple(x.shape)}")
        x = subsample_temporal(x, self.NATIVE_T)
        x = resize_spatial(x, self.NATIVE_HW)
        x = (x - self._mean) / self._std
        return self.net(x)
