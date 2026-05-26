"""SlowFast R50 backbone wrapper (PyTorchVideo, Kinetics-400 pretrained).

The PyTorchVideo SlowFast model takes a `[slow, fast]` list, not a single
tensor — we construct both pathways from one (B, C, T, H, W) clip:

  - fast: T_FAST = 32 frames at 224x224.
  - slow: T_SLOW = 8 frames at 224x224, subsampled from the fast pathway
          (alpha=4). This matches `pytorchvideo.transforms.PackPathway` but
          works batched.

Normalization stats and head-strip are the same pattern as the other
PyTorchVideo backbones (X3D, I3D).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ._video_ops import resize_spatial, subsample_temporal


class SlowFastBackbone(nn.Module):
    NATIVE_T_FAST: int = 32
    NATIVE_T_SLOW: int = 8
    ALPHA: int = 4
    NATIVE_HW: int = 224
    MEAN: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    STD: Tuple[float, float, float] = (0.225, 0.225, 0.225)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        assert self.NATIVE_T_FAST == self.NATIVE_T_SLOW * self.ALPHA, (
            "NATIVE_T_FAST must equal NATIVE_T_SLOW * ALPHA"
        )
        self.net = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=pretrained,
        )

        head = self.net.blocks[-1]
        if not (hasattr(head, "proj") and isinstance(head.proj, nn.Linear)):
            raise RuntimeError(
                f"Unexpected SlowFast head structure (proj={type(getattr(head, 'proj', None))}); "
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
            slow_d = torch.zeros(1, 3, self.NATIVE_T_SLOW, self.NATIVE_HW, self.NATIVE_HW)
            fast_d = torch.zeros(1, 3, self.NATIVE_T_FAST, self.NATIVE_HW, self.NATIVE_HW)
            out = self.net([
                (slow_d - self._mean) / self._std,
                (fast_d - self._mean) / self._std,
            ])
        if was_training:
            self.train()
        if out.ndim != 2:
            raise RuntimeError(f"SlowFast stripped head returned ndim={out.ndim}, expected 2.")
        self.feature_dim: int = out.size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W); got shape {tuple(x.shape)}")
        # Fast pathway: NATIVE_T_FAST frames at NATIVE_HW.
        fast = subsample_temporal(x, self.NATIVE_T_FAST)
        fast = resize_spatial(fast, self.NATIVE_HW)
        # Slow pathway: subsample fast → NATIVE_T_SLOW (PackPathway-equivalent).
        slow = subsample_temporal(fast, self.NATIVE_T_SLOW)
        # Normalize each pathway with the same Kinetics stats.
        fast = (fast - self._mean) / self._std
        slow = (slow - self._mean) / self._std
        return self.net([slow, fast])
