"""X3D-M backbone wrapper.

Input: `(B, C=3, T, H, W)` raw clip in [0, 1] from the dataset (T defaults to
32, H=224, W=398). Internally subsamples to 16 frames, resizes spatially to
224x224, normalizes with X3D's Kinetics-400 stats, then runs the spatiotemporal
CNN with the classifier head stripped. Returns `(B, feature_dim)` per clip.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class X3DBackbone(nn.Module):
    NATIVE_T: int = 16
    NATIVE_HW: int = 224
    MEAN: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    STD: Tuple[float, float, float] = (0.225, 0.225, 0.225)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # Pulls weights from torch.hub on first call; cached afterward.
        self.net = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=pretrained,
        )

        # Strip the classifier head: keep pre/post conv-norm-act + pooling,
        # drop the per-class Linear and Softmax.
        head = self.net.blocks[-1]
        if not (hasattr(head, "proj") and isinstance(head.proj, nn.Linear)):
            raise RuntimeError(
                f"Unexpected X3D head structure (proj={type(getattr(head, 'proj', None))}); "
                "cannot strip classifier."
            )
        head.proj = nn.Identity()
        if getattr(head, "activation", None) is not None:
            head.activation = nn.Identity()

        # Normalization buffers (move with .to(device)).
        self.register_buffer(
            "_mean", torch.tensor(self.MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(self.STD).view(1, 3, 1, 1, 1), persistent=False
        )

        # Compute feature_dim by dummy forward. Done before any .to(device)
        # so the buffers are still on CPU and dummy is also on CPU.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.NATIVE_T, self.NATIVE_HW, self.NATIVE_HW)
            dummy = (dummy - self._mean) / self._std
            out = self.net(dummy)
        if was_training:
            self.train()
        if out.ndim != 2:
            raise RuntimeError(f"X3D stripped head returned ndim={out.ndim}, expected 2.")
        self.feature_dim: int = out.size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W); got shape {tuple(x.shape)}")
        B, C, T, H, W = x.shape

        # Temporal subsample: 32 → 16 (or whatever T → NATIVE_T).
        if T != self.NATIVE_T:
            idx = torch.linspace(0, T - 1, self.NATIVE_T, device=x.device).round().long()
            x = x.index_select(dim=2, index=idx)

        # Spatial resize to 224x224 (per-frame bilinear). Squeezes the 16:9
        # horizontal extent to 1:1 — accepted aspect distortion in exchange for
        # not discarding side-scroller context via a center crop.
        if H != self.NATIVE_HW or W != self.NATIVE_HW:
            # (B, C, T, H, W) → (B*T, C, H, W) → resize → reshape back.
            x = x.transpose(1, 2).reshape(B * self.NATIVE_T, C, H, W)
            x = F.interpolate(
                x, size=(self.NATIVE_HW, self.NATIVE_HW),
                mode="bilinear", align_corners=False,
            )
            x = x.reshape(B, self.NATIVE_T, C, self.NATIVE_HW, self.NATIVE_HW).transpose(1, 2)

        x = (x - self._mean) / self._std
        return self.net(x)
