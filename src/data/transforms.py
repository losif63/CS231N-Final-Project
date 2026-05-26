"""Raw-clip transforms applied by the dataset (NOT backbone normalization).

The dataset emits "raw" clips: aspect-preserving resize to a canonical
(H=short_side, W=canon_width) per the spec, float in [0,1]. Per-backbone
subsampling / native-resolution resize / mean-std normalization is owned by
each backbone wrapper, not here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RawClipTransform:
    """(K, T, H, W, C) uint8  →  (K, T, C, H, W) float in [0,1].

    - Aspect-preserving resize so the short side becomes `short_side`.
    - Width forced to `canon_width` by horizontal center-crop (if wider) or
      symmetric zero-pad (if narrower). No vertical crop; no aspect distortion
      for sources at or near 16:9 (most YouTube gameplay).
    - Optional mild color jitter; same jitter values applied to every frame
      within a clip, fresh values per clip.
    """

    short_side: int = 224
    canon_width: int = 398
    color_jitter: float = 0.2  # 0.0 disables

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype != torch.uint8:
            frames = frames.to(torch.uint8)
        K, T, H, W, C = frames.shape

        # (K*T, C, H, W) float in [0,1].
        x = frames.permute(0, 1, 4, 2, 3).reshape(K * T, C, H, W).float().div_(255.0)

        # Aspect-preserving resize: short side → short_side.
        if H < W:
            new_h = self.short_side
            new_w = max(1, int(round(W * self.short_side / H)))
        else:
            new_w = self.short_side
            new_h = max(1, int(round(H * self.short_side / W)))
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Canonical width: center-crop or symmetric zero-pad horizontally.
        x = self._fix_width(x, self.canon_width)

        if self.color_jitter > 0.0:
            x = x.view(K, T, C, new_h, self.canon_width)
            for k in range(K):
                x[k] = self._color_jitter(x[k], self.color_jitter)
        else:
            x = x.view(K, T, C, new_h, self.canon_width)

        return x.contiguous()

    @staticmethod
    def _fix_width(x: torch.Tensor, target_w: int) -> torch.Tensor:
        cur_w = x.shape[-1]
        if cur_w == target_w:
            return x
        if cur_w > target_w:
            left = (cur_w - target_w) // 2
            return x[..., left : left + target_w]
        pad = target_w - cur_w
        left = pad // 2
        right = pad - left
        # F.pad order for last two dims: (W_left, W_right, H_top, H_bottom)
        return F.pad(x, (left, right, 0, 0), mode="constant", value=0.0)

    @staticmethod
    def _color_jitter(clip: torch.Tensor, amount: float) -> torch.Tensor:
        # clip: (T, C, H, W). Brightness, contrast, saturation each in [1-a, 1+a].
        b = 1.0 + random.uniform(-amount, amount)
        c = 1.0 + random.uniform(-amount, amount)
        s = 1.0 + random.uniform(-amount, amount)

        # Brightness: scale.
        clip = clip * b

        # Contrast: blend toward clip-wide mean luma.
        mean = clip.mean(dim=(-1, -2, -3), keepdim=True)
        clip = (clip - mean) * c + mean

        # Saturation: blend toward grayscale.
        gray = (
            0.2989 * clip[:, 0:1]
            + 0.5870 * clip[:, 1:2]
            + 0.1140 * clip[:, 2:3]
        ).expand_as(clip)
        clip = (clip - gray) * s + gray

        return clip.clamp_(0.0, 1.0)
