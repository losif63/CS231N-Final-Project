"""Shared per-backbone preprocessing helpers.

These are intentionally batched: input is always `(B, C, T, H, W)`. Each
helper is a no-op fast-path when the shape already matches the target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def subsample_temporal(x: torch.Tensor, n: int) -> torch.Tensor:
    """Uniformly subsample T → n along dim=2 of `(B, C, T, H, W)`."""
    T = x.size(2)
    if T == n:
        return x
    idx = torch.linspace(0, T - 1, n, device=x.device).round().long()
    return x.index_select(dim=2, index=idx)


def resize_spatial(x: torch.Tensor, hw: int) -> torch.Tensor:
    """Per-frame bilinear resize to `hw x hw` on `(B, C, T, H, W)`."""
    B, C, T, H, W = x.shape
    if H == hw and W == hw:
        return x
    # (B, C, T, H, W) → (B*T, C, H, W) → resize → reshape back.
    x = x.transpose(1, 2).reshape(B * T, C, H, W)
    x = F.interpolate(x, size=(hw, hw), mode="bilinear", align_corners=False)
    return x.reshape(B, T, C, hw, hw).transpose(1, 2)
