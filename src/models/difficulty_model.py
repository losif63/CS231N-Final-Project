"""DifficultyModel: backbone + K-clip aggregation + MLP head → 10 logits.

Input from the dataloader: `(B, K, T, C, H, W)`. K may differ between train
and eval modes but is constant within a batch. The model:

  1. Reshapes K clips into the backbone batch dim: `(B*K, C, T, H, W)`.
  2. Runs the backbone to get per-clip features `(B*K, D)`.
  3. Aggregates over K with the configured strategy → `(B, D)`.
  4. Runs the MLP head → `(B, 10)` logits.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


Aggregation = Literal["mean", "max", "topk_mean"]


class DifficultyModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        aggregation: Aggregation = "max",
        top_k: int = 3,
        head_hidden: int = 256,
        dropout: float = 0.3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if aggregation not in ("mean", "max", "topk_mean"):
            raise ValueError(f"Unknown aggregation: {aggregation!r}")
        if not hasattr(backbone, "feature_dim"):
            raise AttributeError("Backbone must expose `feature_dim`.")

        self.backbone = backbone
        self.aggregation: Aggregation = aggregation
        self.top_k = top_k

        D = int(backbone.feature_dim)
        self.head = nn.Sequential(
            nn.Linear(D, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 6:
            raise ValueError(
                f"Expected clips of shape (B, K, T, C, H, W); got {tuple(clips.shape)}"
            )
        B, K, T, C, H, W = clips.shape

        # (B, K, T, C, H, W) → (B*K, C, T, H, W) for the backbone.
        x = clips.permute(0, 1, 3, 2, 4, 5).reshape(B * K, C, T, H, W)
        feats = self.backbone(x)              # (B*K, D)
        if feats.ndim != 2 or feats.size(0) != B * K:
            raise RuntimeError(
                f"Backbone returned shape {tuple(feats.shape)}; expected (B*K, D)."
            )
        feats = feats.reshape(B, K, -1)       # (B, K, D)
        agg = self._aggregate(feats)          # (B, D)
        return self.head(agg)                  # (B, num_classes)

    def _aggregate(self, feats: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return feats.mean(dim=1)
        if self.aggregation == "max":
            return feats.max(dim=1).values
        # topk_mean: per-feature top-k along K, then mean.
        k = min(self.top_k, feats.size(1))
        vals, _ = feats.topk(k, dim=1)
        return vals.mean(dim=1)
