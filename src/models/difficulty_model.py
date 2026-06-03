"""DifficultyModel: backbone + K-clip aggregation + MLP head.

Input from the dataloader: `(B, K, T, C, H, W)`. K may differ between train
and eval modes but is constant within a batch. The model:

  1. Reshapes K clips into the backbone batch dim: `(B*K, C, T, H, W)`.
  2. Runs the backbone to get per-clip features `(B*K, D)`.
  3. Aggregates over K with the configured strategy → `(B, D)`.
  4. Runs the MLP head → `(B, num_outputs)` logits.

Two head kinds are supported:

  - "softmax" (default): `num_classes` logits, trained with cross-entropy.
  - "ordinal" (CORN; Shi, Cao, Raschka 2023): `num_classes - 1` logits. Logit
    k models `P(y > k | y > k-1)` via sigmoid; the conditional masking in the
    loss makes the chained marginals `P(y > k) = prod_{j<=k} sigmoid(logit_j)`
    rank-consistent at inference.

Use `compute_loss(logits, labels, head_kind, num_classes)` for training and
`decode_logits(logits, head_kind)` to recover `(pred_class_0idx, ev_in_stars)`
for metrics; both keep the metric math identical across head kinds.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Aggregation = Literal["mean", "max", "topk_mean"]
HeadKind = Literal["softmax", "ordinal"]


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
        head_kind: HeadKind = "softmax",
    ) -> None:
        super().__init__()
        if aggregation not in ("mean", "max", "topk_mean"):
            raise ValueError(f"Unknown aggregation: {aggregation!r}")
        if head_kind not in ("softmax", "ordinal"):
            raise ValueError(f"Unknown head_kind: {head_kind!r}")
        if not hasattr(backbone, "feature_dim"):
            raise AttributeError("Backbone must expose `feature_dim`.")

        self.backbone = backbone
        self.aggregation: Aggregation = aggregation
        self.top_k = top_k
        self.head_kind: HeadKind = head_kind
        self.num_classes = num_classes

        num_outputs = num_classes if head_kind == "softmax" else num_classes - 1

        D = int(backbone.feature_dim)
        self.head = nn.Sequential(
            nn.Linear(D, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_outputs),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 6:
            raise ValueError(
                f"Expected clips of shape (B, K, T, C, H, W); got {tuple(clips.shape)}"
            )
        B, K, T, C, H, W = clips.shape

        x = clips.permute(0, 1, 3, 2, 4, 5).reshape(B * K, C, T, H, W)
        feats = self.backbone(x)              # (B*K, D)
        if feats.ndim != 2 or feats.size(0) != B * K:
            raise RuntimeError(
                f"Backbone returned shape {tuple(feats.shape)}; expected (B*K, D)."
            )
        feats = feats.reshape(B, K, -1)       # (B, K, D)
        agg = self._aggregate(feats)          # (B, D)
        return self.head(agg)                 # (B, num_outputs)

    def _aggregate(self, feats: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return feats.mean(dim=1)
        if self.aggregation == "max":
            return feats.max(dim=1).values
        k = min(self.top_k, feats.size(1))
        vals, _ = feats.topk(k, dim=1)
        return vals.mean(dim=1)


def corn_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """CORN loss (Shi, Cao, Raschka 2023).

    `logits` is `(B, K-1)` where logit k models P(y > k | y > k-1). For
    threshold k, only samples with `y >= k` are eligible (they're the ones
    that have "made it past" threshold k-1). The conditional masking is what
    distinguishes CORN from the naive Frank-and-Hall cumulative-BCE and is
    what guarantees rank-consistent chained marginals at inference.
    """
    K = num_classes
    device = logits.device
    ks = torch.arange(K - 1, device=device).unsqueeze(0)          # (1, K-1)
    labels_exp = labels.unsqueeze(1)                              # (B, 1)
    mask = (labels_exp >= ks).float()                             # eligible: y >= k
    target = (labels_exp >= ks + 1).float()                       # target=1 iff y >= k+1
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * mask).sum() / mask.sum().clamp(min=1.0)


def decode_logits(
    logits: torch.Tensor, head_kind: HeadKind
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map raw logits to `(pred, ev)` in a head-kind-agnostic way.

    Returns
    -------
    pred : (B,) long tensor in `0..num_classes-1` (argmax-style class).
    ev   : (B,) float tensor on the *stars* scale `1..num_classes`, so MAE
           against `true_label + 1` is directly comparable across head kinds.
    """
    if head_kind == "softmax":
        K = logits.size(-1)
        ranks = torch.arange(1, K + 1, device=logits.device, dtype=torch.float32)
        probs = F.softmax(logits.float(), dim=-1)
        ev = (probs * ranks).sum(dim=-1)
        pred = logits.argmax(dim=-1)
        return pred, ev
    if head_kind == "ordinal":
        # logits: (B, K-1). sigmoid(logit_k) = P(y > k | y > k-1).
        p_cond = torch.sigmoid(logits.float())
        p_marginal = torch.cumprod(p_cond, dim=-1)                # P(y > k)
        pred = (p_marginal > 0.5).sum(dim=-1).long()              # 0..K-1
        ev = p_marginal.sum(dim=-1) + 1.0                          # stars: 1..K
        return pred, ev
    raise ValueError(f"Unknown head_kind: {head_kind!r}")


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    head_kind: HeadKind,
    num_classes: int,
) -> torch.Tensor:
    if head_kind == "softmax":
        return F.cross_entropy(logits, labels)
    if head_kind == "ordinal":
        return corn_loss(logits, labels, num_classes)
    raise ValueError(f"Unknown head_kind: {head_kind!r}")
