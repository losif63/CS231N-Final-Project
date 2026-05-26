"""Build a backbone by short name. Keeps training scripts backbone-agnostic."""

from __future__ import annotations

import torch.nn as nn

from .i3d import I3DBackbone
from .r2plus1d import R2Plus1DBackbone
from .slowfast import SlowFastBackbone
from .x3d import X3DBackbone

BACKBONES = {
    "x3d": X3DBackbone,
    "r2plus1d": R2Plus1DBackbone,
    "slowfast": SlowFastBackbone,
    "i3d": I3DBackbone,
}


def build_backbone(name: str, *, pretrained: bool = True) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {name!r}. Options: {sorted(BACKBONES)}"
        )
    return BACKBONES[name](pretrained=pretrained)
