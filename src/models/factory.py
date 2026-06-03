"""Build a backbone by short name. Keeps training scripts backbone-agnostic.

Imports are lazy because we deliberately maintain two conda envs:

  - `cs231n`     : pytorchvideo + torchvision CNN backbones (x3d, r2plus1d,
                   slowfast, i3d). pytorchvideo pins torch to 2.1.x.
  - `cs231n-vit` : transformers >= 4.57 (+ peft) for Qwen3-VL. Needs torch
                   >= 2.2 and so does *not* have pytorchvideo installed.

A non-lazy `from .x3d import X3DBackbone` at module load would crash the VLM
env (no pytorchvideo) and a non-lazy `from .qwen_vl import QwenVLBackbone`
would crash the CNN env (no transformers). Deferring the import until
`build_backbone(name)` is called means each env only tries to import the
backbones it can actually run.
"""

from __future__ import annotations

import torch.nn as nn

BACKBONES = ("x3d", "r2plus1d", "slowfast", "i3d", "qwen_vl")


def build_backbone(name: str, *, pretrained: bool = True, **qwen_kwargs) -> nn.Module:
    """Build a backbone by short name.

    `qwen_kwargs` are forwarded only to `QwenVLBackbone` (train_mode,
    lora_r/alpha/dropout, lora_target_modules, dtype, model_id). Passing
    them to a CNN backbone raises — they're meaningless there and silently
    ignoring would mask typos.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}. Options: {sorted(BACKBONES)}")

    if name == "qwen_vl":
        from .qwen_vl import QwenVLBackbone
        return QwenVLBackbone(pretrained=pretrained, **qwen_kwargs)

    if qwen_kwargs:
        raise TypeError(
            f"Backbone {name!r} does not accept Qwen-VL kwargs "
            f"{sorted(qwen_kwargs)}; only `qwen_vl` does."
        )

    if name == "x3d":
        from .x3d import X3DBackbone
        return X3DBackbone(pretrained=pretrained)
    if name == "r2plus1d":
        from .r2plus1d import R2Plus1DBackbone
        return R2Plus1DBackbone(pretrained=pretrained)
    if name == "slowfast":
        from .slowfast import SlowFastBackbone
        return SlowFastBackbone(pretrained=pretrained)
    if name == "i3d":
        from .i3d import I3DBackbone
        return I3DBackbone(pretrained=pretrained)
    raise AssertionError("unreachable")
