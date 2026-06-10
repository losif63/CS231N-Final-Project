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

Written using Claude Code
"""

from __future__ import annotations

import torch.nn as nn

BACKBONES = ("x3d", "r2plus1d", "slowfast", "i3d", "qwen_vl")


def build_backbone(
    name: str,
    *,
    pretrained: bool = True,
    checkpoint_path: str | None = None,
    **qwen_kwargs
) -> nn.Module:
    """Build a backbone by short name.

    Args:
        name: Backbone name (x3d, r2plus1d, slowfast, i3d, qwen_vl)
        pretrained: Whether to load internet-weight pretrained weights
        checkpoint_path: Optional path to custom checkpoint weights (overrides pretrained)
        **qwen_kwargs: Additional kwargs for qwen_vl backbone

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

    # Build backbone
    backbone = None
    if name == "x3d":
        from .x3d import X3DBackbone
        backbone = X3DBackbone(pretrained=pretrained)
    elif name == "r2plus1d":
        from .r2plus1d import R2Plus1DBackbone
        backbone = R2Plus1DBackbone(pretrained=pretrained)
    elif name == "slowfast":
        from .slowfast import SlowFastBackbone
        backbone = SlowFastBackbone(pretrained=pretrained)
    elif name == "i3d":
        from .i3d import I3DBackbone
        backbone = I3DBackbone(pretrained=pretrained)
    else:
        raise AssertionError("unreachable")

    # Load custom checkpoint if provided
    if checkpoint_path:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        print(f"\n[DEBUG] Checkpoint keys: {list(checkpoint.keys())[:10]}")  # Print first 10 keys

        # Handle full training checkpoint (with metadata) vs bare state_dict
        if isinstance(checkpoint, dict):
            # Check for model state keys (could be "model_state" or "model")
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
                print(f"[DEBUG] Extracted 'model_state' with keys: {list(state_dict.keys())[:10]}")
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
                print(f"[DEBUG] Extracted 'model' with keys: {list(state_dict.keys())[:10]}")
            else:
                # Might be a bare state_dict, check if it has model keys
                if any(k.startswith(("net.", "backbone.", "encoder.")) for k in checkpoint.keys()):
                    state_dict = checkpoint
                    print(f"[DEBUG] Using checkpoint as bare state_dict with keys: {list(state_dict.keys())[:10]}")
                else:
                    # Has metadata keys but no model state - try model_state as fallback
                    state_dict = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
                    print(f"[DEBUG] Using fallback state_dict with keys: {list(state_dict.keys())[:10]}")
        else:
            state_dict = checkpoint

        print(f"[DEBUG] Loading state_dict with {len(state_dict) if isinstance(state_dict, dict) else 'unknown'} keys")

        # If it's a full model checkpoint, extract just the backbone part
        if state_dict and isinstance(state_dict, dict) and any(k.startswith("backbone.") for k in state_dict.keys()):
            # Model has backbone prefix, extract it
            backbone_state = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
            print(f"[DEBUG] Extracted backbone keys: {list(backbone_state.keys())[:10]}")
            backbone.load_state_dict(backbone_state)
        else:
            # Already bare backbone state_dict
            print(f"[DEBUG] Loading as bare state_dict. Keys sample: {list(state_dict.keys())[:5] if isinstance(state_dict, dict) else 'N/A'}")
            backbone.load_state_dict(state_dict)

    return backbone
