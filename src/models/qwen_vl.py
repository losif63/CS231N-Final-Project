"""Qwen3-VL-2B vision-encoder backbone (transformers >= 4.57).

Loads `Qwen/Qwen3-VL-2B-Instruct`, discards the LLM, and keeps only the
407M-param ViT (`model.model.visual`, class `Qwen3VLVisionModel`). The ViT
encodes a clip jointly across (T, H, W) using a temporal patch size of 2 and
a spatial patch size of 16, then a 2x2 spatial-merge projector lifts the
1024-dim hidden states to 2048-dim "post-merger" tokens (the same features
Qwen3-VL would feed into its language model). We mean-pool these tokens
within each clip to produce one `(feature_dim=2048)` vector.

Input from the dataset matches every other backbone: `(B, C=3, T, H, W)` in
[0, 1]. The wrapper normalises with the OpenAI/CLIP statistics that Qwen-VL
was trained on, resizes spatially to a square `NATIVE_HW` (default 224, a
multiple of `patch_size * spatial_merge_size = 32`), pads `T` up to a
multiple of `temporal_patch_size=2`, hand-rolls Qwen's patchifier
(matching `Qwen3VLVisionPatchEmbed`'s expected memory layout), and calls
`visual(flat_patches, grid_thw)`.

Three training modes:

  - "frozen" (default) : every backbone param has `requires_grad=False`;
    only the downstream MLP head trains. Fastest, safest with our 4,400
    videos; useless if the pretrained features can't separate difficulty.
  - "lora"             : freeze the base ViT, then add LoRA adapters on the
    fused `qkv` Linear inside every `Qwen3VLVisionAttention` block via
    `peft.get_peft_model`. About 790K trainable params at r=8 (vs 407M
    frozen). Cheap, well-regularised middle ground between frozen and full.
  - "full"             : every base param trains. Expect overfit on 4,400
    videos and very heavy memory; included for completeness.

Requires a different conda env (`cs231n-vit`) than the CNN backbones, since
transformers >= 4.57 needs torch >= 2.2, but pytorchvideo (the CNN-backbone
loader) is pinned to torch 2.1.x.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn

from ._video_ops import resize_spatial


TrainMode = Literal["frozen", "lora", "full"]


class QwenVLBackbone(nn.Module):
    MODEL_ID: str = "Qwen/Qwen3-VL-2B-Instruct"
    NATIVE_HW: int = 224  # must be a multiple of patch_size * spatial_merge_size = 32
    # OpenAI / CLIP normalization — what every Qwen-VL variant uses.
    MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        pretrained: bool = True,
        *,
        train_mode: TrainMode = "frozen",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Tuple[str, ...] = ("qkv",),
        dtype: torch.dtype = torch.bfloat16,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        if train_mode not in ("frozen", "lora", "full"):
            raise ValueError(
                f"train_mode must be one of frozen/lora/full; got {train_mode!r}"
            )

        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        except ImportError as e:
            raise ImportError(
                "QwenVLBackbone needs `transformers>=4.57`. Activate the "
                "`cs231n-vit` conda env (created separately from `cs231n`)."
            ) from e

        mid = model_id or self.MODEL_ID

        if pretrained:
            full = Qwen3VLForConditionalGeneration.from_pretrained(mid, dtype=dtype)
            visual = full.model.visual
            del full  # drop the LLM before we move anything to CUDA
        else:
            cfg = AutoConfig.from_pretrained(mid)
            visual = Qwen3VLVisionModel._from_config(cfg.vision_config).to(dtype)

        # Cache patchifier constants before we (possibly) wrap with PEFT,
        # since PEFT shimming can shadow attribute access.
        vc = visual.config
        self.patch_size: int = int(vc.patch_size)
        self.temporal_patch_size: int = int(vc.temporal_patch_size)
        self.spatial_merge_size: int = int(vc.spatial_merge_size)
        self.feature_dim: int = int(vc.out_hidden_size)  # 2048

        if self.NATIVE_HW % (self.patch_size * self.spatial_merge_size) != 0:
            raise RuntimeError(
                f"NATIVE_HW={self.NATIVE_HW} must be a multiple of "
                f"patch_size*spatial_merge_size="
                f"{self.patch_size * self.spatial_merge_size}."
            )

        # Mode wiring.
        self.train_mode: TrainMode = train_mode
        if train_mode == "frozen":
            for p in visual.parameters():
                p.requires_grad = False
            self._base_frozen = True
        elif train_mode == "full":
            self._base_frozen = False
        else:  # lora
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise ImportError(
                    "train_mode='lora' needs `pip install peft`."
                ) from e
            # Freeze the base; PEFT will mark only adapter params as trainable.
            for p in visual.parameters():
                p.requires_grad = False
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            )
            visual = get_peft_model(visual, lora_cfg)
            self._base_frozen = True
            self.lora_r = lora_r
            self.lora_alpha = lora_alpha
            self.lora_dropout = lora_dropout

        self.visual = visual
        self.dtype = dtype

        self.register_buffer(
            "_mean", torch.tensor(self.MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(self.STD).view(1, 3, 1, 1, 1), persistent=False
        )

        n_total = sum(p.numel() for p in self.visual.parameters())
        n_trainable = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)
        print(
            f"  QwenVL backbone ({train_mode}): "
            f"total={n_total/1e6:.1f}M  trainable={n_trainable/1e6:.3f}M"
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # In frozen mode, also keep BN/dropout in the visual encoder in eval.
        # In LoRA mode, keep the base ViT in eval but let PEFT's LoraLayer
        # dropouts run normally — PEFT already only puts LoraLayers in train
        # mode under .train(True), so the standard call does the right thing.
        if self.train_mode == "frozen":
            self.visual.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W); got shape {tuple(x.shape)}")
        B, C, T, _, _ = x.shape
        if C != 3:
            raise ValueError(f"QwenVL expects 3-channel RGB; got C={C}.")

        # 1) Spatial resize to NATIVE_HW x NATIVE_HW.
        x = resize_spatial(x, self.NATIVE_HW)

        # 2) Pad T up to a multiple of temporal_patch_size by repeating the last
        #    frame.
        if T % self.temporal_patch_size != 0:
            pad = self.temporal_patch_size - (T % self.temporal_patch_size)
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
            T = x.size(2)

        # 3) CLIP-style normalize, cast to model dtype.
        x = (x - self._mean) / self._std
        x = x.to(self.dtype)

        # 4) Patchify to match Qwen3VLVisionPatchEmbed's expected layout.
        H = x.size(3)
        W = x.size(4)
        T_p = T // self.temporal_patch_size
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        x = x.view(
            B, C, T_p, self.temporal_patch_size,
            H_p, self.patch_size, W_p, self.patch_size,
        )
        # (B, T_p, H_p, W_p, C, t_ps, p_s, p_s)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(
            B * T_p * H_p * W_p,
            C * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        # 5) grid_thw is the per-clip patch grid.
        grid_thw = torch.tensor(
            [[T_p, H_p, W_p]] * B, device=x.device, dtype=torch.long
        )

        # 6) Forward. pooler_output is post-merger
        #    (B * T_p * H_p/2 * W_p/2, 2048) — same tokens Qwen3-VL feeds into
        #    its LLM. Mean-pool per clip for one feature vector per clip.
        #    In "frozen" mode wrap in no_grad to skip activation storage; in
        #    "lora"/"full" let autograd record so the trainable params get
        #    gradients.
        if self.train_mode == "frozen":
            with torch.no_grad():
                out = self.visual(x, grid_thw=grid_thw)
        else:
            out = self.visual(x, grid_thw=grid_thw)
        post = out.pooler_output

        tokens_per_clip = T_p * (H_p // self.spatial_merge_size) * (W_p // self.spatial_merge_size)
        feat = post.view(B, tokens_per_clip, self.feature_dim).mean(dim=1)
        # Cast back to float32 for the downstream head (AMP / fp32).
        return feat.float()
