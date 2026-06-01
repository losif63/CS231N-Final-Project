#!/bin/bash
# Qwen3-VL-2B vision encoder runs. MUST be launched in the `cs231n-vit`
# conda env (the CNN-backbone env is on torch 2.1, which can't load
# transformers >= 4.57). Activate first:
#
#   conda activate cs231n-vit
#   ./experiment_qwen.sh
#
# 4 cells: (head_kind in {softmax, ordinal}) x (backbone-train in {frozen, lora}).
# `full` fine-tune is skipped — 407M params on ~4,400 videos overfits hard and
# the runtime is prohibitive. Add a row if you want it.
#
# Defaults are tighter than the CNN sweeps because Qwen-VL is much heavier per
# clip: batch=1, clips-train=4, clips-eval=8, AMP off (base is already bf16).

set -e

COMMON="--backbone qwen_vl --batch-size 2 --grad-accum 2 --num-workers 8 --clips-train 8 --clips-eval 24 --no-amp --wandb-project gd-difficulty"

# LoRA adapters on attention qkv (r=8 default).
python src/train.py $COMMON --head-kind softmax --backbone-train lora \
    --wandb-run-name qwen-lora-softmax-v1 --out-dir runs/qwen_lora_softmax

python src/train.py $COMMON --head-kind ordinal --backbone-train lora \
    --wandb-run-name qwen-lora-ordinal-v1 --out-dir runs/qwen_lora_ordinal

# # Frozen backbone (head-only).
# python src/train.py $COMMON --head-kind softmax --backbone-train frozen \
#     --wandb-run-name qwen-frozen-softmax-v1 --out-dir runs/qwen_frozen_softmax

# python src/train.py $COMMON --head-kind ordinal --backbone-train frozen \
#     --wandb-run-name qwen-frozen-ordinal-v1 --out-dir runs/qwen_frozen_ordinal