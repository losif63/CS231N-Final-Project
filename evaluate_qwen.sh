#!/bin/bash
# Evaluate the two Qwen-VL LoRA runs on val + test.
# Must be launched in the `cs231n-vit` conda env:
#
#   conda activate cs231n-vit
#   ./evaluate_qwen.sh
#
# Writes <ckpt-dir>/eval_<split>/{metrics.json, confusion_matrix.png,
# confusion_matrix.npy} per (ckpt, split) combination — so 4 result folders
# total. The eval script reads backbone/head_kind/train_mode/LoRA hyperparams
# from each checkpoint's saved args, so no need to pass them on the CLI.

set -e

for ckpt_dir in runs/qwen_lora_softmax runs/qwen_lora_ordinal; do
    for split in val test; do
        echo "=== $ckpt_dir on $split ==="
        python src/eval.py \
            --ckpt "$ckpt_dir/best.pt" \
            --split "$split" \
            --clips-eval 24 \
            --num-workers 8
    done
done
