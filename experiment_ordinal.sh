#!/bin/bash

python src/train.py --backbone slowfast --head-kind ordinal --wandb-project gd-difficulty --wandb-run-name slowfast-ordinal-v1 --out-dir runs/slowfast_ordinal
python src/train.py --backbone x3d --head-kind ordinal --wandb-project gd-difficulty --wandb-run-name x3d-ordinal-v1 --out-dir runs/x3d_ordinal
python src/train.py --backbone r2plus1d --head-kind ordinal --wandb-project gd-difficulty --wandb-run-name r2plus1d-ordinal-v1 --out-dir runs/r2plus1d_ordinal
python src/train.py --backbone i3d --head-kind ordinal --wandb-project gd-difficulty --wandb-run-name i3d-ordinal-v1 --out-dir runs/i3d_ordinal
