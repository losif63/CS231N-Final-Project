#!/bin/bash

python src/train.py --backbone x3d --wandb-project gd-difficulty --wandb-run-name x3d-v1 --out-dir runs/x3d
python src/train.py --backbone r2plus1d --wandb-project gd-difficulty --wandb-run-name r2plus1d-v1 --out-dir runs/r2plus1d
python src/train.py --backbone slowfast --wandb-project gd-difficulty --wandb-run-name slowfast-v1 --out-dir runs/slowfast
python src/train.py --backbone i3d --wandb-project gd-difficulty --wandb-run-name i3d-v1 --out-dir runs/i3d