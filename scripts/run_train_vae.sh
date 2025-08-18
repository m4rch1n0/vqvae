#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=./
python3 src/training/train_vae.py \
  seed=42 device=auto max_epochs=20 lr=1e-3 amp=true \
  ckpt_dir=./experiments/vae_mnist/checkpoints \
  out_dir=./experiments/vae_mnist