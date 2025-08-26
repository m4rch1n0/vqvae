#!/usr/bin/env bash
set -euo pipefail

# runs from repo root
cd "$(dirname "$0")/.."

export PYTHONPATH=./

# place outputs under experiments/vae_<dataset>/ and checkpoints accordingly.
# Read all hyperparameters from configs/train.yaml; allow optional CLI overrides via "$@".
python3 src/training/train_vae.py "$@"