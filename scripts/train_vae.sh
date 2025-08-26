#!/usr/bin/env bash
set -euo pipefail

# runs from repo root
cd "$(dirname "$0")/.."

export PYTHONPATH=./

# place outputs under experiments/vae_<dataset>/ and checkpoints accordingly.
# Rely on configs/train.yaml for defaults; allow user overrides via "$@".
python3 src/training/train_vae.py "$@"