#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/select_dataset.sh <mnist|fashion|cifar10>

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CFG_DIR="$ROOT_DIR/configs"
PRESETS_DIR="$CFG_DIR/presets"

if [ $# -ne 1 ]; then
  echo "Usage: $0 <mnist|fashion|cifar10>"
  exit 1
fi

CHOICE="$1"
case "$CHOICE" in
  mnist|fashion|cifar10) ;;
  *) echo "Invalid dataset: $CHOICE"; exit 1;;
esac

# Ensure preset exists
if [ ! -d "$PRESETS_DIR/$CHOICE" ]; then
  echo "Missing preset directory: $PRESETS_DIR/$CHOICE"
  exit 1
fi

echo "Switching active dataset to: $CHOICE"

cd "$CFG_DIR"

# Copy from preset into root-level config files
cp -f "presets/$CHOICE/data.yaml" data.yaml
cp -f "presets/$CHOICE/vae.yaml" vae.yaml
cp -f "presets/$CHOICE/train.yaml" train.yaml

# Keep dataset-specific convenience links for backward compatibility
cp -f "presets/fashion/data.yaml" data_fashion.yaml
cp -f "presets/fashion/vae.yaml" vae_fashion.yaml
cp -f "presets/fashion/train.yaml" train_fashion.yaml

cp -f "presets/cifar10/quantize.yaml" quantize_cifar10.yaml || true
cp -f "presets/cifar10/quantize_k1024.yaml" quantize_cifar10_k1024.yaml || true
cp -f "presets/fashion/quantize_k1024.yaml" quantize_fashion_k1024.yaml || true

# Default quantize config points to MNIST preset unless dataset provides one
if [ -f "presets/$CHOICE/quantize.yaml" ]; then
  cp -f "presets/$CHOICE/quantize.yaml" quantize.yaml
else
  cp -f "presets/mnist/quantize.yaml" quantize.yaml
fi

# Write active dataset marker
echo "$CHOICE" > active_dataset.txt

echo "Active configs now copied from presets/$CHOICE"



