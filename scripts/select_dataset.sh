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

# No longer creating duplicate dataset-specific files at root.
# Use presets/ directly in scripts to avoid redundancy.

# Default quantize config points to MNIST preset unless dataset provides one
if [ -f "presets/$CHOICE/quantize.yaml" ]; then
  cp -f "presets/$CHOICE/quantize.yaml" quantize.yaml
else
  cp -f "presets/mnist/quantize.yaml" quantize.yaml
fi

# Write active dataset marker
echo "$CHOICE" > active_dataset.txt

echo "Active configs now copied from presets/$CHOICE"



