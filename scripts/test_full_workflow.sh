#!/usr/bin/env bash
set -euo pipefail

# End-to-end workflow test for MNIST, CIFAR10, and Fashion-MNIST
# Steps per dataset:
# 1) Select dataset preset (copies configs)
# 2) Train VAE (uses configs/{data,vae,train}.yaml)
# 3) Build codebook (uses dataset-specific quantize config)
# 4) Evaluate quantized reconstructions and save a check image
#
# Usage:
#   bash scripts/test_full_workflow.sh
# Optional:
#   FAST=1 bash scripts/test_full_workflow.sh   # quick smoke test (few epochs)

# Run from repo root
cd "$(dirname "$0")/.."

export PYTHONPATH=./

# Activate conda env if available (non-fatal if conda missing)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null || true)"
  conda activate rocm_env 2>/dev/null || true
fi

FAST=${FAST:-0}

yaml_get() {
  local file="$1"; shift
  local dotted="$1"; shift || true
  python3 - "$file" "$dotted" <<'PY'
import sys, yaml
file, dotted = sys.argv[1], sys.argv[2]
with open(file, 'r') as f:
    cfg = yaml.safe_load(f) or {}
val = cfg
for k in dotted.split('.'):
    val = val[k]
print(val)
PY
}

run_dataset() {
  local dataset="$1"
  local quant_cfg="$2"

  echo "\n=============================="
  echo "Dataset: $dataset"
  echo "=============================="

  bash scripts/select_dataset.sh "$dataset"

  if [ "$FAST" = "1" ]; then
    echo "Training VAE (FAST mode) ..."
    python3 src/training/train_vae.py max_epochs=2 scheduler.t_max=2 || {
      echo "VAE training failed for $dataset"; exit 1; }
  else
    echo "Training VAE ..."
    python3 src/training/train_vae.py || {
      echo "VAE training failed for $dataset"; exit 1; }
  fi

  local ds_slug
  case "$dataset" in
    cifar10) ds_slug="vae_cifar10" ;;
    fashion) ds_slug="vae_fashion" ;;
    mnist)   ds_slug="vae_mnist" ;;
    *) echo "Unknown dataset: $dataset"; exit 1 ;;
  esac

  local ckpt="experiments/${ds_slug}/checkpoints/best.pt"
  local z_train="experiments/${ds_slug}/latents_train/z.pt"
  local z_val="experiments/${ds_slug}/latents_val/z.pt"

  for f in "$ckpt" "$z_train" "$z_val"; do
    if [ ! -f "$f" ]; then
      echo "Missing expected training artifact: $f"; exit 1
    fi
  done
  echo "✓ Training artifacts present for $dataset"

  local out_dir
  out_dir="$(yaml_get "$quant_cfg" out.dir)"
  echo "Building codebook with $quant_cfg → $out_dir"
  python3 src/training/build_codebook.py --config "$quant_cfg" || {
    echo "Codebook build failed for $dataset"; exit 1; }

  for f in "$out_dir/codebook.pt" "$out_dir/codes.npy"; do
    if [ ! -e "$f" ]; then
      echo "Missing codebook artifact: $f"; exit 1
    fi
  done
  echo "✓ Codebook artifacts present for $dataset"

  mkdir -p demo_outputs/workflow_checks
  local eval_out="demo_outputs/workflow_checks/${dataset}_eval_quantized.png"
  echo "Evaluating quantized reconstructions → $eval_out"
  python3 src/eval/eval_quantized.py \
    --checkpoint "$ckpt" \
    --latents "$z_train" \
    --codes "$out_dir/codes.npy" \
    --codebook "$out_dir/codebook.pt" \
    --out "$eval_out" || {
      echo "Quantized evaluation failed for $dataset"; exit 1; }

  if [ ! -f "$eval_out" ]; then
    echo "Missing evaluation image: $eval_out"; exit 1
  fi
  echo "✓ Quantized evaluation image saved: $eval_out"

  # Absolute evaluation vs ground truth (use geodesic val codes if available)
  local codes_val="$out_dir/codes_val.npy"
  if [ -f "$codes_val" ]; then
    echo "Absolute evaluation (geodesic val assignments)"
    python3 src/eval/eval_quantized_abs.py \
      --checkpoint "$ckpt" \
      --latents "experiments/${ds_slug}/latents_val/z.pt" \
      --codebook "$out_dir/codebook.pt" \
      --codes "$codes_val" \
      --out_json "demo_outputs/workflow_checks/${dataset}_abs_metrics.json" \
      --out_png  "demo_outputs/workflow_checks/${dataset}_abs_grid.png" || {
        echo "Absolute evaluation failed for $dataset"; exit 1; }
  else
    echo "Absolute evaluation (fallback Euclidean NN assignments)"
    python3 src/eval/eval_quantized_abs.py \
      --checkpoint "$ckpt" \
      --latents "experiments/${ds_slug}/latents_val/z.pt" \
      --codebook "$out_dir/codebook.pt" \
      --out_json "demo_outputs/workflow_checks/${dataset}_abs_metrics.json" \
      --out_png  "demo_outputs/workflow_checks/${dataset}_abs_grid.png" || {
        echo "Absolute evaluation failed for $dataset"; exit 1; }
  fi
}

# MNIST
run_dataset mnist configs/presets/mnist/quantize.yaml

# CIFAR10 (use preset quantize config)
run_dataset cifar10 configs/presets/cifar10/quantize.yaml

# Fashion-MNIST (use preset K=1024 quantize config)
run_dataset fashion configs/presets/fashion/quantize_k1024.yaml

echo "\nAll datasets completed successfully. Artifacts under experiments/ and demo_outputs/workflow_checks/"


