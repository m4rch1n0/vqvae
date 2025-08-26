#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=./

# VQ-VAE Experiments Orchestration Script
# Run complete experimental pipeline

echo "Starting VQ-VAE experimental pipeline..."

# Function to check if previous step completed successfully
check_completion() {
    if [ $? -ne 0 ]; then
        echo "Error in: $1"
        exit 1
    fi
    echo "Completed: $1"
}

# Train VAE model (always respects dataset from configs/data.yaml)
echo "Training VAE model..."
./train_vae.sh
check_completion "VAE training"

# Run Riemannian sanity checks
echo "Running Riemannian sanity checks..."
python3 experiments/geo/riemann_sanity_check.py
check_completion "Riemannian sanity checks"

# Run Riemannian graph effects analysis
echo "Analyzing Riemannian graph effects..."
python3 experiments/geo/run_riemann_experiments.py
check_completion "Riemannian graph effects analysis"

echo "All experiments completed successfully!"
echo "Results available in experiments/geo/"

# Geodesic K-medoids analysis (post-hoc quantization study)
echo "Running Geodesic K-medoids analysis..."
python3 demos/kmedoids_geodesic_analysis.py
check_completion "Geodesic K-medoids analysis"

# Codebook comparison demo (Euclidean vs Geodesic quantization)
echo "Running Codebook Comparison Demo..."
python3 demos/codebook_comparison.py
check_completion "Codebook comparison demo"

echo "Demo outputs available in demo_outputs/"
