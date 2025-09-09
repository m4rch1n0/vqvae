#!/bin/bash

# Complete CIFAR-10 Pipeline Runner
# Runs all approaches with standard configs from configs/cifar10/

set -e  # Exit on any error

echo "=============================================================================="
echo "COMPLETE CIFAR-10 VQ-VAE PIPELINE"
echo "=============================================================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Ensure we're in the scripts directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"

# Function to run commands with timing
run_step() {
    local description="$1"
    local command="$2"
    
    echo ""
    echo "======================================================================"
    echo "STEP: $description"
    echo "COMMAND: $command"
    echo "TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
    
    start_time=$(date +%s)
    
    if eval "$command"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        seconds=$((duration % 60))
        echo "SUCCESS: $description completed!"
        printf "Duration: %02d:%02d:%02d\n" $hours $minutes $seconds
    else
        echo "ERROR: $description failed!"
        exit 1
    fi
}

echo ""
echo "Using standard configs from configs/cifar10/ for all approaches..."

# Step 1: Run Vanilla Euclidean Pipeline (standard configs)
run_step "Vanilla Euclidean Pipeline" "python run_cifar10_vanilla_euclidean_pipeline.py"

# Step 2: Run Vanilla Geodesic Pipeline (standard configs)  
run_step "Vanilla Geodesic Pipeline" "python run_cifar10_vanilla_geodesic_pipeline.py"

# Step 3: Run Spatial Geodesic Pipeline (standard configs)
run_step "Spatial Geodesic Pipeline" "python run_cifar10_spatial_geodesic_pipeline.py"

# Step 4: Run Baseline Pipeline (includes training and evaluation)
run_step "Baseline VQ-VAE Pipeline" "python run_baseline_pipeline.py --skip-comparison"

# Step 5: Run Comprehensive Comparison
run_step "Comprehensive Comparison Analysis" "python compare_all_approaches.py --approaches baseline vanilla_euclidean vanilla_geodesic spatial_geodesic"

echo ""
echo "All pipelines completed using standard configs."

# Final summary
echo ""
echo "=============================================================================="
echo "COMPLETE CIFAR-10 PIPELINE FINISHED!"
echo "=============================================================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results locations:"
echo "  - Vanilla Euclidean: ../experiments/cifar10/vanilla/euclidean/"
echo "  - Vanilla Geodesic:  ../experiments/cifar10/vanilla/geodesic/"
echo "  - Spatial Geodesic:  ../experiments/cifar10/spatial/geodesic/"
echo "  - Baseline VQ-VAE:   ../experiments/cifar10/baseline_vqvae/"
echo "  - Comparison Report: ../experiments/cifar10/comparison/"
echo ""
echo "Key outputs:"
echo "  - Comparison table:     ../experiments/cifar10/comparison/comparison_results.csv"
echo "  - Summary report:       ../experiments/cifar10/comparison/comparison_report.md"
echo "  - Visualizations:       ../experiments/cifar10/comparison/comparison_charts.png"
echo "  - Generated samples:    ../experiments/cifar10/comparison/generated_samples.png"
echo ""
echo "Next steps:"
echo "  1. Review comparison results in the comparison directory"
echo "  2. Analyze the performance differences between approaches"
echo "  3. Check generated samples for visual quality assessment"
echo "=============================================================================="
