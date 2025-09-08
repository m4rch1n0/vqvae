#!/usr/bin/env python3
"""
Fast Testing Pipeline for FashionMNIST Vanilla VAE Euclidean 
Ultra-fast configuration for testing evaluation scripts integration
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description, critical=True):
    """Run a command with optional critical handling"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, text=True)
    
    if result.returncode != 0:
        if critical:
            print(f"ERROR: {description} failed! (Exit code: {result.returncode})")
            sys.exit(1)
        else:
            print(f"WARNING: {description} failed! (Exit code: {result.returncode}) - continuing...")
            return False
    
    print(f"SUCCESS: {description} completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fast Test Pipeline with Evaluation Integration")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE training")
    parser.add_argument("--skip-vae-check", action="store_true", help="Skip VAE quality check")
    parser.add_argument("--skip-codebook", action="store_true", help="Skip codebook building")
    parser.add_argument("--skip-quantization-analysis", action="store_true", help="Skip quantization analysis")
    parser.add_argument("--skip-codebook-health", action="store_true", help="Skip codebook health check")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip transformer training")
    parser.add_argument("--skip-generation", action="store_true", help="Skip sample generation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip final evaluation")
    
    args = parser.parse_args()

    # Set up sandbox paths
    base_dir = "experiments/sandbox-fashion/euclidean"
    
    # Create directory structure
    os.makedirs(f"{base_dir}/vae", exist_ok=True)
    os.makedirs(f"{base_dir}/codebook", exist_ok=True)
    os.makedirs(f"{base_dir}/transformer", exist_ok=True)
    os.makedirs(f"{base_dir}/evaluation", exist_ok=True)

    print(f"STARTING FAST TEST PIPELINE")
    print(f"Base directory: {base_dir}")
    print(f"Configuration: Ultra-fast (15 epochs VAE, 15 epochs Transformer)")
    print(f"Purpose: Test evaluation scripts integration")

    # Step 1: Train VAE (Fast: 15 epochs, small model)
    if not args.skip_vae:
        run_command(
            f"python src/scripts/train_vanilla_vae.py --config configs/sandbox-fashion/euclidean/vae.yaml",
            "Training Fast Vanilla VAE"
        )

    # Step 1.5: VAE Quality Check  
    if not args.skip_vae_check:
        vae_quality_ok = run_command(
            f"python src/eval/evaluate_vae_quality.py --experiment {base_dir}",
            "VAE Quality Assessment",
            critical=False  # Don't stop if fails
        )
        if not vae_quality_ok:
            print("WARNING: VAE Quality check failed - check manually if needed")

    # Step 2: Build Codebook (Fast: K=128)
    if not args.skip_codebook:
        run_command(
            f"python src/training/build_codebook_legacy.py --config configs/sandbox-fashion/euclidean/codebook.yaml",
            "Building Fast Euclidean Codebook"
        )

    # Step 2.5: Quantization Loss Analysis
    if not args.skip_quantization_analysis:
        quant_analysis_ok = run_command(
            f"python src/eval/evaluate_quantization_loss.py --experiment {base_dir} --dataset fashionmnist",
            "Quantization Loss Analysis", 
            critical=False
        )
        if not quant_analysis_ok:
            print("WARNING: Quantization analysis failed - check manually if needed")

    # Step 2.6: Codebook Health Check
    if not args.skip_codebook_health:
        codebook_health_ok = run_command(
            f"python src/eval/evaluate_codebook_health.py --experiment {base_dir} --dataset fashionmnist",
            "Codebook Health Check",
            critical=False
        )
        if not codebook_health_ok:
            print("WARNING: Codebook health check failed - check manually if needed")

    # Step 3: Train Transformer (Fast: 15 epochs, small model)
    if not args.skip_transformer:
        run_command(
            f"python src/scripts/train_transformer.py --config configs/sandbox-fashion/euclidean/transformer.yaml",
            "Training Fast Transformer"
        )

    # Step 4: Generate Samples (Fast: 2 per class)
    if not args.skip_generation:
        run_command(
            f"python src/scripts/generate_samples.py --config configs/sandbox-fashion/euclidean/generate.yaml",
            "Generating Fast Samples"
        )

    # Step 5: Evaluate Results (Current pipeline evaluation)
    if not args.skip_evaluation:
        run_command(
            f"python src/eval/evaluate_model.py --config configs/sandbox-fashion/euclidean/evaluate.yaml",
            "Final Evaluation"
        )

    print(f"\n{'='*60}")
    print("FAST TEST PIPELINE COMPLETED!")
    print(f"Results saved to: {base_dir}/")
    print(f"Check evaluation outputs:")
    print(f"   - VAE Quality: {base_dir}/vae/vae_quality_assessment.json")
    print(f"   - Quantization Analysis: {base_dir}/evaluation/quantization_analysis.json") 
    print(f"   - Codebook Health: {base_dir}/evaluation/codebook_health.json")
    print(f"   - Generated Samples: {base_dir}/evaluation/generated_samples.png")
    print(f"   - Comparison Grid: {base_dir}/evaluation/comparison_grid.png")
    print(f"   - Final Metrics: {base_dir}/evaluation/metrics.yaml")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
