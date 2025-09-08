#!/usr/bin/env python3
"""
FashionMNIST Vanilla VAE Geodesic Pipeline
Complete pipeline from VAE training to evaluation
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
    parser = argparse.ArgumentParser(description="Run FashionMNIST Vanilla VAE Geodesic Pipeline with Enhanced Evaluation")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE training (use existing model)")
    parser.add_argument("--skip-vae-check", action="store_true", help="Skip VAE quality assessment")
    parser.add_argument("--skip-codebook", action="store_true", help="Skip codebook building (use existing codebook)")
    parser.add_argument("--skip-quantization-analysis", action="store_true", help="Skip quantization loss analysis")
    parser.add_argument("--skip-codebook-health", action="store_true", help="Skip codebook health check")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip transformer training (use existing model)")
    parser.add_argument("--skip-generation", action="store_true", help="Skip sample generation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip final evaluation")
    
    args = parser.parse_args()

    # Set up paths according to new structure
    base_dir = "../experiments/fashionmnist/vanilla/geodesic"
    
    # Create directory structure
    os.makedirs(f"{base_dir}/vae", exist_ok=True)
    os.makedirs(f"{base_dir}/codebook", exist_ok=True)
    os.makedirs(f"{base_dir}/transformer", exist_ok=True)
    os.makedirs(f"{base_dir}/evaluation", exist_ok=True)

    # Use current environment (should already be rocm_env)
    conda_cmd = ""

    # Step 1: Train VAE
    if not args.skip_vae:
        run_command(
            f"{conda_cmd}python ../src/scripts/train_vanilla_vae.py --config ../configs/fashionmnist/vanilla/geodesic/vae.yaml",
            "Training Vanilla VAE for FashionMNIST"
        )

    # Step 1.5: VAE Quality Assessment  
    if not args.skip_vae_check:
        vae_quality_ok = run_command(
            f"{conda_cmd}python ../src/eval/evaluate_vae_quality.py --experiment ../{base_dir}",
            "VAE Quality Assessment",
            critical=False
        )
        if not vae_quality_ok:
            print("WARNING: VAE Quality check failed - check manually if needed")

    # Step 2: Build Geodesic Codebook
    if not args.skip_codebook:
        run_command(
            f"{conda_cmd}python ../src/training/build_riemannian_codebook_legacy.py --config ../configs/fashionmnist/vanilla/geodesic/codebook.yaml",
            "Building Geodesic Codebook"
        )

    # Step 2.5: Quantization Loss Analysis
    if not args.skip_quantization_analysis:
        quant_analysis_ok = run_command(
            f"{conda_cmd}python ../src/eval/evaluate_quantization_loss.py --experiment ../{base_dir} --dataset fashionmnist",
            "Quantization Loss Analysis", 
            critical=False
        )
        if not quant_analysis_ok:
            print("WARNING: Quantization analysis failed - check manually if needed")

    # Step 2.6: Codebook Health Check
    if not args.skip_codebook_health:
        codebook_health_ok = run_command(
            f"{conda_cmd}python ../src/eval/evaluate_codebook_health.py --experiment ../{base_dir} --dataset fashionmnist",
            "Codebook Health Check",
            critical=False
        )
        if not codebook_health_ok:
            print("WARNING: Codebook health check failed - check manually if needed")

    # Step 3: Train Transformer
    if not args.skip_transformer:
        run_command(
            f"{conda_cmd}python ../src/scripts/train_transformer.py --config ../configs/fashionmnist/vanilla/geodesic/transformer.yaml",
            "Training Transformer on Geodesic Codes"
        )

    # Step 4: Generate Samples
    if not args.skip_generation:
        run_command(
            f"{conda_cmd}python ../src/scripts/generate_samples.py --config ../configs/fashionmnist/vanilla/geodesic/generate.yaml",
            "Generating Samples"
        )

    # Step 5: Evaluate Results
    if not args.skip_evaluation:
        run_command(
            f"{conda_cmd}python ../src/eval/evaluate_model.py --config ../configs/fashionmnist/vanilla/geodesic/evaluate.yaml",
            "Evaluating Generated Samples"
        )

    print(f"\n{'='*60}")
    print("FASHIONMNIST VANILLA VAE GEODESIC PIPELINE COMPLETED!")
    print(f"Results saved to: {base_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

