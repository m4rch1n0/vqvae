
"""
CIFAR-10 Vanilla VAE Geodesic Pipeline
Complete pipeline from VAE training to evaluation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show real-time progress"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed! (Exit code: {result.returncode})")
        sys.exit(1)
    
    print(f"SUCCESS: {description} completed!")

def main():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 Vanilla VAE Geodesic Pipeline")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE training (use existing model)")
    parser.add_argument("--skip-codebook", action="store_true", help="Skip codebook building (use existing codebook)")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip transformer training (use existing model)")
    parser.add_argument("--skip-generation", action="store_true", help="Skip sample generation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    
    args = parser.parse_args()

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Changed to project root: {project_root}")

    base_dir = "experiments/cifar10/vanilla/geodesic"
    
    # Create directory structure
    os.makedirs(f"{base_dir}/vae", exist_ok=True)
    os.makedirs(f"{base_dir}/codebook", exist_ok=True)
    os.makedirs(f"{base_dir}/transformer", exist_ok=True)
    os.makedirs(f"{base_dir}/evaluation", exist_ok=True)

    
    conda_cmd = ""

    # Step 1: Train VAE
    if not args.skip_vae:
        run_command(
            f"{conda_cmd}python src/scripts/train_vanilla_vae.py --config configs/cifar10/vanilla/geodesic/vae.yaml",
            "Training Vanilla VAE for CIFAR-10"
        )

    # Step 2: Build Geodesic Codebook
    if not args.skip_codebook:
        run_command(
            f"{conda_cmd}python src/training/build_riemannian_codebook_legacy.py --config configs/cifar10/vanilla/geodesic/codebook.yaml",
            "Building Geodesic Codebook"
        )

    # Step 3: Train Transformer
    if not args.skip_transformer:
        run_command(
            f"{conda_cmd}python src/scripts/train_transformer.py --config configs/cifar10/vanilla/geodesic/transformer.yaml",
            "Training Transformer on Geodesic Codes"
        )

    # Step 4: Generate Samples
    if not args.skip_generation:
        run_command(
            f"{conda_cmd}python src/scripts/generate_samples.py --config configs/cifar10/vanilla/geodesic/generate.yaml",
            "Generating Samples"
        )

    # Step 5: Evaluate Results
    if not args.skip_evaluation:
        run_command(
            f"{conda_cmd}python src/eval/evaluate_model.py --config configs/cifar10/vanilla/geodesic/evaluate.yaml",
            "Evaluating Generated Samples"
        )

    print(f"\n{'='*60}")
    print("CIFAR-10 VANILLA VAE GEODESIC PIPELINE COMPLETED!")
    print(f"Results saved to: {base_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

