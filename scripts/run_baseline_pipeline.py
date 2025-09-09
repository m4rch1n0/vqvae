
"""
Run Baseline VQ-VAE Pipeline
Train baseline model and run evaluation for comparison with other approaches
"""

import os
import sys
import subprocess
import argparse


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
    parser = argparse.ArgumentParser(description="Run Baseline VQ-VAE Pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (use existing model)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip comparison with other approaches")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    
    args = parser.parse_args()

    # Setup paths - make them absolute to avoid path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    baseline_dir = os.path.join(project_root, "baseline VQVAE", "vqvae_cifar10_clean")
    output_dir = os.path.join(project_root, "experiments", "cifar10", "baseline_vqvae")
    
    # Create output directory structure
    os.makedirs(f"{output_dir}/evaluation", exist_ok=True)
    
    print(f"=== BASELINE VQ-VAE PIPELINE ===")
    print(f"Baseline directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")

    # Step 1: Train Baseline VQ-VAE
    if not args.skip_training:
        # Change to baseline directory for training
        original_dir = os.getcwd()
        os.chdir(baseline_dir)
        
        try:
            train_cmd = "python train.py --config config.yaml"
            if args.epochs is not None:
                train_cmd += f" --epochs {args.epochs}"
            if args.batch_size is not None:
                train_cmd += f" --batch_size {args.batch_size}"
            
            run_command(train_cmd, "Training Baseline VQ-VAE")
        finally:
            # Return to original directory
            os.chdir(original_dir)

    # Step 2: Evaluate Baseline Model
    if not args.skip_evaluation:
        eval_cmd = f"python evaluate_baseline_simple.py " \
                  f"--baseline_dir '{baseline_dir}' " \
                  f"--checkpoint 'outputs/checkpoints/ckpt_best.pt' " \
                  f"--out_dir '{output_dir}/evaluation'"
        
        run_command(eval_cmd, "Evaluating Baseline VQ-VAE")

    # Step 3: Run Comparison with Other Approaches
    if not args.skip_comparison:
        comparison_dir = os.path.join(project_root, "experiments", "cifar10", "comparison")
        comparison_cmd = f"python compare_all_approaches.py " \
                        f"--out_dir '{comparison_dir}' " \
                        f"--approaches baseline vanilla_euclidean vanilla_geodesic spatial_geodesic"
        
        run_command(comparison_cmd, "Comparing All Approaches", critical=False)

    print(f"\n{'='*60}")
    print("BASELINE VQ-VAE PIPELINE COMPLETED!")
    print(f"Results saved to: {output_dir}/")
    print(f"Comparison results: {comparison_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
