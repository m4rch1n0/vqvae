
"""
FashionMNIST Spatial VAE Geodesic Pipeline
Complete pipeline from VAE training to evaluation
"""
import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and show real-time progress"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    # Set PYTHONPATH to include project root for subprocess
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = project_root + ':' + env.get('PYTHONPATH', '')
    
    result = subprocess.run(cmd, shell=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed! (Exit code: {result.returncode})")
        sys.exit(1)
    
    print(f"SUCCESS: {description} completed!")

def main():
    parser = argparse.ArgumentParser(description="Run FashionMNIST Spatial VAE Geodesic Pipeline")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE training (use existing model)")
    parser.add_argument("--skip-codebook", action="store_true", help="Skip codebook building (use existing codebook)")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip transformer training (use existing model)")
    parser.add_argument("--skip-generation", action="store_true", help="Skip sample generation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    
    args = parser.parse_args()

    # Change to project root directory and add it to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    base_dir = "experiments/fashionmnist/spatial/geodesic"
    
    # Create directory structure
    os.makedirs(f"{base_dir}/vae", exist_ok=True)
    os.makedirs(f"{base_dir}/codebook", exist_ok=True)
    os.makedirs(f"{base_dir}/transformer", exist_ok=True)
    os.makedirs(f"{base_dir}/evaluation", exist_ok=True)

    # Use current environment (should already be rocm_env)
    conda_cmd = ""

    # Step 1: Train Spatial VAE
    if not args.skip_vae:
        run_command(
            f"{conda_cmd}python src/scripts/train_vae.py --config configs/fashionmnist/spatial/geodesic/vae.yaml",
            "Training Spatial VAE for FashionMNIST"
        )

    # Step 2: Build Geodesic Codebook
    if not args.skip_codebook:
        cmd = f"{conda_cmd}python src/scripts/build_codebook.py " \
              f"--latents_path experiments/fashionmnist/spatial/geodesic/vae/spatial_vae_fashionmnist/latents_train/z.pt " \
              f"--vae_ckpt_path experiments/fashionmnist/spatial/geodesic/vae/spatial_vae_fashionmnist/checkpoints/best.pt " \
              f"--out_dir experiments/fashionmnist/spatial/geodesic/codebook " \
              f"--in_channels 1 --output_image_size 28 --latent_dim 16 " \
              f"--enc_channels 64 128 256 --dec_channels 256 128 64 " \
              f"--recon_loss mse --norm_type batch " \
              f"--mse_use_sigmoid " \
              f"--k 20 --sym union --K 512 --init kpp --seed 42 --batch_size 512"
        run_command(cmd, "Building Geodesic Spatial Codebook")

    # Step 3: Train Transformer
    if not args.skip_transformer:
        run_command(
            f"{conda_cmd}python src/scripts/train_transformer.py --config configs/fashionmnist/spatial/geodesic/transformer.yaml",
            "Training Transformer on Spatial Geodesic Codes"
        )

    # Step 4: Generate Samples
    if not args.skip_generation:
        run_command(
            f"{conda_cmd}python src/scripts/generate_samples.py --config configs/fashionmnist/spatial/geodesic/generate.yaml",
            "Generating Samples"
        )

    # Step 5: Evaluate Results
    if not args.skip_evaluation:
        run_command(
            f"{conda_cmd}python src/eval/evaluate_model.py --config configs/fashionmnist/spatial/geodesic/evaluate.yaml",
            "Evaluating Generated Samples"
        )

    print(f"\n{'='*60}")
    print("FASHIONMNIST SPATIAL VAE GEODESIC PIPELINE COMPLETED!")
    print(f"Results saved to: {base_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
