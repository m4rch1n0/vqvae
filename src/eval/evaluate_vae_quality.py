
"""
Evaluate VAE reconstruction quality
"""
import argparse
import yaml
import torch
from pathlib import Path
from src.models.vae import VAE
from src.eval.metrics import psnr, ssim_simple


def load_config(config_path: str) -> dict:
    """Load unified YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_vae_model(checkpoint_path: str, vae_config: dict, device: torch.device) -> VAE:
    """Load trained VAE model from checkpoint."""
    model = VAE(
        in_channels=int(vae_config.get("in_channels", 1)),
        enc_channels=vae_config.get("enc_channels", [64, 128, 256]),
        dec_channels=vae_config.get("dec_channels", [256, 128, 64]),
        latent_dim=int(vae_config.get("latent_dim", 128)),
        recon_loss=str(vae_config.get("recon_loss", "mse")),
        output_image_size=int(vae_config.get("output_image_size", 28)),
        norm_type=str(vae_config.get("norm_type", "batch")),
        mse_use_sigmoid=bool(vae_config.get("mse_use_sigmoid", True)),
    ).to(device).eval()

    # Load checkpoint with robust loading
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "unknown")
    return model, epoch


def evaluate_latent_reconstructions(
    model: VAE, latents: torch.Tensor, data_config: dict, device: torch.device,
    max_samples: int = 1000, batch_size: int = 512
) -> torch.Tensor:
    """Decode latents and return reconstructed images."""
    dataset_name = str(data_config.get("name", "")).upper()

    # Apply proper denormalization/activation based on dataset and loss
    recon_loss = str(model.recon_loss).lower()
    mse_use_sigmoid = bool(model.mse_use_sigmoid)
    apply_sigmoid = (recon_loss == "bce") or mse_use_sigmoid

    reconstructions = []
    n_samples = min(len(latents), max_samples)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            z_batch = latents[i : i + batch_size].to(device)
            x_recon = model.decoder(z_batch)

            # Apply appropriate post-processing
            if dataset_name == "CIFAR10" and not apply_sigmoid:
                # CIFAR-10 with MSE: unnormalize
                mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x_recon.device).view(1, 3, 1, 1)
                std = torch.tensor([0.2470, 0.2430, 0.2610], device=x_recon.device).view(1, 3, 1, 1)
                x_recon = (x_recon * std + mean).clamp(0, 1)
            else:
                # Fashion-MNIST, MNIST: sigmoid if BCE, else clamp
                x_recon = torch.sigmoid(x_recon) if apply_sigmoid else x_recon.clamp(0, 1)

            reconstructions.append(x_recon.cpu())

    return torch.cat(reconstructions, 0)


def assess_quality(psnr_value: float, ssim_value: float) -> tuple[str, bool]:
    """Assess VAE quality and provide recommendation."""
    if psnr_value > 20:
        quality = "excellent"
        proceed = True
    elif psnr_value > 15:
        quality = "good"  
        proceed = True
    elif psnr_value > 10:
        quality = "acceptable"
        proceed = True
    else:
        quality = "poor"
        proceed = False

    return quality, proceed


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE reconstruction quality")
    parser.add_argument("--experiment", required=True, help="Experiment directory")
    parser.add_argument("--config", help="Config file path (auto-detected if not provided)")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Auto-detect config path if not provided
    if args.config:
        config_path = args.config
    else:
        config_path = f"{args.experiment}/../../configs/sandbox-fashion/euclidean/vae.yaml"
        if not Path(config_path).exists():
            # Try alternative paths
            config_path = f"configs/sandbox-fashion/euclidean/vae.yaml"

    # Load configuration
    try:
        config = load_config(config_path)
        vae_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return 1

    dataset_name = data_cfg.get("name", "Unknown")

    # Load model
    checkpoint_path = f"{args.experiment}/vae/checkpoints/best.pt"
    try:
        model, epoch = load_vae_model(checkpoint_path, vae_cfg, device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 1

    # Load latents
    latents_dir = Path(f"{args.experiment}/vae/latents_val")
    try:
        mu_val = torch.load(latents_dir / "mu.pt", map_location="cpu").float()
        z_val = torch.load(latents_dir / "z.pt", map_location="cpu").float() 
    except Exception as e:
        print(f"Error loading latents: {e}")
        return 1

    # Evaluate reconstruction quality
    x_from_z = evaluate_latent_reconstructions(model, z_val, data_cfg, device, args.max_samples, args.batch_size)
    x_from_mu = evaluate_latent_reconstructions(model, mu_val, data_cfg, device, args.max_samples, args.batch_size)

    # Compare z vs mu reconstructions
    z_mu_psnr = psnr(x_from_z, x_from_mu)
    z_mu_ssim = ssim_simple(x_from_z, x_from_mu)

    print(f"PSNR: {z_mu_psnr:.2f} dB, SSIM: {z_mu_ssim:.4f}")

    # Quality assessment
    quality, proceed = assess_quality(z_mu_psnr, z_mu_ssim)

    print(f"Quality: {quality.upper()}")
    print(f'Recommendation: {"PROCEED" if proceed else "RETRAIN"}')

    # Save results
    results = {
        "dataset": dataset_name,
        "checkpoint_epoch": epoch,
        "psnr_db": float(z_mu_psnr),
        "ssim": float(z_mu_ssim),
        "quality_rating": quality,
        "recommendation": "proceed" if proceed else "retrain",
        "samples_evaluated": len(x_from_mu),
    }

    output_file = Path(args.experiment) / "vae" / "vae_quality_assessment.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    return 0 if proceed else 1


if __name__ == "__main__":
    exit(main())
