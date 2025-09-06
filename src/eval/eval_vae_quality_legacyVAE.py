#!/usr/bin/env python3
"""
Evaluate VAE continuous reconstruction quality.
Assesses whether a trained VAE produces high-quality latent representations
by comparing z vs mu reconstructions and analyzing reconstruction metrics.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from src.models.vae import VAE
from src.eval.metrics import psnr, ssim_simple


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_vae_model(checkpoint_path: str, vae_config: dict, device: torch.device) -> VAE:
    """Load trained VAE model from checkpoint."""
    model = (
        VAE(
            in_channels=int(vae_config.get("in_channels", 1)),
            enc_channels=vae_config.get("enc_channels", [64, 128, 256]),
            dec_channels=vae_config.get("dec_channels", [256, 128, 64]),
            latent_dim=int(vae_config.get("latent_dim", 128)),
            recon_loss=str(vae_config.get("recon_loss", "mse")),
            output_image_size=int(vae_config.get("output_image_size", 28)),
            norm_type=str(vae_config.get("norm_type", "batch")),
            mse_use_sigmoid=bool(vae_config.get("mse_use_sigmoid", False)),
        )
        .to(device)
        .eval()
    )

    # Load checkpoint with robust loading
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "unknown")
    return model, epoch


def evaluate_latent_reconstructions(
    model: VAE,
    latents: torch.Tensor,
    data_config: dict,
    device: torch.device,
    max_samples: int = 2048,
    batch_size: int = 512,
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
                mean = torch.tensor(
                    [0.4914, 0.4822, 0.4465], device=x_recon.device
                ).view(1, 3, 1, 1)
                std = torch.tensor(
                    [0.2470, 0.2430, 0.2610], device=x_recon.device
                ).view(1, 3, 1, 1)
                x_recon = (x_recon * std + mean).clamp(0, 1)
            else:
                # Fashion-MNIST, MNIST: sigmoid if BCE, else clamp
                x_recon = (
                    torch.sigmoid(x_recon) if apply_sigmoid else x_recon.clamp(0, 1)
                )

            reconstructions.append(x_recon.cpu())

    return torch.cat(reconstructions, 0)


def assess_quality(psnr_value: float, ssim_value: float) -> tuple[str, str, bool]:
    """Assess VAE quality and provide recommendation."""
    if psnr_value > 20:
        quality = "excellent"
        status = "✓ EXCELLENT: PSNR > 20 dB indicates high-quality latent space"
        proceed = True
    elif psnr_value > 15:
        quality = "good"
        status = "✓ GOOD: PSNR > 15 dB indicates decent latent space"
        proceed = True
    elif psnr_value > 10:
        quality = "acceptable"
        status = "~ ACCEPTABLE: PSNR > 10 dB, but could be improved"
        proceed = True
    else:
        quality = "poor"
        status = "✗ POOR: PSNR < 10 dB, retraining recommended"
        proceed = False

    return quality, status, proceed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VAE continuous reconstruction quality"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint")
    parser.add_argument(
        "--latents_dir", required=True, help="Directory containing mu.pt and z.pt"
    )
    parser.add_argument(
        "--vae_config", default="configs/vae.yaml", help="VAE config file"
    )
    parser.add_argument(
        "--data_config", default="configs/data.yaml", help="Data config file"
    )
    parser.add_argument(
        "--max_samples", type=int, default=2048, help="Max samples to evaluate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for inference"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configurations
    vae_cfg = load_config(args.vae_config)
    data_cfg = load_config(args.data_config)
    dataset_name = data_cfg.get("name", "Unknown")

    print(f"=== {dataset_name.upper()} VAE QUALITY ASSESSMENT ===")

    # Load model
    try:
        model, epoch = load_vae_model(args.checkpoint, vae_cfg, device)
        print(f"✓ Loaded checkpoint from epoch {epoch}")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return 1

    # Load latents
    latents_dir = Path(args.latents_dir)
    try:
        mu_val = torch.load(latents_dir / "mu.pt", map_location="cpu").float()
        z_val = torch.load(latents_dir / "z.pt", map_location="cpu").float()
        print(f"✓ Loaded validation latents: mu {mu_val.shape}, z {z_val.shape}")
    except Exception as e:
        print(f"✗ Error loading latents: {e}")
        return 1

    # Evaluate continuous reconstruction quality
    print("\nEvaluating continuous reconstruction quality...")

    x_from_z = evaluate_latent_reconstructions(
        model, z_val, data_cfg, device, args.max_samples, args.batch_size
    )
    x_from_mu = evaluate_latent_reconstructions(
        model, mu_val, data_cfg, device, args.max_samples, args.batch_size
    )

    # Compare z vs mu reconstructions (continuous baseline)
    z_mu_psnr = psnr(x_from_z, x_from_mu)
    z_mu_ssim = ssim_simple(x_from_z, x_from_mu)

    print(f"\n=== RESULTS ===")
    print(
        f"Continuous baseline (z vs mu): PSNR {z_mu_psnr:.2f} dB, SSIM {z_mu_ssim:.4f}"
    )
    print(f"Reconstruction range: [{x_from_mu.min():.3f}, {x_from_mu.max():.3f}]")
    print(f"Evaluated {len(x_from_mu)} samples")

    # Quality assessment
    quality, status, proceed = assess_quality(z_mu_psnr, z_mu_ssim)

    print(f"\n=== QUALITY ASSESSMENT ===")
    print(status)
    print(f"\nQUALITY RATING: {quality.upper()}")
    print(
        f'RECOMMENDATION: {"PROCEED with codebook construction" if proceed else "RETRAIN VAE with more epochs/better hyperparameters"}'
    )

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

    output_file = latents_dir.parent / "vae_quality_assessment.json"
    import json

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")

    return 0 if proceed else 1


if __name__ == "__main__":
    exit(main())
