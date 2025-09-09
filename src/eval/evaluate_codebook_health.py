
"""
Evaluate codebook health
"""
import argparse
from pathlib import Path
import torch

from src.utils.checkpoint_utils import load_vae_from_checkpoint
from src.eval.metrics import psnr, ssim_simple, codebook_stats


def nearest_medoid_assign(z: torch.Tensor, z_medoid: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Assign latent vectors to nearest medoids"""
    codes_list = []
    z_medoid_t = z_medoid.t().contiguous()
    b2 = (z_medoid ** 2).sum(dim=1).view(1, -1)
    
    for i in range(0, z.size(0), batch_size):
        zi = z[i:i+batch_size]
        a2 = (zi ** 2).sum(dim=1, keepdim=True)
        ab = zi @ z_medoid_t
        d2 = a2 + b2 - 2.0 * ab
        codes_list.append(d2.argmin(dim=1).cpu())
    
    return torch.cat(codes_list, 0).long()


def unnormalize_images(x: torch.Tensor, dataset_name: str, apply_sigmoid: bool) -> torch.Tensor:
    """Apply appropriate denormalization based on dataset and loss type"""
    if dataset_name.upper() == "CIFAR10" and not apply_sigmoid:
        # CIFAR-10 with MSE: unnormalize using dataset statistics
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2430, 0.2610], device=x.device).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)
    else:
        # FashionMNIST/MNIST: sigmoid if BCE, else clamp
        return torch.sigmoid(x) if apply_sigmoid else x.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate codebook health")
    parser.add_argument("--experiment", required=True, help="Experiment directory")
    parser.add_argument("--dataset", default="fashionmnist", help="Dataset name") 
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--n_vis", type=int, default=32, help="Number of samples for visualization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = Path(args.experiment)

    # Load VAE model
    checkpoint_path = experiment_dir / "vae" / "checkpoints" / "best.pt"
    vae, vae_config = load_vae_from_checkpoint(str(checkpoint_path), device=device)
    
    if vae is None:
        print("Error: Failed to load VAE model")
        return 1

    # Load data
    try:
        z_val = torch.load(experiment_dir / "vae" / "latents_val" / "z.pt", map_location="cpu", weights_only=False).float()
        codebook = torch.load(experiment_dir / "codebook" / "codebook.pt", map_location="cpu", weights_only=False)
        z_medoid = codebook["z_medoid"].float()
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Compute validation codes
    try:
        codes = nearest_medoid_assign(z_val, z_medoid, batch_size=8192)
    except Exception as e:
        print(f"Error computing codes: {e}")
        return 1

    # Build quantized latents
    zq_val = z_medoid[codes]

    # Determine post-processing  
    recon_loss = vae_config.get('recon_loss', 'mse').lower()
    mse_use_sigmoid = vae_config.get('mse_use_sigmoid', True) 
    apply_sigmoid = (recon_loss == "bce") or mse_use_sigmoid

    # Compute reconstructions
    x_cont_list, x_quant_list = [], []
    
    with torch.no_grad():
        for i in range(0, len(z_val), args.batch_size):
            z_batch = z_val[i:i+args.batch_size].to(device)
            zq_batch = zq_val[i:i+args.batch_size].to(device)
            
            x_cont = vae.decoder(z_batch)
            x_quant = vae.decoder(zq_batch)
            
            x_cont = unnormalize_images(x_cont, args.dataset, apply_sigmoid)
            x_quant = unnormalize_images(x_quant, args.dataset, apply_sigmoid)
            
            x_cont_list.append(x_cont.cpu())
            x_quant_list.append(x_quant.cpu())

    x_continuous = torch.cat(x_cont_list, 0)
    x_quantized = torch.cat(x_quant_list, 0)

    # Compute metrics
    cont_quant_psnr = psnr(x_continuous, x_quantized)
    cont_quant_ssim = ssim_simple(x_continuous, x_quantized)
    cb_stats = codebook_stats(codes, K=z_medoid.shape[0])

    # Results summary
    usage_percent = 100 * cb_stats['used'] / z_medoid.shape[0]
    print(f"PSNR: {cont_quant_psnr:.2f} dB, SSIM: {cont_quant_ssim:.4f}")
    print(f"Entropy: {cb_stats['entropy']:.3f}, Usage: {usage_percent:.1f}%")

    # Health assessment
    if cb_stats['entropy'] > 4.5 and usage_percent > 80:
        health = "EXCELLENT"
    elif cb_stats['entropy'] > 3.5 and usage_percent > 60:
        health = "GOOD"
    elif cb_stats['entropy'] > 2.5 and usage_percent > 40:
        health = "MODERATE"
    else:
        health = "POOR"

    print(f"Health: {health}")

    # Save results
    output_dir = experiment_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "dataset": args.dataset,
        "samples_evaluated": len(x_continuous),
        "codebook_size": int(z_medoid.shape[0]),
        "psnr_continuous_vs_quantized": float(f"{cont_quant_psnr:.6f}"),
        "ssim_continuous_vs_quantized": float(f"{cont_quant_ssim:.6f}"),
        "entropy": float(f"{cb_stats['entropy']:.6f}"),
        "used_codes": int(cb_stats['used']),
        "dead_codes": int(cb_stats['dead_codes']),
        "usage_percent": float(f"{usage_percent:.2f}"),
        "health_assessment": health
    }
    
    import json
    with open(output_dir / "codebook_health.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}/codebook_health.json")
    return 0


if __name__ == "__main__":
    exit(main())
