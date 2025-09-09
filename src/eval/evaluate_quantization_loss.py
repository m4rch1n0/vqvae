
"""
Evaluate quantization loss
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision import transforms

from src.utils.checkpoint_utils import load_vae_from_checkpoint
from src.eval.metrics import psnr, ssim_simple


def load_dataset_samples(dataset_name: str, num_samples: int = 1000) -> torch.Tensor:
    """Load real dataset samples for comparison"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
    ])
    
    if dataset_name.lower() == "fashionmnist":
        dataset = FashionMNIST(root="./data", train=False, download=True, transform=transform)
    elif dataset_name.lower() == "cifar10":
        dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Sample from validation set
    indices = torch.randperm(len(dataset))[:num_samples]
    samples = [dataset[i][0] for i in indices]
    return torch.stack(samples)


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
    parser = argparse.ArgumentParser(description="Evaluate quantization loss")
    parser.add_argument("--experiment", required=True, help="Experiment directory")
    parser.add_argument("--dataset", default="fashionmnist", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = Path(args.experiment)

    # Load VAE model
    checkpoint_path = experiment_dir / "vae" / "checkpoints" / "best.pt"
    vae, vae_config = load_vae_from_checkpoint(str(checkpoint_path), device=device)
    
    if vae is None:
        print("Error: Failed to load VAE model")
        return 1

    # Load latents and codebook
    try:
        z_val = torch.load(experiment_dir / "vae" / "latents_val" / "z.pt", map_location="cpu", weights_only=False).float()
        codebook = torch.load(experiment_dir / "codebook" / "codebook.pt", map_location="cpu", weights_only=False)
        z_medoid = codebook["z_medoid"].float()
    except Exception as e:
        print(f"Error loading latents/codebook: {e}")
        return 1

    # Load or compute codes
    codes_path = experiment_dir / "codebook" / "codes.npy"
    if codes_path.exists():
        codes_np = np.load(codes_path)
        if codes_np.ndim > 1:
            codes_np = codes_np.reshape(-1)
        codes = torch.from_numpy(codes_np).long()
    else:
        codes = nearest_medoid_assign(z_val, z_medoid, batch_size=8192)

    # Load real dataset samples
    try:
        x_real = load_dataset_samples(args.dataset, args.max_samples)
        # Ensure channel consistency with VAE reconstructions
        if vae_config['in_channels'] == 1 and x_real.size(1) == 3:
            x_real = x_real.mean(dim=1, keepdim=True)  # Convert RGB to grayscale
        elif vae_config['in_channels'] == 3 and x_real.size(1) == 1:
            x_real = x_real.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Build quantized latents
    zq_val = z_medoid[codes]

    # Determine post-processing
    recon_loss = vae_config.get('recon_loss', 'mse').lower()
    mse_use_sigmoid = vae_config.get('mse_use_sigmoid', True)
    apply_sigmoid = (recon_loss == "bce") or mse_use_sigmoid

    # Compute reconstructions
    x_cont_list, x_quant_list = [], []
    n_samples = min(len(z_val), args.max_samples)
    
    with torch.no_grad():
        for i in range(0, n_samples, args.batch_size):
            end_idx = min(i + args.batch_size, n_samples)
            z_batch = z_val[i:end_idx].to(device)
            zq_batch = zq_val[i:end_idx].to(device)
            
            x_cont = vae.decoder(z_batch)
            x_quant = vae.decoder(zq_batch)
            
            # Apply appropriate post-processing
            x_cont = unnormalize_images(x_cont, args.dataset, apply_sigmoid)
            x_quant = unnormalize_images(x_quant, args.dataset, apply_sigmoid)
            
            x_cont_list.append(x_cont.cpu())
            x_quant_list.append(x_quant.cpu())

    x_continuous = torch.cat(x_cont_list, 0)[:n_samples]
    x_quantized = torch.cat(x_quant_list, 0)[:n_samples]
    x_real = x_real[:n_samples].to(device)
    x_real = unnormalize_images(x_real, args.dataset, apply_sigmoid).cpu()

    # Compute metrics
    metrics = {
        "dataset": args.dataset,
        "samples_evaluated": n_samples,
        "codebook_size": int(z_medoid.shape[0]),
        "psnr_real_vs_continuous": float(f"{psnr(x_real, x_continuous):.6f}"),
        "psnr_real_vs_quantized": float(f"{psnr(x_real, x_quantized):.6f}"),
        "psnr_continuous_vs_quantized": float(f"{psnr(x_continuous, x_quantized):.6f}"),
        "ssim_real_vs_continuous": float(f"{ssim_simple(x_real, x_continuous):.6f}"),
        "ssim_real_vs_quantized": float(f"{ssim_simple(x_real, x_quantized):.6f}"),
        "ssim_continuous_vs_quantized": float(f"{ssim_simple(x_continuous, x_quantized):.6f}"),
    }

    # Results summary
    print(f"Real vs Continuous: PSNR {metrics['psnr_real_vs_continuous']:.2f} dB, SSIM {metrics['ssim_real_vs_continuous']:.4f}")
    print(f"Real vs Quantized: PSNR {metrics['psnr_real_vs_quantized']:.2f} dB, SSIM {metrics['ssim_real_vs_quantized']:.4f}")
    print(f"Continuous vs Quantized: PSNR {metrics['psnr_continuous_vs_quantized']:.2f} dB, SSIM {metrics['ssim_continuous_vs_quantized']:.4f}")

    # Assessment
    cont_quant_psnr = metrics['psnr_continuous_vs_quantized']
    if cont_quant_psnr > 25:
        assessment = "EXCELLENT"
    elif cont_quant_psnr > 20:
        assessment = "GOOD"
    elif cont_quant_psnr > 15:
        assessment = "MODERATE"
    else:
        assessment = "HIGH"

    print(f"Quantization loss: {assessment}")

    # Save results
    output_dir = experiment_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "quantization_analysis.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {output_dir}/quantization_analysis.json")
    return 0


if __name__ == "__main__":
    exit(main())
