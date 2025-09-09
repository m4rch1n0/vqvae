
"""
Simple Baseline Evaluation
Direct evaluation of baseline VQ-VAE without complex path imports
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image
from pathlib import Path

# Simple metrics implementation
@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((x - y) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(torch.tensor(max_val**2) / mse))

@torch.no_grad()
def ssim_simple(x: torch.Tensor, y: torch.Tensor, C1=0.01**2, C2=0.03**2) -> float:
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x = x.var(unbiased=False)
    sigma_y = y.var(unbiased=False)
    sigma_xy = ((x - mu_x)*(y - mu_y)).mean()
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return float((num/den).clamp(0,1))

@torch.no_grad()
def codebook_stats(codes: torch.Tensor, K: int) -> dict:
    if codes.dim() != 1:
        codes = codes.view(-1)
    codes = codes.long()
    valid = codes >= 0
    if valid.any():
        hist = torch.bincount(codes[valid], minlength=K).float()
    else:
        hist = torch.zeros(K, dtype=torch.float32)
    p = (hist / hist.sum().clamp_min(1e-12)).clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum())
    dead = int((hist == 0).sum())
    return {"entropy": entropy, "dead_codes": dead, "used": int((hist>0).sum())}


def main():
    parser = argparse.ArgumentParser(description="Simple Baseline VQ-VAE Evaluation")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_baseline_dir = os.path.join(project_root, "baseline VQVAE", "vqvae_cifar10_clean")
    default_out_dir = os.path.join(project_root, "experiments", "cifar10", "baseline_vqvae", "evaluation")
    
    parser.add_argument("--baseline_dir", default=default_baseline_dir)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/ckpt_best.pt")
    parser.add_argument("--out_dir", default=default_out_dir)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--gen_samples", type=int, default=100)
    
    args = parser.parse_args()
    
    # Change to baseline directory to use their imports
    original_dir = os.getcwd()
    baseline_dir = Path(args.baseline_dir).resolve()
    
    if not baseline_dir.exists():
        print(f"ERROR: Baseline directory not found: {baseline_dir}")
        return 1
    
    try:
        print(f"Changing to baseline directory: {baseline_dir}")
        os.chdir(baseline_dir)
        
        # Import from baseline directory
        sys.path.insert(0, str(baseline_dir))
        from models.vqvae import VQVAE
        from utils import set_seed
        
        print("Successfully imported baseline modules")
        
        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        set_seed(config["seed"])
        
        # Load model
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return 1
            
        print(f"Loading checkpoint: {checkpoint_path}")
        
        model = VQVAE(
            in_channels=config["model"]["in_channels"],
            z_channels=config["model"]["z_channels"], 
            hidden=config["model"]["hidden"],
            n_res_blocks=config["model"]["n_res_blocks"],
            n_codes=config["model"]["n_codes"],
            beta=config["model"]["beta"],
            ema_decay=config["model"]["ema_decay"],
            ema_eps=config["model"]["ema_eps"],
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        
        print(f"Model loaded successfully")
        print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   Codebook size: {config['model']['n_codes']}")
        
        # Get test data
        print("Loading test data...")
        transform = transforms.Compose([
            transforms.Resize(config["data"]["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(config["data"]["normalize_mean"], config["data"]["normalize_std"]),
        ])
        
        test_dataset = datasets.CIFAR10(root=config["data"]["root"], train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Evaluate reconstruction quality
        print(f"Evaluating reconstruction on {args.max_samples} samples...")
        all_originals = []
        all_reconstructions = [] 
        all_codes = []
        
        samples_processed = 0
        with torch.no_grad():
            for x, _ in test_loader:
                if samples_processed >= args.max_samples:
                    break
                    
                x = x.to(device)
                x_rec, loss_vq, idx, z_q, z_e = model(x)
                
                # Unnormalize for evaluation ([-1,1] -> [0,1])
                x_eval = (x + 1.0) / 2.0
                x_rec_eval = (x_rec + 1.0) / 2.0
                
                all_originals.append(x_eval.cpu())
                all_reconstructions.append(x_rec_eval.cpu())
                all_codes.append(idx.view(-1).cpu())
                
                samples_processed += x.size(0)
                if samples_processed % 500 == 0:
                    print(f"   Processed {samples_processed}/{args.max_samples} samples")
        
        # Concatenate results
        originals = torch.cat(all_originals, 0)[:args.max_samples]
        reconstructions = torch.cat(all_reconstructions, 0)[:args.max_samples] 
        codes = torch.cat(all_codes, 0)[:args.max_samples]
        
        # Compute reconstruction metrics
        print("Computing reconstruction metrics...")
        psnr_recon = psnr(originals, reconstructions)
        ssim_recon = ssim_simple(originals, reconstructions)
        cb_stats = codebook_stats(codes, model.quant.n_codes)
        
        print(f"Reconstruction Results:")
        print(f"   PSNR: {psnr_recon:.4f} dB")
        print(f"   SSIM: {ssim_recon:.4f}")
        print(f"   Entropy: {cb_stats['entropy']:.4f}")
        print(f"   Usage: {cb_stats['used']}/{model.quant.n_codes} ({100*cb_stats['used']/model.quant.n_codes:.1f}%)")
        
        # Generate samples
        print(f"Generating {args.gen_samples} samples...")
        generated_images = []
        samples_per_class = args.gen_samples // 10
        
        with torch.no_grad():
            for class_id in range(10):
                for sample_id in range(samples_per_class):
                    # Sample random codes (8x8 spatial grid for CIFAR-10)
                    random_codes = torch.randint(0, model.quant.n_codes, (1, 8, 8), device=device)
                    z_q = model.quant.embed[random_codes].view(1, 8, 8, -1)
                    z_q = z_q.permute(0, 3, 1, 2).contiguous()
                    
                    x_gen = model.dec(z_q)
                    x_gen = (x_gen + 1.0) / 2.0  # Unnormalize
                    generated_images.append(x_gen.cpu())
        
        generated_images = torch.cat(generated_images, 0)
        
        # Get real samples for comparison
        print("Loading real samples for comparison...")
        real_transform = transforms.Compose([transforms.ToTensor()])
        real_dataset = datasets.CIFAR10(root=config["data"]["root"], train=False, download=True, transform=real_transform)
        
        # Get samples by class
        class_samples = {i: [] for i in range(10)}
        for img, label in real_dataset:
            if len(class_samples[label]) < samples_per_class:
                class_samples[label].append(img)
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break
        
        real_images = []
        for class_id in range(10):
            real_images.extend(class_samples[class_id][:samples_per_class])
        real_images = torch.stack(real_images)
        
        # Compute generation metrics
        print("Computing generation metrics...")
        gen_psnr = psnr(real_images, generated_images)
        gen_ssim = ssim_simple(real_images, generated_images)
        
        # Compute LPIPS (if available)
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            
            # Resize and prepare for LPIPS
            gen_lpips = F.interpolate(generated_images, size=(64, 64), mode='bilinear', align_corners=False)
            real_lpips = F.interpolate(real_images, size=(64, 64), mode='bilinear', align_corners=False) 
            gen_lpips = (gen_lpips * 2 - 1).to(device)  # [0,1] -> [-1,1]
            real_lpips = (real_lpips * 2 - 1).to(device)
            
            lpips_scores = []
            batch_size = 32
            for i in range(0, len(gen_lpips), batch_size):
                end_idx = min(i + batch_size, len(gen_lpips))
                lpips_batch = lpips_fn(gen_lpips[i:end_idx], real_lpips[i:end_idx])
                lpips_scores.append(lpips_batch.cpu())
            
            lpips_score = torch.cat(lpips_scores, 0).mean().item()
            print(f"LPIPS: {lpips_score:.4f}")
            
        except ImportError:
            lpips_score = None
            print("WARNING: LPIPS not available (install lpips package)")
        
        print(f"Generation Results:")
        print(f"   PSNR (vs Real): {gen_psnr:.4f} dB") 
        print(f"   SSIM (vs Real): {gen_ssim:.4f}")
        if lpips_score is not None:
            print(f"   LPIPS (vs Real): {lpips_score:.4f}")
        
        # Save results
        print("Saving results...")
        
        # Create output directory (relative to original directory)
        os.chdir(original_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save generated samples
        save_image(generated_images, out_dir / "generated_samples.png", nrow=samples_per_class,
                  normalize=True, scale_each=True)
        
        # Save comparison grid
        comparison_samples = []
        for class_id in range(10):
            start_idx = class_id * samples_per_class
            if start_idx + 1 < len(real_images) and start_idx + 1 < len(generated_images):
                real_class = real_images[start_idx:start_idx + 2]
                gen_class = generated_images[start_idx:start_idx + 2]
                for i in range(2):
                    comparison_samples.append(real_class[i])
                    comparison_samples.append(gen_class[i])
        
        if comparison_samples:
            comparison_grid = torch.stack(comparison_samples)
            save_image(comparison_grid, out_dir / "comparison_grid.png", nrow=4, 
                      normalize=True, scale_each=True)
        
        # Compile results
        results = {
            "model_type": "baseline_vqvae",
            "dataset": "cifar10",
            "reconstruction_quality": {
                "psnr": float(f"{psnr_recon:.6f}"),
                "ssim": float(f"{ssim_recon:.6f}"),
                "samples_evaluated": len(originals)
            },
            "generation_quality": {
                "psnr": float(f"{gen_psnr:.6f}"),
                "ssim": float(f"{gen_ssim:.6f}"),
                "samples_generated": len(generated_images),
                "samples_per_class": samples_per_class,
            },
            "codebook_health": {
                "entropy": float(f"{cb_stats['entropy']:.6f}"),
                "used_codes": int(cb_stats["used"]),
                "dead_codes": int(cb_stats["dead_codes"]),
                "usage_percent": float(f"{100 * cb_stats['used'] / model.quant.n_codes:.2f}"),
                "codebook_size": model.quant.n_codes,
            }
        }
        
        if lpips_score is not None:
            results["generation_quality"]["lpips"] = float(f"{lpips_score:.6f}")
        
        # Save in multiple formats for compatibility
        with open(out_dir / "metrics.yaml", 'w') as f:
            metrics_yaml = {
                "PSNR": f"{gen_psnr:.4f}",
                "SSIM": f"{gen_ssim:.4f}",
            }
            if lpips_score is not None:
                metrics_yaml["LPIPS"] = f"{lpips_score:.4f}"
            yaml.dump(metrics_yaml, f)
        
        with open(out_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(out_dir / "codebook_health.json", 'w') as f:
            json.dump(results["codebook_health"], f, indent=2)
        
        print(f"Results saved to: {out_dir}")
        print(f"   Generated samples: generated_samples.png")
        print(f"   Comparison grid: comparison_grid.png") 
        print(f"   Metrics: metrics.yaml, evaluation_results.json")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Always return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    exit(main())
