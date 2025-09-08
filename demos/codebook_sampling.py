"""Generate reconstruction grids using a built codebook and a trained VAE.

Saves a grid of original reconstructions (from continuous z) vs quantized reconstructions
(decode nearest codebook medoid for each selected sample).

This does NOT recompute geodesic assignments for the selected set; it uses
Euclidean nearest medoids to the precomputed geodesic codebook medoids, which is
adequate for visualization (also i'm not sure about this, might give a try with)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import torchvision
import torchvision.utils as vutils

from src.models.vae import VAE
from src.models.spatial_vae import SpatialVAE


def load_vae_model(checkpoint_path: Path, device: torch.device) -> tuple[object, dict]:
    """Load VAE model from checkpoint and return model + config for activation decisions."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract config with simplified fallback
    vae_cfg = ckpt.get("config") or ckpt.get("model_config")
    state_dict = ckpt["model_state_dict"]
    
    if vae_cfg is None:
        print("Warning: No config found in checkpoint, inferring from state dict...")
        # Quick inference of critical parameters from state dict
        is_spatial_check = any("conv_in" in k or "deconv_layers" in k for k in state_dict.keys())
        
        if is_spatial_check:
            # SpatialVAE: latent_dim from fc_mu
            latent_dim = state_dict.get("encoder.fc_mu.weight", torch.zeros(16, 1)).shape[0]
        else:
            # Vanilla VAE: latent_dim from decoder first layer input
            decoder_fc_key = next((k for k in state_dict.keys() if "decoder" in k and "0.weight" in k), None)
            if decoder_fc_key:
                latent_dim = state_dict[decoder_fc_key].shape[1]
            else:
                latent_dim = 128  # fallback
        
        # Infer in_channels from encoder first conv
        encoder_conv_key = next((k for k in state_dict.keys() if "encoder" in k and "0.weight" in k), None)
        in_channels = state_dict[encoder_conv_key].shape[1] if encoder_conv_key else 1
        
        vae_cfg = {
            "in_channels": in_channels, "latent_dim": latent_dim, 
            "enc_channels": [64, 128, 256], "dec_channels": [256, 128, 64], 
            "recon_loss": "mse", "output_image_size": 28 if in_channels == 1 else 32, 
            "norm_type": "batch", "mse_use_sigmoid": in_channels == 1
        }
    
    # Detect model type from state dict keys
    is_spatial = any("conv_in" in k or "deconv_layers" in k for k in state_dict.keys())
    
    # Default parameters for both model types
    model_params = {
        "in_channels": int(vae_cfg.get("in_channels", 1)),
        "enc_channels": vae_cfg.get("enc_channels", [64, 128, 256]),
        "dec_channels": vae_cfg.get("dec_channels", [256, 128, 64]),
        "latent_dim": int(vae_cfg.get("latent_dim", 16 if is_spatial else 128)),
        "recon_loss": str(vae_cfg.get("recon_loss", "mse")),
        "output_image_size": int(vae_cfg.get("output_image_size", 28)),
        "norm_type": str(vae_cfg.get("norm_type", "batch")),
        "mse_use_sigmoid": bool(vae_cfg.get("mse_use_sigmoid", True)),
    }
    
    # Create appropriate model
    if is_spatial:
        print("Detected SpatialVAE checkpoint")
        model = SpatialVAE(**model_params)
    else:
        print("Detected vanilla VAE checkpoint")
        model = VAE(**model_params)
    
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, vae_cfg


def select_indices(N: int, num: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=min(num, N), replace=False)
    return np.sort(idx)


def nearest_medoid_indices(z_sel: np.ndarray, z_medoid: np.ndarray) -> np.ndarray:
    # Compute distances for the selected samples only for efficiency
    # z_sel: (M, D), z_medoid: (K, D)
    # returns: (M,) argmin over medoids
    # Use (a-b)^2 = a^2 + b^2 - 2ab to avoid large allocations if needed
    a2 = np.sum(z_sel**2, axis=1, keepdims=True)  # (M,1)
    b2 = np.sum(z_medoid**2, axis=1, keepdims=True).T  # (1,K)
    ab = z_sel @ z_medoid.T  # (M,K)
    d2 = a2 + b2 - 2.0 * ab
    return np.argmin(d2, axis=1)


def make_grid(x_top: torch.Tensor, x_bottom: torch.Tensor) -> torch.Tensor:
    # x_top and x_bottom are (B,C,H,W) already on CPU in [0,1]
    grid = vutils.make_grid(torch.cat([x_top, x_bottom], dim=0), nrow=x_top.size(0))
    return grid


def auto_detect_paths(experiment_dir: Path) -> dict:
    """Auto-detect all necessary paths from experiment directory structure."""
    exp_dir = Path(experiment_dir)
    
    # Find codebook directory
    codebook_dir = exp_dir / "codebook"
    if not codebook_dir.exists():
        raise FileNotFoundError(f"Codebook directory not found: {codebook_dir}")
    
    # Find VAE directory and checkpoint
    vae_dir = exp_dir / "vae"
    if not vae_dir.exists():
        raise FileNotFoundError(f"VAE directory not found: {vae_dir}")
    
    # Look for checkpoint in VAE directory structure
    checkpoint_paths = list(vae_dir.rglob("checkpoints/best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"VAE checkpoint not found in: {vae_dir}")
    checkpoint_path = checkpoint_paths[0]
    
    # Look for latents in VAE directory structure  
    latents_paths = list(vae_dir.rglob("latents_val/z.pt"))
    if not latents_paths:
        raise FileNotFoundError(f"Validation latents not found in: {vae_dir}")
    latents_path = latents_paths[0]
    
    return {
        "codebook_dir": codebook_dir,
        "checkpoint_path": checkpoint_path,
        "latents_path": latents_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Reconstruction grid from codebook")
    p.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (e.g., experiments/fashionmnist/vanilla/euclidean)",
    )
    p.add_argument("--num_samples", type=int, default=16, help="Number of samples to visualize")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    p.add_argument("--out", type=str, default="reconstruction_grid_quantized.png", help="Output filename")
    p.add_argument(
        "--out_dir", type=str, default="", help="Optional output directory (default: codebook_dir)"
    )
    args = p.parse_args()
    
    try:
        paths = auto_detect_paths(args.experiment_dir)
        print(f"Auto-detected paths:")
        print(f"  Codebook: {paths['codebook_dir']}")
        print(f"  Checkpoint: {paths['checkpoint_path']}")
        print(f"  Latents: {paths['latents_path']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codebook_dir = paths["codebook_dir"]
    # PyTorch 2.6 sets weights_only=True by default; our file is a trusted dict
    codebook_path = codebook_dir / "codebook.pt"
    try:
        codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        # For older torch versions where weights_only is not supported
        codebook = torch.load(codebook_path, map_location="cpu")
    z_medoid = codebook["z_medoid"].float().numpy()  # (K, D)

    z = torch.load(paths["latents_path"], map_location="cpu")
    if isinstance(z, dict) and "z" in z:
        z = z["z"]
    z = z.float().numpy()  # (N, D) or (N, C, H, W) for spatial

    # Handle spatial vs vanilla latents
    is_spatial_latents = z.ndim == 4
    
    if is_spatial_latents:
        print(f"Detected spatial latents: {z.shape}")
        N, C, H, W = z.shape
        # For spatial latents, select full images first
        idx = select_indices(N=N, num=args.num_samples, seed=args.seed)
        z_sel_spatial = z[idx]  # (num_samples, C, H, W)
        
        # Then flatten for codebook matching
        z_flat = z.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        print(f"Flattened to: {z_flat.shape} for codebook matching")
        
        # Check dimensional compatibility
        if C != z_medoid.shape[-1]:
            print(f"ERROR: Dimensional mismatch!")
            print(f"  Latents dimension: {C} (spatial shape: {z.shape})")
            print(f"  Codebook dimension: {z_medoid.shape[-1]} (shape: {z_medoid.shape})")
            print(f"\nEnsure latents and codebook come from compatible experiments.")
            return
        
        # For spatial latents, we need to handle reconstruction differently
        z_sel = z_sel_spatial  # Keep spatial format for reconstruction
        
    else:
        print(f"Detected vanilla latents: {z.shape}")
        # Check dimensional compatibility
        if z.shape[-1] != z_medoid.shape[-1]:
            print(f"ERROR: Dimensional mismatch!")
            print(f"  Latents dimension: {z.shape[-1]} (shape: {z.shape})")
            print(f"  Codebook dimension: {z_medoid.shape[-1]} (shape: {z_medoid.shape})")
            print(f"\nEnsure latents and codebook come from compatible experiments.")
            return

        idx = select_indices(N=z.shape[0], num=args.num_samples, seed=args.seed)
        z_sel = z[idx]

    codes_path = codebook_dir / "codes.npy"
    medoid_idx = None
    if codes_path.exists():
        codes = np.load(codes_path)
        if is_spatial_latents:
            # For spatial latents, codes should match original spatial shape
            if codes.shape == z.shape[:3]:  # (N, H, W)
                print("Using precomputed spatial geodesic assignments from codes.npy")
                medoid_idx = codes[idx]  # (num_samples, H, W)
        else:
            # For vanilla latents, codes should match flattened shape
            if codes.shape[0] == z.shape[0]:
                print("Using precomputed geodesic assignments from codes.npy")
                medoid_idx = codes[idx]

    if medoid_idx is None:
        print("Computing nearest medoids using Euclidean distance for visualization")
        if is_spatial_latents:
            # For spatial latents, flatten selected samples and compute medoids
            z_sel_flat = z_sel_spatial.transpose(0, 2, 3, 1).reshape(-1, C)
            medoid_idx_flat = nearest_medoid_indices(z_sel_flat, z_medoid)
            # Reshape back to spatial format
            medoid_idx = medoid_idx_flat.reshape(args.num_samples, H, W)
        else:
            medoid_idx = nearest_medoid_indices(z_sel, z_medoid)

    # Get quantized latents
    if is_spatial_latents:
        # For spatial: create quantized spatial latents
        zq_sel = np.zeros_like(z_sel)
        for i in range(args.num_samples):
            for h in range(H):
                for w in range(W):
                    zq_sel[i, :, h, w] = z_medoid[medoid_idx[i, h, w]]
    else:
        zq_sel = z_medoid[medoid_idx]

    model, vae_cfg = load_vae_model(paths["checkpoint_path"], device)

    # decide visualization activation based on config from checkpoint
    recon_loss = str(vae_cfg.get("recon_loss", "mse")).lower()
    mse_use_sigmoid = bool(vae_cfg.get("mse_use_sigmoid", True))
    apply_sigmoid = (recon_loss == "bce") or mse_use_sigmoid

    with torch.no_grad():
        z_sel_t = torch.from_numpy(z_sel).to(device)
        zq_sel_t = torch.from_numpy(zq_sel).to(device)
        
        # SpatialVAE already has correct shape (B, C, H, W), vanilla VAE has (B, D)
        x_orig_logits = model.decoder(z_sel_t)
        x_quant_logits = model.decoder(zq_sel_t)
        x_orig = torch.sigmoid(x_orig_logits) if apply_sigmoid else x_orig_logits
        x_quant = torch.sigmoid(x_quant_logits) if apply_sigmoid else x_quant_logits
        x_orig = x_orig.cpu()
        x_quant = x_quant.cpu()

    # Apply CIFAR-10 denormalization if we have RGB images and no sigmoid
    # (heuristic: 3 channels + no sigmoid activation suggests CIFAR-10 normalization)
    is_rgb = vae_cfg.get("in_channels", 1) == 3
    if is_rgb and not apply_sigmoid:
        print("Applying CIFAR-10 denormalization (RGB + no sigmoid activation)")
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2430, 0.2610]).view(1, 3, 1, 1)
        x_orig = (x_orig * std + mean).clamp(0, 1)
        x_quant = (x_quant * std + mean).clamp(0, 1)

    grid = make_grid(x_top=x_orig, x_bottom=x_quant)
    # If out_dir is set, write there; else, default under codebook_dir
    base_out_dir = Path(args.out_dir) if args.out_dir else codebook_dir
    base_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_out_dir / args.out
    torchvision.utils.save_image(grid, out_path)
    print(f"Saved reconstruction grid to: {out_path}")


if __name__ == "__main__":
    main()
