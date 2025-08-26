"""Generate reconstruction grids using a built codebook and a trained VAE.

Saves a grid of original reconstructions (from continuous z) vs quantized reconstructions
(decode nearest codebook medoid for each selected sample).

This does NOT recompute geodesic assignments for the selected set; it uses
Euclidean nearest medoids to the precomputed geodesic codebook medoids, which is
adequate for visualization.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
import torchvision
import torchvision.utils as vutils

from src.models.vae import VAE


def _load_vae_model(checkpoint_path: Path, device: torch.device) -> VAE:
    with open(Path("configs/vae.yaml"), "r") as f:
        vae_cfg = yaml.safe_load(f) or {}

    model = VAE(
        in_channels=int(vae_cfg.get("in_channels", 1)),
        enc_channels=vae_cfg.get("enc_channels", [32, 64, 128]),
        dec_channels=vae_cfg.get("dec_channels", [128, 64, 32]),
        latent_dim=int(vae_cfg.get("latent_dim", 16)),
        recon_loss=str(vae_cfg.get("recon_loss", "bce")),
        output_image_size=int(vae_cfg.get("output_image_size", 28)),
        norm_type=str(vae_cfg.get("norm_type", "none")),
        mse_use_sigmoid=bool(vae_cfg.get("mse_use_sigmoid", True)),
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])  # type: ignore[index]
    model.to(device).eval()
    return model


def _select_indices(N: int, num: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=min(num, N), replace=False)
    return np.sort(idx)


def _nearest_medoid_indices(z_sel: np.ndarray, z_medoid: np.ndarray) -> np.ndarray:
    # Compute distances for the selected samples only for efficiency
    # z_sel: (M, D), z_medoid: (K, D)
    # returns: (M,) argmin over medoids
    # Use (a-b)^2 = a^2 + b^2 - 2ab to avoid large allocations if needed
    a2 = np.sum(z_sel**2, axis=1, keepdims=True)         # (M,1)
    b2 = np.sum(z_medoid**2, axis=1, keepdims=True).T    # (1,K)
    ab = z_sel @ z_medoid.T                              # (M,K)
    d2 = a2 + b2 - 2.0 * ab
    return np.argmin(d2, axis=1)


def _make_grid(x_top: torch.Tensor, x_bottom: torch.Tensor) -> torch.Tensor:
    # x_top and x_bottom are (B,C,H,W) already on CPU in [0,1]
    grid = vutils.make_grid(torch.cat([x_top, x_bottom], dim=0), nrow=x_top.size(0))
    return grid


def main() -> None:
    p = argparse.ArgumentParser(description="Reconstruction grid from codebook")
    p.add_argument("--codebook_dir", type=str, required=True,
                   help="Directory containing codebook.pt and codes.npy")
    p.add_argument("--latents_path", type=str, default="experiments/vae_cifar10/latents_val/z.pt",
                   help="Path to latents tensor (z.pt)")
    p.add_argument("--checkpoint", type=str, default="experiments/vae_cifar10/checkpoints/best.pt",
                   help="Path to VAE checkpoint (best.pt)")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="reconstruction_grid_quantized.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codebook_dir = Path(args.codebook_dir)
    # PyTorch 2.6 sets weights_only=True by default; our file is a trusted dict
    codebook_path = codebook_dir / "codebook.pt"
    try:
        codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        # For older torch versions where weights_only is not supported
        codebook = torch.load(codebook_path, map_location="cpu")
    z_medoid = codebook["z_medoid"].float().numpy()  # (K, D)

    z = torch.load(Path(args.latents_path), map_location="cpu")
    if isinstance(z, dict) and "z" in z:
        z = z["z"]
    z = z.float().numpy()  # (N, D)

    idx = _select_indices(N=z.shape[0], num=args.num_samples, seed=args.seed)
    z_sel = z[idx]

    codes_path = codebook_dir / "codes.npy"
    medoid_idx = None
    if codes_path.exists():
        codes = np.load(codes_path)
        if codes.shape[0] == z.shape[0]:
            print("Using precomputed geodesic assignments from codes.npy")
            medoid_idx = codes[idx]

    if medoid_idx is None:
        print("Computing nearest medoids using Euclidean distance for visualization")
        medoid_idx = _nearest_medoid_indices(z_sel, z_medoid)

    zq_sel = z_medoid[medoid_idx]

    model = _load_vae_model(Path(args.checkpoint), device)

    # decide visualization activation based on config
    with open(Path("configs/vae.yaml"), "r") as f:
        vae_cfg_vis = yaml.safe_load(f) or {}
    recon_loss = str(vae_cfg_vis.get("recon_loss", "mse")).lower()
    mse_use_sigmoid = bool(vae_cfg_vis.get("mse_use_sigmoid", True))
    apply_sigmoid = (recon_loss == "bce") or mse_use_sigmoid

    with torch.no_grad():
        z_sel_t = torch.from_numpy(z_sel).to(device)
        zq_sel_t = torch.from_numpy(zq_sel).to(device)
        x_orig_logits = model.decoder(z_sel_t)
        x_quant_logits = model.decoder(zq_sel_t)
        x_orig = torch.sigmoid(x_orig_logits) if apply_sigmoid else x_orig_logits
        x_quant = torch.sigmoid(x_quant_logits) if apply_sigmoid else x_quant_logits
        x_orig = x_orig.cpu()
        x_quant = x_quant.cpu()

    # unnormalize for CIFAR10 if we didn’t use sigmoid
    with open(Path("configs/data.yaml"), "r") as f:
        data_cfg = yaml.safe_load(f) or {}
    if str(data_cfg.get("name", "")).strip().upper() == "CIFAR10" and not apply_sigmoid:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
        std  = torch.tensor([0.2470, 0.2430, 0.2610]).view(1,3,1,1)
        x_orig = (x_orig * std + mean).clamp(0, 1)
        x_quant = (x_quant * std + mean).clamp(0, 1)

    grid = _make_grid(x_top=x_orig, x_bottom=x_quant)
    out_path = codebook_dir / args.out
    torchvision.utils.save_image(grid, out_path)
    print(f"Saved reconstruction grid to: {out_path}")


if __name__ == "__main__":
    main()


