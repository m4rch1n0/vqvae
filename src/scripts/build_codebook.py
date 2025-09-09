import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from src.geo.kmeans_optimized import fit_kmedoids_optimized
from src.geo.knn_graph_optimized import build_knn_graph_auto, largest_connected_component
from src.models import SpatialVAE
from src.geo.riemannian_metric import edge_lengths_riemannian


def main(args):
    """Builds a spatial codebook using a geodesic metric and saves artifacts."""
    latents_path = Path(args.latents_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_cfg = {
        'in_channels': args.in_channels, 'output_image_size': args.output_image_size,
        'latent_dim': args.latent_dim, 'enc_channels': args.enc_channels,
        'dec_channels': args.dec_channels, 'recon_loss': args.recon_loss,
        'norm_type': args.norm_type, 'mse_use_sigmoid': args.mse_use_sigmoid
    }
    vae = SpatialVAE(**vae_cfg).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt_path)['model_state_dict'])
    decoder = vae.decoder

    z = torch.load(latents_path, map_location="cpu").float()
    N, C, H, W = z.shape
    print(f"Loaded spatial latents: N={N}, C={C}, H={H}, W={W}")

    z_flat = z.permute(0, 2, 3, 1).reshape(-1, C).numpy()
    print(f"Reshaped to: {z_flat.shape}")

    W_euclidean, _ = build_knn_graph_auto(
        z_flat, k=args.k, metric="euclidean",
        mode="connectivity", sym=args.sym
    )
    
    rows, cols = W_euclidean.nonzero()
    edge_indices = np.stack((rows, cols), axis=1)
    edge_indices = edge_indices[rows < cols]
    
    z_start = torch.from_numpy(z_flat[edge_indices[:, 0]]).float()
    z_end = torch.from_numpy(z_flat[edge_indices[:, 1]]).float()

    print(f"Re-weighting {len(z_start)} edges using Riemannian metric...")
    geo_lengths = edge_lengths_riemannian(decoder, z_start, z_end, batch_size=args.batch_size)
    
    W_geo = sparse.csr_matrix((geo_lengths.cpu().numpy(), (edge_indices[:, 0], edge_indices[:, 1])), shape=W_euclidean.shape)
    W_geo = W_geo + W_geo.T

    mask_lcc = largest_connected_component(W_geo)
    if mask_lcc.sum() < W_geo.shape[0]:
        print(f"Using LCC: {mask_lcc.sum()}/{W_geo.shape[0]} nodes")
        W_lcc = W_geo[mask_lcc][:, mask_lcc]
        z_lcc = z_flat[mask_lcc]
    else:
        W_lcc = W_geo
        z_lcc = z_flat

    sparse.save_npz(out_dir / "knn_graph_geodesic.npz", W_lcc)

    quant_cfg = {
        "K": args.K, "init": args.init, "seed": args.seed
    }
    medoids, assign_lcc, qe = fit_kmedoids_optimized(
        W_lcc, K=quant_cfg["K"], init=quant_cfg["init"], seed=quant_cfg["seed"]
    )

    assign_flat = np.full(z_flat.shape[0], -1, dtype=np.int32)
    assign_flat[mask_lcc] = assign_lcc
    codes = assign_flat.reshape(N, H, W)
    
    z_medoid = z_lcc[medoids]
    codebook = {
        "medoid_indices": medoids.astype(np.int32),
        "z_medoid": torch.from_numpy(z_medoid).float(),
        "config": {
            "latents_path": args.latents_path,
            "out_dir": args.out_dir,
            "vae_ckpt_path": args.vae_ckpt_path,
            "in_channels": args.in_channels,
            "output_image_size": args.output_image_size,
            "latent_dim": args.latent_dim,
            "enc_channels": args.enc_channels,
            "dec_channels": args.dec_channels,
            "recon_loss": args.recon_loss,
            "norm_type": args.norm_type,
            "mse_use_sigmoid": args.mse_use_sigmoid,
            "k": args.k,
            "sym": args.sym,
            "K": args.K,
            "init": args.init,
            "seed": args.seed,
            "batch_size": args.batch_size
        },
    }
    torch.save(codebook, out_dir / "codebook.pt")
    np.save(out_dir / "codes.npy", codes)

    print(f"Quantization error: {qe:.3f}")
    print(f"Saved artifacts to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a geodesic spatial codebook.")
    # File Paths
    parser.add_argument("--latents_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--vae_ckpt_path", type=str, required=True)
    # VAE Config
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--output_image_size", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--enc_channels", type=int, nargs='+', required=True)
    parser.add_argument("--dec_channels", type=int, nargs='+', required=True)
    parser.add_argument("--recon_loss", type=str, required=True)
    parser.add_argument("--norm_type", type=str, required=True)
    parser.add_argument("--mse_use_sigmoid", action='store_true')
    # Graph Config
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sym", type=str, default="union")
    # Quantize Config
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--init", type=str, default="kpp")
    parser.add_argument("--seed", type=int, default=42)
    # System Config
    parser.add_argument("--batch_size", type=int, default=512)
    
    args = parser.parse_args()
    main(args)
