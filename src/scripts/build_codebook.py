import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import sparse

from src.geo.kmeans_optimized import fit_kmedoids_optimized
from src.geo.knn_graph_optimized import build_knn_graph_auto, largest_connected_component
from src.models import SpatialVAE
from src.geo.riemannian_metric import edge_lengths_riemannian


def build_and_save_spatial_geodesic(config: dict) -> Path:
    """Builds a spatial codebook using a geodesic metric and saves artifacts."""
    latents_path = Path(config["data"]["latents_path"])
    out_dir = Path(config["out"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE model
    vae_cfg = config['vae']
    vae = SpatialVAE(
        in_channels=vae_cfg['in_channels'], enc_channels=tuple(vae_cfg['enc_channels']),
        dec_channels=tuple(vae_cfg['dec_channels']), latent_dim=vae_cfg['latent_dim'],
        recon_loss=vae_cfg['recon_loss'], output_image_size=vae_cfg['output_image_size'],
        norm_type=vae_cfg['norm_type']
    ).to(device)
    vae.load_state_dict(torch.load(config['vae_ckpt_path'])['model_state_dict'])
    decoder = vae.decoder

    z = torch.load(latents_path, map_location="cpu").float()
    N, C, H, W = z.shape
    print(f"Loaded spatial latents: N={N}, C={C}, H={H}, W={W}")

    z_flat = z.permute(0, 2, 3, 1).reshape(-1, C).numpy()
    print(f"Reshaped to: {z_flat.shape}")

    graph_cfg = config["graph"]
    W_euclidean, _ = build_knn_graph_auto(
        z_flat, k=graph_cfg["k"], metric="euclidean",
        mode="connectivity", sym=graph_cfg["sym"]
    )

    # Re-weight edges with Riemannian metric
    rows, cols = W_euclidean.nonzero()
    edge_indices = np.stack((rows, cols), axis=1)
    
    # Filter out self-loops and duplicate edges
    edge_indices = edge_indices[rows < cols]
    
    z_start = torch.from_numpy(z_flat[edge_indices[:, 0]]).float()
    z_end = torch.from_numpy(z_flat[edge_indices[:, 1]]).float()

    print(f"Re-weighting {len(z_start)} edges using Riemannian metric...")
    geo_lengths = edge_lengths_riemannian(decoder, z_start, z_end, batch_size=config['system']['batch_size'])
    
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

    quant_cfg = config["quantize"]
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
        "config": config,
    }
    torch.save(codebook, out_dir / "codebook.pt")
    np.save(out_dir / "codes.npy", codes)

    print(f"Quantization error: {qe:.3f}")
    print(f"Saved artifacts to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    build_and_save_spatial_geodesic(cfg)
