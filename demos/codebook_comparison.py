"""Geodesic vs Euclidean codebook comparison."""
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.geo.kmeans_precomputed import fit_kmedoids_precomputed
from src.geo.knn_graph import build_knn_graph, largest_connected_component
from src.geo.geo_shortest_paths import dijkstra_multi_source
from src.models.vae import VAE


def parse_args():
    """Parse command line arguments for configuration selection."""
    parser = argparse.ArgumentParser(description="Geodesic vs Euclidean codebook comparison")
    parser.add_argument(
        "--config",
        type=str,
        default="test2",
        choices=["test1", "test2", "fashion_k1024"],
        help="Configuration to use (default: test2)"
    )
    return parser.parse_args()


# Available configurations:
# Standard:           --config test1
# Experimental:       --config test2


def load_latents(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "z" in obj:
        z = obj["z"]
    elif torch.is_tensor(obj):
        z = obj
    else:
        raise ValueError("Expected dict with 'z' key or tensor")
    return z.float()


def load_vae_model(checkpoint_path: Path, device: torch.device) -> VAE:
    """Load VAE with architecture from config to match checkpoint."""
    # Load root-level VAE config only
    with open(Path("configs/vae.yaml"), "r") as f:
        vae_cfg = yaml.safe_load(f) or {}

    in_channels = vae_cfg.get("in_channels", 1)
    latent_dim = vae_cfg.get("latent_dim", 16)
    enc_channels = vae_cfg.get("enc_channels", [32, 64, 128])
    dec_channels = vae_cfg.get("dec_channels", [128, 64, 32])
    recon_loss = vae_cfg.get("recon_loss", "bce")
    output_image_size = int(vae_cfg.get("output_image_size", 28))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = VAE(
        in_channels=in_channels,
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        latent_dim=latent_dim,
        recon_loss=recon_loss,
        output_image_size=output_image_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[index]
    model.to(device)
    model.eval()
    return model


def build_euclidean_codebook(z: np.ndarray, K: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    assign = kmeans.fit_predict(z)
    centroids = kmeans.cluster_centers_
    return centroids, assign


def build_geodesic_codebook(z: np.ndarray, K: int, k_graph: int = 10, seed: int = 42, metric: str = "euclidean", sym: str = "mutual", mode: str = "distance") -> Tuple[np.ndarray, np.ndarray, object, np.ndarray, np.ndarray]:
    W, _ = build_knn_graph(z, k=k_graph, metric=metric, mode=mode, sym=sym)
    
    mask_lcc = largest_connected_component(W)
    if mask_lcc.sum() < W.shape[0]:
        W = W[mask_lcc][:, mask_lcc]
        z_lcc = z[mask_lcc]
    else:
        z_lcc = z
        mask_lcc = np.ones(len(z), dtype=bool)
    
    medoids, assign_lcc, _ = fit_kmedoids_precomputed(W, K=K, init="kpp", seed=seed, chunk_size=1000)
    
    assign = np.full(len(z), fill_value=-1, dtype=np.int32)
    assign[mask_lcc] = assign_lcc
    
    centroids = z_lcc[medoids]
    # Return additional artifacts to allow geodesic QE computation in caller
    return centroids, assign, W, mask_lcc, medoids


def compute_reconstruction_quality(model: VAE, z_original: torch.Tensor, z_quantized: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        z_orig_dev = z_original.to(device)
        z_quant_dev = z_quantized.to(device)
        
        x_orig = torch.sigmoid(model.decoder(z_orig_dev))
        x_quant = torch.sigmoid(model.decoder(z_quant_dev))
        
        mse = torch.nn.functional.mse_loss(x_quant, x_orig).item()
    return mse


def compute_perplexity(assign: np.ndarray, K: int) -> float:
    valid_mask = assign >= 0
    if not valid_mask.any():
        return 0.0
    
    assign_valid = assign[valid_mask]
    counts = np.bincount(assign_valid, minlength=K)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    
    entropy = -np.sum(nz * np.log(nz + 1e-12))
    return float(np.exp(entropy))


def save_comparison_plot(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['Euclidean', 'Geodesic']
    colors = ['#1f77b4', '#ff7f0e']
    
    # Reconstruction quality
    recon_errors = [metrics['euclidean']['reconstruction_mse'], metrics['geodesic']['reconstruction_mse']]
    axes[0].bar(methods, recon_errors, color=colors)
    axes[0].set_ylabel('Reconstruction MSE')
    axes[0].set_title('Reconstruction Quality\n(Lower is Better)')
    
    # Perplexity
    perplexities = [metrics['euclidean']['perplexity'], metrics['geodesic']['perplexity']]
    axes[1].bar(methods, perplexities, color=colors)
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Code Usage Diversity\n(Higher is Better)')
    
    # Quantization error
    qe_errors = [metrics['euclidean']['quantization_error'], metrics['geodesic']['quantization_error']]
    axes[2].bar(methods, qe_errors, color=colors)
    axes[2].set_ylabel('Quantization Error')
    axes[2].set_title('Clustering Quality\n(Lower is Better)')
    
    plt.tight_layout()
    plt.savefig(out_dir / "codebook_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config_path = Path(f"configs/codebook_comparison/{args.config}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    latents_path = Path(config["data"]["latents_path"])
    checkpoint_path = Path(config["data"]["checkpoint_path"])
    K = config["quantization"]["K"]
    k_graph = config["graph"]["k"]
    seed = config["quantization"]["seed"]
    
    # Setup output directory
    if config["output"]["timestamp"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"{config['output']['base_dir']}_{timestamp}")
    else:
        out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Device selection
    if config["experiment"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["experiment"]["device"])
    print(f"Using device: {device}")
    
    z = load_latents(latents_path).numpy()
    model = load_vae_model(checkpoint_path, device)
    N, D = z.shape
    print(f"Loaded {N} latents (dim={D})")
    
    print(f"Building Euclidean codebook (K={K})...")
    centroids_euc, assign_euc = build_euclidean_codebook(z, K, seed)
    
    print(f"Building geodesic codebook (K={K}, k_graph={k_graph})...")
    centroids_geo, assign_geo, W_lcc, mask_lcc, medoids_lcc = build_geodesic_codebook(
        z, K, k_graph, seed, 
        metric=config["graph"]["metric"], 
        sym=config["graph"]["sym"], 
        mode=config["graph"]["mode"]
    )
    
    z_tensor = torch.from_numpy(z).float()
    
    z_quant_euc = torch.from_numpy(centroids_euc[assign_euc]).float()
    
    valid_geo = assign_geo >= 0
    z_quant_geo = z_tensor.clone()
    if valid_geo.any():
        z_quant_geo[valid_geo] = torch.from_numpy(centroids_geo[assign_geo[valid_geo]]).float()
    
    recon_mse_euc = compute_reconstruction_quality(model, z_tensor, z_quant_euc, device)
    recon_mse_geo = compute_reconstruction_quality(model, z_tensor[valid_geo], z_quant_geo[valid_geo], device)
    
    perp_euc = compute_perplexity(assign_euc, K)
    perp_geo = compute_perplexity(assign_geo, K)
    
    qe_euc = np.mean(np.linalg.norm(z - z_quant_euc.numpy(), axis=1) ** 2)

    # Geodesic quantization error computed on the graph using shortest-path distances
    # Only nodes in the largest connected component are valid for geodesic distances
    if mask_lcc.any():
        # Distances from each medoid to all nodes in the LCC: shape (K, N_lcc)
        D_geo = dijkstra_multi_source(W_lcc, medoids_lcc)
        # Recover assignments on the LCC to index D_geo row-wise
        assign_lcc = assign_geo[mask_lcc]
        idx = np.arange(assign_lcc.shape[0])
        dmin = D_geo[assign_lcc, idx]
        finite_mask = np.isfinite(dmin)
        qe_geo = float(np.mean((dmin[finite_mask]) ** 2)) if finite_mask.any() else float("inf")
    else:
        qe_geo = float("inf")
    
    metrics = {
        'euclidean': {
            'reconstruction_mse': float(recon_mse_euc),
            'perplexity': float(perp_euc),
            'quantization_error': float(qe_euc),
            'valid_samples': int(N)
        },
        'geodesic': {
            'reconstruction_mse': float(recon_mse_geo),
            'perplexity': float(perp_geo),
            'quantization_error': float(qe_geo),
            'valid_samples': int(valid_geo.sum())
        }
    }
    
    # Save results based on configuration
    if config["experiment"]["save_plots"]:
        save_comparison_plot(metrics, out_dir)
    
    if config["experiment"]["save_metrics"]:
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Save configuration for reproducibility
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"\nComparison Results (K={K}):")
    print(f"Euclidean  - MSE: {recon_mse_euc:.6f}, Perplexity: {perp_euc:.2f}, QE: {qe_euc:.2f}")
    print(f"Geodesic   - MSE: {recon_mse_geo:.6f}, Perplexity: {perp_geo:.2f}, QE: {qe_geo:.2f}")
    print(f"Results saved to: {out_dir}")
    
    return out_dir


if __name__ == "__main__":
    main()
