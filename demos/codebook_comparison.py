"""Simplified Geodesic vs Euclidean codebook comparison."""
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

from src.geo.kmeans_optimized import fit_kmedoids_optimized
from src.geo.knn_graph_optimized import build_knn_graph, largest_connected_component
from src.geo.geo_shortest_paths import dijkstra_multi_source
from src.models.vae import VAE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Geodesic vs Euclidean codebook comparison")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--K", type=int, default=64, help="Codebook size (default: 64)")
    parser.add_argument("--k_graph", type=int, default=10, help="k-NN connectivity (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def auto_detect_paths(experiment_dir: str) -> dict:
    """Auto-detect latents and checkpoint paths from experiment directory."""
    exp_dir = Path(experiment_dir)
    vae_dir = exp_dir / "vae"
    
    if not vae_dir.exists():
        raise FileNotFoundError(f"VAE directory not found: {vae_dir}")
    
    checkpoint_paths = list(vae_dir.rglob("checkpoints/best.pt"))
    latents_paths = list(vae_dir.rglob("latents_val/z.pt"))
    
    if not checkpoint_paths:
        raise FileNotFoundError(f"VAE checkpoint not found in: {vae_dir}")
    if not latents_paths:
        raise FileNotFoundError(f"Validation latents not found in: {vae_dir}")
    
    return {"checkpoint_path": checkpoint_paths[0], "latents_path": latents_paths[0]}


def load_vae_model(checkpoint_path: Path, device: torch.device) -> VAE:
    """Load VAE with simplified config handling."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Simple config extraction with reasonable defaults
    vae_cfg = checkpoint.get("config") or checkpoint.get("model_config") or {}
    
    defaults = {
        "in_channels": 1, "latent_dim": 128, "enc_channels": [64, 128, 256],
        "dec_channels": [256, 128, 64], "recon_loss": "mse", 
        "output_image_size": 28, "norm_type": "batch", "mse_use_sigmoid": True
    }
    
    # Use config values or defaults
    model_params = {k: vae_cfg.get(k, defaults[k]) for k in defaults}
    model = VAE(**model_params)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def build_euclidean_codebook(z: np.ndarray, K: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build K-means codebook."""
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    assign = kmeans.fit_predict(z)
    return kmeans.cluster_centers_, assign


def build_geodesic_codebook(z: np.ndarray, K: int, k_graph: int, seed: int) -> Tuple[np.ndarray, np.ndarray, object, np.ndarray, np.ndarray]:
    """Build geodesic k-medoids codebook."""
    # Build k-NN graph with fixed parameters
    W, _ = build_knn_graph(z, k=k_graph, metric="euclidean", mode="distance", sym="mutual")
    
    # Extract largest connected component
    mask_lcc = largest_connected_component(W)
    if mask_lcc.sum() < W.shape[0]:
        W_lcc = W[mask_lcc][:, mask_lcc]
        z_lcc = z[mask_lcc]
    else:
        W_lcc = W
        z_lcc = z
        mask_lcc = np.ones(len(z), dtype=bool)
    
    # Run k-medoids
    medoids, assign_lcc, _ = fit_kmedoids_optimized(W_lcc, K=K, init="kpp", seed=seed)
    
    # Map back to full dataset
    assign_full = np.full(len(z), fill_value=-1, dtype=np.int32)
    assign_full[mask_lcc] = assign_lcc
    
    centroids = z_lcc[medoids]
    return centroids, assign_full, W_lcc, mask_lcc, medoids


def compute_metrics(model: VAE, z: np.ndarray, centroids: np.ndarray, assign: np.ndarray, 
                   W_lcc: object, mask_lcc: np.ndarray, medoids: np.ndarray, 
                   K: int, device: torch.device, is_geodesic: bool = False) -> Dict[str, float]:
    """Compute all metrics for a codebook."""
    
    # Reconstruction quality
    z_tensor = torch.from_numpy(z).float()
    if is_geodesic:
        valid_mask = assign >= 0
        z_quant = z_tensor.clone()
        if valid_mask.any():
            z_quant[valid_mask] = torch.from_numpy(centroids[assign[valid_mask]]).float()
        recon_mse = compute_reconstruction_mse(model, z_tensor[valid_mask], z_quant[valid_mask], device)
        valid_samples = int(valid_mask.sum())
    else:
        z_quant = torch.from_numpy(centroids[assign]).float()
        recon_mse = compute_reconstruction_mse(model, z_tensor, z_quant, device)
        valid_samples = len(z)
    
    # Perplexity
    valid_assign = assign[assign >= 0] if is_geodesic else assign
    counts = np.bincount(valid_assign, minlength=K)
    probs = counts / max(1, counts.sum())
    nz_probs = probs[probs > 0]
    entropy = -np.sum(nz_probs * np.log(nz_probs + 1e-12))
    perplexity = float(np.exp(entropy))
    
    # Quantization error
    if is_geodesic and mask_lcc.any():
        # Geodesic QE using shortest paths
        D_geo = dijkstra_multi_source(W_lcc, medoids)
        assign_lcc = assign[mask_lcc]
        idx = np.arange(len(assign_lcc))
        dmin = D_geo[assign_lcc, idx]
        finite_mask = np.isfinite(dmin)
        qe = float(np.mean(dmin[finite_mask] ** 2)) if finite_mask.any() else float("inf")
    else:
        # Euclidean QE
        qe = float(np.mean(np.linalg.norm(z - z_quant.numpy(), axis=1) ** 2))
    
    return {
        "reconstruction_mse": float(recon_mse),
        "perplexity": perplexity,
        "quantization_error": qe,
        "valid_samples": valid_samples
    }


def compute_reconstruction_mse(model: VAE, z_orig: torch.Tensor, z_quant: torch.Tensor, device: torch.device) -> float:
    """Compute reconstruction MSE between original and quantized latents."""
    model.eval()
    with torch.no_grad():
        x_orig = torch.sigmoid(model.decoder(z_orig.to(device)))
        x_quant = torch.sigmoid(model.decoder(z_quant.to(device)))
        return torch.nn.functional.mse_loss(x_quant, x_orig).item()


def save_comparison_plot(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    """Save comparison bar plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = ['Euclidean', 'Geodesic']
    colors = ['#1f77b4', '#ff7f0e']
    
    # Plot each metric
    metric_configs = [
        ('reconstruction_mse', 'Reconstruction MSE', 'Reconstruction Quality\n(Lower is Better)'),
        ('perplexity', 'Perplexity', 'Code Usage Diversity\n(Higher is Better)'),
        ('quantization_error', 'Quantization Error', 'Clustering Quality\n(Lower is Better)')
    ]
    
    for i, (metric_key, ylabel, title) in enumerate(metric_configs):
        values = [metrics['euclidean'][metric_key], metrics['geodesic'][metric_key]]
        axes[i].bar(methods, values, color=colors)
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(title)
    
    plt.tight_layout()
    plt.savefig(out_dir / "codebook_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Auto-detect paths
    try:
        paths = auto_detect_paths(args.experiment_dir)
        print(f"Auto-detected paths:")
        print(f"  Checkpoint: {paths['checkpoint_path']}")
        print(f"  Latents: {paths['latents_path']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = Path(args.experiment_dir).name
    out_dir = Path(f"demo_outputs/codebook_comparison_{exp_name}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Using device: {device}")
    
    # Load data
    latents_obj = torch.load(paths["latents_path"], map_location="cpu")
    z = (latents_obj["z"] if isinstance(latents_obj, dict) else latents_obj).float().numpy()
    model = load_vae_model(paths["checkpoint_path"], device)
    
    print(f"[demo] Loaded {len(z)} latents (dim={z.shape[1]})")
    
    # Build codebooks
    print(f"[demo] Building Euclidean codebook (K={args.K})...")
    centroids_euc, assign_euc = build_euclidean_codebook(z, args.K, args.seed)
    
    print(f"[demo] Building geodesic codebook (K={args.K}, k_graph={args.k_graph})...")
    centroids_geo, assign_geo, W_lcc, mask_lcc, medoids = build_geodesic_codebook(z, args.K, args.k_graph, args.seed)
    
    # Compute metrics
    metrics_euc = compute_metrics(model, z, centroids_euc, assign_euc, None, None, None, args.K, device, False)
    metrics_geo = compute_metrics(model, z, centroids_geo, assign_geo, W_lcc, mask_lcc, medoids, args.K, device, True)
    
    metrics = {"euclidean": metrics_euc, "geodesic": metrics_geo}
    
    # Save results
    save_comparison_plot(metrics, out_dir)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    config_used = {
        "experiment_dir": args.experiment_dir, "K": args.K, 
        "k_graph": args.k_graph, "seed": args.seed,
        "latents_path": str(paths["latents_path"]),
        "checkpoint_path": str(paths["checkpoint_path"])
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config_used, f, default_flow_style=False, indent=2)
    
    # Print results
    print(f"\n[demo] Comparison (K={args.K}):")
    print(f"[demo] Euclidean  MSE={metrics_euc['reconstruction_mse']:.6f}  PPL={metrics_euc['perplexity']:.2f}  QE={metrics_euc['quantization_error']:.2f}")
    print(f"[demo] Geodesic   MSE={metrics_geo['reconstruction_mse']:.6f}  PPL={metrics_geo['perplexity']:.2f}  QE={metrics_geo['quantization_error']:.2f}")
    print(f"[demo] Results saved to: {out_dir}")
    
    return out_dir


if __name__ == "__main__":
    main()
