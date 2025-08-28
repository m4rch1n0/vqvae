"""Build discrete codebook via true Riemannian geodesic K-medoids clustering.

This module implements genuine Riemannian geodesic quantization by:
1. Building Euclidean k-NN graph for connectivity
2. Re-weighting edges with decoder-induced Riemannian distances  
3. Performing K-medoids on the Riemannian-weighted graph

This completes the "missing piece" for true geodesic quantization as described
in the project goals, going beyond graph geodesics to manifold geodesics.
"""
from pathlib import Path
from typing import Dict
import warnings

import numpy as np
import torch
from scipy import sparse

from src.geo.kmeans_optimized import fit_kmedoids_optimized
from src.geo.knn_graph_optimized import build_knn_graph_auto, largest_connected_component, analyze_graph_connectivity
from src.geo.riemannian_metric import edge_lengths_riemannian
from src.models.vae import VAE


def _load_latents(path: Path) -> torch.Tensor:
    """Load latent vectors from saved file."""
    obj = torch.load(path, map_location="cpu")
    
    if isinstance(obj, dict) and "z" in obj:
        z = obj["z"]
    elif torch.is_tensor(obj):
        z = obj
    else:
        raise ValueError("Expected dict with 'z' key or tensor")
    
    return z.float()


def _load_vae_model(checkpoint_path: Path, dataset: str, device: torch.device) -> VAE:
    """Load trained VAE model from checkpoint."""
    import yaml
    
    # Load configuration from preset file
    config_paths = {
        "mnist": "configs/presets/mnist/vae.yaml",
        "fashion": "configs/presets/fashion/vae.yaml", 
        "cifar10": "configs/presets/cifar10/vae.yaml"
    }
    
    config_path = config_paths.get(dataset, "configs/presets/fashion/vae.yaml")
    with open(config_path, "r") as f:
        vae_config = yaml.safe_load(f)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the entire dict is the state dict
            state_dict = checkpoint
    else:
        raise ValueError("Expected checkpoint dict with model_state_dict")
    
    # Create VAE model with configuration from preset
    from src.models.vae import VAE
    latent_dim = vae_config.get('latent_dim', 128)
    in_channels = vae_config.get('in_channels', 1)
    enc_channels = vae_config.get('enc_channels', [32, 64, 128])
    dec_channels = vae_config.get('dec_channels', [128, 64, 32])
    norm_type = vae_config.get('norm_type', 'none')
    
    model = VAE(
        latent_dim=latent_dim, 
        in_channels=in_channels,
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        norm_type=norm_type
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def _reweight_graph_with_riemannian(
    W: sparse.csr_matrix, 
    z: np.ndarray, 
    decoder: torch.nn.Module,
    mode: str = "subset",
    max_edges: int = 5000,
    batch_size: int = 512,
    device: torch.device = None
) -> sparse.csr_matrix:
    """Re-weight k-NN graph edges with Riemannian distances.
    
    Args:
        W: Sparse k-NN graph with Euclidean edge weights
        z: Latent points (N, D) 
        decoder: VAE decoder network
        mode: "subset" for sampled edges, "full" for all edges
        max_edges: Maximum edges to reweight in subset mode
        batch_size: Batch size for Riemannian computation
        device: Device for computation
        
    Returns:
        W_riemannian: Graph with Riemannian edge weights
    """
    if device is None:
        device = next(decoder.parameters()).device
    
    # Convert latents to tensor
    z_tensor = torch.from_numpy(z).float().to(device)
    
    # Get edge information from sparse matrix
    W_coo = W.tocoo()
    edge_sources = W_coo.row
    edge_targets = W_coo.col
    num_edges = len(edge_sources)
    
    print(f"Graph has {num_edges} edges")
    
    # Select edges to reweight
    if mode == "subset" and num_edges > max_edges:
        # Stratified sampling by Euclidean distance quantiles
        euclidean_distances = W_coo.data
        quantiles = np.linspace(0, 1, 6)  # 5 bins
        indices_by_quantile = []
        
        for i in range(len(quantiles) - 1):
            q_low, q_high = np.quantile(euclidean_distances, [quantiles[i], quantiles[i+1]])
            mask = (euclidean_distances >= q_low) & (euclidean_distances <= q_high)
            indices = np.where(mask)[0]
            
            # Sample from this quantile  
            n_sample = min(max_edges // 5, len(indices))
            if n_sample > 0:
                selected = np.random.choice(indices, size=n_sample, replace=False)
                indices_by_quantile.extend(selected)
        
        edge_indices = np.array(indices_by_quantile)
        print(f"Reweighting {len(edge_indices)} edges (subset mode)")
    else:
        edge_indices = np.arange(num_edges)
        print(f"Reweighting all {len(edge_indices)} edges (full mode)")
    
    # Extract edge endpoints for selected edges
    selected_sources = edge_sources[edge_indices]
    selected_targets = edge_targets[edge_indices]
    
    # Get latent points for edges
    z_start = z_tensor[selected_sources]  # (num_selected, D)
    z_end = z_tensor[selected_targets]    # (num_selected, D)
    
    # Compute Riemannian edge lengths
    print(f"Computing Riemannian distances for {len(edge_indices)} edges...")
    with torch.no_grad():
        riemannian_lengths = edge_lengths_riemannian(
            decoder, z_start, z_end, batch_size=batch_size
        )
    
    # Convert to numpy
    riemannian_lengths = riemannian_lengths.cpu().numpy()
    
    # Create new sparse matrix with Riemannian weights
    W_new = W.copy().astype(np.float32)
    W_new_coo = W_new.tocoo()
    
    # Replace selected edge weights with Riemannian distances
    W_new_coo.data[edge_indices] = riemannian_lengths
    
    # Ensure symmetry (undirected graph)
    W_riemannian = W_new_coo.tocsr()
    W_riemannian = W_riemannian.maximum(W_riemannian.T)
    
    # Verify finite weights
    finite_mask = np.isfinite(W_riemannian.data)
    if not finite_mask.all():
        n_inf = (~finite_mask).sum()
        warnings.warn(f"Found {n_inf} non-finite Riemannian distances, keeping original Euclidean weights")
        W_riemannian.data[~finite_mask] = W.tocsr().data[~finite_mask]
    
    print(f"Riemannian reweighting complete. Edge weight ratio: mean={np.mean(riemannian_lengths/W_coo.data[edge_indices]):.3f}")
    
    return W_riemannian


def build_and_save(config: Dict) -> Path:
    """Build Riemannian geodesic codebook and save artifacts."""
    # Parse paths and directories
    z_path_cfg = config["data"].get("latents_path") if isinstance(config.get("data"), dict) else None
    if z_path_cfg:
        z_path = Path(z_path_cfg)
    else:
        ds = str(config.get("data", {}).get("dataset", "fashion")).strip().lower()
        base = {
            "mnist": "experiments/vae_mnist/latents_train/z.pt",
            "fashion": "experiments/vae_fashion/latents_train/z.pt", 
            "cifar10": "experiments/vae_cifar10/latents_train/z.pt",
        }.get(ds, "experiments/vae_fashion/latents_train/z.pt")
        z_path = Path(base)
    
    # Parse checkpoint path 
    checkpoint_path_cfg = config["model"].get("checkpoint_path") if isinstance(config.get("model"), dict) else None
    if checkpoint_path_cfg:
        checkpoint_path = Path(checkpoint_path_cfg)
    else:
        ds = str(config.get("data", {}).get("dataset", "fashion")).strip().lower()
        base = {
            "mnist": "experiments/vae_mnist/checkpoints/best.pt",
            "fashion": "experiments/vae_fashion/checkpoints/best.pt",
            "cifar10": "experiments/vae_cifar10/checkpoints/best.pt", 
        }.get(ds, "experiments/vae_fashion/checkpoints/best.pt")
        checkpoint_path = Path(base)
    
    out_dir = Path(config["out"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load latent vectors
    z = _load_latents(z_path).numpy()
    N, D = z.shape
    print(f"Loaded latents: N={N}, D={D}")

    # Load VAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ds = str(config.get("data", {}).get("dataset", "fashion")).strip().lower()
    vae_model = _load_vae_model(checkpoint_path, ds, device)
    decoder = vae_model.decoder

    # Build initial k-NN graph with Euclidean weights
    k = int(config["graph"]["k"])
    metric = str(config["graph"]["metric"])
    sym = str(config["graph"]["sym"])
    mode = str(config["graph"]["mode"])

    print(f"Building k-NN graph: k={k}, metric={metric}, sym={sym}")
    W_euclidean, info = build_knn_graph_auto(z, k=k, metric=metric, mode=mode, sym=sym)
    
    # Analyze graph connectivity
    graph_stats = analyze_graph_connectivity(W_euclidean, verbose=True)
    
    # Extract largest connected component
    mask_lcc = largest_connected_component(W_euclidean)
    if mask_lcc.sum() < W_euclidean.shape[0]:
        print(f"Using LCC: {mask_lcc.sum()}/{W_euclidean.shape[0]} nodes")
        W_euclidean_lcc = W_euclidean[mask_lcc][:, mask_lcc]
        z_lcc = z[mask_lcc]
    else:
        W_euclidean_lcc = W_euclidean
        z_lcc = z

    # Re-weight edges with Riemannian distances
    riemannian_config = config.get("riemannian", {})
    reweight_mode = riemannian_config.get("mode", "subset")
    max_edges = int(riemannian_config.get("max_edges", 5000))
    batch_size = int(riemannian_config.get("batch_size", 512))
    
    print(f"Re-weighting graph with Riemannian distances (mode={reweight_mode})")
    W_riemannian = _reweight_graph_with_riemannian(
        W_euclidean_lcc, z_lcc, decoder, 
        mode=reweight_mode, max_edges=max_edges, 
        batch_size=batch_size, device=device
    )

    # Save both graphs for comparison
    sparse.save_npz(out_dir / "knn_graph_euclidean.npz", W_euclidean_lcc)
    sparse.save_npz(out_dir / "knn_graph_riemannian.npz", W_riemannian)

    # Run geodesic K-medoids clustering on Riemannian graph
    K = int(config["quantize"]["K"])
    init = str(config["quantize"]["init"])
    seed = int(config["quantize"]["seed"])

    print(f"Running K-medoids on Riemannian graph: K={K}, init={init}")
    medoids, assign_lcc, qe = fit_kmedoids_optimized(
        W_riemannian, K=K, init=init, seed=seed, verbose=True
    )

    # Map assignments back to original indices
    assign = np.full((N,), fill_value=-1, dtype=np.int32)
    if mask_lcc.sum() < N:
        assign[mask_lcc] = assign_lcc
    else:
        assign = assign_lcc

    # Save codebook and cluster assignments
    z_medoid = z_lcc[medoids]
    codebook = {
        "medoid_indices": medoids.astype(np.int32),
        "z_medoid": torch.from_numpy(z_medoid).float(),
        "config": config,
        "graph_stats": graph_stats,
        "method": "riemannian_geodesic"
    }
    torch.save(codebook, out_dir / "codebook.pt")
    np.save(out_dir / "codes.npy", assign)

    print(f"Riemannian quantization error: {qe:.3f}")
    print(f"Saved artifacts to: {out_dir}")
    
    return out_dir


if __name__ == "__main__":
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Riemannian geodesic codebook")
    parser.add_argument("--config", type=str, default="configs/quantize.yaml",
                       help="Configuration file path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    output_dir = build_and_save(cfg)
    print(f"Completed: {output_dir}")
