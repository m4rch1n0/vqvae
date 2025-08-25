"""Build discrete codebook via geodesic K-medoids clustering."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from scipy import sparse

from src.geo.kmeans_precomputed import fit_kmedoids_precomputed
from src.geo.knn_graph import build_knn_graph, largest_connected_component


def _load_latents(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    
    if isinstance(obj, dict) and "z" in obj:
        z = obj["z"]
    elif torch.is_tensor(obj):
        z = obj
    else:
        raise ValueError(f"Expected dict with 'z' key or tensor")
    
    return z.float()


def build_and_save(config: Dict[str, Any]) -> Path:
    """Build geodesic codebook and save artifacts."""
    z_path = Path(config["data"]["latents_path"])
    out_dir = Path(config["out"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load latent vectors
    z = _load_latents(z_path).numpy()
    N, D = z.shape
    print(f"Loaded latents: N={N}, D={D}")

    # Build k-NN graph
    k = int(config["graph"]["k"])
    metric = str(config["graph"]["metric"])
    sym = str(config["graph"]["sym"])
    mode = str(config["graph"]["mode"])

    W, info = build_knn_graph(z, k=k, metric=metric, mode=mode, sym=sym)
    print(f"Built k-NN graph: k={k}, edges={W.nnz}")
    
    mask_lcc = largest_connected_component(W)
    if mask_lcc.sum() < W.shape[0]:
        print(f"Using LCC: {mask_lcc.sum()}/{W.shape[0]} nodes")
        W = W[mask_lcc][:, mask_lcc]
        z_lcc = z[mask_lcc]
    else:
        z_lcc = z

    sparse.save_npz(out_dir / "knn_graph.npz", W)

    # Run geodesic K-medoids clustering
    K = int(config["quantize"]["K"])
    init = str(config["quantize"]["init"])
    seed = int(config["quantize"]["seed"])

    medoids, assign_lcc, distance_matrix = fit_kmedoids_precomputed(
        W, K=K, init=init, seed=seed, chunk_size=1000
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
    }
    torch.save(codebook, out_dir / "codebook.pt")
    np.save(out_dir / "codes.npy", assign)

    dmin = distance_matrix[medoids].min(axis=0)
    qe = float(np.sum(dmin[np.isfinite(dmin)] ** 2))
    print(f"Quantization error: {qe:.3f}")
    print(f"Saved artifacts to: {out_dir}")
    
    return out_dir


if __name__ == "__main__":
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/quantize.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    output_dir = build_and_save(cfg)
    print(f"Completed: {output_dir}")
