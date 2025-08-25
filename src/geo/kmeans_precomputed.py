"""
K-medoids clustering using precomputed geodesic distances.
Computes distances in chunks to handle large graphs efficiently.

Full documentation: see docs/geo/kmeans_precomputed.md
"""
from typing import List, Tuple
import numpy as np
from scipy import sparse
import psutil
from src.geo.geo_shortest_paths import dijkstra_multi_source


def _check_ram_requirements(N: int) -> None:
    required_gb = (N * N * 4) / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    if required_gb > available_gb * 0.8: # tolerance for other processes
        raise MemoryError(f"Insufficient RAM: need {required_gb:.1f} GB, have {available_gb:.1f} GB")


def _kpp_init_precomputed(distance_matrix: np.ndarray, K: int, seed: int = 42) -> List[int]:
    """K-means++ initialization using precomputed distances."""
    N = distance_matrix.shape[0]
    rng = np.random.RandomState(seed)
    centers: List[int] = [int(rng.randint(0, N))]
    
    for _ in range(1, K):
        center_indices = np.array(centers, dtype=int)
        dmin = distance_matrix[center_indices].min(axis=0)
        
        # Handle infinite distances
        max_finite = np.max(dmin[np.isfinite(dmin)]) if np.isfinite(dmin).any() else 1.0
        dmin = np.where(np.isfinite(dmin), dmin, max_finite * 2.0)
        
        probs = dmin ** 2
        probs[center_indices] = 0.0
        
        if probs.sum() > 0:
            probs /= probs.sum()
            next_center = int(rng.choice(N, p=probs))
        else:
            # Fallback: random choice from remaining candidates
            candidates = [i for i in range(N) if i not in centers]
            next_center = int(rng.choice(candidates))
        centers.append(next_center)
    
    return centers


def _compute_distance_matrix_chunked(W: sparse.spmatrix, chunk_size: int = 1000) -> np.ndarray:
    """Compute pairwise geodesic distances in chunks to manage memory."""
    N = W.shape[0]
    D = np.full((N, N), fill_value=np.inf, dtype=np.float32)
    
    print(f"Computing {N}x{N} distance matrix...")
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sources = list(range(start, end))
        D_chunk = dijkstra_multi_source(W, sources, dtype=np.float32)
        D[start:end] = D_chunk
        
        # Progress update every 5 chunks
        if (start // chunk_size) % 5 == 0:
            print(f"Progress: {end}/{N} ({100*end/N:.0f}%)")
    
    return D


def fit_kmedoids_precomputed(
    W: sparse.spmatrix,
    K: int = 512,
    init: str = "kpp",
    seed: int = 42,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = W.shape[0]
    _check_ram_requirements(N)
    distance_matrix = _compute_distance_matrix_chunked(W, chunk_size=chunk_size)

    if init == "kpp":
        medoids = np.array(_kpp_init_precomputed(distance_matrix, K, seed=seed), dtype=int)
    elif init == "random":
        rng = np.random.RandomState(seed)
        medoids = rng.choice(N, K, replace=False).astype(int)
    else:
        raise ValueError("init must be 'kpp' or 'random'")

    # Assign points to nearest medoids
    assign = distance_matrix[medoids].argmin(axis=0).astype(int)
    print(f"K-medoids completed: {K} clusters, {N} points")
    
    return medoids, assign, distance_matrix


