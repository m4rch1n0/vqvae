"""K-medoids clustering on graph distances using geodesic metrics."""
from typing import List, Tuple
import numpy as np
from scipy import sparse
from src.geo.geo_shortest_paths import dijkstra_multi_source


def _kpp_init_on_graph(W: sparse.spmatrix, K: int, seed: int = 42) -> List[int]:
    """K-means++ initialization using geodesic distances."""
    assert K > 0 and W.shape[0] >= K, f"K={K} must be in range [1, {W.shape[0]}]"
    
    rng = np.random.RandomState(seed)
    N = W.shape[0]
    centers: List[int] = []
    
    centers.append(int(rng.randint(0, N)))
    
    for _ in range(1, K):
        D = dijkstra_multi_source(W, centers)  # (len(centers), N)
        dmin = D.min(axis=0)  # distance to nearest center
        
        # handle disconnected components
        if not np.isfinite(dmin).all():
            max_finite = np.max(dmin[np.isfinite(dmin)]) if np.isfinite(dmin).any() else 1.0
            dmin = np.where(np.isfinite(dmin), dmin, max_finite * 2.0)
        
        probs = dmin ** 2
        if len(centers) > 0:
            probs = probs.copy()
            probs[np.array(centers, dtype=int)] = 0.0

        s = probs.sum()
        if np.isfinite(s) and s > 0:
            probs /= s
            next_center = int(rng.choice(N, p=probs))
        else:
            # fallback: random choice from remaining nodes
            candidates = [i for i in range(N) if i not in centers]
            next_center = int(rng.choice(candidates))
        
        centers.append(next_center)
    
    return centers


def fit_kmedoids_graph(W: sparse.spmatrix, K: int = 512, init: str = "kpp", seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """K-medoids clustering on weighted graph using geodesic distances."""
    assert sparse.isspmatrix(W), "W must be a scipy sparse matrix"
    N = W.shape[0]
    assert 0 < K <= N, f"K={K} must be in range [1, {N}]"
    
    if init == "random":
        rng = np.random.RandomState(seed)
        medoids = rng.choice(N, K, replace=False).astype(int)
    elif init == "kpp":
        medoids = np.array(_kpp_init_on_graph(W, K, seed=seed), dtype=int)
    else:
        raise ValueError(f"init='{init}' not supported. Use 'random' or 'kpp'")
    
    distances = dijkstra_multi_source(W, medoids)  # (K, N)
    assign = distances.argmin(axis=0).astype(int)
    
    return medoids, assign
