"""
Graph-based geodesic K-medoids clustering.

Avoids full distance matrix computation by using graph structure directly.

Full documentation: see docs/geo/kmeans_optimized.md
""" 
from typing import List, Tuple
import numpy as np
from scipy import sparse
from src.geo.geo_shortest_paths import dijkstra_multi_source, dijkstra_single_source


def kpp_initialization_graph(W: sparse.spmatrix, K: int, seed: int = 42) -> List[int]:
    """
    K-means++ initialization using graph distances.

    Iteratively computes shortest paths from each center to avoid
    full distance matrix computation.

    Args:
        W: Sparse adjacency matrix (CSR format preferred)
        K: Number of clusters
        seed: Random seed for reproducibility

    Returns:
        List of medoid indices
    """
    N = W.shape[0]
    rng = np.random.RandomState(seed)
    
    # Choose first center randomly
    centers = [int(rng.randint(0, N))]
    
    # Keep track of minimum distances to any center
    d_min = np.full(N, np.inf, dtype=np.float32)
    
    print(f"[kpp] Selecting {K} centers among {N} nodes")
    
    for i in range(1, K):
        # Update distances from new center
        last_center = centers[-1]
        distances_from_center = dijkstra_single_source(W, last_center, dtype=np.float32)
        d_min = np.minimum(d_min, distances_from_center)
        
        # Handle infinite distances (disconnected components)
        finite_mask = np.isfinite(d_min)
        if finite_mask.any():
            max_finite = np.max(d_min[finite_mask])
            d_min_safe = np.where(finite_mask, d_min, max_finite * 2.0)
        else:
            # Fallback if all distances are infinite
            d_min_safe = np.ones_like(d_min)
        
        # Compute probabilities for k-means++ (squared distances)
        probs = d_min_safe ** 2
        probs[centers] = 0.0  # Don't select existing centers
        
        if probs.sum() > 0:
            probs /= probs.sum()
            next_center = int(rng.choice(N, p=probs))
        else:
            # Fallback: random choice from remaining candidates
            candidates = [i for i in range(N) if i not in centers]
            if candidates:
                next_center = int(rng.choice(candidates))
            else:
                print(f"Warning: Could not find {K} valid centers, stopping at {len(centers)}")
                break
        
        centers.append(next_center)
    
    print(f"[kpp] Selected {len(centers)} centers")
    return centers


def assign_points_to_medoids(W: sparse.spmatrix, medoids: np.ndarray) -> np.ndarray:
    """
    Assign points to nearest medoids using multi-source Dijkstra.

    Computes shortest paths from all medoids simultaneously,
    then assigns each point to its closest medoid.

    Args:
        W: Sparse adjacency matrix
        medoids: Array of medoid indices

    Returns:
        Array of cluster assignments (shape: N,)
    """
    N = W.shape[0]
    K = len(medoids)
    
    print(f"[assign] {N} points to {K} medoids")
    
    # Compute distances from all medoids to all points in one shot
    distances_from_medoids = dijkstra_multi_source(W, medoids, dtype=np.float32)
    
    # Assign each point to nearest medoid
    assign = distances_from_medoids.argmin(axis=0).astype(int)
    
    # Count assignments per cluster
    cluster_counts = np.bincount(assign, minlength=K)
    print(f"[assign] sizes min={cluster_counts.min()}, max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")
    
    return assign


def compute_quantization_error(W: sparse.spmatrix, medoids: np.ndarray, assign: np.ndarray) -> float:
    """
    Compute quantization error as sum of squared geodesic distances.

    Measures clustering quality by summing squared distances from each point
    to its assigned medoid.

    Args:
        W: Sparse adjacency matrix
        medoids: Array of medoid indices
        assign: Array of cluster assignments

    Returns:
        Quantization error (float)
    """
    # Compute distances from medoids to all points
    distances_from_medoids = dijkstra_multi_source(W, medoids, dtype=np.float32)
    
    # For each point, get distance to its assigned medoid
    point_indices = np.arange(len(assign))
    distances_to_assigned = distances_from_medoids[assign, point_indices]
    
    # Sum of squared distances (finite distances only)
    finite_mask = np.isfinite(distances_to_assigned)
    if finite_mask.any():
        qe = float(np.sum(distances_to_assigned[finite_mask] ** 2))
    else:
        qe = float('inf')
    
    return qe


def fit_kmedoids_optimized(
    W: sparse.spmatrix,
    K: int = 512,
    init: str = "kpp",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Graph-based geodesic K-medoids clustering.

    Performs K-medoids without computing full distance matrix.
    Uses graph structure directly for all computations.

    Args:
        W: Sparse adjacency matrix (CSR format preferred)
        K: Number of clusters
        init: Initialization method ("kpp" for K-means++, "random" for random)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (medoid_indices, assignments, quantization_error)
    """
    N = W.shape[0]
    
    print(f"[kmedoids] N={N}, K={K}, edges={W.nnz}, avg_deg={W.nnz/max(1,N):.1f}")
    
    # Initialize medoids
    if init == "kpp":
        medoids = np.array(kpp_initialization_graph(W, K, seed=seed), dtype=int)
    elif init == "random":
        rng = np.random.RandomState(seed)
        medoids = rng.choice(N, size=min(K, N), replace=False).astype(int)
    else:
        raise ValueError("init must be 'kpp' or 'random'")
    
    # Assign points to medoids
    assign = assign_points_to_medoids(W, medoids)
    
    # Compute quantization error
    qe = compute_quantization_error(W, medoids, assign)
    
    print(f"[kmedoids] Done: clusters={len(medoids)}, qe={qe:.3f}")
    
    return medoids, assign, qe


def fit_kmedoids_with_connectivity_check(
    W: sparse.spmatrix,
    K: int = 512,
    init: str = "kpp",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Fit K-medoids with connectivity analysis.

    Analyzes graph components before clustering and returns metadata
    about connectivity and clustering results.

    Returns:
        Tuple of (medoid_indices, assignments, quantization_error, metadata)
    """
    from scipy.sparse.csgraph import connected_components
    
    N = W.shape[0]
    
    # Analyze connectivity
    n_components, labels = connected_components(W, directed=False)
    
    metadata = {
        "n_nodes": N,
        "n_edges": W.nnz,
        "n_components": n_components,
        "largest_component_size": np.bincount(labels).max() if n_components > 0 else N
    }
    
    print(f"[graph] components={n_components}, largest={metadata['largest_component_size']}")
    
    # Run optimized clustering
    medoids, assign, qe = fit_kmedoids_optimized(W, K=K, init=init, seed=seed)
    
    # Add performance metadata
    metadata.update({
        "n_medoids": len(medoids),
        "quantization_error": qe,
        "method": "optimized_kmedoids"
    })
    
    return medoids, assign, qe, metadata


if __name__ == "__main__":
    """Test optimized K-medoids clustering."""
    import numpy as np
    from src.geo.knn_graph_optimized import build_knn_graph_auto

    # Generate test data
    np.random.seed(42)
    z = np.random.randn(1000, 64).astype(np.float32)

    # Build k-NN graph and test clustering
    W, _ = build_knn_graph_auto(z, k=10)
    medoids, assign, qe = fit_kmedoids_optimized(W, K=50, init="kpp", seed=42)

    print(f"\nTest completed:")
    print(f"  Medoids: {len(medoids)}")
    print(f"  Assignments: {len(assign)} points")
    print(f"  Quantization error: {qe:.3f}")
    print(f"  Cluster sizes: {np.bincount(assign)}")

