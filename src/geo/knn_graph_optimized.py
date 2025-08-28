"""
Optimized k-NN graph construction with automatic sklearn/FAISS selection.

Automatically chooses the fastest method based on dataset size:
- sklearn for small-medium datasets (< 50k samples)
- FAISS for large datasets (>= 50k samples)

Full documentation: see docs/models/knn_graph.md
"""
from typing import Tuple, Dict, Optional
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
import faiss


def build_knn_graph_sklearn(z: np.ndarray, k: int = 10, metric: str = "euclidean", 
                           mode: str = "distance", sym: str = "mutual") -> Tuple[sparse.csr_matrix, Dict[str, np.ndarray]]:
    """Build k-NN graph using scikit-learn (original implementation)."""
    assert z.ndim == 2, "z must be (N,D)"
    N = z.shape[0]
    if N == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32), {"distances": np.empty((0, 0), np.float32), "indices": np.empty((0, 0), dtype=int)}

    # cap k to N-1 (no self)
    k_eff = max(0, min(k, N - 1))
    if k_eff == 0:
        W = sparse.csr_matrix((N, N), dtype=np.float32)
        return W, {"distances": np.empty((N, 0), np.float32), "indices": np.empty((N, 0), dtype=int)}

    n_neighbors = min(k_eff + 1, N)  # query one extra (self)
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric=metric)
    nn.fit(z)
    distances, indices = nn.kneighbors(z, return_distance=True)

    # drop self (distance 0). fallback: drop row-wise argmin
    if (indices[:, 0] == np.arange(N)).all():
        distances, indices = distances[:, 1:], indices[:, 1:]
    else:
        argmin = np.argmin(distances, axis=1)
        mask = np.ones_like(distances, dtype=bool)
        mask[np.arange(N), argmin] = False
        distances = distances[mask].reshape(N, -1)
        indices = indices[mask].reshape(N, -1)

    data = distances.ravel() if mode == "distance" else np.ones_like(distances).ravel()
    rows = np.repeat(np.arange(N), indices.shape[1])
    cols = indices.ravel()

    W = sparse.csr_matrix((data.astype(np.float32), (rows, cols)), shape=(N, N))
    if sym == "mutual":
        W = W.minimum(W.T)              # undirected
    elif sym == "union":
        W = W.maximum(W.T)              # undirected
    else:
        raise ValueError(f"Invalid symmetry mode: {sym}")
    W.setdiag(0.0)                  # ensure no diagonal entries
    W.eliminate_zeros()
    return W, {"distances": distances.astype(np.float32, copy=False), "indices": indices}


def build_knn_graph_faiss(z: np.ndarray, k: int = 10, metric: str = "euclidean",
                         mode: str = "distance", sym: str = "mutual") -> Tuple[sparse.csr_matrix, Dict[str, np.ndarray]]:
    """Build k-NN graph using FAISS for large datasets."""
    N, D = z.shape
    
    # Create FAISS index
    if metric == "euclidean":
        index = faiss.IndexFlatL2(D)
    elif metric == "cosine":
        index = faiss.IndexFlatIP(D)  # Inner product for cosine after normalization
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)  # Normalize for cosine
    else:
        raise ValueError(f"FAISS metric {metric} not implemented")
    
    # Ensure data is contiguous and float32
    z_faiss = np.ascontiguousarray(z.astype(np.float32))
    index.add(z_faiss)
    
    # Search for k+1 neighbors (including self)
    k_search = min(k + 1, N)
    distances, indices = index.search(z_faiss, k_search)
    
    # Remove self-connections (first column should be self with distance 0)
    if indices.shape[1] > 1:
        # Check if first column is self-connections
        if (indices[:, 0] == np.arange(N)).all():
            distances = distances[:, 1:]  # Remove self
            indices = indices[:, 1:]      # Remove self
        else:
            # Fallback: find and remove self-connections
            self_mask = indices == np.arange(N)[:, None]
            if self_mask.any():
                # Remove self-connections by masking
                for i in range(N):
                    self_idx = np.where(self_mask[i])[0]
                    if len(self_idx) > 0:
                        # Remove first self-connection found
                        remove_idx = self_idx[0]
                        distances[i] = np.concatenate([distances[i][:remove_idx], distances[i][remove_idx+1:]])
                        indices[i] = np.concatenate([indices[i][:remove_idx], indices[i][remove_idx+1:]])
    
    # Convert to CSR matrix
    actual_k = indices.shape[1]
    data = distances.ravel() if mode == "distance" else np.ones(N * actual_k, dtype=np.float32)
    rows = np.repeat(np.arange(N), actual_k)
    cols = indices.ravel()
    
    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    
    # Make symmetric (mutual or union connections)
    if sym == "mutual":
        W = W.minimum(W.T)              # undirected
    elif sym == "union":
        W = W.maximum(W.T)              # undirected
    else:
        raise ValueError(f"Invalid symmetry mode: {sym}")
    
    W.setdiag(0.0)                  # ensure no diagonal entries
    W.eliminate_zeros()
    
    return W, {"distances": distances, "indices": indices}


def build_knn_graph_auto(z: np.ndarray, k: int = 10, metric: str = "euclidean",
                        mode: str = "distance", sym: str = "mutual", 
                        force_method: Optional[str] = None, 
                        size_threshold: int = 50000) -> Tuple[sparse.csr_matrix, Dict[str, np.ndarray]]:
    """
    Automatically choose between sklearn and FAISS based on dataset size.
    
    Args:
        z: Input data (N, D)
        k: Number of neighbors
        metric: Distance metric ('euclidean', 'cosine')
        mode: Edge weight mode ('distance' or 'connectivity')
        sym: Symmetry mode ('mutual' or 'union')
        force_method: Force specific method ('sklearn' or 'faiss')
        size_threshold: Switch to FAISS above this many samples
        
    Returns:
        Tuple of (sparse_matrix, info_dict)
    """
    N = z.shape[0]
    
    # Determine method
    if force_method == "sklearn":
        method = "sklearn"
    elif force_method == "faiss":
        method = "faiss"
    else:
        # Auto-select based on size
        if N >= size_threshold:
            method = "faiss"
        else:
            method = "sklearn"
    
    print(f"Building k-NN graph: N={N}, k={k}, method={method}")
    
    # Build graph with selected method
    if method == "faiss":
        return build_knn_graph_faiss(z, k=k, metric=metric, mode=mode, sym=sym)
    else:
        return build_knn_graph_sklearn(z, k=k, metric=metric, mode=mode, sym=sym)


def largest_connected_component(W: sparse.csr_matrix) -> np.ndarray:
    """Return boolean mask of nodes in the largest connected component."""
    n_comp, labels = connected_components(W, directed=False)
    if n_comp <= 1:
        return np.ones(W.shape[0], dtype=bool)
    # take the most frequent label
    counts = np.bincount(labels)
    lcc = np.argmax(counts)
    return labels == lcc


def analyze_graph_connectivity(W: sparse.csr_matrix, verbose: bool = True) -> Dict:
    """Analyze k-NN graph connectivity and return statistics."""
    N = W.shape[0]
    n_components, labels = connected_components(W, directed=False)
    
    if n_components > 1:
        component_sizes = np.bincount(labels)
        largest_size = component_sizes.max()
        connectivity_ratio = largest_size / N
    else:
        component_sizes = np.array([N])
        largest_size = N
        connectivity_ratio = 1.0
    
    # Compute average degree
    degrees = np.array(W.sum(axis=1)).flatten()
    avg_degree = degrees.mean()
    
    stats = {
        "n_nodes": N,
        "n_edges": W.nnz,
        "n_components": n_components,
        "largest_component_size": largest_size,
        "connectivity_ratio": connectivity_ratio,
        "avg_degree": avg_degree,
        "min_degree": degrees.min(),
        "max_degree": degrees.max()
    }
    
    if verbose:
        print(f"Graph connectivity analysis:")
        print(f"  Nodes: {N}, Edges: {W.nnz}, Avg degree: {avg_degree:.1f}")
        print(f"  Components: {n_components}, Largest: {largest_size} ({100*connectivity_ratio:.1f}%)")
        if n_components > 1:
            print(f"Graph is disconnected - applying LCC filter")
    
    return stats


# Backward compatibility - use auto method by default
def build_knn_graph(z: np.ndarray, k: int = 10, metric: str = "euclidean",
                   mode: str = "distance", sym: str = "mutual") -> Tuple[sparse.csr_matrix, Dict[str, np.ndarray]]:
    """
    Build k-NN graph with automatic method selection (backward compatible).
    
    This function maintains compatibility with existing code while providing
    automatic optimization based on dataset size.
    """
    return build_knn_graph_auto(z, k=k, metric=metric, mode=mode, sym=sym)


if __name__ == "__main__":
    """Test optimized k-NN graph construction."""
    import time
    
    # Test with different sizes
    test_sizes = [1000, 10000, 60000]
    
    for N in test_sizes:
        print(f"\n🔹 Testing N={N}")
        
        # Generate test data
        np.random.seed(42)
        z = np.random.randn(N, 64).astype(np.float32)
        
        # Test auto method
        start_time = time.time()
        W, info = build_knn_graph_auto(z, k=10)
        auto_time = time.time() - start_time
        
        # Analyze connectivity
        stats = analyze_graph_connectivity(W)
        
        print(f"  Time: {auto_time:.2f}s")
        print(f"  Edges: {W.nnz}, Connectivity: {100*stats['connectivity_ratio']:.1f}%")
