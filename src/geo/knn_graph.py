"""
Builds an undirected k-nearest neighbors (k-NN) graph from latent vectors.

Full documentation: see docs/knn_graph.md
"""
from typing import Tuple, Dict
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

def build_knn_graph(z: np.ndarray, k: int = 10, metric = "euclidean", mode = "distance", sym = "mutual") -> Tuple[sparse.csr_matrix, Dict[str, np.ndarray]]:
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

def largest_connected_component(W: sparse.csr_matrix) -> np.ndarray:
    """Return boolean mask of nodes in the largest connected component."""
    n_comp, labels = connected_components(W, directed=False)
    if n_comp <= 1:
        return np.ones(W.shape[0], dtype=bool)
    # take the most frequent label
    counts = np.bincount(labels)
    lcc = np.argmax(counts)
    return labels == lcc
