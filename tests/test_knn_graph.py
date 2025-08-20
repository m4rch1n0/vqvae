import numpy as np
import scipy.sparse as sp
import pytest

from src.geo.knn_graph import build_knn_graph

def random_latents(N=200, D=8, scale=1.0, seed=0):
    r = np.random.RandomState(seed)
    return (r.randn(N, D) * scale).astype(np.float32)

def test_empty_input():
    """Test kNN graph construction with empty input."""
    z = np.empty((0, 8), dtype=np.float32)
    W, nbrs = build_knn_graph(z, k=10)
    assert isinstance(W, sp.csr_matrix)
    assert W.shape == (0, 0)
    assert nbrs["distances"].shape == (0, 0)
    assert nbrs["indices"].shape == (0, 0)

def test_edge_cases_and_shapes():
    """Test edge cases: single point, k=0, and k capping."""
    # Single point
    z = random_latents(N=1, D=8)
    W, nbrs = build_knn_graph(z, k=10)
    assert W.shape == (1, 1)
    assert float(W.diagonal().sum()) == 0.0
    assert nbrs["distances"].shape == (1, 0)
    assert nbrs["indices"].shape == (1, 0)
    
    # k=0 case
    z = random_latents(25, 4)
    W, nbrs = build_knn_graph(z, k=0)
    assert W.nnz == 0
    assert nbrs["indices"].shape == (25, 0)
    
    # k capping at N-1
    N, D = 5, 4
    z = random_latents(N, D)
    W, nbrs = build_knn_graph(z, k=10)  # k > N-1
    assert nbrs["distances"].shape == (N, N - 1)
    assert nbrs["indices"].shape == (N, N - 1)
    assert W.shape == (N, N)

def test_no_self_in_neighbors():
    """Test that self-neighbors are excluded from neighbor lists."""
    N, D = 50, 6
    z = random_latents(N, D)
    _, nbrs = build_knn_graph(z, k=5)
    idx = nbrs["indices"]
    rows = np.repeat(np.arange(N), idx.shape[1]).reshape(N, idx.shape[1])
    assert not np.any(idx == rows), "Self neighbors must be removed"

def test_symmetry_and_no_diagonal():
    """Test that graph is symmetric with zero diagonal."""
    z = random_latents(200, 8)
    W, _ = build_knn_graph(z, k=10)
    assert (W - W.T).nnz == 0
    assert float(W.diagonal().sum()) == 0.0

def test_modes_weights():
    """Test different weight modes: distance vs connectivity."""
    z = random_latents(120, 6)
    Wd, _ = build_knn_graph(z, k=8, mode="distance")
    Wc, _ = build_knn_graph(z, k=8, mode="connectivity")

    assert isinstance(Wd, sp.csr_matrix) and isinstance(Wc, sp.csr_matrix)

    if Wc.nnz > 0:
        assert np.allclose(Wc.data, 1.0), "Connectivity weights must be 1"

    if Wd.nnz > 0:
        assert np.all(Wd.data >= 0.0), "Distance weights must be non-negative"

# Note: Connected component and metric tests moved to visualizations/knn_experiments.py
# for better educational value through visual exploration
