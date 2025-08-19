import numpy as np
import scipy.sparse as sp
import pytest

from src.geo.knn_graph import build_knn_graph

def random_latents(N=200, D=8, scale=1.0, seed=0):
    r = np.random.RandomState(seed)
    return (r.randn(N, D) * scale).astype(np.float32)

def test_empty_input():
    z = np.empty((0, 8), dtype=np.float32)
    W, nbrs = build_knn_graph(z, k=10)
    assert isinstance(W, sp.csr_matrix)
    assert W.shape == (0, 0)
    assert nbrs["distances"].shape == (0, 0)
    assert nbrs["indices"].shape == (0, 0)

def test_single_point():
    z = random_latents(N=1, D=8)
    W, nbrs = build_knn_graph(z, k=10)
    assert W.shape == (1, 1)
    assert float(W.diagonal().sum()) == 0.0
    assert nbrs["distances"].shape == (1, 0)
    assert nbrs["indices"].shape == (1, 0)

def test_k_capped_and_shapes():
    N, D = 5, 4
    z = random_latents(N, D)
    W, nbrs = build_knn_graph(z, k=10)  # k > N-1
    assert nbrs["distances"].shape == (N, N - 1)
    assert nbrs["indices"].shape == (N, N - 1)
    assert W.shape == (N, N)

def test_no_self_in_neighbors():
    N, D = 50, 6
    z = random_latents(N, D)
    _, nbrs = build_knn_graph(z, k=5)
    idx = nbrs["indices"]
    rows = np.repeat(np.arange(N), idx.shape[1]).reshape(N, idx.shape[1])
    assert not np.any(idx == rows), "Self neighbors must be removed"

def test_symmetry_and_no_diagonal():
    z = random_latents(200, 8)
    W, _ = build_knn_graph(z, k=10)
    assert (W - W.T).nnz == 0
    assert float(W.diagonal().sum()) == 0.0

def test_modes_weights():
    z = random_latents(120, 6)
    Wd, _ = build_knn_graph(z, k=8, mode="distance")
    Wc, _ = build_knn_graph(z, k=8, mode="connectivity")

    assert isinstance(Wd, sp.csr_matrix) and isinstance(Wc, sp.csr_matrix)

    if Wc.nnz > 0:
        assert np.allclose(Wc.data, 1.0), "Connectivity weights must be 1"

    if Wd.nnz > 0:
        assert np.all(Wd.data >= 0.0), "Distance weights must be non-negative"

def test_two_blobs_no_cross_edges():
    # Two far blobs; mutual k-NN must not create inter-blob edges.
    N1, N2, D = 60, 40, 4
    r = np.random.RandomState(0)
    A = (r.randn(N1, D) * 0.2).astype(np.float32)          # near 0
    B = (r.randn(N2, D) * 0.2 + 100.0).astype(np.float32)  # far away
    z = np.vstack([A, B])

    W, _ = build_knn_graph(z, k=3, mode="distance")
    # Off-diagonal blocks must be empty
    assert W[:N1, N1:].nnz == 0 and W[N1:, :N1].nnz == 0


def test_largest_connected_component_two_blobs():
    N1, N2, D = 30, 20, 4
    r = np.random.RandomState(1)
    A = (r.randn(N1, D) * 0.2).astype(np.float32)
    B = (r.randn(N2, D) * 0.2 + 50.).astype(np.float32)
    z = np.vstack([A, B])
    W, _ = build_knn_graph(z, k=3, mode="distance", sym="union")  # mutual does not assure that the two blobs are connected
    from src.geo.knn_graph import largest_connected_component
    mask = largest_connected_component(W)
    in_A = mask[:N1].sum(); in_B = mask[N1:].sum()
    assert (in_A == 0) ^ (in_B == 0)
    assert mask.sum() in {N1, N2}

def test_cosine_metric_smoke():
    z = random_latents(150, 6)
    W, nbrs = build_knn_graph(z, k=5, metric="cosine", mode="distance")
    assert (W - W.T).nnz == 0
    assert W.shape == (150, 150)
    assert nbrs["indices"].shape[1] == 5

def test_monotonicity_in_k():
    z = random_latents(300, 6)
    W5, _ = build_knn_graph(z, k=5)
    W8, _ = build_knn_graph(z, k=8)
    P5 = (W5 > 0).astype(int)
    P8 = (W8 > 0).astype(int)
    diff = P5 - (P5.multiply(P8))  # edges in W5 that are not in W8
    assert diff.nnz == 0

def test_same_pattern_between_modes():
    z = random_latents(180, 6)
    Wd, _ = build_knn_graph(z, k=7, mode="distance")
    Wc, _ = build_knn_graph(z, k=7, mode="connectivity")
    Pd = (Wd > 0).astype(int)
    Pc = (Wc > 0).astype(int)
    assert (Pd != Pc).nnz == 0

def test_k_zero():
    z = random_latents(25, 4)
    W, nbrs = build_knn_graph(z, k=0)
    assert W.nnz == 0
    assert nbrs["indices"].shape == (25, 0)

def test_edge_count_upper_bound():
    z = random_latents(120, 5)
    k = 6
    W, _ = build_knn_graph(z, k=k)
    assert W.nnz <= 2 * 120 * k
