import numpy as np
import pytest
from scipy import sparse

from src.geo.knn_graph_optimized import build_knn_graph, largest_connected_component
from src.geo.geo_shortest_paths import (
    dijkstra_multi_source,
    dijkstra_single_source,
    distances_between,
)

def random_latents(N=200, D=16, seed=0):
    """Generate random latent vectors for testing."""
    r = np.random.RandomState(seed)
    return r.randn(N, D).astype(np.float32)

def test_knn_lcc_dijkstra_end_to_end():
    """Test full pipeline: kNN graph -> LCC filtering -> Dijkstra."""
    z = random_latents(N=240, D=12, seed=1)
    # Small k to force disconnected components
    W, _ = build_knn_graph(z, k=1, mode="distance", sym="mutual")
    # Full graph should have some infinite distances
    D_full = dijkstra_multi_source(W, sources=[0, 10])
    assert np.isinf(D_full).any()

    # Filter to largest connected component
    lcc = largest_connected_component(W)
    W_lcc = W[lcc][:, lcc]
    sources_lcc = [0, max(1, W_lcc.shape[0] // 2)]
    D = dijkstra_multi_source(W_lcc, sources=sources_lcc)
    assert D.shape == (len(sources_lcc), W_lcc.shape[0])
    # All distances should be finite in LCC
    assert np.isfinite(D).all()
    # Distance from each source to itself should be zero
    for r, s in enumerate(sources_lcc):
        assert np.allclose(D[r, s], 0.0)

def test_sources_row_order_and_single_wrapper():
    """Test source ordering and consistency between single/multi-source."""
    z = random_latents(N=150, D=8, seed=2)
    W, _ = build_knn_graph(z, k=5, mode="distance", sym="mutual")
    sources = [5, 1, 19]
    D = dijkstra_multi_source(W, sources)
    # Rows should respect source order
    assert np.allclose(D[0, sources[0]], 0.0)
    assert np.allclose(D[1, sources[1]], 0.0)
    assert np.allclose(D[2, sources[2]], 0.0)

    # Single-source should match multi-source result
    d1 = dijkstra_single_source(W, sources[1])
    assert d1.shape == (W.shape[0],)
    np.testing.assert_allclose(d1, D[1])

def test_distances_between_matches_full_integration():
    """Test that distances_between returns correct subset of full distances."""
    z = random_latents(N=180, D=10, seed=3)
    W, _ = build_knn_graph(z, k=6, mode="distance", sym="mutual")
    sources = [0, 50]
    targets = [1, 5, 25, 100, 179]
    D_full = dijkstra_multi_source(W, sources)
    D_sub = distances_between(W, sources, targets)
    assert D_sub.shape == (len(sources), len(targets))
    np.testing.assert_allclose(D_sub, D_full[:, targets])

# Note: Weighted vs unweighted comparison moved to visualizations for better insight

def test_negative_edge_raises_integration():
    """Test that negative edge weights raise ValueError."""
    z = random_latents(N=40, D=6, seed=5)
    W, _ = build_knn_graph(z, k=4, mode="distance", sym="mutual")
    W = W.tolil()
    # Introduce negative symmetric edge weight
    W[0, 1] = -1.0
    W[1, 0] = -1.0
    W = W.tocsr()
    with pytest.raises(ValueError):
        _ = dijkstra_multi_source(W, sources=[0])

def test_lcc_masking_removes_inf():
    """Test that LCC masking eliminates infinite distances."""
    z = random_latents(N=160, D=10, seed=6)
    W, _ = build_knn_graph(z, k=1, mode="distance", sym="mutual")
    D_all = dijkstra_multi_source(W, sources=[0, 10])
    assert np.isinf(D_all).any()

    lcc = largest_connected_component(W)
    W_lcc = W[lcc][:, lcc]
    D_lcc = dijkstra_multi_source(W_lcc, sources=[0, min(10, W_lcc.shape[0]-1)])
    assert np.isfinite(D_lcc).all()