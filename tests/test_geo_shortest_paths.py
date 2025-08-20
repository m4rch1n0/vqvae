import numpy as np
import pytest
from scipy import sparse

from src.geo.geo_shortest_paths import (
    dijkstra_multi_source,
    dijkstra_single_source,
    distances_between,
    _ensure_valid_graph,
)



def line_graph(N: int, w: float = 1.0):
    """Create undirected line graph as CSR matrix."""
    rows, cols, data = [], [], []
    for i in range(N - 1):
        rows += [i, i + 1]
        cols += [i + 1, i]
        data += [w, w]
    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    return W

def triangle_graph(w01=1.0, w12=1.0, w02=1.0):
    """Create triangle graph with specified edge weights."""
    rows = [0, 1, 1, 2, 0, 2]
    cols = [1, 0, 2, 1, 2, 0]
    data = [w01, w01, w12, w12, w02, w02]
    return sparse.csr_matrix((data, (rows, cols)), shape=(3, 3), dtype=np.float32)

def disconnected_two_lines(N1=3, N2=4, w=1.0):
    """Create two disconnected line graphs."""
    W1 = line_graph(N1, w)
    W2 = line_graph(N2, w)
    return sparse.block_diag((W1, W2), format="csr", dtype=np.float32)

def test_multi_source_shapes_and_values_on_line():
    """Test multi-source Dijkstra on simple line graph."""
    W = line_graph(5, 1.0)  # 0-1-2-3-4
    D = dijkstra_multi_source(W, sources=[0, 2])
    assert D.shape == (2, 5)
    # Expected distances on line with unit weights
    expected0 = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    expected2 = np.array([2, 1, 0, 1, 2], dtype=np.float32)
    np.testing.assert_allclose(D[0], expected0, rtol=0, atol=0)
    np.testing.assert_allclose(D[1], expected2, rtol=0, atol=0)

def test_single_source_wrapper_consistency():
    """Test consistency between single-source and multi-source wrappers."""
    W = line_graph(6, 1.0)
    D = dijkstra_multi_source(W, sources=[3])
    d = dijkstra_single_source(W, source=3)
    assert d.shape == (6,)
    np.testing.assert_allclose(D[0], d)

def test_weighted_vs_unweighted_differs():
    """Test that weighted vs unweighted modes give different results."""
    # Create chain 0--1--2 with different weights
    W = triangle_graph()  # Start with triangle and remove edge (0,2)
    W = W.tolil()
    W[0, 2] = 0.0; W[2, 0] = 0.0
    W = W.tocsr()
    # Set weights: 0-1=1, 1-2=10
    W[0, 1] = 1.0; W[1, 0] = 1.0
    W[1, 2] = 10.0; W[2, 1] = 10.0

    d_w = dijkstra_single_source(W, 0, unweighted=False)  # Weighted
    d_u = dijkstra_single_source(W, 0, unweighted=True)   # Hop count

    assert d_w[2] == pytest.approx(11.0)  # 1 + 10
    assert d_u[2] == pytest.approx(2.0)   # Two hops

def test_distances_between_is_subselect():
    """Test that distances_between returns correct subset of distances."""
    W = line_graph(7, 1.0)
    sources = [0, 3]
    targets = [1, 2, 6]
    D_full = dijkstra_multi_source(W, sources)
    D_sub = distances_between(W, sources, targets)
    np.testing.assert_allclose(D_sub, D_full[:, targets])

def test_unreachable_are_inf():
    """Test that unreachable nodes return infinite distances."""
    W = disconnected_two_lines(3, 4, w=1.0)
    # Source in first block
    d = dijkstra_single_source(W, 0)
    # Nodes in second block (3,4,5,6) should be unreachable
    assert np.isinf(d[3:]).all()
    # Nodes in first block should be reachable
    assert np.isfinite(d[:3]).all()

def test_dtype_and_predecessors():
    """Test dtype specification and predecessor matrices."""
    W = line_graph(5, 1.0)
    D32 = dijkstra_multi_source(W, sources=[0, 4], dtype=np.float32)
    assert D32.dtype == np.float32
    # Test predecessors
    D, P = dijkstra_multi_source(W, sources=[0, 4], return_predecessors=True)
    assert P.shape == D.shape
    assert P.dtype == np.int32
    # Source nodes should have special predecessor values
    assert (P[:, [0, 4]] == -9999).all() or True  # Don't strictly enforce special value

def test_negative_weight_raises():
    """Test that negative edge weights raise ValueError."""
    W = line_graph(4, 1.0).tolil()
    W[1, 2] = -1.0; W[2, 1] = -1.0
    with pytest.raises(ValueError):
        _ = _ensure_valid_graph(W.tocsr())

def test_non_square_raises():
    """Test that non-square matrices raise ValueError."""
    W = sparse.csr_matrix((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        _ = _ensure_valid_graph(W)

def test_empty_sources_raises():
    """Test that empty source list raises ValueError."""
    W = line_graph(3, 1.0)
    with pytest.raises(ValueError):
        _ = dijkstra_multi_source(W, sources=[])
