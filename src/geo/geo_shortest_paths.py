"""
Geodesic shortest paths using SciPy's Dijkstra.

Full documentation: see docs/geo_shortest_paths.md
"""
from typing import Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra


# From knn_graph.py we expect W to be CSR, square and with all weights positive but for debug reasons we check here
def _ensure_valid_graph(W: sparse.spmatrix) -> sparse.spmatrix:
    if not sparse.isspmatrix(W):
        raise TypeError("W must be a scipy sparse matrix")
    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    if W.nnz > 0 and (W.data < 0).any():
        raise ValueError("Negative weights")
    # Prefer CSR for csgraph ops
    return W.tocsr()


def dijkstra_multi_source(W: sparse.spmatrix, sources, directed: bool = False, unweighted: bool = False, return_predecessors: bool = False, dtype=np.float32) -> Tuple:
    """Multi-source geodesic distances"""
    if len(sources) == 0:
        raise ValueError("sources must be a non-empty sequence of node indices")
    W = _ensure_valid_graph(W)
    sources = np.asarray(sources, dtype=int)

    if unweighted and W.nnz > 0:
        W = W.copy()
        W.data.fill(1.0)

    if return_predecessors:
        D, P = csgraph_dijkstra(
            csgraph=W,
            directed=directed,
            indices=sources,
            return_predecessors=True,
        )
        return D.astype(dtype, copy=False), P.astype(np.int32, copy=False)
    else:
        D = csgraph_dijkstra(
            csgraph=W,
            directed=directed,
            indices=sources,
            return_predecessors=False,
        )
        return D.astype(dtype, copy=False)


def dijkstra_single_source(W: sparse.spmatrix, source: int, directed: bool = False, unweighted: bool = False, return_predecessors: bool = False, dtype=np.float32) -> Tuple:
    """Single-source wrapper returning 1D arrays"""
    result = dijkstra_multi_source(
        W, [int(source)], directed=directed, unweighted=unweighted,
        return_predecessors=return_predecessors, dtype=dtype
    )
    if return_predecessors:
        D, P = result
        return D[0], P[0]
    else:
        return result[0]


def distances_between(W: sparse.spmatrix, sources, targets, directed: bool = False, unweighted: bool = False, dtype=np.float32) -> np.ndarray:
    """Compact (S x T) distance matrix via multi-source + sub-index"""
    if len(sources) == 0 or len(targets) == 0:
        raise ValueError("sources and targets must be non-empty.")
    sources = np.asarray(sources, dtype=int)
    targets = np.asarray(targets, dtype=int)
    D_full = dijkstra_multi_source(
        W, sources, directed=directed, unweighted=unweighted,
        return_predecessors=False, dtype=dtype
    )
    return D_full[:, targets]
