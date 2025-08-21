
## build_knn_graph
Builds an undirected k-nearest neighbors graph from latent vectors.

**Signature:** `build_knn_graph(z, k=10, metric="euclidean", mode="distance", sym="mutual")`

**Arguments:**
- `z`: array of shape (N, D)
- `k`: neighbors per node (default: 10)
- `metric`: distance metric, sklearn-compatible (euclidean will fit our case)
- `mode`: "distance" (weights = distances) or "connectivity" (weights = 1)
- `sym`: symmetrization method - "mutual" or "union" (mutual keeps mutual neighbors, union keeps any)

**Returns:**
- `W`: sparse CSR matrix (N, N), undirected with zero diagonal
- `neighbors`: dict with `"distances"` and `"indices"` arrays (N, k') where k' $\leq$ k


## largest_connected_component
Returns boolean mask of nodes in the largest connected component.

**Signature:** `largest_connected_component(W)`

**Arguments:**
- `W`: sparse CSR adjacency matrix (undirected)

**Returns:**
- `mask`: boolean array (N,)

**Note:** Use to restrict graph to single connected subgraph for methods requiring connectivity (shortest paths, spectral analysis, etc.).
