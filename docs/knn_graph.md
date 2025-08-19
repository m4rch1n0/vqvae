
Builds an undirected k-nearest neighbors (k-NN) graph from latent vectors.

#### Signature
`build_knn_graph(z, k=10, metric="euclidean", mode="distance", sym="mutual")`

#### Arguments
- **z**: array of shape (N, D)
- **k**: neighbors per node
- **metric**: any metric supported by sklearn (euclidean will fit our case)
- **mode**: "distance" (weights = distances) or "connectivity" (weights = 1)
- **sym**: graph symmetrization. "mutual" (keep mutual neighbors) or "union" (keep any)

#### Returns
- **W**: `scipy.sparse.csr_matrix` of shape (N, N); undirected with zero diagonal
- **neighbors**: dict with `"distances"` (float32) and `"indices"`, each of shape (N, k') where k' ≤ k

#### Details
- Uses `sklearn.neighbors.NearestNeighbors` for neighbor search
- Caps `k` to `N-1`. Queries `k+1` then drops self
- For `N=0`, returns empty CSR and empty neighbor arrays


Computes the largest connected component mask.

#### Signature
`largest_connected_component(W)`

#### Arguments
- **W**: `scipy.sparse.csr_matrix` adjacency (undirected)

#### Returns
- **mask**: boolean array of shape (N,)

#### Details
- Use it to restrict the graph (and data) to a single connected subgraph so graph methods assuming connectivity (e.g., shortest paths, diffusion, spectral) behave well
- If already connected, returns all `True`
- Uses `scipy.sparse.csgraph.connected_components`

new test
