## fit_kmedoids_optimized

Graph-based geodesic K-medoids clustering using sparse graph algorithms.

**Signature:** `fit_kmedoids_optimized(W, K=512, init="kpp", seed=42, verbose=True)`

**Arguments:**
- `W`: sparse CSR matrix (N, N) with non-negative edge weights (geodesic distances)
- `K`: number of clusters (medoids) to select
- `init`: initialization method - `"kpp"` (K-means++) or `"random"`
- `seed`: random seed for reproducible initialization
- `verbose`: if True, prints progress information

**Returns:**
- `medoids`: array (K,) of selected medoid indices
- `assign`: array (N,) of cluster assignments [0, K-1]
- `quantization_error`: float sum of squared geodesic distances to assigned medoids

## Algorithm

Performs K-medoids clustering on weighted graphs using direct graph algorithms.

**Core approach:**
1. **Initialization**: K-means++ using iterative single-source Dijkstra
2. **Assignment**: Multi-source Dijkstra from all medoids simultaneously
3. **Iteration**: No iterations - single-shot assignment after initialization

**Key advantage**: Avoids O(N²) distance matrix computation and storage.

**Previous implementation**: Computed full N×N distance matrix before clustering.

## Usage Example

```python
from src.geo.kmeans_optimized import fit_kmedoids_optimized

# Build k-NN graph from latent vectors
W, _ = build_knn_graph(z, k=10, metric="euclidean")

# Perform geodesic K-medoids clustering
medoids, assignments, qe = fit_kmedoids_optimized(W, K=512, init="kpp")
```

## Notes

- Graph should be connected for meaningful geodesic distances
- Use `largest_connected_component()` if graph has multiple components
- Memory usage scales with O(K×N) for distance computations vs O(N²) for matrix approach