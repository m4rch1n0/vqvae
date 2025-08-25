## fit_kmedoids_graph

K-medoids clustering on weighted graph using a precomputed geodesic distance matrix (chunked).

**Signature:** `fit_kmedoids_precomputed(W, K=512, init="kpp", seed=42, chunk_size=1000)`

**Arguments:**
- `W`: sparse matrix (N, N) with non-negative edge weights
- `K`: number of clusters (medoids)
- `init`: `"random"` or `"kpp"` (k-means++)
- `seed`: random seed
- `chunk_size`: number of sources per Dijkstra batch when computing the full distance matrix

**Returns:**
- `medoids`: array (K,) containing indices of selected medoids
- `assign`: array (N,) with cluster assignment `[0, K-1]`
- `distance_matrix`: array (N, N) float32 of all-pairs geodesic distances

**Algorithm:**
1. Precompute full `(N × N)` geodesic distance matrix via chunked multi-source Dijkstra
2. Initialize K medoids with k-means++ on the precomputed matrix
3. Assign each node to the nearest medoid using the precomputed distances


## _kpp_init_precomputed

K-means++ initialization using the precomputed distance matrix.

**Signature:** `_kpp_init_precomputed(distance_matrix, K, seed=42)`

**Arguments:**
- `distance_matrix`: array (N, N) of geodesic distances (float32)
- `K`: number of centers
- `seed`: random seed

**Returns:**
- `centers`: list of K node indices selected as initial centers