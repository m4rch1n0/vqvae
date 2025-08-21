## fit_kmedoids_graph

K-medoids clustering on weighted graph using geodesic distances.

**Signature:** `fit_kmedoids_graph(W, K=512, init="kpp", seed=42)`

**Arguments:**
- `W`: sparse matrix (N, N) with non-negative edge weights
- `K`: number of clusters (medoids) to find
- `init`: initialization method - `"random"` or `"kpp"` (k-means++)
- `seed`: random seed for reproducible results

**Returns:**
- `medoids`: array (K,) containing indices of selected medoid nodes
- `assign`: array (N,) with cluster assignment [0, K-1] for each node

**Algorithm:**
1. Initialize K medoids (random or k-means++)
2. Assign each node to nearest medoid using geodesic distances


## _kpp_init_on_graph

K-means++ initialization using geodesic distances.

**Signature:** `_kpp_init_on_graph(W, K, seed=42)`

**Arguments:**
- `W`: sparse matrix (N, N) representing weighted graph
- `K`: number of centers to select
- `seed`: random seed

**Returns:**
- `centers`: list of K node indices selected as initial centers

**Note:** Automatically handles disconnected graphs by mapping infinite distances to finite values.
