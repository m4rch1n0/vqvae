
## build_knn_graph

Builds an undirected k-nearest neighbors graph with automatic method selection for optimal performance.

**Signature:** `build_knn_graph(z, k=10, metric="euclidean", mode="distance", sym="mutual")`

**Arguments:**
- `z`: array of shape (N, D) - input data points
- `k`: neighbors per node (default: 10)
- `metric`: distance metric ("euclidean" or "cosine")
- `mode`: "distance" (weights = distances) or "connectivity" (weights = 1)
- `sym`: symmetrization method - "mutual" (keeps mutual neighbors) or "union" (keeps any neighbors)

**Returns:**
- `W`: sparse CSR matrix (N, N), undirected with zero diagonal
- `neighbors`: dict with `"distances"` and `"indices"` arrays (N, k') where k' ≤ k

## Implementation

**Automatic Method Selection:**
- **< 50,000 samples**: Uses scikit-learn (exact, precise, widely compatible)
- **≥ 50,000 samples**: Uses FAISS if available (exact with IndexFlat*, fast for large datasets)
- **Fallback**: Automatically falls back to sklearn if FAISS is not installed

**Backend Details:**
- **sklearn**: Uses `NearestNeighbors` with exact algorithms, supports all metrics
- **FAISS**: Uses `IndexFlatL2` (euclidean) and `IndexFlatIP` (cosine), both exact methods
- **Compatibility**: Both backends return identical results (within float32 precision)

**Key advantages:**
- Memory-efficient sparse graph construction
- Automatic optimization based on dataset size and availability
- Graceful fallback when FAISS is unavailable
- Support for both exact methods with consistent results

## build_knn_graph_auto

Advanced k-NN graph construction with explicit method control.

**Signature:** `build_knn_graph_auto(z, k=10, metric="euclidean", mode="distance", sym="mutual", force_method=None, size_threshold=50000)`

**Arguments:**
- `z`: array of shape (N, D) - input data points
- `k`: neighbors per node (default: 10)
- `metric`: distance metric ("euclidean" or "cosine")
- `mode`: "distance" (weights = distances) or "connectivity" (weights = 1)
- `sym`: symmetrization method - "mutual" or "union" 
- `force_method`: force specific method ("sklearn" or "faiss")
- `size_threshold`: switch to FAISS above this sample count (default: 50,000)

**Returns:**
- `W`: sparse CSR matrix (N, N), undirected with zero diagonal
- `neighbors`: dict with `"distances"` and `"indices"` arrays

**Implementation:**
- Allows manual control over method selection via `force_method` parameter
- Configurable `size_threshold` for automatic selection
- Automatic fallback to sklearn if FAISS unavailable
- Maintains backward compatibility with existing code

## largest_connected_component

Returns boolean mask of nodes in the largest connected component.

**Signature:** `largest_connected_component(W)`

**Arguments:**
- `W`: sparse CSR adjacency matrix (undirected)

**Returns:**
- `mask`: boolean array (N,) - True for nodes in largest component

**Algorithm:** Uses scipy's connected_components to identify graph components, then selects the largest one by frequency count.

## analyze_graph_connectivity

Analyzes k-NN graph connectivity and returns comprehensive statistics.

**Signature:** `analyze_graph_connectivity(W)`

**Arguments:**
- `W`: sparse CSR adjacency matrix
**Returns:**
- `stats`: dict with connectivity statistics:
  - `n_nodes`: total number of nodes
  - `n_edges`: total number of edges
  - `n_components`: number of connected components
  - `largest_component_size`: size of largest component
  - `connectivity_ratio`: fraction of nodes in largest component
  - `avg_degree`: average node degree
  - `min_degree`: minimum node degree
  - `max_degree`: maximum node degree

**Note:** Essential for understanding graph structure before geodesic computations. Helps identify disconnected components that may affect shortest path calculations.
