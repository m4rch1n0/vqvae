
## dijkstra_multi_source
Computes geodesic shortest paths from multiple sources using Dijkstra's algorithm.

**Signature:** `dijkstra_multi_source(W, sources, directed=False, unweighted=False, return_predecessors=False, dtype=np.float32)`

**Arguments:**
- `W`: sparse matrix (N, N) with non-negative weights
- `sources`: source node indices 
- `directed`: treat as directed graph
- `unweighted`: use topology only, ignore weights (can be useful for ablation check)
- `return_predecessors`: also return predecessor tree
- `dtype`: output data type (np.float32)

**Returns:**
- `D`: distances array (S, N); `np.inf` for unreachable nodes
- `P`: predecessors array (S, N) if `return_predecessors=True`


## dijkstra_single_source
Single-source wrapper that returns 1D arrays.

**Signature:** `dijkstra_single_source(W, source, directed=False, unweighted=False, return_predecessors=False, dtype=np.float32)`

**Arguments:**
- `W`: sparse matrix (N, N) with non-negative weights
- `source`: source node index
- Other parameters: same as `dijkstra_multi_source`

**Returns:**
- `d`: distances array (N,)
- `p`: predecessors array (N,) if `return_predecessors=True`


## distances_between
Computes compact distance matrix between source and target node sets.

**Signature:** `distances_between(W, sources, targets, directed=False, unweighted=False, dtype=np.float32)`

**Arguments:**
- `W`: sparse matrix (N, N)
- `sources`: source node indices
- `targets`: target node indices
- Other parameters: same as `dijkstra_multi_source`

**Returns:**
- `D_sub`: distances array (S, T); `np.inf` for unreachable pairs

**Note:** Efficient when T $\ll$ N since it runs Dijkstra once and extracts target columns.

