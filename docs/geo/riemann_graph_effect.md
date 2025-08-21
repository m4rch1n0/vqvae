# Riemannian Graph Effects

## riemann_graph_analysis
Evaluates the impact of re-weighting k-NN graph edges with decoder-induced Riemannian lengths versus Euclidean weights on graph connectivity and shortest-path distances.

**Objective:** Support the choice of using geodesics in a-posteriori quantization by analyzing: (i) graph **connectivity**, (ii) average **shortest-path distances**.

**Data:** MNIST validation set using trained VAE model (`experiments/vae_mnist/checkpoints/best.pt`) with latent dimension 16.

**Graph Construction:** k-NN graph with k=10 built using Euclidean distances in latent space, then re-weighted with Riemannian edge lengths computed via decoder JVP.

**Modes:**
- `subset` (default): stratified re-weighting on edge subset (distance quantiles)
- `full`: re-weighting of all k-NN graph edges

**Formula:** Ratio metric $R = \frac{\text{mean\_sp\_riem}}{\text{mean\_sp\_euc}}$ where $R > 1$ indicates average path dilation.

## experimental_procedure
Standard procedure for comparing Euclidean vs Riemannian graph properties.

**Steps:**
1. Build Euclidean k-NN graph on latent points `z`
2. Select source nodes in largest connected component (LCC) and estimate baseline Euclidean shortest-path distances
3. Compute Riemannian edge lengths on selected edges (subset/full mode) and replace corresponding weights symmetrically
4. Recalculate connected components, LCC size, and shortest-path distances with Riemannian weights
5. Report ratio metric and connectivity changes

**Parameters:**
- `K_NEIGHBORS`: k-NN parameter (default: 10)
- `SAMPLE_EDGES`: number of edges for subset mode (default: 5000)
- `NUM_SOURCES`: source nodes for shortest-path estimation (default: 100)

## metrics_and_results
Key metrics for evaluating Riemannian vs Euclidean graph properties.

**Connectivity Metrics:**
- `ncomp`: number of connected components
- `lcc_size`: size of largest connected component

**Distance Metrics:**
- `mean_sp_euc`: average shortest-path distance (Euclidean weights)
- `mean_sp_riem`: average shortest-path distance (Riemannian weights)
- `ratio_sp`: ratio $\frac{\text{mean\_sp\_riem}}{\text{mean\_sp\_euc}}$

## experimental_results
Concrete results from stratified subset re-weighting experiment.

**Configuration:**
- Graph: k=10 neighbors, stratified subset re-weighting
- Edges processed: 5,000 (subset mode)
- Source nodes: 8 (for shortest-path estimation)

**Connectivity Analysis:**
- Euclidean: `ncomp = 298`, `lcc_size = 9,662`
- Riemannian: `ncomp = 298`, `lcc_size = 9,662`

**Shortest-Path Analysis:**
- `mean_sp_euc = 20.2284`
- `mean_sp_riem = 23.3099`
- `ratio_sp = 1.152` (15.2% path dilation)

**Interpretation:** Riemannian re-weighting dilates average path lengths without affecting global connectivity (same LCC size and component count). The moderate increase is expected in subset mode since shortest paths can bypass re-weighted edges.

**Note:** Effect magnitude should increase significantly with full re-weighting mode as fewer alternative paths remain available.

**Configuration Options:**
- Switch to full re-weighting: edit `REWEIGHT_MODE = "full"` in script
- Adjust parameters: modify `SAMPLE_EDGES`, `NUM_SOURCES`, `K_NEIGHBORS`

**Output:** Experiment results saved to `experiments/geo/riemann_graph_effects/` including metrics and visualization plots.
