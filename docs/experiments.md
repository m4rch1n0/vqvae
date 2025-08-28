# Experiments

Research experiments for analyzing Riemannian geometry effects in VAE latent spaces and validating geodesic distance computations.

## riemann_sanity_check

Validation experiment comparing Riemannian vs Euclidean edge lengths on k-NN graph connections.

**Signature:** `python experiments/geo/riemann_sanity_check.py`

**Objective:** Verify that decoder-induced Riemannian distances provide meaningful geometric information beyond Euclidean distances in latent space.

**Configuration:**
```python
LATENTS_PATH = "experiments/vae_mnist/latents_val/z.pt"
CHECKPOINT_PATH = "experiments/vae_mnist/checkpoints/best.pt"
OUTPUT_DIR = "experiments/geo/riemann_sanity"
K_NEIGHBORS = 10
MAX_EDGES = 2000
LATENT_DIM = 16
IMAGE_SHAPE = (1, 28, 28)
```

**Procedure:**
1. Load pre-trained VAE model and validation latents
2. Build k-NN graph using Euclidean distances
3. Sample random subset of edges (MAX_EDGES)
4. Compute both Euclidean and Riemannian lengths for sampled edges
5. Generate comparative visualizations and statistics

**Metrics:**
- **Ratio Distribution**: $\frac{L_{riem}}{L_{euc}}$ for all sampled edges
- **Correlation Analysis**: Scatter plot of Riemannian vs Euclidean lengths
- **Statistical Summary**: Mean, std, percentiles of ratio distribution

**Expected Results:**
- Ratio $> 1$: Riemannian distances typically longer due to manifold curvature
- Moderate correlation: Some relationship but meaningful differences
- Outliers: Edges where geometric structure differs significantly

**Output Files:**
- `scatter_riem_vs_euc.png` - Correlation visualization
- `hist_ratio.png` - Ratio distribution histogram  
- `sanity_stats.npz` - Numerical results for further analysis

## run_riemann_experiments

Comprehensive analysis of Riemannian edge re-weighting effects on k-NN graph connectivity and shortest path distances.

**Signature:** `python experiments/geo/run_riemann_experiments.py`

**Objective:** Quantify how replacing Euclidean with Riemannian edge weights affects graph connectivity and geodesic path lengths.

**Configuration:**
```python
K_NEIGHBORS = 10
REWEIGHT_MODE = "subset"  # "subset" or "full"
SAMPLE_EDGES = 5000
NUM_BINS = 5
NUM_SOURCES = 8
```

**Modes:**
- subset: Re-weight stratified sample of edges (distance quantiles)
- full: Re-weight all k-NN graph edges

**Experimental Procedure:**
1. Build Euclidean k-NN graph on validation latents
2. Extract largest connected component (LCC)
3. Baseline Analysis: Compute connectivity metrics and shortest paths
4. Edge Selection: Choose edges for Riemannian re-weighting
5. Riemannian Re-weighting: Replace selected edge weights
6. Impact Analysis: Compare connectivity and path length changes

**Connectivity Metrics:**
- `ncomp`: Number of connected components
- `lcc_size`: Size of largest connected component
- `connectivity_preserved`: Boolean for maintained connectivity

**Distance Metrics:**
- `mean_sp_euc`: Average shortest path (Euclidean weights)
- `mean_sp_riem`: Average shortest path (Riemannian weights)  
- `ratio_sp`: $\frac{\text{mean\_sp\_riem}}{\text{mean\_sp\_euc}}$

**Stratified Sampling:** Edges binned by Euclidean distance quantiles to ensure representative coverage across distance scales.

**Expected Results:**
- Subset Mode: Moderate path dilation (10-20%), preserved connectivity
- Full Mode: Significant path changes, potential connectivity fragmentation
- Ratio > 1: Riemannian paths typically longer due to manifold geometry

**Output Files:**
- `graph_effects.png` - Visualization of connectivity and distance changes
- `graph_effects.npz` - Complete numerical results
- Console output with detailed metrics

## codebook_comparison

Direct comparison of geodesic vs Euclidean quantization methods for post-hoc VQ-VAE codebook construction.

**Signature:** `python demos/codebook_comparison.py`

**Objective:** Evaluate reconstruction quality and code usage efficiency when quantizing VAE latents using geodesic K-medoids vs standard Euclidean K-means.

**Configuration:** Configuration files can be found in configs/codebook_comparison

**Methodology:**
1. Load pre-trained VAE model and latent representations (N=10000, dim=16)
2. Build Euclidean codebook: `sklearn.KMeans(K=64)` 
3. Build geodesic codebook: k-NN graph + `fit_kmedoids_optimized(K=64)`
4. Quantize latents using both codebooks
5. Compare VAE reconstruction quality via decoder MSE

**Key Metrics:**
- **Reconstruction MSE**: Quality of image reconstruction from quantized latents
- **Perplexity**: Code usage balance (`exp(entropy)`)
- **Quantization Error**: Distance to assigned centroids/medoids

**Current Results (MNIST):**
```
Euclidean  - Reconstruction MSE: 0.044484, Perplexity: 62.60
Geodesic   - Reconstruction MSE: 0.071040, Perplexity: 43.21
```

**Interpretation:** Counter-intuitive finding that Euclidean quantization outperforms geodesic on MNIST. Suggests manifold complexity may not justify computational overhead for simple datasets.

**Output Files:**
- `codebook_comparison.png` - 3-panel comparison visualization
- `metrics.json` - Detailed numerical results
- `config.yaml` - Configuration used for the experiment (reproducibility)

**Recommended Configurations for Better Geodesic Performance:**

**Default Configuration:** `configs/test1.yaml`
- Standard parameters for baseline comparison

**Experimental Configuration:** `configs/test2.yaml`
- Higher connectivity: `k=20` (vs default `k=10`)
- Larger codebook: `K=128` (vs default `K=64`)  
- Broader connectivity: `sym="union"` (vs default `sym="mutual"`)


## Experimental Pipeline

**Complete Workflow:**
```bash
# Run full experimental pipeline
./scripts/run_experiments.sh
```

**Manual Execution:**
```bash
# 1. Train VAE (if needed)
./scripts/train_vae.sh

# 2. Sanity check validation
python experiments/geo/riemann_sanity_check.py

# 3. Graph effects analysis  
python experiments/geo/run_riemann_experiments.py

# 4. Codebook comparison (geodesic vs Euclidean quantization)
python demos/codebook_comparison.py

# 5. Geodesic k-medoids analysis (post-hoc quantization)
python demos/kmedoids_geodesic_analysis.py
```

**Prerequisites:**
- Trained VAE model with saved checkpoints
- Extracted latent representations
- MNIST validation dataset

**Dependencies:**
- PyTorch (with grad computation)
- SciPy (sparse matrices, Dijkstra)
- NumPy, matplotlib
- Project modules: geo, models

## Geodesic K-medoids Analysis (Post-hoc Quantization)

Runs k-medoids clustering on a k-NN graph (Euclidean edge weights) and evaluates code usage and optional label alignment.

Signature: `python demos/kmedoids_geodesic_analysis.py`

Environment variables (optional):

```bash
KM_Z_PATH=experiments/vae_mnist/latents_val/z.pt \
KM_Y_PATH=experiments/vae_mnist/latents_val/y.pt \
KM_K_GRAPH=10 \
KM_GRAPH_SYM=mutual \  # mutual|union
KM_K_VALUES=32,64,128 \
KM_INITS=kpp,random \
KM_SEED=42 \
python demos/kmedoids_geodesic_analysis.py
```

Outputs (in `demo_outputs/kmedoids_geodesic_*`):
- `metrics.csv|json`: per-(K,init) metrics: geodesic QE (finite nodes), finite fraction, purity, NMI, ARI, perplexity
- `elbow.png`: QE vs K curves (one line per init)
- `pca_clusters_euclidean_K{K}_{init}.png`: PCA of distance-to-medoids features with medoids highlighted (first configuration)
- `code_usage_euclidean_K{K}_{init}.png`: code usage histogram with perplexity (first configuration)

Interpretation tips:
- Lower geodesic QE at fixed K suggests better codebook fit to manifold geometry
- Higher purity/NMI/ARI indicates stronger label alignment (if labels available)
- Higher finite fraction means more stable geodesic assignments
- Perplexity close to K implies balanced code usage; very low perplexity suggests code collapse

## Configuration Tuning

**Performance vs Accuracy Trade-offs:**

Memory Management:
- Reduce `MAX_EDGES` for memory constraints
- Increase `batch_size` in Riemannian computation for efficiency
- Use subset mode for faster experimentation

Statistical Significance:
- Increase `NUM_SOURCES` for more robust shortest path estimates
- Use full re-weighting mode for maximum effect measurement
- Increase `SAMPLE_EDGES` for better edge coverage

Graph Parameters:
- Modify `K_NEIGHBORS` to study connectivity effects
- Experiment with different latent dimensions
- Test on different datasets beyond MNIST

## Result Interpretation

**Ratio Analysis:**
- $R \approx 1$: Minimal geometric distortion
- $R = 1.1-1.3$: Moderate manifold curvature effects  
- $R > 1.5$: Significant geometric structure differences

**Connectivity Preservation:**
- Same LCC size: Robust graph structure
- Fragmentation: Strong geometric constraints from Riemannian metric
- Component increase: Local connectivity changes

**Research Implications:**
- Validates geometric meaningfulness of learned latent spaces
- Quantifies trade-offs in post-hoc vs end-to-end quantization
- Guides hyperparameter selection for geodesic clustering
