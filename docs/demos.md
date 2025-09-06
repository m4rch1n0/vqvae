# Demos

Interactive examples demonstrating VAE latent space analysis with geodesic metrics.

## vae_knn_analysis

Comprehensive latent space analysis demo that trains a VAE and visualizes geodesic distance patterns across MNIST digit classes.

**Signature:** `python demos/vae_knn_analysis.py`

**Functionality:**
- Trains 2D VAE on MNIST (5 epochs for demo purposes)
- Extracts latent representations for validation set
- Builds k-NN graphs with multiple k values [3, 4, 5, 6]
- Computes geodesic distances from digit centroids
- Generates comprehensive visualizations

**Configuration:**
```python
{
    'seed': 42,
    'device': 'auto',
    'max_epochs': 5,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'early_stop': False,
    'max_samples': 300,
    'k_values': [3, 4, 5, 6]
}
```

**Output Directory:** `demo_outputs/vae_mnist_geodesic_{timestamp}/`

**Generated Files:**
- `geodesic_analysis_k{k}.png` - Distance heatmaps for each k value
- `latent_distribution.png` - 2D latent space scatter plot with digit classes

**Visualization Features:**
- Latent Distribution: Scatter plot showing digit separation in 2D latent space
- Geodesic Distance Heatmaps: From each digit centroid to all latent points
- k-NN Connectivity: Analysis of neighborhood structure for different k values

**Note:** Uses limited epochs (5) and samples (300) for fast demonstration. Increase for research-quality analysis.

## interactive_exploration

Interactive visualization tool for exploring k-NN graph connectivity and geodesic distance patterns using synthetic data.

**Signature:** `python demos/interactive_exploration.py`

**Data Generation:**
- Creates synthetic 2D latent data simulating MNIST digit distributions
- 10 clusters arranged in circular pattern with realistic variance
- 30 points per digit class (300 total points)

**Interactive Features:**
- k-NN Parameter Control: Adjust k value dynamically [1-20]
- Distance Metric Toggle: Switch between Euclidean/geodesic visualization
- Click Exploration: Click points to see distance patterns
- Real-time Updates: Immediate graph and distance recalculation

**Synthetic Data Pattern:**
```python
# Digit centers arranged in circle
for i in range(10):
    angle = 2 * np.pi * i / 10
    cx = 3 * np.cos(angle)
    cy = 3 * np.sin(angle)
    centers.append((cx, cy))
```

**Note:** Uses synthetic data for responsiveness. Real VAE latents can be loaded by modifying data generation function.

## kmedoids_geodesic_analysis

Post-hoc geodesic k-medoids clustering on a latent space using a k-NN graph. Uses precomputed latents (and optional labels) to evaluate codebook quality.

**Signature:** `python demos/kmedoids_geodesic_analysis.py`

**Configuration (env vars):**
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

**Output Directory:** `demo_outputs/kmedoids_geodesic_{timestamp}/`

**Generated Files:**
- `metrics.csv` and `metrics.json` with: `K`, `init`, `qe_geo_finite`, `finite_fraction`, `purity`, `nmi`, `ari`, `perplexity`
- `elbow.png` — QE vs K curves (per init)
- `pca_clusters_euclidean_K{K}_{init}.png` — PCA of distance-to-medoids features with medoids highlighted (first configuration)
- `code_usage_euclidean_K{K}_{init}.png` — code usage histogram with perplexity (first configuration)

## Running Demos

**Prerequisites:**
1. Complete environment setup: `./scripts/setup_env.sh`
2. Download MNIST data: `./scripts/download_data.sh`

**Execution:**
```bash
# Full latent space analysis (requires VAE training)
python demos/vae_knn_analysis.py

# Interactive exploration (synthetic data, no training needed)
python demos/interactive_exploration.py

# Geodesic k-medoids (post-hoc quantization on precomputed latents)
python demos/kmedoids_geodesic_analysis.py


# Reconstruct grid using a codebook (supports optional --out_dir)
python demos/codebook_sampling.py \
  --codebook_dir experiments/geo/codebook_<dataset>_k1024/ \
  --latents_path experiments/vae_<dataset>/latents_val/z.pt \
  --checkpoint experiments/vae_<dataset>/checkpoints/best.pt \
  --out_dir demo_outputs/codebook_sampling_<dataset>/
```

**Dependencies:**
- Trained VAE model (automatically trained by vae_knn_analysis.py)
- matplotlib, numpy, sklearn
- Interactive demo requires plotly/bokeh backend

**Customization:**
- Modify `get_config()` in vae_knn_analysis.py for different training parameters
- Adjust k_values range for different connectivity analysis
- Change max_samples for faster/more comprehensive analysis
- Edit synthetic data generation in interactive_exploration.py for different patterns

**Output Interpretation:**
- Tight clusters: Good latent space organization by digit class
- Smooth geodesic transitions: Well-structured manifold learned by VAE
- k-NN connectivity: Higher k values show broader neighborhood relationships
- Distance heatmaps: Reveal geometric structure and class separability
