# VQ-VAE with Geodesic Quantization

A research implementation of vector quantization using geodesic distances in VAE latent spaces, exploring post-hoc discrete coding with Riemannian metrics instead of Euclidean distances.

## Overview

This project revisits the classic VQ-VAE pipeline by separating the continuous representation learning from the quantization step. Instead of jointly learning a discrete latent space, we:

1. **Train a standard continuous VAE** (Done on MNIST, Cifar10)
2. **Apply geodesic K-means clustering** in the latent space using Riemannian distances
3. **Build a discrete codebook** from geodesic centroids  
4. **Compare reconstruction quality** vs. standard VQ-VAE

**Key Innovation:** Replace Euclidean distance with geodesic metrics that better reflect the curvature and topology of the learned latent manifold.

## Project Structure

```
vqvae/
├── configs/                   # Hydra configuration files
│   ├── data/                  # Dataset configs
│   ├── model/                 # Base model architecture configs
│   └── presets/               # Runnable experiment presets
│       └── fashion_spatial_geodesic/ # The main workflow for this project
├── docs/                      # Documentation and summaries
├── experiments/               # Default output directory for models and artifacts
├── qualitative_results/       # Saved generated image grids for comparison
├── src/                       # Core implementation
│   ├── data/                  # Dataset loaders and logic
│   ├── geo/                   # Geodesic computation modules
│   ├── models/                # VAE and Transformer architectures
│   ├── training/              # Core training engine logic
│   ├── scripts/               # Main executable scripts
│   └── utils/                 # System and logging utilities
└── README.md
```

## Quick Start: Main Pipeline (Spatial VAE + Transformer)

This guide runs the entire pipeline for the `fashion_spatial_geodesic` experiment.

### 1. Environment Setup

Ensure you have the required dependencies installed and the conda environment activated.

```bash
# First time setup: pip install -r requirements.txt
conda activate rocm_env
```

### 2. Run the Full Pipeline

The project uses Hydra for configuration. Each step is a Python script that loads its configuration from the `configs/presets/fashion_spatial_geodesic/` directory.

**Step 1: Train the Spatial VAE**
This script will train the VAE and save the model checkpoint and latent representations to the `experiments/` directory.

```bash
python src/scripts/train_vae.py
```

**Step 2: Build the Geodesic Codebook**
Using the latents from the VAE, this script builds the k-NN graph, re-weights it with the Riemannian metric, and performs k-medoids clustering to create the codebook.

```bash
python src/scripts/build_codebook.py --config-name presets/fashion_spatial_geodesic/2_build_codebook
```

**Step 3: Train the Autoregressive Transformer**
This trains the Transformer on the sequences of discrete codes generated in the previous step.

```bash
python src/scripts/train_transformer.py
```

**Step 4: Generate Samples**
Use the trained Transformer and VAE to generate a grid of new image samples.

```bash
python src/scripts/generate_samples.py --config-name presets/fashion_spatial_geodesic/4_generate
```

**Step 5: Evaluate the Generated Samples**
Calculate PSNR, SSIM, and LPIPS metrics for the generated images.

```bash
python src/scripts/evaluate_model.py --config-name presets/fashion_spatial_geodesic/5_evaluate
```

## Legacy Quick Start (Original VAE)

Setup environment and download data:
```bash
./scripts/setup_env.sh
./scripts/download_data.sh
```

Select dataset (default: MNIST) in `configs/data.yaml`:
```yaml
# name: one of [MNIST, FashionMNIST]
name: MNIST
```

Train VAE model:
```bash
./scripts/train_vae.sh
```

This saves checkpoints and latents to `experiments/vae_mnist/` (paths are configurable in `configs/train.yaml`).

Build geodesic codebook (post‑hoc quantization on latents):
```bash
python src/scripts/build_codebook.py --config configs/quantize.yaml
```

Compare Euclidean vs Geodesic codebooks:
```bash
python demos/codebook_comparison.py --config test2
```

Run other demos:
```bash
python demos/vae_knn_analysis.py
python demos/interactive_exploration.py
python demos/codebook_comparison.py
python demos/kmedoids_geodesic_analysis.py
```

Full experimental pipeline:
```bash
./scripts/run_experiments.sh
```

## Key Modules

-   **`src/scripts/`**: Contains the five main entry points for the main pipeline.
-   **`src/models/spatial_vae.py`**: The VAE architecture with a spatial latent grid output.
-   **`src/models/transformer.py`**: The decoder-only Transformer for autoregressive modeling.
-   **`src/geo/riemannian_metric.py`**: Computes decoder-induced Riemannian distances using Jacobian-vector products.
-   **`src/geo/kmeans_optimized.py`**: Implements graph-based geodesic K-medoids clustering.

Riemannian Metric (`src/geo/riemannian_metric.py`)
- Computes decoder-induced Riemannian distances using Jacobian-vector products
- Formula: $L_{ij} \approx 0.5 \cdot (\|J(z_i)(z_j - z_i)\|_2 + \|J(z_j)(z_j - z_i)\|_2)$

k-NN Graph (`src/geo/knn_graph_optimized.py`)  
- Builds connectivity graphs in latent space
- Supports both Euclidean and Riemannian edge weights

Geodesic Shortest Paths (`src/geo/geo_shortest_paths.py`)
- Dijkstra's algorithm for multi-source shortest paths
- Efficient computation on sparse graphs

K-medoids Clustering (`src/geo/kmeans_optimized.py`)
- Graph-based geodesic K-medoids without full distance matrix computation
- Iterative Dijkstra-based initialization and multi-source assignment
- Scales to large datasets through algorithmic efficiency

Codebook Builder (`src/scripts/build_codebook.py`)
- Post-hoc discrete codebook construction via geodesic clustering
- Saves quantized representations and cluster assignments

## Research Questions

- Geometry: How does the VAE latent manifold structure affect quantization quality?
- Connectivity: Do Riemannian distances preserve better neighborhood relationships?
- Reconstruction: Does geodesic clustering improve reconstruction vs. Euclidean methods?
- Comparative Analysis: When do geodesic methods outperform standard Euclidean quantization?
- Generative Quality: How does post-hoc quantization compare to end-to-end VQ-VAE training?

## Configuration

Main configuration files in `configs/`:
- `configs/data.yaml` - Dataset configuration (set `name: MNIST` or `FashionMNIST`)
- `configs/vae.yaml` - VAE architecture
- `configs/train.yaml` - Training parameters

## Results

Experimental outputs saved to:
- `experiments/geo/` - Riemannian analysis results
- `demo_outputs/` - Demo visualizations and quantization comparisons
- `experiments/vae_mnist/` - Trained models and latent representations

Notes:
- Dataset presets live under `configs/presets/<dataset>/`. Root files (`configs/data.yaml`, `configs/vae.yaml`, `configs/train.yaml`) are real copies of the chosen preset.
- Switch dataset quickly:
  ```bash
  bash scripts/select_dataset.sh mnist   # or fashion, cifar10
  ```
  Then run training/quantization as usual.
- Ensure `in_channels` matches the dataset (`1` for MNIST/Fashion, `3` for CIFAR10).
  - `output_image_size`: 28 for MNIST/Fashion, 32 for CIFAR10
- Training auto-clusters outputs per dataset name (mnist/fashion/cifar10) for checkpoints and out dirs.

## Dependencies

- PyTorch (with ROCm/CUDA support)
- NumPy, SciPy, matplotlib
- scikit-learn (for clustering comparisons)
- Hydra (configuration management)
- MLflow (experiment tracking)

See `requirements.txt` for complete dependencies.

## Documentation

Detailed technical documentation available in `docs/`:
- [Riemannian Metrics](docs/geo/riemannian_metric.md) - Mathematical formulation and implementation
- [Geodesic Shortest Paths](docs/geo/geo_shortest_paths.md) - Graph algorithms
- [k-NN Graph Construction](docs/models/knn_graph.md) - Connectivity analysis

## Citation

This work explores the intersection of differential geometry and discrete representation learning, building on foundations from VQ-VAE and Riemannian geometry in deep learning.
