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
├── src/                    # Core implementation
│   ├── data/              # Dataset loaders (MNIST)
│   ├── models/            # VAE architecture
│   ├── geo/               # Geodesic computation modules
│   ├── training/          # Training pipeline
│   └── utils/             # System utilities
├── demos/                  # Interactive examples
│   ├── vae_knn_analysis.py           # Main latent space analysis demo
│   ├── interactive_exploration.py    # Interactive k-NN visualization
│   ├── codebook_comparison.py        # Geodesic vs Euclidean quantization comparison
│   └── kmedoids_geodesic_analysis.py # Geodesic K-medoids (post-hoc quantization)
├── experiments/           # Research experiments  
│   ├── geo/              # Riemannian geometry experiments
│   └── vae_mnist/        # Trained models and latents
├── scripts/              # Utility scripts
│   ├── setup_env.sh      # Environment setup
│   ├── download_data.sh  # Data download
│   ├── train_vae.sh      # VAE training
│   └── run_experiments.sh  # Full experimental pipeline
├── docs/                 # Technical documentation
│   ├── geo/              # Riemannian geometry docs
│   └── models/           # Model architecture docs
└── visualizations/       # Plotting and interaction tools
```

## Quick Start

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
python src/training/build_codebook.py --config configs/quantize.yaml
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

Riemannian Metric (`src/geo/riemannian_metric.py`)
- Computes decoder-induced Riemannian distances using Jacobian-vector products
- Formula: $L_{ij} \approx 0.5 \cdot (\|J(z_i)(z_j - z_i)\|_2 + \|J(z_j)(z_j - z_i)\|_2)$

k-NN Graph (`src/geo/knn_graph.py`)  
- Builds connectivity graphs in latent space
- Supports both Euclidean and Riemannian edge weights

Geodesic Shortest Paths (`src/geo/geo_shortest_paths.py`)
- Dijkstra's algorithm for multi-source shortest paths
- Efficient computation on sparse graphs

K-medoids Clustering (`src/geo/kmeans_precomputed.py`)
- Geodesic K-medoids using precomputed distance matrices
- Chunked computation for memory-efficient large graph processing

Codebook Builder (`src/training/build_codebook.py`)
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
- Set `configs/vae.yaml` to match the dataset:
  - `in_channels`: 1 for MNIST/Fashion, 3 for CIFAR10
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
