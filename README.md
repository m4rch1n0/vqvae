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
├── configs/                   # YAML configuration files (no Hydra required)
│   ├── fashionmnist/
│   │   └── spatial/
│   │       └── geodesic/
│   │           ├── vae.yaml
│   │           ├── codebook.yaml
│   │           ├── transformer.yaml
│   │           ├── generate.yaml
│   │           └── evaluate.yaml
│   ├── cifar10/
│   │   ├── spatial/geodesic/...           # Same layout as above
│   │   └── vanilla/{euclidean,geodesic}/... 
│   ├── sandbox-fashion/...                # Sandbox presets
│   ├── improved_cifar10/...               # Experimental improved configs
│   ├── data.yaml                          # Default dataset selector for legacy flow
│   └── vae.yaml                           # Default VAE config for legacy flow
├── docs/                      # Documentation and summaries
├── experiments/               # Output directory for models and artifacts
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

## Quick Start: Main Pipeline (via `scripts/`)

The end-to-end pipelines are orchestrated by the scripts in `scripts/`. These scripts expect to be run from within the `scripts/` directory (paths inside use `../`).

### 0. Environment creation

```bash
conda create -n "environment_name" python=3.11
```

### 1. Environment Setup

```bash
conda activate "environment_name"
./scripts/setup_env.sh


# Install project in development mode (required for demos)
pip install -e .
```

> **Note:** If you encounter a "Permission denied" error when running any shell script (e.g., `./scripts/setup_env.sh`), you may need to make it executable first:
>
> ```bash
> chmod +x ./scripts/setup_env.sh
> ```
> Repeat this for any other script as needed.
```


### 2. Download Data

```bash
./scripts/download_data.sh fashion   # or: mnist | cifar10
```

### 3. Run a Pipeline

Change into the `scripts/` directory before launching any pipeline:

```bash
cd scripts
```

- FashionMNIST Spatial Geodesic (full pipeline):
```bash
python run_fashionmnist_spatial_geodesic_pipeline.py \
  [--skip-vae] [--skip-codebook] [--skip-transformer] [--skip-generation] [--skip-evaluation]
```

- CIFAR-10 Spatial Geodesic (full pipeline):
```bash
python run_cifar10_spatial_geodesic_pipeline.py \
  [--skip-vae] [--skip-codebook] [--skip-transformer] [--skip-generation] [--skip-evaluation]
```

- FashionMNIST Vanilla Euclidean pipeline:
```bash
python run_fashionmnist_vanilla_euclidean_pipeline.py \
  [--skip-vae] [--skip-codebook] [--skip-transformer] [--skip-generation] [--skip-evaluation]
```

- FashionMNIST Vanilla Geodesic pipeline (with extra checks):
```bash
python run_fashionmnist_vanilla_geodesic_pipeline.py \
  [--skip-vae] [--skip-vae-check] [--skip-codebook] [--skip-quantization-analysis] \
  [--skip-codebook-health] [--skip-transformer] [--skip-generation] [--skip-evaluation]
```

- CIFAR-10 Vanilla Euclidean pipeline:
```bash
python run_cifar10_vanilla_euclidean_pipeline.py \
  [--skip-vae] [--skip-codebook] [--skip-transformer] [--skip-generation] [--skip-evaluation]
```

Outputs are written under `experiments/<dataset>/<variant>/<distance>/...` and include `vae/`, `codebook/`, `transformer/`, and `evaluation/` subfolders.

### Manual Pipeline

If you prefer running each step manually, use the per-step scripts and configs:

1) Train Spatial VAE
```bash
python src/scripts/train_vae.py --config configs/fashionmnist/spatial/geodesic/vae.yaml
```

2) Build Geodesic Codebook (CLI mirrors `configs/.../codebook.yaml`)
```bash
python src/scripts/build_codebook.py \
  --latents_path experiments/fashionmnist/spatial/geodesic/vae/spatial_vae_fashionmnist/latents_train/z.pt \
  --vae_ckpt_path experiments/fashionmnist/spatial/geodesic/vae/spatial_vae_fashionmnist/checkpoints/best.pt \
  --out_dir experiments/fashionmnist/spatial/geodesic/codebook \
  --in_channels 1 --output_image_size 28 --latent_dim 16 \
  --enc_channels 64 128 256 --dec_channels 256 128 64 \
  --recon_loss mse --norm_type batch --mse_use_sigmoid \
  --k 20 --sym union --K 512 --init kpp --seed 42 --batch_size 512
```

3) Train Transformer
```bash
python src/scripts/train_transformer.py --config configs/fashionmnist/spatial/geodesic/transformer.yaml
```

4) Generate Samples
```bash
python src/scripts/generate_samples.py --config configs/fashionmnist/spatial/geodesic/generate.yaml
```

5) Evaluate
```bash
python src/eval/evaluate_model.py --config configs/fashionmnist/spatial/geodesic/evaluate.yaml
```

## Legacy Quick Start (Vanilla VAE)

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

Train vanilla VAE model:
```bash
python src/scripts/train_vanilla_vae.py --config configs/fashionmnist/vanilla/euclidean/vae.yaml
```

Build codebook (post‑hoc quantization on latents) for the chosen preset by passing the explicit flags analogous to the corresponding `configs/.../codebook.yaml`.

Compare Euclidean vs Geodesic codebooks:
```bash
python demos/codebook_comparison.py --config test2
```

Available demos:
```bash
python demos/interactive_exploration.py
python demos/codebook_comparison.py
python demos/kmedoids_geodesic_analysis.py
python demos/codebook_sampling.py
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

Configuration lives under `configs/<dataset>/<variant>/<distance>/` and is per‑step:
- `vae.yaml` – Spatial or vanilla VAE training
- `codebook.yaml` – Reference values/paths for codebook building (pass via CLI flags)
- `transformer.yaml` – Transformer training on codes
- `generate.yaml` – Sample generation settings
- `evaluate.yaml` – Quantitative evaluation settings

Legacy helpers:
- `configs/data.yaml` – quick dataset selector for older scripts
- `configs/vae.yaml` – default VAE config for legacy flow

## Results

Experimental outputs are organized by dataset/variant under `experiments/`:
- `experiments/fashionmnist/spatial/geodesic/vae/` – Spatial VAE checkpoints and latents
- `experiments/fashionmnist/spatial/geodesic/codebook/` – k‑NN graphs, codebook, codes.npy
- `experiments/fashionmnist/spatial/geodesic/transformer/` – Transformer checkpoints
- `experiments/fashionmnist/spatial/geodesic/evaluation/` – Generated grids and metrics

Notes:
- Ensure `in_channels` matches the dataset (`1` for MNIST/FashionMNIST, `3` for CIFAR10)
- `output_image_size`: 28 for MNIST/FashionMNIST, 32 for CIFAR10
- Use the dataset‑specific configs under `configs/cifar10/...` to replicate CIFAR10 runs

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

