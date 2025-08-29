# VQ-VAE with Geodesic Quantization

A research implementation of vector quantization using geodesic distances in VAE latent spaces, exploring post-hoc discrete coding with Riemannian metrics instead of Euclidean distances.

## Overview

This project revisits the classic VQ-VAE pipeline by separating the continuous representation learning from the quantization step. Instead of jointly learning a discrete latent space, we:

1. **Train a standard continuous VAE** (Done on MNIST, Cifar10, Fashion-MNIST)
2. **Apply geodesic K-means clustering** in the latent space using Riemannian distances
3. **Build a discrete codebook** from geodesic centroids  
4. **Train autoregressive transformer** on discrete codes for image generation
5. **Compare reconstruction quality** vs. standard VQ-VAE

**Key Innovation:** Replace Euclidean distance with geodesic metrics that better reflect the curvature and topology of the learned latent manifold, plus autoregressive generation without joint training.

## Project Structure

```
vqvae/
├── src/                   
│   ├── data/              # Dataset loaders (MNIST, FashionMNIST, cifar10)
│   ├── models/            # VAE architecture
│   ├── geo/               # Geodesic computation modules
│   ├── training/          # Training pipeline
│   └── utils/             
├── demos/                  # Interactive examples
│   ├── vae_knn_analysis.py           # Main latent space analysis demo
│   ├── interactive_exploration.py    # Interactive k-NN visualization
│   ├── codebook_comparison.py        # Geodesic vs Euclidean quantization comparison
│   └── kmedoids_geodesic_analysis.py # Geodesic K-medoids (post-hoc quantization)
├── experiments/           # Research experiments  
│   ├── geo/              # Riemannian geometry experiments
│   └── vae_mnist/        # Trained models and latents
├── scripts/              
│   ├── setup_env.sh      # Environment setup
│   ├── download_data.sh  
│   ├── train_vae.sh
│   └── run_experiments.sh 
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
# name: one of [MNIST, FashionMNIST, CIFAR10]
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

Train autoregressive transformer on discrete codes:
```bash
python src/training/train_transformer.py --config-name transformer/mnist_small
```

Generate images from trained transformer:
```bash
python src/generation/generate_images.py \
  --transformer_ckpt experiments/transformer/mnist_small/checkpoints/best.pt \
  --transformer_cfg configs/transformer/mnist_small.yaml \
  --vae_ckpt experiments/vae_mnist/checkpoints/best.pt \
  --codebook_pt experiments/geo/test_optimized_mnist/codebook.pt \
  --H 28 --W 28 --num_samples 16 --temperature 1.0 --top_k 50
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

### VAE Architecture (`src/models/vae.py`)
- Convolutional encoder-decoder with configurable regularization
- Support for MNIST (1×28×28), Fashion-MNIST (1×28×28), CIFAR-10 (3×32×32)
- Configurable latent dimensions and loss functions

### Transformer Model (`src/models/transformer.py`)
- GPT-style architecture with causal self-attention
- Class-conditional generation support
- Configurable model size (layers, heads, embedding dimension)
- Automatic weight initialization following GPT-2 style

### Sequence Dataset (`src/data/sequence_datasets.py`)
- Converts discrete codes to training sequences
- Automatic special token assignment (BOS, EOS, PAD)
- Supports both 1D and 2D code arrays
- Handles class labels for conditional generation

### Geodesic Quantization (`src/geo/`)
- **Riemannian Metric**: Computes decoder-induced Riemannian distances using Jacobian-vector products
- **k-NN Graph**: Builds connectivity graphs in latent space
- **Geodesic Shortest Paths**: Dijkstra's algorithm for multi-source shortest paths
- **K-medoids Clustering**: Graph-based geodesic K-medoids without full distance matrix computation

### Generation Pipeline (`src/generation/`)
- **Sampling Strategies**: Greedy, top-k, top-p (nucleus), temperature-controlled
- **Image Generation**: End-to-end generation from discrete codes to images
- **Conditional Generation**: Class-conditional image synthesis

## Research Questions

- **Geometry**: How does the VAE latent manifold structure affect quantization quality?
- **Connectivity**: Do Riemannian distances preserve better neighborhood relationships?
- **Reconstruction**: Does geodesic clustering improve reconstruction vs. Euclidean methods?
- **Generation**: Can autoregressive models learn meaningful patterns from discrete codes?
- **Comparative Analysis**: When do geodesic methods outperform standard Euclidean quantization?
- **Post-hoc vs Joint**: How does post-hoc quantization compare to end-to-end VQ-VAE training?

## Configuration

Main configuration files in `configs/`:
- `configs/data.yaml` - Dataset configuration
- `configs/vae.yaml` - VAE architecture
- `configs/train.yaml` - Training parameters
- `configs/transformer/*.yaml` - Transformer training configs

**Dataset presets** live under `configs/presets/<dataset>/`. Root files are real copies of the chosen preset.

**Switch dataset quickly**:
```bash
bash scripts/select_dataset.sh mnist   # or fashion, cifar10
```

## Results

Experimental outputs saved to:
- `experiments/geo/` - Riemannian analysis results
- `experiments/transformer/` - Trained transformer models
- `demo_outputs/` - Demo visualizations and quantization comparisons
- `experiments/vae_*/` - Trained models and latent representations

**Key findings**:
- **CIFAR-10**: Euclidean K-means outperforms geodesic methods by ~1.4 dB PSNR
- **Fashion-MNIST**: Both methods achieve >30 dB PSNR (exceptional quantization performance)
- **Dataset dependency**: Structured datasets (Fashion-MNIST) favor post-hoc approaches
- **Method consistency**: Euclidean superiority maintained across dataset complexity levels

## Dependencies

- PyTorch (with ROCm/CUDA support)
- NumPy, SciPy, matplotlib
- scikit-learn (for clustering comparisons)
- Hydra (configuration management)
- MLflow (experiment tracking)

See `requirements.txt` for complete dependencies.

## Documentation

Detailed technical documentation available in `docs/`:
- [Transformer Implementation](docs/transformer.md) - Complete transformer guide
- [Autoregressive Pipeline](docs/autoregressive_pipeline.md) - Autoregressive training pipeline
- [Riemannian Metrics](docs/geo/riemannian_metric.md) - Mathematical formulation
- [Geodesic Shortest Paths](docs/geo/geo_shortest_paths.md) - Graph algorithms
- [k-NN Graph Construction](docs/models/knn_graph.md) - Connectivity analysis

## Citation

This work explores the intersection of differential geometry and discrete representation learning, building on foundations from VQ-VAE and Riemannian geometry in deep learning. The post-hoc approach enables autoregressive generation without joint training, providing insights into the relationship between manifold structure and quantization quality.
