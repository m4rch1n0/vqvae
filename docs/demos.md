# Analysis & Visualization Demos

This document describes interactive and analytical scripts for exploring the VAE latent space and quantization methods.

## `interactive_exploration.py`

An interactive visualization tool for exploring k-NN graph connectivity and geodesic distance patterns using synthetic data. Because it uses synthetic data, it does not require a pre-trained model.

**Signature:** `python demos/interactive_exploration.py`

**Features:**
- Adjust the `k` value for the k-NN graph dynamically.
- Switch between Euclidean and geodesic distance visualizations.
- Click on points to see distance patterns in real-time.

## `kmedoids_geodesic_analysis.py`

Performs comprehensive geodesic k-medoids clustering analysis on a latent space using a k-NN graph. Evaluates codebook quality through multiple metrics and visualizations.

**Signature:** `python demos/kmedoids_geodesic_analysis.py <experiment_directory> [options]`

**Example Usage:**
```bash
# Basic analysis with default parameters (K=[32,64,128], k_graph=10)
python demos/kmedoids_geodesic_analysis.py experiments/fashionmnist/vanilla/euclidean

# Custom codebook sizes and connectivity
python demos/kmedoids_geodesic_analysis.py experiments/fashionmnist/vanilla/euclidean \
  --K_values 64,128,256 --k_graph 20

# Higher connectivity with union symmetry
python demos/kmedoids_geodesic_analysis.py experiments/fashionmnist/vanilla/euclidean \
  --k_graph 15 --graph_sym union --K_values 32,64
```

**Output Directory:** `demo_outputs/kmedoids_geodesic_{timestamp}/`

**Generated Files:**
- `metrics.csv` and `metrics.json` with detailed clustering quality metrics (purity, NMI, ARI, perplexity).
- `elbow.png`: A plot showing Quantization Error vs. Codebook Size (K).
- `pca_clusters...` and `code_usage...` plots for visual analysis.

**Note:** The script automatically detects latents and labels from the experiment directory. If labels are found, it computes additional clustering quality metrics (purity, NMI, ARI).

## `codebook_comparison.py`

Compares geodesic vs. Euclidean quantization methods for post-hoc VQ-VAE codebook construction.

**Signature:** `python demos/codebook_comparison.py <experiment_directory> [--K size] [--k_graph connectivity]`

**Objective:** To evaluate reconstruction quality (Reconstruction MSE), code usage (Perplexity), and clustering fit (Quantization Error) for the two methods.

**Example Usage:**
```bash
# Basic comparison with default parameters (K=64, k_graph=10)
python demos/codebook_comparison.py experiments/fashionmnist/vanilla/euclidean

# Larger codebook for more detailed comparison
python demos/codebook_comparison.py experiments/fashionmnist/vanilla/euclidean --K 256

# Higher connectivity for geodesic method
python demos/codebook_comparison.py experiments/fashionmnist/vanilla/euclidean --K 128 --k_graph 20
```

**Output:** Creates timestamped directory in `demo_outputs/` with comparison plot, metrics JSON, and configuration file.

For configuration details, see `docs/codebook_comparison_tests.md`.

## `codebook_sampling.py`

Creates a comparison grid showing original vs. quantized reconstructions. This is useful for visually inspecting codebook quality.

**Signature:**
```bash
python demos/codebook_sampling.py <experiment_directory> [--num_samples N]
```

**Output Format:** 
- **Top row**: N original reconstructions (from continuous latents)
- **Bottom row**: N quantized reconstructions (from discrete codebook)
- **Total images**: 2×N (comparison pairs)

**Example Usage:**
```bash
# FashionMNIST Vanilla Euclidean
python demos/codebook_sampling.py experiments/fashionmnist/vanilla/euclidean

# FashionMNIST Spatial Geodesic  
python demos/codebook_sampling.py experiments/fashionmnist/spatial/geodesic

# CIFAR-10 with custom options
python demos/codebook_sampling.py experiments/cifar10/vanilla/geodesic \
  --num_samples 8 --out_dir demo_outputs/my_comparison
```

**Note:** The script automatically detects all necessary paths (codebook, checkpoint, latents) from the experiment directory structure. Quantized reconstructions (bottom row) are expected to be lower quality than originals (top row) due to discrete compression.
