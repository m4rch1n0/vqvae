# CIFAR-10 Post-Hoc VQ-VAE Analysis: Geodesic vs Euclidean Quantization

## Executive Summary

This document reports a systematic investigation of post-hoc vector quantization for CIFAR-10 using geodesic distances versus traditional Euclidean clustering. **Key finding: Euclidean K-means consistently outperforms geodesic methods by ~1.4 dB PSNR**, challenging the hypothesis that manifold-aware quantization improves reconstruction quality.

## Objective

Improve post-hoc VQ-VAE reconstruction quality on CIFAR-10 by replacing Euclidean distance with geodesic distances computed on k-NN graphs in the latent space. Expected benefits:
- Better preservation of manifold structure
- Improved code assignment consistency
- Higher reconstruction fidelity

## Methodology

### Initial State
- **Base VAE**: Trained on CIFAR-10 with MSE loss, 128D latent space
- **Continuous baseline**: 24.04 dB PSNR, 0.9411 SSIM (mu vs z reconstructions)
- **Initial symptoms**: Severe quantization degradation (11-13 dB PSNR)

### Systematic Investigation

#### 1. Assignment Pipeline Diagnosis
**Problem**: Training used geodesic codebook, validation used Euclidean nearest-neighbor assignments
- **Solution**: Implemented robust geodesic assignment (`assign_codes_val_geodesic.py`)
- **Improvement**: +2 dB PSNR (11.11 → 13.23 dB)

#### 2. Latent Representation Choice
**Hypothesis**: Posterior means (μ) provide smoother manifold than stochastic samples (z)
- **Test**: Built codebooks using `mu.pt` vs `z.pt`
- **Result**: +0.46 dB improvement (13.27 → 13.73 dB PSNR)

#### 3. Graph Hyperparameter Optimization
**Variables tested**:
- **Connectivity**: k=30 vs k=50 neighbors
- **Metric**: Euclidean vs cosine distance  
- **Symmetry**: union vs mutual symmetrization

**Results**:
- k=50: +0.29 dB improvement (13.73 → 14.02 dB)
- Cosine metric: Similar performance, lower entropy (14.04 dB)
- Union symmetry: Better connectivity, consistent across all tests

#### 4. Euclidean Baseline Comparison
**Implementation**: Standard K-means clustering with K=1024, K=2048
- **K=1024**: 15.47 dB PSNR, 0.3828 SSIM
- **K=2048**: 15.48 dB PSNR, 0.3857 SSIM (minimal improvement)

## Results

### Performance Hierarchy
| Method | PSNR (dB) | SSIM | Code Usage | Dead Codes |
|--------|-----------|------|------------|------------|
| **Continuous (μ)** | **24.04** | **0.9411** | - | - |
| **Euclidean K-means** | **15.47** | **0.3828** | 406/1024 | 618 |
| Geodesic (μ, k=50, euclidean) | 14.02 | 0.2269 | 529/1024 | 495 |
| Geodesic (μ, k=30, cosine) | 14.04 | 0.2117 | 442/1024 | 582 |
| Geodesic (μ, k=30, euclidean) | 13.73 | 0.2123 | 507/1024 | 517 |

### Key Findings

1. **Euclidean superiority**: Simple K-means outperforms all geodesic methods by ~1.4 dB
2. **Capacity plateau**: Increasing K from 1024→2048 yields negligible improvement (0.01 dB)
3. **Fundamental gap**: ~8.5 dB difference between continuous and best quantized reconstruction
4. **Graph artifacts**: Geodesic methods show lower code usage and entropy, suggesting over-conservative clustering

## Analysis of Failure Modes

### Why Geodesic Methods Underperformed

1. **Manifold assumption violation**: CIFAR-10 latent space may not form a coherent low-dimensional manifold
2. **Graph connectivity artifacts**: k-NN graphs introduce discretization noise vs smooth Euclidean space
3. **K-medoids limitations**: Medoid constraint forces centers to existing points rather than optimal centroids
4. **Distance distortion**: Shortest path distances may amplify local irregularities in the latent space

### Fundamental Quantization Gap

The 8.5 dB gap between continuous and quantized reconstruction suggests:
- **VAE latent space limitation**: CIFAR-10 complexity may require higher-dimensional continuous representations
- **Post-hoc quantization penalty**: Joint VQ-VAE training would learn quantization-aware representations
- **Expected range**: Literature suggests joint training achieves 18-22 dB PSNR on CIFAR-10

## Configurations Used

### Key Configuration Files (Cleaned Codebase)
- **`configs/quantize_cifar10_k1024.yaml`**: Main CIFAR-10 geodesic config (K=1024, k=30)
- **`configs/quantize_cifar10.yaml`**: General CIFAR-10 geodesic config (K=512, k=48)  
- **`configs/vae.yaml`**: VAE architecture (128D latent, MSE loss, no sigmoid)
- **`configs/data.yaml`**: CIFAR-10 dataset configuration
- **`configs/train.yaml`**: Training hyperparameters

### Optimal Configuration (for reference)
```yaml
# Best geodesic result: modify configs/quantize_cifar10_k1024.yaml to use mu.pt
data:
  latents_path: experiments/vae_cifar10/latents_train/mu.pt  # Change from z.pt
graph:
  k: 50          # Increase from 30
  metric: euclidean
  sym: union
  mode: distance
quantize:
  K: 1024
```

### Euclidean Baseline (Best Overall)
Standard K-means clustering achieves 15.47 dB PSNR:
- **Latents**: `experiments/vae_cifar10/latents_train/mu.pt` 
- **Method**: `sklearn.KMeans(n_clusters=1024, random_state=42)`
- **Output**: `experiments/euclidean_baseline_k1024/`

## Recommendations

### For Immediate Use
- **Use Euclidean K-means with K=1024** on posterior means (μ) for best reconstruction quality
- **Configuration**: `experiments/euclidean_baseline_k1024/`

### For Research Directions
1. **Joint VQ-VAE training**: Expected 3-7 dB improvement over post-hoc methods
2. **Alternative geodesic approaches**: Learned graph weights, diffusion distances, local isometry preservation
3. **Class-conditional codebooks**: Exploit CIFAR-10 labels for improved locality
4. **Hybrid methods**: Geodesic-guided initialization with Euclidean refinement

### Geodesic Method Value
While reconstruction quality lags, geodesic methods may excel at:
- **Semantic consistency**: Better preservation of meaningful latent neighborhoods
- **Sample quality**: Improved generation from discrete codes
- **Interpretability**: Manifold-aware code organization

## Technical Artifacts

### Verified Implementations
- `src/training/assign_codes_val_geodesic.py`: Robust geodesic assignment with connectivity safeguards
- `src/training/assign_codes.py`: General Euclidean assignment utility
- `src/training/build_codebook.py`: Geodesic codebook construction pipeline
- `src/eval/eval_quantized.py`: Consistent evaluation pipeline with proper normalization
- `demos/codebook_sampling.py`: Visualization tools for reconstruction comparison

### Experimental Outputs
- `experiments/geo/codebook_cifar10_k1024_mu_k50/`: Best geodesic result (14.02 dB)
- `experiments/euclidean_baseline_k1024/`: Best overall result (15.47 dB)

## Conclusion

This investigation successfully diagnosed and resolved assignment pipeline issues, achieving metrically consistent quantization. However, **geodesic quantization does not improve reconstruction quality over simple Euclidean clustering for CIFAR-10**. The negative result provides valuable insight: manifold-aware quantization benefits may be task-dependent and require careful evaluation against simpler baselines.

The established baseline (15.47 dB PSNR) and systematic methodology provide a solid foundation for future research into alternative geometric quantization approaches.
