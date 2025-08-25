# Codebook Comparison Test Configurations

This document describes the different test configurations for comparing Euclidean vs Geodesic codebooks and what to expect from each.

## Test Configurations Overview

### 1. `test1.yaml` - Standard Configuration
**Purpose**: Baseline comparison between Euclidean and Geodesic approaches
**Parameters**:
- K = 64 (medium codebook)
- k = 10 (moderate connectivity)
- sym = "mutual" (balanced graph)

**Expected Results**:
- **Execution time**: ~2-5 minutes
- **Memory usage**: ~1-2 GB
- **Quality**: Baseline for comparison
- **Performance**: Moderate differences between approaches

### 2. `test2.yaml` - Experimental Configuration
**Purpose**: Optimized for geodesic performance
**Parameters**:
- K = 256 (large codebook)
- k = 10 (moderate connectivity)
- sym = "mutual" (balanced connectivity)

**Expected Results**:
- **Execution time**: ~5-10 minutes
- **Memory usage**: ~3-5 GB
- **Quality**: Significant improvement for geodesic approach
- **Performance**: Clear geodesic advantage

### 3. `test3.yaml` - High Resolution Configuration
**Purpose**: Maximum quantization quality
**Parameters**:
- K = 256 (very large codebook)
- k = 30 (high connectivity)
- metric = "cosine" (angular similarity)

**Expected Results**:
- **Execution time**: ~15-30 minutes
- **Memory usage**: ~8-12 GB
- **Quality**: Excellent, but with diminishing returns
- **Performance**: 
  - Reconstruction MSE: ~20-30% better than test2
  - Perplexity: ~15-25% higher
  - Quantization Error: ~25-35% lower

### 4. `test4.yaml` - Memory Efficient Configuration
**Purpose**: Fast and efficient testing
**Parameters**:
- K = 32 (small codebook)
- k = 8 (low connectivity)
- chunk_size = 500 (memory management)

**Expected Results**:
- **Execution time**: ~30 seconds - 2 minutes
- **Memory usage**: ~200-500 MB
- **Quality**: Low but sufficient for quick tests
- **Performance**: 
  - Reconstruction MSE: ~2-3x worse than test2
  - Perplexity: ~40-60% lower
  - Quantization Error: ~3-4x worse

### 5. `test5.yaml` - Multi-Metric Configuration
**Purpose**: Statistical robustness and comprehensive analysis
**Parameters**:
- K = 128 (balanced codebook)
- k = 15 (moderate connectivity)
- multiple_seeds: 5 different seeds

**Expected Results**:
- **Execution time**: ~25-40 minutes (5x test2)
- **Memory usage**: ~3-5 GB per execution
- **Quality**: Statistically robust
- **Performance**: 
  - Confidence intervals for all metrics
  - Stability analysis across seeds
  - Saved codebooks for detailed analysis

## Key Metrics

### 1. Reconstruction MSE (Lower is Better)
- **Expected range**: 0.001 - 0.1
- **Interpretation**: Mean squared error in reconstruction
- **Geodesic advantage**: 10-40% better for k > 15

### 2. Perplexity (Higher is Better)
- **Expected range**: 2 - 128 (for K = 32-256)
- **Interpretation**: Diversity in codebook usage
- **Geodesic advantage**: 15-50% better for K > 64

### 3. Quantization Error (Lower is Better)
- **Expected range**: 0.1 - 10.0
- **Interpretation**: Average quantization error
- **Geodesic advantage**: 20-60% better for k > 10

## Usage Recommendations

### Development and Debugging
- **Configuration**: `test4.yaml`
- **Reason**: Fast execution, minimal memory usage
- **Time**: < 2 minutes

### Baseline Testing
- **Configuration**: `test1.yaml` or `test2.yaml`
- **Reason**: Good quality/speed balance
- **Time**: 2-10 minutes

### Production and Analysis
- **Configuration**: `test3.yaml` or `test5.yaml`
- **Reason**: Maximum quality or statistical robustness
- **Time**: 15-40 minutes

### Research and Experimentation
- **Configuration**: `test5.yaml`
- **Reason**: Statistical robustness, comprehensive analysis
- **Time**: 25-40 minutes

## Resource Monitoring

### Memory
- **Monitor**: `htop` or `nvidia-smi` (if GPU)
- **Critical threshold**: > 80% RAM used
- **Solutions**: Reduce K, k, or chunk_size

### Execution Time
- **Estimate**: 1-2 minutes per 1000 points × K/64
- **Optimizations**: Reduce k, use metric="euclidean"

### Quality Check
- **Verify**: Reconstruction MSE < 0.01 for K > 128
- **Warning**: Perplexity < K/4 indicates underutilized codebook

## Common Troubleshooting

### Memory Error
```bash
# Reduce parameters
K: 64 -> 32
k: 20 -> 10
chunk_size: 1000 -> 500
```

### Excessive Execution Time
```bash
# Optimize speed
metric: "cosine" -> "euclidean"
k: 30 -> 15
sym: "union" -> "mutual"
```

### Insufficient Quality
```bash
# Increase quality
K: 64 -> 128
k: 10 -> 20
metric: "euclidean" -> "cosine"
```
