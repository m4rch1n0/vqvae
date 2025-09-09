# Riemannian vs Euclidean Metric Sanity Check

## Objective

Evaluate whether Euclidean distances in the latent space reflect the "perceptual" metric induced by the VAE decoder, to motivate the use of geodesics in quantization.

## Experimental Setup

### Data and Model
- **Dataset**: MNIST validation split
- **Model**: VAE trained model (checkpoint: `experiments/vae_mnist/checkpoints/best.pt`)
- **Latent space**: 16 dimensions

### Procedure
1. Build k-NN graph with k=10 on latent space z
2. Random sampling of 2,000 edges from the graph
3. Compare:
   - **Euclidean length**: standard distance in latent space
   - **Riemannian length**: approximated via decoder JVP using pullback metric $G = J^T J$

### Riemannian Formula
The Riemannian length is approximated as:

$L_ij \approx \frac{1}{2} (\|J(z_i)\Delta z\|_2 + \|J(z_j)\Delta z\|_2)$

where J is the Jacobian of the post-sigmoid decoder and $\Delta z = z_j - z_i$.

## Results

### Key Metrics
- **Pearson correlation**: corr(R,E) = **0.422**
- **Mean ratio**: dr/de = **2.571**

### Interpretation
- The moderate correlation (0.422) indicates that Euclidean distances in latent space do not faithfully reflect the decoder-induced metric
- The mean ratio > 1 shows a pattern of non-uniform **dilation**: the decoder expands distances on average
- Many points above the y=x diagonal in the scatter plot confirm that the Riemannian metric does not preserve Euclidean ordering well

## Conclusions

**Since the decoder induces a non-Euclidean geometry, quantization based on geodesics (graph distances) is more coherent than simple Euclidean K-means in latent space.**

This result provides empirical motivation for using quantization methods that account for the intrinsic geometry of the latent space rather than assuming a Euclidean structure.

## Generated Files

- `riemann_analysis.png`: Combined plot with scatter plot and histogram
- `sanity_stats.npz`: NumPy file with all numerical data and statistics

### Configuration

The script uses hardcoded configuration values at the top of the file. To modify parameters, edit these constants:

- `LATENTS_PATH`: Path to latent vectors file
- `CHECKPOINT_PATH`: Path to VAE checkpoint
- `K_NEIGHBORS`: Number of neighbors in k-NN graph (default: 10)
- `MAX_EDGES`: Maximum number of edges to sample (default: 2000)
- `OUTPUT_DIR`: Output directory (default: experiments/geo/riemann_sanity)

## Technical Notes

- The script uses JVP (Jacobian-Vector Product) to efficiently compute the metric pullback
- The decoder is evaluated in `eval()` mode for consistency
- Computations are optimized for batch processing to handle large numbers of edges
- The script automatically detects GPU availability and uses it if available
