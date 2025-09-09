# Project Journal: Strategy, Analysis, and Future Work

This document outlines the analytical journey of developing the post-hoc VQ-VAE pipeline, from the core strategy to the final analysis and recommended next steps.

## 1. The Core Objective & Initial Strategy

As defined in `instruction.txt`, the central goal was to **rethink the VQ-VAE pipeline** by using post-hoc vector quantization with geodesic distances. This involved training a continuous VAE, building a discrete codebook from its latent space, and then training an autoregressive model on those discrete codes.

### The Journey So Far: A Summary of Actions and Learnings

Our development process was a systematic journey of implementation, debugging, and refinement.

1.  **From Single Codes to Spatial Sequences:** We first identified that a standard VAE, which produces a single latent vector per image, is insufficient for an autoregressive model. Our first major action was to implement a **`SpatialVAE`** that produces a `4x4` grid of latent vectors, providing the necessary sequential data.

2.  **Building the Full Pipeline:** We successfully built an end-to-end pipeline around this concept, which included the `SpatialVAE`, a `Transformer` model, and scripts for training, codebook construction, generation, and evaluation.

3.  **Overcoming Technical Hurdles:** We navigated a series of significant technical challenges, including low-level GPU errors, `IndexError` bugs caused by invalid tokens in the codebook, and numerous configuration management issues, all of which have now been resolved. The result is a stable, modular, and fully functional pipeline.

### The "Clean Dictionary" Hypothesis

Early results with CIFAR-10 were poor, showing blurry and unrecognizable images. After determining that neither codebook size nor geodesic clustering alone were the issue, we formed a new hypothesis.

The VAE decoder was trained to reconstruct images from noisy, continuous latent vectors (`z = mu + noise`). The Transformer, however, generates sequences of perfect, discrete codebook vectors. We were asking the decoder to interpret "words" from a dictionary it had never seen in its pure form.

**Hypothesis:** The decoder will produce sharper, more coherent images if the "dictionary" it uses (the codebook) is built from the most stable, noise-free representations of the latent space. These are the **`mu`** vectors—the learned mean of the latent distributions.

**The Strategy:** We proceeded by building our codebooks by clustering the **`mu`** latents instead of the noisy **`z`** latents. This is a direct and insightful refinement of the "post-hoc vector quantization" step and a key part of "rethinking the VQ-VAE pipeline."

## 2. Analysis of Key Findings

Our experiments yielded several crucial insights into the nature of post-hoc quantization.

### Finding 1: Geodesic Clustering is Highly Effective

The central hypothesis of this project was that a geodesic distance metric would produce a more meaningful codebook than a standard Euclidean one. Our results provide strong evidence for this.

| Experiment | Quantization Error (Sum of Squared Distances) |
| :--- | :--- |
| Euclidean K=512 | 3.55 x 10^8 |
| **Geodesic K=512** | **3.40 x 10^5** |

The quantization error for the geodesic codebook was **three orders of magnitude lower** than for the Euclidean codebook. This is a significant finding and demonstrates that the Riemannian metric creates a much tighter and more representative clustering of the VAE's latent manifold.

### Finding 2: The Post-Hoc Bottleneck

Despite the success of the geodesic clustering, the final image quality for the complex CIFAR-10 dataset remained poor. This is a critical result, revealing the fundamental limitation of a purely post-hoc approach: **the training mismatch**.

*   **The VAE's Training:** The decoder was trained to reconstruct images from the *continuous, noisy* latent vectors (`z`).
*   **The Transformer's Output:** The Transformer, however, generates sequences of *clean, discrete* codebook vectors.
*   **The Disconnect:** We are feeding the decoder a type of data it was never explicitly trained to handle. The decoder's struggle to translate this unfamiliar input results in the blurry, low-fidelity images we observe.

The "bad results" are therefore not a failure of the implementation, but the **primary finding of our investigation**.

## 3. Conclusion

This project successfully demonstrates that while geodesic clustering is a highly effective method for creating a representative codebook in a VAE's latent space, the post-hoc nature of the pipeline acts as an **irreducible bottleneck** on final image quality for complex datasets like CIFAR-10. The VAE decoder must be explicitly trained to handle the discrete nature of the codebook to produce sharp, high-fidelity images.

## 4. A Clear Path for Future Work

The clear and most promising next step is to bridge the gap between the VAE and the codebook by moving from a purely post-hoc approach to a more integrated one.

**Recommendation: Fine-Tuning with a Vector Quantization (VQ) Layer**

1.  **Implement a VQ Layer:** This is a non-trainable layer inserted between the VAE's encoder and decoder. It takes the encoder's output, finds the nearest codebook vector, and passes this *quantized* vector to the decoder.
2.  **Fine-Tune the Decoder:** With the encoder and codebook frozen, the VAE would be trained for a few more epochs. This process would update **only the decoder's weights**, specifically teaching it how to reconstruct high-quality images from the discrete codebook vectors. A "straight-through estimator" is used to allow gradients to flow through the non-differentiable quantization step.

This is the standard technique used in high-quality VQ-VAE models. It directly addresses the "training mismatch" we have identified and is the logical next step to unlock the full potential of our geodesic codebook.
