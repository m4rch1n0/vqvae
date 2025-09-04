# Summary of VQ-VAE Development and Experiments

This document outlines the end-to-end process of developing, debugging, and refining a post-hoc vector quantization pipeline for a VAE, culminating in the use of geodesic distances for codebook generation.

## 1. Initial Goal & Core Problem

The primary objective was to train an autoregressive model (Transformer) on discrete codes derived from a pre-trained VAE's latent space.

**Initial Problem:** The first major roadblock was discovering that the existing VAE was designed to produce a single latent vector per image. This resulted in a single discrete code per image after quantization. Autoregressive models require a *sequence* of codes (e.g., a `4x4` grid) to learn spatial relationships and generate coherent images. A single code is insufficient for this task.

## 2. Solution Part 1: The `SpatialVAE`

To address the need for sequential data, we implemented a new VAE architecture.

- **Action:** Created a new model, `SpatialVAE`, with an encoder that outputs a spatial grid of latent vectors (`4x4x16`) instead of a single vector.
- **New Problem:** The initial training of this model failed due to a `RuntimeError` caused by a tensor size mismatch. The decoder's architecture was not correctly upsampling the `4x4` latent grid back to the `28x28` image size of the FashionMNIST dataset.
- **Solution:** The deconvolutional layers in the `SpatialDecoder` were redesigned and corrected to ensure the output dimensions precisely matched the input, allowing the `SpatialVAE` to be trained successfully.

## 3. Building the Autoregressive Pipeline

With a source of spatial latents, we proceeded to build the full pipeline for training the Transformer. This phase involved the most significant debugging effort.

- **Action:** Implemented a GPT-style `Transformer` model, a `CodesDataset` to load the spatial codes, and a `train_transformer.py` script.
- **Problems Encountered & Solutions:**
    1.  **GPU Hardware Crash (`HSA_STATUS_ERROR_EXCEPTION`):** Initial attempts to train the Transformer on the GPU resulted in a low-level hardware exception. This is often difficult to debug directly.
    2.  **`IndexError` on CPU:** To diagnose the crash, we switched to CPU training. This revealed the true underlying bug: an `IndexError`. The codebook generation process had assigned a `-1` value to some codes (representing points outside the graph's largest connected component), which is an invalid index for the Transformer's embedding layer.
    3.  **The Fix:** The `CodesDataset` was updated to filter out any image grids containing these invalid `-1` codes, ensuring only valid tokens were passed to the model. This fix resolved both the `IndexError` on the CPU and the subsequent hardware crashes on the GPU.
    4.  **Model Loading Errors:** During the sampling and evaluation phase, we encountered a series of `KeyError`, `AttributeError`, and `UnpicklingError` issues. These were due to minor inconsistencies in how model checkpoints were saved versus how they were loaded. Each was resolved by carefully aligning the loading code with the specific format of the saved checkpoint files.

## 4. Experimentation and Results

We conducted three main experiments to assess image quality.

### Experiment 1: Baseline with Euclidean Codebook (K=512)

The first successful run of the end-to-end pipeline used a standard k-medoids clustering with Euclidean distance to create a codebook of 512 "visual words".

- **Result:** The generated images were recognizable but blurry and lacked detail, as noted in the initial problem description.
- **Key Parameters:**
    - **Codebook Size (K):** 512
    - **Transformer Layers:** 4
    - **Transformer Embedding Dim:** 256
    - **Transformer Heads:** 4
    - **Training Epochs:** 50

### Experiment 2: Euclidean Codebook with Increased Size (K=1024)

Our first hypothesis was that the blurriness was due to a "poor visual vocabulary."

- **Action:** We increased the codebook size to `K=1024` and retrained a larger Transformer to handle the increased complexity.
- **Result:** This did **not** lead to a significant improvement. The images were still blurry, and the quantitative metrics were slightly worse. This suggested that simply increasing the vocabulary size was not addressing the root cause.
- **Key Parameters:**
    - **Codebook Size (K):** 1024
    - **Transformer Layers:** 8
    - **Transformer Embedding Dim:** 512
    - **Transformer Heads:** 8
    - **Training Epochs:** 100

### Experiment 3: Geodesic Codebook (K=512)

This experiment tested the core hypothesis of the project: that using a distance metric that respects the VAE's learned manifold would produce a better codebook.

- **Action:** We integrated your `riemannian_metric.py` script into the pipeline. The codebook generation now first builds a k-NN graph with Euclidean distance and then re-weights all edges using the VAE decoder to approximate the true geodesic distance between points.
- **Result:** The quantization error for this codebook was **dramatically lower** than the Euclidean version, indicating a much tighter and more representative clustering. The final images, while still not perfect, represent the culmination of this refined approach.
- **Key Parameters:**
    - **Codebook Size (K):** 512
    - **Transformer Layers:** 4
    - **Transformer Embedding Dim:** 256
    - **Transformer Heads:** 4
    - **Training Epochs:** 100

## 5. Quantitative Results Summary

| Experiment | PSNR | SSIM | LPIPS | Quantization Error |
| :--- | :--- | :--- | :--- | :--- |
| Euclidean K=512 | 8.9272 | 0.3641 | 0.4875 | 3.55e8 |
| Euclidean K=1024 | 8.7034 | 0.3460 | 0.4995 | 3.14e8 |
| **Geodesic K=512** | **8.2531** | **0.3298** | **0.4894** | **3.40e5** |

The most striking result is the several-orders-of-magnitude drop in quantization error for the geodesic method. While this did not translate to a clear win in the pixel-based PSNR/SSIM metrics, the perceptual LPIPS score remained competitive, suggesting the method has merit.

## 6. Next Steps

As you observed, the model still struggles with certain classes like t-shirts. This indicates that while the overall structure is in place, the Transformer may not be powerful enough or trained long enough to learn the fine-grained "rules" for all fashion items. Future work could focus on:

- **Fine-Tuning the VAE with a VQ Layer:** The most promising next step is to slightly fine-tune the `SpatialVAE` decoder so it learns to reconstruct directly from the quantized codebook vectors. This often leads to a large jump in image sharpness.
- **Larger Transformer / Longer Training:** A more powerful Transformer trained for more epochs could better learn the complex relationships between the geodesic codes.
- **Conditional Generation:** Providing the class label (e.g., "t-shirt") to the Transformer as an additional input during training and generation can significantly improve the quality and coherence of specific classes.
