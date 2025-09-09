# Appendix

## A. Model Architectures

The detailed architectures for the main models are provided below.

#### Spatial VAE Encoder

| Layer Type          | Channels In/Out | Kernel Size | Stride | Padding | Activation | Normalization |
| ------------------- | --------------- | ----------- | ------ | ------- | ---------- | ------------- |
| Conv2d              | `in_channels`/64  | 3x3         | 2      | 1       | ReLU       | Batch Norm    |
| Conv2d              | 64/128          | 3x3         | 2      | 1       | ReLU       | Batch Norm    |
| Conv2d              | 128/256         | 3x3         | 2      | 1       | ReLU       | Batch Norm    |
| Conv2d (for `mu`)   | 256/`latent_dim`  | 1x1         | 1      | 0       | -          | -             |
| Conv2d (for `logvar`)| 256/`latent_dim`  | 1x1         | 1      | 0       | -          | -             |

*Note: `in_channels` is 1 for FashionMNIST and 3 for CIFAR-10. `latent_dim` is 16 for FashionMNIST and 32 for CIFAR-10.*

#### Spatial VAE Decoder

| Layer Type         | Channels In/Out | Kernel Size | Stride | Padding | Activation | Normalization |
| ------------------ | --------------- | ----------- | ------ | ------- | ---------- | ------------- |
| Conv2d             | `latent_dim`/256| 1x1         | 1      | 0       | -          | -             |
| ConvTranspose2d    | 256/128         | 4x4         | 2      | 1       | ReLU       | Batch Norm    |
| ConvTranspose2d    | 128/64          | 4x4         | 2      | 1       | ReLU       | Batch Norm    |
| ConvTranspose2d    | 64/`out_channels`| 4x4         | 2      | 3 or 1  | -          | -             |

*Note: The final padding is 3 for FashionMNIST (28x28) and 1 for CIFAR-10 (32x32).*

#### Transformer

| Parameter              | Value     | Description                                |
| ---------------------- | --------- | ------------------------------------------ |
| `num_tokens`           | 512       | Size of the discrete codebook vocabulary.  |
| `embed_dim`            | 256       | Internal embedding dimension.              |
| `n_layers`             | 4         | Number of Transformer blocks.              |
| `n_head`               | 4         | Number of attention heads.                 |
| `max_seq_len`          | 16        | Sequence length from the 4x4 latent grid.  |
| `dropout`              | 0.1       | Dropout rate.                              |

## B. Training Hyperparameters

#### VAE Training

| Parameter              | Value     | Description                           |
| ---------------------- | --------- | ------------------------------------- |
| Optimizer              | AdamW     | -                                     |
| Epochs                 | 200       | With early stopping patience of 20.   |
| Batch Size             | 256       | -                                     |
| Learning Rate          | 1e-3      | -                                     |
| LR Schedule            | Cosine    | With `t_max` of 200.                  |
| Weight Decay           | 1e-5      | -                                     |
| Gradient Clipping      | 1.0       | Max norm for gradient clipping.       |
| $\beta$ (KL weight)    | 1.0       | -                                     |

#### Transformer Training

| Parameter              | Value     | Description                           |
| ---------------------- | --------- | ------------------------------------- |
| Optimizer              | AdamW     | -                                     |
| Epochs                 | 200       | -                                     |
| Batch Size             | 256       | -                                     |
| Learning Rate          | 3e-4      | -                                     |
| Weight Decay           | 0.01      | -                                     |
| Label Smoothing        | 0.1       | -                                     |

## C. Geodesic K-Medoids Algorithm

The geodesic quantization process is a core contribution and is performed as follows, using the mean `mu` vectors from the VAE as input features. The codebook size `K` was set to 512 for all FashionMNIST comparisons.

1.  **Build k-NN Graph:** We first construct a k-Nearest Neighbors graph from the raw `mu` latent vectors using Euclidean distance. The number of neighbors `k` is a hyperparameter set to 50, which was found to produce a single large connected component for both datasets. Symmetrization is performed using the `union` method to maximize connectivity.
2.  **Re-weight with Riemannian Metric:** The weight of graph edges is updated according to the specific method. For our **Partial Riemannian** method, a stratified subset of 5,000 edges is re-weighted. For our **Full Riemannian** method, all unique edges are re-weighted using our approximation of the decoder-induced Riemannian metric. The formula is given in the main text.
3.  **Find Largest Connected Component (LCC):** To ensure all points are reachable, we identify the LCC of the graph. In our experiments, this component contained over 99% of all data points. Clustering is performed only on the nodes within this component.
4.  **K-Medoids Clustering:** We perform K-Medoids clustering on the (potentially re-weighted) graph.
    *   **Initialization:** The initial medoids are chosen using a K-means++ strategy, where distances are calculated as shortest paths on the graph via Dijkstra's algorithm.
    *   **Assignment:** Each node in the LCC is assigned to its closest medoid, again using shortest path distances. This single-shot assignment is highly efficient as it avoids the iterative update steps of traditional K-Medoids.

This graph-based approach avoids computing the full $N \times N$ distance matrix, making it scalable to large datasets.

## D. Dataset Preprocessing

Input images for both FashionMNIST and CIFAR-10 were normalized to the range `[-1, 1]`. For CIFAR-10, standard data augmentation (random horizontal flips and crops) was used during VAE training. No augmentation was used for FashionMNIST.

## E. Evaluation Metrics

The image reconstruction metrics (PSNR and SSIM) were calculated using the custom implementations found in `src/eval/metrics.py`.

*   **PSNR:** The Peak Signal-to-Noise Ratio is computed with the standard formula, assuming a maximum pixel value of 1.0. This implies that input images are expected to be normalized to the range `[0, 1]`.
*   **SSIM:** The Structural Similarity Index is calculated on a per-image basis for the entire batch and then averaged.

As noted in the main text, while these metrics were applied consistently across all experiments, any potential mismatch between the data's normalization range (e.g., `[-1, 1]`) and the metric's expected range (`[0, 1]`) would affect the absolute values reported. Therefore, the primary value of these metrics in our analysis is for the *relative comparison* between methods.
