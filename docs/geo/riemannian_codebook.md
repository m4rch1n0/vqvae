# Building a Geodesic Codebook

This document explains the modern process for building a geodesic codebook, which is handled by the `src/scripts/build_codebook.py` script. This script is a core part of the automated pipelines.

## How it Works

The process creates a "geodesic" codebook by treating the VAE's latent space as a curved surface (a manifold). Instead of clustering points based on simple straight-line distances, it uses distances that approximate the true path along this curved surface.

The algorithm follows these steps:

1.  **Build a k-NN Graph**: First, it builds a standard k-Nearest Neighbors graph using simple Euclidean (straight-line) distances. This step determines which latent vectors are considered "close" to each other and establishes the local connectivity of the manifold.

2.  **Re-weight with Riemannian Metric**: Next, it re-calculates the distance (the "weight") for each connection in the graph. Instead of the straight-line distance, it uses the **Riemannian metric**, which is a better approximation of the "true" distance along the curved latent surface. This is done using the `edge_lengths_riemannian()` function, which is based on the following formula:

    $L_{ij} \approx 0.5 \cdot (\|J(z_i)(z_j - z_i)\|_2 + \|J(z_j)(z_j - z_i)\|_2)$

    Here, $J(z)$ is the Jacobian of the VAE's decoder. This step effectively "stretches" the connections in the graph to match the geometry learned by the VAE.

3.  **Geodesic K-Medoids Clustering**: Finally, it performs K-medoids clustering on this new, more accurate graph. The shortest paths for the clustering are now calculated along the connections of the graph (using Dijkstra's algorithm), approximating a geodesic path.

This process results in a codebook where the selected "code" vectors (medoids) are chosen based on the underlying geometry of the latent space, which can lead to a more meaningful set of codes.

## How to Run

You typically do not need to run `src/scripts/build_codebook.py` by hand. It is designed to be called automatically as part of the main end-to-end pipelines located in the `scripts/` directory.

The main pipeline scripts (e.g., `run_fashionmnist_spatial_geodesic_pipeline.py`) will call this script with all the necessary parameters taken from the configuration files.

If you need to run it manually for debugging, you can see an example of the required command-line flags in the main `README.md` file under the "Manual Pipeline" section.
