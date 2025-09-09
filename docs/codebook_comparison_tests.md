# How to Use the Codebook Comparison Demo

This guide explains how to use the `demos/codebook_comparison.py` script to compare codebook generation methods.

## What it Does

The script compares two different methods for creating a "codebook" from the latent space of a trained VAE:
1.  **Euclidean K-Means**: The standard clustering method based on straight-line distances.
2.  **Geodesic K-Medoids**: A method that uses distances measured along the curved surface (manifold) of the latent space, which can sometimes create more meaningful clusters.

After creating the codebooks, it calculates metrics to help you decide which method works better for your data.

## Configurations

The demo is controlled by configuration files located in `configs/codebook_comparison/`. Two are provided:

-   `test1.yaml`: A standard configuration with a medium-sized codebook (`K=64`). Good for a baseline comparison.
-   `test2.yaml`: An experimental configuration with a larger codebook (`K=256`). Designed to give the geodesic method a better chance to perform well.

**Note**: These configuration files contain paths to a pre-trained VAE model and latent vectors (e.g., `experiments/vae_mnist/...`). If you want to run this demo on your own model, you will need to **update these paths** inside the YAML file.

## How to Run

1.  Make sure you have a trained VAE and the corresponding latent vectors saved.
2.  Open the config file you want to use (e.g., `configs/codebook_comparison/test1.yaml`) and check that the paths are correct.
3.  Run the script from the project's root directory:

```bash
# Run with the standard configuration
python demos/codebook_comparison.py --config test1

# Run with the experimental configuration
python demos/codebook_comparison.py --config test2
```

## Outputs

The script will create a new directory inside `demo_outputs/`. For example, `demo_outputs/codebook_comparison_test1_20231027_103000/`. Inside, you will find:

-   `codebook_comparison.png`: A bar chart that visually compares the key metrics between the two methods.
-   `metrics.json`: A file containing the detailed numerical results for each method.
-   `config.yaml`: A copy of the configuration file you used, so you can easily reproduce your results.

## Understanding the Metrics

The `metrics.json` file and the plot show three key metrics:

1.  **Reconstruction MSE** (Lower is better)
    -   *What it means*: How much image quality was lost when the original latents were replaced by the "quantized" codebook vectors.
2.  **Perplexity** (Higher is better)
    -   *What it means*: How many of the codebook vectors were actually used. A higher number is better because it means the codebook is being used efficiently.
3.  **Quantization Error** (Lower is better)
    -   *What it means*: How far, on average, the original latent vectors are from their assigned codebook vectors. It measures how well the codebook "fits" the data.
