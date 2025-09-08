# Research & Validation Experiments

This document describes low-level scripts used for deep-dives into the geometric properties of the VAE latent space. These are intended for research and validation, not for running the main end-to-end pipeline.

## `riemann_sanity_check.py`

A validation experiment that compares Riemannian vs. Euclidean edge lengths on connections in a k-NN graph.

**Signature:** `python experiments/geo/riemann_sanity_check.py`

**Objective:** To verify that the decoder-induced Riemannian metric provides meaningful geometric information beyond simple Euclidean distance in the latent space.

**Note:** The default file paths in this script may point to legacy experiment outputs (e.g., `experiments/vae_mnist/...`). You may need to update these paths to point to your own trained VAE model and latents.

## `run_riemann_experiments.py`

A comprehensive analysis of how re-weighting a k-NN graph with Riemannian distances affects graph connectivity and shortest path distances.

**Signature:** `python experiments/geo/run_riemann_experiments.py`

**Objective:** To quantify the impact of using Riemannian edge weights on graph properties, providing justification for its use in geodesic-based methods.

**Note:** The default file paths in this script may point to legacy experiment outputs. You may need to update these paths to point to your own trained VAE model and latents.

## Running End-to-End Experiments

Full, end-to-end experiments (from VAE training to final evaluation) should be run using the automated pipeline scripts located in the `scripts/` directory.

Please refer to the main `README.md` for detailed instructions on how to set up the environment and run these pipelines.
