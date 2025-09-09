# VQ-VAE Comparison Protocol

This document outlines the technical protocol for comparing the end-to-end baseline VQ-VAE against the post-hoc quantization pipelines (Euclidean and Geodesic). The comparison is managed by a set of three core scripts located in the `scripts/` directory.

## System Overview

The framework is designed to produce a fair and reproducible comparison between different generative modeling approaches. It operates in three main stages:

1.  **Baseline Training & Evaluation:** The end-to-end VQ-VAE model is trained and evaluated to produce a set of benchmark metrics.
2.  **Post-Hoc Pipeline Execution:** The various post-hoc pipelines (vanilla Euclidean, vanilla Geodesic, spatial Geodesic) are executed to produce their own evaluation metrics.
3.  **Comparative Analysis:** The results from all approaches are aggregated into a unified report with tables and visualizations.

---

## Core Scripts

### 1. `run_baseline_pipeline.py`

This script is the main entry point for managing the baseline VQ-VAE. It automates the training and evaluation process.

**Purpose:** To train the end-to-end VQ-VAE model from scratch and subsequently evaluate it to generate the necessary metric files for comparison.

**Usage:**
```bash
python scripts/run_baseline_pipeline.py [OPTIONS]
```

**Key Arguments:**
*   `--skip-training`: Use a pre-existing trained model and proceed directly to evaluation.
*   `--skip-evaluation`: Train the model but do not run the evaluation step.
*   `--skip-comparison`: Do not run the final `compare_all_approaches.py` script after evaluation.
*   `--epochs INT`: Override the number of training epochs specified in the config file.

---

### 2. `evaluate_baseline_simple.py`

This script is a standalone utility for evaluating a pre-trained baseline VQ-VAE model. It is typically called by `run_baseline_pipeline.py` but can be run manually.

**Purpose:** To generate evaluation artifacts for a baseline VQ-VAE checkpoint that are in a format consistent with the post-hoc pipelines.

**Usage:**
```bash
python scripts/evaluate_baseline_simple.py --baseline_dir <path> --checkpoint <path> --out_dir <path>
```

**Key Arguments:**
*   `--baseline_dir`: Path to the baseline model's root directory (e.g., `baseline VQVAE/vqvae_cifar10_clean`).
*   `--checkpoint`: Path to the specific model checkpoint file (e.g., `outputs/checkpoints/ckpt_best.pt`).
*   `--out_dir`: The directory where the final evaluation artifacts will be saved.

**Outputs:**
*   `evaluation_results.json`: A comprehensive JSON file with all computed metrics.
*   `metrics.yaml`: A simplified file containing PSNR, SSIM, and LPIPS, for compatibility.
*   `codebook_health.json`: Contains statistics like entropy and codebook usage.
*   `generated_samples.png`: A grid of unconditionally generated samples.
*   `comparison_grid.png`: A grid comparing real test images to their reconstructions.

---

### 3. `compare_all_approaches.py`

This is the final script in the pipeline. It aggregates all results and produces the final comparative analysis.

**Purpose:** To collect evaluation data from all specified approaches (baseline and post-hoc) and generate a unified report.

**Usage:**
```bash
python scripts/compare_all_approaches.py --out_dir <path> --approaches <list_of_approaches>
```

**Key Arguments:**
*   `--out_dir`: The directory where the final comparison report and figures will be saved (e.g., `experiments/cifar10/comparison`).
*   `--approaches`: A space-separated list of the approaches to compare. The default is `baseline vanilla_euclidean vanilla_geodesic spatial_geodesic`.

**Outputs:**
*   `comparison_report.md`: A detailed markdown report with analysis and tables.
*   `comparison_results.csv` / `.json`: The aggregated data in tabular formats.
*   `comparison_charts.png`: Bar charts visualizing the key performance metrics.
*   `entropy_vs_psnr.png`: A scatter plot analyzing the trade-off between codebook usage and reconstruction quality.

---

## Step-by-Step Workflow

To perform the full comparison, follow these steps from the root of the `vqvae` repository:

1.  **Activate Environment:**
    ```bash
    conda activate rocm_env
    ```

2.  **Run Post-Hoc Pipelines:** Ensure that you have already run the desired post-hoc pipelines (e.g., `run_fashionmnist_vanilla_euclidean_pipeline.py`), as their output is required for the final comparison.

3.  **Run the Baseline Pipeline:** Execute the main baseline script. If you have already trained a model, you can skip the training step.
    ```bash
    # To train and evaluate from scratch
    python scripts/run_baseline_pipeline.py

    # To evaluate a pre-existing model
    python scripts/run_baseline_pipeline.py --skip-training
    ```
    This script will automatically run the evaluation and the final comparison at the end.

4.  **Review Results:** The final, unified comparison will be located in `experiments/cifar10/comparison/`.

## Expected Directory Structure

For the comparison scripts to work correctly, they expect the results of each pipeline to be organized as follows:

```
experiments/cifar10/
├── baseline_vqvae/
│   └── evaluation/
│       ├── evaluation_results.json
│       └── metrics.yaml
├── vanilla/
│   ├── euclidean/
│   │   └── evaluation/
│   │       └── metrics.yaml
│   └── geodesic/
│       └── evaluation/
│           └── metrics.yaml
└── spatial/
    └── geodesic/
        └── evaluation/
            └── metrics.yaml
```
