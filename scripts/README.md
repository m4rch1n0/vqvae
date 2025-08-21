# Scripts Directory

Utility scripts for project setup, data management, and pipeline orchestration.

## Available Scripts

### Environment Setup
- **`setup_env.sh`** - Configure Python environment and verify GPU support
- **`download_data.sh`** - Download MNIST dataset

### Training and Execution  
- **`train_vae.sh`** - Train VAE model with default parameters
- **`run_experiments.sh`** - Execute complete experimental pipeline

## Usage

**Quick Start:**
```bash
# Setup environment
./scripts/setup_env.sh

# Download data  
./scripts/download_data.sh

# Run complete pipeline
./scripts/run_experiments.sh
```

**Individual Components:**
```bash
# Train VAE only
./scripts/train_vae.sh

# Run specific demos
python demos/vae_knn_analysis.py
python demos/interactive_exploration.py
python demos/kmedoids_geodesic_analysis.py

# Run specific experiments
python experiments/geo/riemann_sanity_check.py
python experiments/geo/run_riemann_experiments.py
```

See main [README.md](../README.md) for complete project documentation.
