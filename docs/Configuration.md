# Config Directory Structure

## New Organized Structure

The config directory has been reorganized to mirror the experiment structure, making it easier to understand which configs belong to which experimental approach.

### Directory Structure
```
configs/
└── cifar10/
    ├── vanilla/
    │   ├── euclidean/
    │   │   ├── vae.yaml         # VAE training config
    │   │   ├── codebook.yaml    # Euclidean codebook config
    │   │   ├── transformer.yaml # Transformer training config
    │   │   ├── generate.yaml    # Sample generation config
    │   │   └── evaluate.yaml    # Evaluation config
    │   └── geodesic/
    │       ├── vae.yaml         # VAE training config
    │       ├── codebook.yaml    # Geodesic codebook config
    │       ├── transformer.yaml # Transformer training config
    │       ├── generate.yaml    # Sample generation config
    │       └── evaluate.yaml    # Evaluation config
    └── spatial/
        └── geodesic/
            ├── vae.yaml         # Spatial VAE training config
            ├── codebook.yaml    # Geodesic spatial codebook config
            ├── transformer.yaml # Spatial transformer training config
            ├── generate.yaml    # Sample generation config
            └── evaluate.yaml    # Evaluation config
```

### Benefits of New Structure

1. **Clear Hierarchy**: Dataset → Model Type → Distance Metric
2. **Matches Experiment Structure**: Mirrors `experiments/cifar10/...`
3. **Easy Navigation**: Find configs by following logical path
4. **Reduced Redundancy**: File names are shorter (no prefixes needed)
5. **Scalable**: Easy to add new datasets/approaches

### Pipeline Script Updates

All three pipeline scripts have been updated to use the new structure:

- `run_cifar10_vanilla_euclidean_pipeline.py` → `configs/cifar10/vanilla/euclidean/`
- `run_cifar10_vanilla_geodesic_pipeline.py` → `configs/cifar10/vanilla/geodesic/`  
- `run_cifar10_spatial_geodesic_pipeline.py` → `configs/cifar10/spatial/geodesic/`

### Usage Examples

```bash
# Run vanilla euclidean pipeline
python run_cifar10_vanilla_euclidean_pipeline.py

# Skip VAE training for spatial geodesic
python run_cifar10_spatial_geodesic_pipeline.py --skip-vae

# Run only evaluation for vanilla geodesic
python run_cifar10_vanilla_geodesic_pipeline.py --skip-vae --skip-codebook --skip-transformer --skip-generation
```

### Future Additions

To add new experiments, follow the same pattern:

```
configs/
└── {dataset}/         # e.g., fashionmnist, celeba
    └── {model_type}/   # e.g., vanilla, spatial, hierarchical
        └── {distance}/ # e.g., euclidean, geodesic, cosine
            ├── vae.yaml
            ├── codebook.yaml
            ├── transformer.yaml
            ├── generate.yaml
            └── evaluate.yaml
```
