## build_and_save
Builds Riemannian geodesic codebook using decoder-induced edge weights.

**Signature:** `build_and_save(config: Dict[str, Any]) -> Path`

**Arguments:**
- `config`: configuration dict with data, model, graph, riemannian, and quantize parameters

**Returns:**
- `output_dir`: path to saved codebook artifacts

**Algorithm:**
1. Build Euclidean k-NN graph for connectivity
2. Re-weight edges with Riemannian distances using `edge_lengths_riemannian()`
3. Apply K-medoids clustering on Riemannian-weighted graph
4. Save codebook and assignments

**Note:** Implements true manifold geodesics vs graph geodesics in `build_codebook.py`.


## _reweight_graph_with_riemannian
Re-weights k-NN graph edges with decoder-induced Riemannian distances.

**Signature:** `_reweight_graph_with_riemannian(W, z, decoder, mode="subset", max_edges=5000, batch_size=512, device=None)`

**Arguments:**
- `W`: sparse k-NN graph with Euclidean edge weights
- `z`: latent points (N, D)
- `decoder`: VAE decoder network
- `mode`: "subset" for stratified sampling, "full" for all edges
- `max_edges`: maximum edges to reweight in subset mode
- `batch_size`: batch size for Riemannian computation
- `device`: computation device

**Returns:**
- `W_riemannian`: graph with Riemannian edge weights

**Formula:** $L_{ij} \approx 0.5 \cdot (\|J(z_i)(z_j - z_i)\|_2 + \|J(z_j)(z_j - z_i)\|_2)$

**Note:** Uses stratified sampling by Euclidean distance quantiles for representative edge coverage.


## Usage Example

```python
# Build Riemannian codebook
import yaml
from src.training.build_riemannian_codebook import build_and_save

with open("configs/riemannian_quantize.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = build_and_save(config)
```

**Configuration:**
```yaml
data:
  latents_path: "experiments/vae_fashion/latents_train/mu.pt"
model:
  checkpoint_path: "experiments/vae_fashion/checkpoints/best.pt"
riemannian:
  mode: "subset"     # or "full" 
  max_edges: 5000    # computational budget
  batch_size: 512    # memory management
```

**Performance:**
- Subset mode: ~2-5 minutes, moderate geometric improvement
- Full mode: ~15-30 minutes, maximum geometric fidelity
