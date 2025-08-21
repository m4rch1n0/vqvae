## decoder_logits_to_img
Converts decoder logits to image space [0,1] using sigmoid activation.

**Signature:** `decoder_logits_to_img(logits: torch.Tensor) -> torch.Tensor`

**Arguments:**
- `logits`: decoder output logits tensor

**Returns:**
- `torch.Tensor`: images in [0,1] range after sigmoid

**Note:** Applied with `@torch.no_grad()` for efficiency during inference.


## edge_lengths_riemannian
Calculates Riemannian edge lengths between latent points using decoder-induced metric.

**Signature:** `edge_lengths_riemannian(decoder, z_start: torch.Tensor, z_end: torch.Tensor, batch_size: int = 512) -> torch.Tensor`

**Arguments:**
- `decoder`: VAE decoder network
- `z_start`: starting latent points (num_edges, latent_dim)
- `z_end`: ending latent points (num_edges, latent_dim)  
- `batch_size`: process edges in batches to manage memory (default: 512)

**Returns:**
- `edge_lengths`: tensor of Riemannian distances (num_edges,)

**Formula:** $L_{ij} ≈ 0.5 * (||J(z_i)(z_j - z_i)||_2 + ||J(z_j)(z_j - z_i)||_2)$

**Note:** Averages lengths computed from both endpoints for better geodesic approximation. The metric is induced by the decoder: $G(z) = J(z)^T J(z)$.


## _compute_jacobian_vector_product
Internal function that computes Jacobian-vector products using automatic differentiation.

**Signature:** `_compute_jacobian_vector_product(decoder, z: torch.Tensor, direction: torch.Tensor) -> torch.Tensor`

**Arguments:**
- `decoder`: VAE decoder network
- `z`: latent points (batch_size, latent_dim)
- `direction`: displacement vectors (batch_size, latent_dim)

**Returns:**
- `torch.Tensor`: $J(z)$ * direction flattened to (batch_size, num_pixels)

**Note:** Uses `torch.autograd.functional.jvp` to avoid computing the full Jacobian matrix. For small displacement $\delta z: local\_length ≈ ||J(z) \delta z||_2$.
