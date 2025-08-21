# src/geo/riemannian_metric.py
"""Riemannian metric for VAE latent space."""
import torch

@torch.no_grad()
def decoder_logits_to_img(logits: torch.Tensor) -> torch.Tensor:
    """Convert decoder logits to image space [0,1] using sigmoid."""
    return torch.sigmoid(logits)

def _compute_jacobian_vector_product(decoder, z: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Compute J(z) * direction using automatic differentiation."""
    z = z.requires_grad_(True)
    
    def decode_to_image(latent_point):
        decoder_output = decoder(latent_point)
        image = torch.sigmoid(decoder_output)
        return image.view(image.size(0), -1)
    
    _, jacobian_vector_product = torch.autograd.functional.jvp(
        decode_to_image, (z,), (direction,)
    )
    return jacobian_vector_product

@torch.no_grad()
def edge_lengths_riemannian(
    decoder,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    """Calculate Riemannian edge lengths between latent points."""
    assert z_start.shape == z_end.shape, "Start and end points must have same shape"
    
    device = next(decoder.parameters()).device
    z_start = z_start.to(device)
    z_end = z_end.to(device)
    
    displacement = z_end - z_start
    edge_lengths = []
    num_edges = z_start.size(0)
    
    for batch_start in range(0, num_edges, batch_size):
        batch_end = min(batch_start + batch_size, num_edges)
        
        start_batch = z_start[batch_start:batch_end]
        end_batch = z_end[batch_start:batch_end]
        displacement_batch = displacement[batch_start:batch_end]
        
        jvp_at_start = _compute_jacobian_vector_product(decoder, start_batch, displacement_batch)
        jvp_at_end = _compute_jacobian_vector_product(decoder, end_batch, displacement_batch)
        
        length_from_start = torch.linalg.vector_norm(jvp_at_start, dim=1)
        length_from_end = torch.linalg.vector_norm(jvp_at_end, dim=1)
        
        batch_lengths = 0.5 * (length_from_start + length_from_end)
        edge_lengths.append(batch_lengths)

    return torch.cat(edge_lengths).to(torch.float32)
