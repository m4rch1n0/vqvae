import torch
import pytest
from src.geo.riemannian_metric import edge_lengths_riemannian


class DummyDec(torch.nn.Module):
    """Dummy decoder for testing: (B,D) -> (B,1,28,28) logits (we are considering MNIST as sandbox)."""
    
    def __init__(self, d=16, hw=28*28):
        super().__init__()
        self.lin = torch.nn.Linear(d, hw)
    
    def forward(self, z):
        return self.lin(z).view(z.size(0), 1, 28, 28)


def _make_pairs(M=64, D=16, eps=0.1, seed=42, device="cpu"):
    """Generate pairs of latent points for testing edge length computation."""
    g = torch.Generator(device=device).manual_seed(seed)
    z_i = torch.randn(M, D, generator=g, device=device)
    noise = torch.randn(M, D, generator=g, device=device)
    z_j = z_i + eps * noise
    return z_i, z_j

def test_basic_shape_and_nonneg_cpu():
    """Test basic shape, dtype, and non-negativity of edge lengths."""
    dec = DummyDec().eval()
    z_i, z_j = _make_pairs()
    L = edge_lengths_riemannian(dec, z_i, z_j, batch_size=32)
    assert L.shape == (z_i.size(0),)
    assert L.dtype == torch.float32
    assert torch.all(L >= 0)

def test_symmetry_swap_pairs_cpu():
    """Test that edge lengths are symmetric when swapping point pairs."""
    dec = DummyDec().eval()
    z_i, z_j = _make_pairs()
    L_ij = edge_lengths_riemannian(dec, z_i, z_j, batch_size=64)
    L_ji = edge_lengths_riemannian(dec, z_j, z_i, batch_size=64)
    assert torch.allclose(L_ij, L_ji, rtol=1e-4, atol=1e-6)

def test_scaling_step_cpu():
    """Test that edge lengths scale approximately with step size."""
    dec = DummyDec().eval()
    M, D = 64, 16
    g = torch.Generator().manual_seed(0)
    z_i = torch.randn(M, D, generator=g)
    v = 0.05 * torch.randn(M, D, generator=g)  # Small displacement vector
    L1 = edge_lengths_riemannian(dec, z_i, z_i + v, batch_size=64)
    Lh = edge_lengths_riemannian(dec, z_i, z_i + 0.5 * v, batch_size=64)
    # Half step should give smaller distances and roughly half the average
    ratio = (Lh / (L1 + 1e-8)).mean().item()
    assert torch.all(Lh <= L1 + 1e-6)
    assert 0.3 < ratio < 0.7

def test_batch_sizeing_consistency_cpu():
    """Test that different batch sizes produce consistent results."""
    dec = DummyDec().eval()
    z_i, z_j = _make_pairs(M=127)  # Non-multiple of batch_size to test final batch
    L_small = edge_lengths_riemannian(dec, z_i, z_j, batch_size=16)
    L_big = edge_lengths_riemannian(dec, z_i, z_j, batch_size=1024)
    assert torch.allclose(L_small, L_big, rtol=1e-5, atol=1e-7)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda/rocm not available")
def test_gpu_rocm_optional():
    """Test edge length computation on GPU when available."""
    device = "cuda"
    dec = DummyDec().eval().to(device)
    z_i, z_j = _make_pairs(device=device)
    L = edge_lengths_riemannian(dec, z_i, z_j, batch_size=64)
    assert L.is_cuda and L.dtype == torch.float32 and L.numel() == z_i.size(0)
