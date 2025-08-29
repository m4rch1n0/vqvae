import torch
import torch.nn as nn


class SinusoidalPositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding that can be flattened to 1D sequence.

    Produces a tensor of shape (H*W, D) given H, W, and embedding dim D.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @staticmethod
    def _get_1d_sin_cos(pos: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim % 2 == 0, "positional dimension must be even"
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
        sinusoid_inp = pos.unsqueeze(-1) * inv_freq
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return emb  # (N, dim)

    def forward(self, H: int, W: int, device=None) -> torch.Tensor:
        device = device or torch.device('cpu')
        d_h = self.embed_dim // 2
        d_w = self.embed_dim - d_h
        assert d_h % 2 == 0 and d_w % 2 == 0, "embed_dim halves must be even"
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        emb_y = self._get_1d_sin_cos(y, d_h)
        emb_x = self._get_1d_sin_cos(x, d_w)
        grid_y = emb_y[:, None, :].expand(H, W, d_h)
        grid_x = emb_x[None, :, :].expand(H, W, d_w)
        pos = torch.cat([grid_y, grid_x], dim=-1).view(H * W, self.embed_dim)
        return pos


class LearnedPositionalEncoding2D(nn.Module):
    def __init__(self, H: int, W: int, embed_dim: int):
        super().__init__()
        self.H = H
        self.W = W
        self.embed = nn.Parameter(torch.zeros(H * W, embed_dim))
        nn.init.normal_(self.embed, mean=0.0, std=0.02)

    def forward(self) -> torch.Tensor:
        return self.embed


