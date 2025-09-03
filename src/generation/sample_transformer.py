from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision

from src.models.transformer import GPTTransformer, TransformerConfig
from src.data.sequences import SPECIAL_TOKENS
from src.models.vae import VAE


@torch.no_grad()
def sample_sequences(
    model: GPTTransformer,
    num_samples: int,
    start_token_id: int,
    max_len: int,
    class_labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = SPECIAL_TOKENS['EOS'],
    grid_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    x = torch.full((num_samples, 1), start_token_id, dtype=torch.long, device=device)
    attn = torch.ones_like(x)
    out = model.generate(
        x, max_new_tokens=max_len - 1, temperature=temperature, top_k=top_k, top_p=top_p,
        class_labels=class_labels, attention_mask=attn, eos_token_id=eos_token_id, grid_size=grid_size
    )
    return out


def tokens_to_grid(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    # tokens: [B, T] -> [B, H, W]
    B, T = tokens.size()
    assert grid_h * grid_w <= T, "Not enough tokens to fill grid"
    return tokens[:, : grid_h * grid_w].view(B, grid_h, grid_w)


@torch.no_grad()
def decode_with_vae(grid_tokens: torch.Tensor, codebook: torch.Tensor, decoder: VAE, out_dir: Path, tag: str) -> None:
    # Simple visualization by mapping tokens to codebook vectors averaged to an image via decoder
    # Assumes codebook: [K, latent_dim]. We first map each token to its code vector, average across grid, and decode.
    device = next(decoder.parameters()).device
    B, H, W = grid_tokens.size()
    K, D = codebook.size()
    flat = grid_tokens.view(B, H * W)  # [B, T]
    codes = codebook[flat]  # [B, T, D]
    z = codes.mean(dim=1)  # [B, D]
    x_logits = decoder.decoder(z.to(device))
    x = torch.sigmoid(x_logits)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(x, out_dir / f"samples_{tag}.png", nrow=8)


def load_transformer(ckpt_path: Path, cfg: TransformerConfig, device: torch.device) -> GPTTransformer:
    model = GPTTransformer(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()
    return model

