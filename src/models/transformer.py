from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1
    use_2d_positions: bool = False
    grid_size: Optional[Tuple[int, int]] = None  # (H, W) when using 2D-aware positions
    num_classes: Optional[int] = None  # when using class conditioning


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("_mask", torch.empty(0), persistent=False)

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # Causal mask: [T, T], True for allowed positions
        if self._mask.numel() == 0 or self._mask.size(0) < T:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            self._mask = mask
        return self._mask[:T, :T].to(device)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))  # [B, nh, T, T]
        causal = self._get_mask(T, x.device)
        att = att.masked_fill(~causal, float('-inf'))
        if attn_mask is not None:
            # attn_mask: [B, 1, 1, T] broadcasted to [B, nh, T, T] for key padding mask
            att = att.masked_fill(~attn_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTTransformer(nn.Module):
    """A simple GPT-style decoder-only transformer for discrete tokens.

    Supports optional class conditioning (additive class embedding) and
    optional 2D-aware positional embeddings (factorized row/col embeddings).
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        if cfg.use_2d_positions:
            assert cfg.grid_size is not None, "grid_size must be provided when use_2d_positions=True"
            H, W = cfg.grid_size
            self.row_emb = nn.Embedding(H, cfg.d_model)
            self.col_emb = nn.Embedding(W, cfg.d_model)
        else:
            self.row_emb = None
            self.col_emb = None

        if cfg.num_classes is not None:
            self.class_emb = nn.Embedding(cfg.num_classes, cfg.d_model)
        else:
            self.class_emb = None

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_key_padding_mask(self, attention_mask: Optional[torch.Tensor], T: int) -> Optional[torch.Tensor]:
        # attention_mask expected shape [B, T] with 1 for valid tokens, 0 for padding
        if attention_mask is None:
            return None
        # Convert to broadcastable mask [B, 1, T, 1] then expand to [B, 1, T, T] during usage
        key_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        key_mask = key_mask.expand(-1, 1, T, -1)  # [B,1,T,T]
        return key_mask.to(dtype=torch.bool)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, class_labels: Optional[torch.Tensor] = None,
                grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        # input_ids: [B, T]
        B, T = input_ids.size()
        assert T <= self.cfg.max_seq_len, "Sequence length exceeds model's max_seq_len"

        tok = self.tok_emb(input_ids)  # [B, T, C]

        if self.cfg.use_2d_positions:
            gs = grid_size if grid_size is not None else self.cfg.grid_size
            assert gs is not None
            H, W = gs
            assert H * W >= T, "Grid size must cover sequence length"
            positions = torch.arange(T, device=input_ids.device)
            rows = positions // W
            cols = positions % W
            pos = self.pos_emb(positions)
            pos = pos + self.row_emb(rows) + self.col_emb(cols)
            pos = pos.unsqueeze(0).expand(B, -1, -1)  # [B, T, C]
        else:
            positions = torch.arange(T, device=input_ids.device)
            pos = self.pos_emb(positions).unsqueeze(0).expand(B, -1, -1)

        x = tok + pos

        if self.class_emb is not None and class_labels is not None:
            cls = self.class_emb(class_labels).unsqueeze(1)  # [B,1,C]
            x = x + cls  # additive conditioning

        x = self.drop(x)

        key_padding_mask = self._build_key_padding_mask(attention_mask, T)
        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, class_labels: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None, eos_token_id: Optional[int] = None,
                 grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        self.eval()
        B, T = input_ids.size()
        seq = input_ids
        attn = attention_mask
        for _ in range(max_new_tokens):
            if seq.size(1) > self.cfg.max_seq_len:
                seq = seq[:, -self.cfg.max_seq_len:]
                if attn is not None:
                    attn = attn[:, -self.cfg.max_seq_len:]
            logits = self.forward(seq, attention_mask=attn, class_labels=class_labels, grid_size=grid_size)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(1, ix, F.softmax(v, dim=-1))
            elif top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > float(top_p)
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                logits_masked = sorted_logits.masked_fill(sorted_mask, float('-inf'))
                probs = F.softmax(logits_masked, dim=-1)
                probs = torch.gather(probs, 1, torch.argsort(sorted_indices))
            else:
                probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)
            if attn is not None:
                pad = torch.ones((B, 1), dtype=attn.dtype, device=attn.device)
                attn = torch.cat([attn, pad], dim=1)
            if eos_token_id is not None:
                if torch.any(next_token.squeeze(-1) == eos_token_id):
                    break
        return seq

