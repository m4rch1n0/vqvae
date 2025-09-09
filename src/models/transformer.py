
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """A simple decoder-only Transformer for autoregressive sequence modeling."""

    def __init__(self, num_classes: int, num_tokens: int, embed_dim: int,
                 n_layers: int, n_head: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.token_emb = nn.Embedding(num_tokens, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_seq_len, embed_dim))
        self.drop = nn.Dropout(dropout)

        # Optional class conditioning
        if self.num_classes > 0:
            self.class_emb = nn.Embedding(self.num_classes, self.embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, max_seq_len, dropout) for _ in range(n_layers)
        ])
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_tokens, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, Transformer):
             torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds model max length {self.max_seq_len}"

        # Token and positional embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(tok_emb + pos_emb)

        # Add class conditioning if provided
        if y is not None:
            class_emb = self.class_emb(y).unsqueeze(1)
            x = x + class_emb

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class Block(nn.Module):
    """A single Transformer block."""

    def __init__(self, embed_dim: int, n_head: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_head, max_seq_len, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Masked Multi-Head Self-Attention."""

    def __init__(self, embed_dim: int, n_head: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert embed_dim % n_head == 0
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.embed_dim = embed_dim
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                     .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Q, K, V projections
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
