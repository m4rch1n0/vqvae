from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, resid_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        self.register_buffer("mask", None, persistent=False)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.mask is None or self.mask.size(0) < seq_len:
            m = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
            self.mask = m
        return self.mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self._get_causal_mask(T, x.device)
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(mlp_ratio * embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, inner),
            nn.GELU(),
            nn.Linear(inner, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, resid_dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, attn_dropout, resid_dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTConfig:
    def __init__(self,
                 vocab_size: int,
                 seq_len: int,
                 embed_dim: int = 256,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 attn_dropout: float = 0.0,
                 resid_dropout: float = 0.0,
                 mlp_ratio: float = 4.0,
                 class_conditional: bool = False,
                 num_classes: int = 0,
                 ):  
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.mlp_ratio = mlp_ratio
        self.class_conditional = class_conditional
        self.num_classes = num_classes


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.seq_len, config.embed_dim)

        if config.class_conditional:
            assert config.num_classes > 0, "num_classes must be > 0 when class_conditional=True"
            self.class_emb = nn.Embedding(config.num_classes, config.embed_dim)
        else:
            self.class_emb = None

        self.drop = nn.Dropout(config.resid_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                attn_dropout=config.attn_dropout,
                resid_dropout=config.resid_dropout,
                mlp_ratio=config.mlp_ratio,
            ) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, *, labels: Optional[torch.Tensor] = None, classes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.seq_len, "Sequence length exceeds model's maximum"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        if self.class_emb is not None and classes is not None:
            class_tokens = self.class_emb(classes).unsqueeze(1)
            x = x + class_tokens

        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss


def generate(model: 'GPT', idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, classes: Optional[torch.Tensor] = None) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.seq_len:]
        with torch.no_grad():
            logits, _ = model(idx_cond, classes=classes)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


