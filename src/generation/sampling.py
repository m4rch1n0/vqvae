from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
    logits = logits / max(1e-8, float(temperature))
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float('Inf')
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cumprobs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits[cutoff] = -float('Inf')
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_autoregressive(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None, classes: Optional[torch.Tensor] = None) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -model.config.seq_len:]
        logits, _ = model(idx_cond, classes=classes)
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


