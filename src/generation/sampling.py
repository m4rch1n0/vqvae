import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k = None, top_p = None) -> torch.Tensor:
    """Sample next token from logits using various decoding strategies."""
    # Apply temperature scaling (higher = more random, lower = more deterministic)
    logits = logits / max(1e-8, float(temperature))
    
    if top_k is not None and top_k > 0:
        # Top-k sampling: only consider top k most likely tokens
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float('Inf')
    
    if top_p is not None and 0.0 < top_p < 1.0:
        # Top-p (nucleus) sampling: sample from tokens with cumulative probability p
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cumprobs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits[cutoff] = -float('Inf')
        # Map back to original positions
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    
    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_autoregressive(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k = None, top_p = None, classes = None) -> torch.Tensor:
    """Generate new tokens autoregressively using the trained transformer."""
    model.eval()
    
    for _ in range(max_new_tokens):
        # Use only the last seq_len tokens to avoid exceeding model capacity
        idx_cond = input_ids[:, -model.config.seq_len:]
        
        # Get logits for next token prediction
        logits, _ = model(idx_cond, classes=classes)
        # Sample next token using specified strategy
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
        # Append new token to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids


