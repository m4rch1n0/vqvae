from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.transformer import GPTTransformer
from src.data.sequences import SPECIAL_TOKENS


@torch.no_grad()
def evaluate_perplexity(model: GPTTransformer, loader, pad_token_id: int = SPECIAL_TOKENS['PAD']) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        input_ids, attention_mask, class_labels = batch
        input_ids = input_ids.to(next(model.parameters()).device)
        attention_mask = attention_mask.to(next(model.parameters()).device)
        if class_labels is not None:
            class_labels = class_labels.to(next(model.parameters()).device)

        logits = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1], class_labels=class_labels)
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=pad_token_id, reduction='sum'
        )
        valid = (labels != pad_token_id).sum().item()
        total_nll += float(loss.item())
        total_tokens += int(valid)
    if total_tokens == 0:
        return float('inf')
    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return float(ppl.item())


@torch.no_grad()
def measure_diversity(samples: torch.Tensor) -> float:
    # samples: [B, T] integer tokens
    flat = samples.cpu().numpy().tolist()
    unique = len({tuple(x) for x in flat})
    return unique / max(1, len(flat))


@torch.no_grad()
def compute_reconstruction_metrics(images_a: torch.Tensor, images_b: torch.Tensor) -> Tuple[float, float]:
    # images in [0,1], shape [B, C, H, W]
    mse = F.mse_loss(images_a, images_b, reduction='none').view(images_a.size(0), -1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    # SSIM placeholder (simple variant); in practice prefer skimage.metrics.structural_similarity
    mu_x = images_a.mean(dim=(2, 3))
    mu_y = images_b.mean(dim=(2, 3))
    sigma_x = images_a.var(dim=(2, 3))
    sigma_y = images_b.var(dim=(2, 3))
    sigma_xy = ((images_a - mu_x.unsqueeze(-1).unsqueeze(-1)) * (images_b - mu_y.unsqueeze(-1).unsqueeze(-1))).mean(dim=(2, 3))
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return float(psnr.mean().item()), float(ssim.mean().item())

