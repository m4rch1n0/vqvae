from typing import Dict

import torch
from torch import Tensor


@torch.no_grad()
def mse(x: Tensor, y: Tensor) -> float:
    return float(torch.mean((x - y) ** 2))


@torch.no_grad()
def ssim_global(x: Tensor, y: Tensor, C1=0.01**2, C2=0.03**2) -> float:
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x = x.var(unbiased=False)
    sigma_y = y.var(unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float((num / den).clamp(0, 1))


def summarize_metrics(x_rec: Tensor, x_ref: Tensor) -> Dict[str, float]:
    return {
        'mse': mse(x_rec, x_ref),
        'ssim': ssim_global(x_rec, x_ref),
    }


