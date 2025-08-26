import torch
from torch import Tensor

@torch.no_grad()
def psnr(x: Tensor, y: Tensor, max_val: float = 1.0) -> float:
    # x,y in [0,1], shape (N,C,H,W)
    mse = torch.mean((x - y) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(torch.tensor(max_val**2) / mse))

@torch.no_grad()
def ssim_simple(x: Tensor, y: Tensor, C1=0.01**2, C2=0.03**2) -> float:
    # semplice SSIM “global”, non finestrato per rapidità
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x = x.var(unbiased=False)
    sigma_y = y.var(unbiased=False)
    sigma_xy = ((x - mu_x)*(y - mu_y)).mean()
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return float((num/den).clamp(0,1))

@torch.no_grad()
def codebook_stats(codes: torch.Tensor, K: int) -> dict:
    # codes: (N,) int64
    hist = torch.bincount(codes, minlength=K).float()
    p = (hist / hist.sum()).clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum())
    dead = int((hist == 0).sum())
    return {"entropy": entropy, "dead_codes": dead, "used": int((hist>0).sum())}


