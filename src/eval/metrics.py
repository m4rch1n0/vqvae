import torch
from torch import Tensor

@torch.no_grad()
def psnr(x: Tensor, y: Tensor, max_val: float = 1.0) -> float:
    # x,y in [0,1], shape (N,C,H,W)
    mse = torch.mean((x - y) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(torch.tensor(max_val**2) / mse))

@torch.no_grad()
def ssim_simple(x: Tensor, y: Tensor, C1=0.01**2, C2=0.03**2) -> float:
    # Calculate SSIM per image and average (fixed from batch-level calculation)
    if x.dim() == 4:  # Batch of images (B, C, H, W)
        batch_size = x.size(0)
        ssim_values = []
        
        for i in range(batch_size):
            # Calculate SSIM for each image pair individually
            xi, yi = x[i], y[i]
            mu_x, mu_y = xi.mean(), yi.mean()
            sigma_x = xi.var(unbiased=False)
            sigma_y = yi.var(unbiased=False)
            sigma_xy = ((xi - mu_x)*(yi - mu_y)).mean()
            
            num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
            den = (mu_x**2 + mu_y**2 + C1) + (sigma_x + sigma_y + C2)
            ssim_i = float((num/den).clamp(0,1))
            ssim_values.append(ssim_i)
        
        return sum(ssim_values) / len(ssim_values)
    else:
        # Single image calculation
        mu_x, mu_y = x.mean(), y.mean()
        sigma_x = x.var(unbiased=False)
        sigma_y = y.var(unbiased=False)
        sigma_xy = ((x - mu_x)*(y - mu_y)).mean()
        num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        return float((num/den).clamp(0,1))

@torch.no_grad()
def codebook_stats(codes: torch.Tensor, K: int) -> dict:
    # codes: (N,) int64, may contain -1 for invalid/unassigned
    if codes.dim() != 1:
        codes = codes.view(-1)
    codes = codes.long()
    valid = codes >= 0
    if valid.any():
        hist = torch.bincount(codes[valid], minlength=K).float()
    else:
        hist = torch.zeros(K, dtype=torch.float32)
    p = (hist / hist.sum().clamp_min(1e-12)).clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum())
    dead = int((hist == 0).sum())
    return {"entropy": entropy, "dead_codes": dead, "used": int((hist>0).sum())}