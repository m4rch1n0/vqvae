from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from src.models.vae import VAE


def save_latents(model: VAE, loader: Iterable, device: torch.device, out_dir: Path) -> None:
    model.eval()
    zs, mus, logvars, ys = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Save latents"):
            x = x.to(device)
            _, mu, logvar, z = model(x)
            zs.append(z.cpu())
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
            ys.append(y)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(zs), out_dir / 'z.pt')
    torch.save(torch.cat(mus), out_dir / 'mu.pt')
    torch.save(torch.cat(logvars), out_dir / 'logvar.pt')
    torch.save(torch.cat(ys), out_dir / 'y.pt')


