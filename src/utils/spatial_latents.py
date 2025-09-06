from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from src.models.spatial_vae import SpatialVAE


def save_spatial_latents(model: SpatialVAE, loader: Iterable, device: torch.device, out_dir: Path) -> None:
    model.eval()
    zs, mus, logvars, ys = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Save spatial latents"):
            x = x.to(device)
            _, mu, logvar, z = model(x)
            zs.append(z.cpu())
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
            ys.append(y)
            
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Concatenate lists of tensors. These will be 4D tensors for SpatialVAE
    z_tensor = torch.cat(zs)
    mu_tensor = torch.cat(mus)
    logvar_tensor = torch.cat(logvars)
    y_tensor = torch.cat(ys)

    # Save tensors
    torch.save(z_tensor, out_dir / 'z.pt')
    torch.save(mu_tensor, out_dir / 'mu.pt')
    torch.save(logvar_tensor, out_dir / 'logvar.pt')
    torch.save(y_tensor, out_dir / 'y.pt')

    print(f"Saved spatial latents to {out_dir}")
