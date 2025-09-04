import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from torchvision.utils import save_image

from src.data import get_data_loaders
from src.models.vae import VAE
from src.eval.metrics import psnr, ssim_simple


def torch_load_trusted(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def find_normalize(transform):
    from torchvision import transforms as T
    if transform is None:
        return None
    if isinstance(transform, T.Normalize):
        return transform
    sub = getattr(transform, "transforms", None)
    if isinstance(sub, (list, tuple)):
        for s in sub:
            n = find_normalize(s)
            if n is not None:
                return n
    nested = getattr(transform, "transform", None)
    if nested is not None:
        return find_normalize(nested)
    return None


@torch.no_grad()
def unnormalize(x: torch.Tensor, norm_module) -> torch.Tensor:
    if norm_module is None:
        return x
    mean = torch.as_tensor(norm_module.mean, device=x.device).view(1, -1, 1, 1)
    std = torch.as_tensor(norm_module.std, device=x.device).view(1, -1, 1, 1)
    return (x * std + mean).clamp(0, 1)


def build_model(device: torch.device) -> Tuple[VAE, dict, dict, bool]:
    vae_cfg = load_cfg("configs/vae.yaml")
    data_cfg = load_cfg("configs/data.yaml")
    model = VAE(
        in_channels=int(vae_cfg.get("in_channels", 1)),
        enc_channels=vae_cfg.get("enc_channels", [64, 128, 256]),
        dec_channels=vae_cfg.get("dec_channels", [256, 128, 64]),
        latent_dim=int(vae_cfg.get("latent_dim", 128)),
        recon_loss=str(vae_cfg.get("recon_loss", "mse")),
        output_image_size=int(vae_cfg.get("output_image_size", 28)),
        norm_type=str(vae_cfg.get("norm_type", "none")),
        mse_use_sigmoid=bool(vae_cfg.get("mse_use_sigmoid", True)),
    ).to(device).eval()

    recon_loss = str(vae_cfg.get("recon_loss", "mse")).lower()
    mse_use_sigmoid = bool(vae_cfg.get("mse_use_sigmoid", True))
    apply_sigmoid = (recon_loss == "bce") or mse_use_sigmoid
    return model, vae_cfg, data_cfg, apply_sigmoid


def nearest_medoid_assign(z: torch.Tensor, z_medoid: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    codes_list = []
    z_medoid_t = z_medoid.t().contiguous()
    b2 = (z_medoid ** 2).sum(dim=1).view(1, -1)
    for i in range(0, z.size(0), batch):
        zi = z[i:i+batch]
        a2 = (zi ** 2).sum(dim=1, keepdim=True)
        ab = zi @ z_medoid_t
        d2 = a2 + b2 - 2.0 * ab
        codes_list.append(d2.argmin(dim=1).cpu())
    return torch.cat(codes_list, 0).long()


def main():
    p = argparse.ArgumentParser(description="Absolute evaluation of quantized reconstructions vs ground truth")
    p.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint (best.pt)")
    p.add_argument("--latents", required=True, help="Path to latents tensor (val z.pt recommended)")
    p.add_argument("--codebook", required=True, help="Path to codebook.pt (contains z_medoid)")
    p.add_argument("--codes", default="", help="Optional codes.npy aligned with latents; if missing, use Euclidean NN")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--out_json", default="abs_metrics.json")
    p.add_argument("--out_png", default="", help="Optional grid image output path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model from active configs
    model, vae_cfg, data_cfg, apply_sigmoid = build_model(device)
    ckpt = torch_load_trusted(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Prepare loaders (val only), order must match saved latents_val
    name = str(data_cfg.get("name", "MNIST"))
    root = str(data_cfg.get("root", "./data"))
    bs = int(data_cfg.get("batch_size", max(64, args.batch_size)))
    _, val_loader = get_data_loaders(
        name=name, root=root, batch_size=bs,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        augment=False,
    )
    norm_mod = find_normalize(getattr(val_loader.dataset, "transform", None))

    # Load latents (z)
    z_all = torch.load(args.latents, map_location="cpu")
    if isinstance(z_all, dict):
        z_all = z_all.get("z", z_all)
    z_all = z_all.float()

    # Load codebook medoids
    cb = torch_load_trusted(args.codebook, map_location="cpu")
    z_medoid = cb["z_medoid"].float()

    # Load or compute codes aligned to latents
    if args.codes and Path(args.codes).exists():
        codes_np = np.load(args.codes)
        if codes_np.ndim > 1:
            codes_np = codes_np.reshape(-1)
        codes = torch.from_numpy(codes_np).long()
        assert len(codes) == len(z_all), "codes.npy and z.pt must have same length/order"
    else:
        codes = nearest_medoid_assign(z_all, z_medoid, batch=8192)

    # Build quantized latents
    zq_all = z_medoid[codes]

    # Decode and compare against ground truth
    x_true_list, x_cont_list, x_quant_list = [], [], []
    offset = 0
    n_total = int(z_all.size(0))
    with torch.no_grad():
        for x_gt, _ in val_loader:
            if offset >= n_total:
                break
            b = x_gt.size(0)
            end = min(offset + b, n_total)
            z = z_all[offset:end].to(device)
            zq = zq_all[offset:end].to(device)
            offset = end

            x_cont = model.decoder(z)
            x_quant = model.decoder(zq)

            if apply_sigmoid:
                x_cont = torch.sigmoid(x_cont)
                x_quant = torch.sigmoid(x_quant)

            x_gt = x_gt.to(device)
            x_gt = unnormalize(x_gt, norm_mod)
            x_cont = unnormalize(x_cont, norm_mod)
            x_quant = unnormalize(x_quant, norm_mod)

            # Ensure valid image range for PSNR/SSIM
            x_gt = x_gt.clamp(0, 1)
            x_cont = x_cont.clamp(0, 1)
            x_quant = x_quant.clamp(0, 1)

            x_true_list.append(x_gt.cpu())
            x_cont_list.append(x_cont.cpu())
            x_quant_list.append(x_quant.cpu())

    x_true = torch.cat(x_true_list, 0)
    x_cont = torch.cat(x_cont_list, 0)
    x_quant = torch.cat(x_quant_list, 0)

    n = min(len(x_true), n_total)
    x_true = x_true[:n]
    x_cont = x_cont[:n]
    x_quant = x_quant[:n]

    metrics = {
        "dataset": name,
        "psnr_abs_cont": float(f"{psnr(x_true, x_cont):.6f}"),
        "psnr_abs_quant": float(f"{psnr(x_true, x_quant):.6f}"),
        "psnr_cont_vs_quant": float(f"{psnr(x_cont, x_quant):.6f}"),
        "ssim_abs_cont": float(f"{ssim_simple(x_true, x_cont):.6f}"),
        "ssim_abs_quant": float(f"{ssim_simple(x_true, x_quant):.6f}"),
        "ssim_cont_vs_quant": float(f"{ssim_simple(x_cont, x_quant):.6f}"),
        "samples_evaluated": int(n),
    }
    print("Absolute metrics:", metrics)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved:", out_json)

    if args.out_png:
        vis = min(16, n)
        idx = torch.randperm(n)[:vis]
        grid = torch.cat([x_true[idx], x_cont[idx], x_quant[idx]], dim=0)
        from torchvision.utils import make_grid
        grid = make_grid(grid, nrow=vis, padding=2)
        save_image(grid, args.out_png)
        print("Saved:", args.out_png)


if __name__ == "__main__":
    main()


