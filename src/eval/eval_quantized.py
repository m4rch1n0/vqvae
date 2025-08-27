import argparse, yaml, torch
from pathlib import Path
from torchvision.utils import save_image
from src.models.vae import VAE
from src.eval.metrics import psnr, ssim_simple, codebook_stats

# Robust torch.load helper for PyTorch 2.6+ where weights_only defaults to True
def _torch_load_trusted(path, map_location="cpu"):
    """
    Load a trusted checkpoint/artifact disabling weights_only (PyTorch >= 2.6).
    Fallback for older versions where the argument does not exist.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch < 2.6 non supporta weights_only
        return torch.load(path, map_location=map_location)

def load_cfg(path): 
    with open(path,"r") as f: return yaml.safe_load(f) or {}

def unnorm_if_needed(x, data_name:str, apply_sigmoid:bool):
    if data_name.upper()=="CIFAR10" and not apply_sigmoid:
        mean = torch.tensor([0.4914,0.4822,0.4465], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.2470,0.2430,0.2610], device=x.device).view(1,3,1,1)
        x = (x*std + mean).clamp(0,1)
    else:
        x = x.sigmoid() if apply_sigmoid else x.clamp(0,1)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latents", required=True)        # z.pt with validation order
    ap.add_argument("--codes", required=True)           # codes.npy same order as z.pt
    ap.add_argument("--codebook", required=True)        # codebook.pt (contains z_medoid)
    ap.add_argument("--out", default="eval_quantized.png")
    ap.add_argument("--nvis", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_cfg = load_cfg("configs/vae.yaml")
    data_cfg = load_cfg("configs/data.yaml")

    # model
    model = VAE(**{
        "in_channels": int(vae_cfg.get("in_channels",3)),
        "enc_channels": vae_cfg.get("enc_channels",[64,128,256]),
        "dec_channels": vae_cfg.get("dec_channels",[256,128,64]),
        "latent_dim": int(vae_cfg.get("latent_dim",32)),
        "recon_loss": str(vae_cfg.get("recon_loss","mse")),
        "output_image_size": int(vae_cfg.get("output_image_size",32)),
        "norm_type": str(vae_cfg.get("norm_type","none")),
        "mse_use_sigmoid": bool(vae_cfg.get("mse_use_sigmoid", False)),
    }).to(device).eval()
    ckpt = _torch_load_trusted(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    recon_loss = str(vae_cfg.get("recon_loss","mse")).lower()
    mse_use_sigmoid = bool(vae_cfg.get("mse_use_sigmoid", False))
    apply_sigmoid = (recon_loss=="bce") or mse_use_sigmoid

    # load z and assignments
    z_all = torch.load(args.latents, map_location="cpu")
    if isinstance(z_all, dict): z_all = z_all.get("z", z_all)
    z_all = z_all.float()

    cb = _torch_load_trusted(args.codebook, map_location="cpu")
    z_medoid = cb["z_medoid"].float()
    import numpy as _np
    codes_np = _np.load(args.codes)
    if codes_np.ndim > 1:
        codes_np = codes_np.reshape(-1)
    codes = torch.from_numpy(codes_np).long()
    assert len(codes)==len(z_all), "codes.npy and z.pt must have same length/order"

    # construct per-sample zq
    zq_all = z_medoid[codes]

    # batch inference for metrics
    bs = 256
    x_cont_list, x_quant_list = [], []
    for i in range(0, len(z_all), bs):
        z = z_all[i:i+bs].to(device)
        zq = zq_all[i:i+bs].to(device)
        with torch.no_grad():
            x_cont = model.decoder(z)
            x_quant = model.decoder(zq)
        x_cont = unnorm_if_needed(x_cont, data_cfg.get("name",""), apply_sigmoid).cpu()
        x_quant = unnorm_if_needed(x_quant, data_cfg.get("name",""), apply_sigmoid).cpu()
        x_cont_list.append(x_cont); x_quant_list.append(x_quant)
    x_cont = torch.cat(x_cont_list,0)
    x_quant = torch.cat(x_quant_list,0)

    # metrics
    print(f"PSNR (cont vs quant): {psnr(x_cont, x_quant):.2f} dB")
    print(f"SSIM (cont vs quant): {ssim_simple(x_cont, x_quant):.4f}")

    # codebook stats
    stats = codebook_stats(codes, K=z_medoid.shape[0])
    print(f"Codebook — entropy: {stats['entropy']:.3f}, used: {stats['used']}/{z_medoid.shape[0]}, dead: {stats['dead_codes']}")

    # visualization grid
    vis_idx = torch.randperm(len(x_cont))[:args.nvis]
    grid = torch.cat([x_cont[vis_idx], x_quant[vis_idx]], dim=0)
    from torchvision.utils import make_grid
    grid = make_grid(grid, nrow=args.nvis, padding=2)
    save_image(grid, args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
