import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.vqvae import VQVAE


def build_loaders(cfg, split="test"):
    mean, std = cfg["data"]["normalize_mean"], cfg["data"]["normalize_std"]
    tfm = transforms.Compose([
        transforms.Resize(cfg["data"]["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    is_train = split == "train"
    ds = datasets.CIFAR10(root=cfg["data"]["root"], train=is_train, download=True, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def compute_codebook_metrics(model, loader, device, n_codes):
    model.eval()
    totals = {"q_mse": 0.0, "perplex": 0.0, "usage": 0.0, "dead": 0.0, "loss": 0.0, "rec": 0.0, "vq": 0.0, "n": 0}
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        x_rec, loss_vq, idx, z_q, z_e = model(x)
        loss_rec = F.l1_loss(x_rec, x)
        loss = loss_rec + loss_vq

        quant_mse = F.mse_loss(z_q.detach(), z_e.detach())
        hist = torch.bincount(idx.view(-1), minlength=n_codes).float().to(x.device)
        usage = (hist > 0).float().mean()
        p = hist / hist.sum().clamp_min(1.0)
        perplex = torch.exp(-(p * (p + 1e-12).log()).sum())
        dead_pct = 1.0 - usage

        bs = x.size(0)
        totals["loss"]   += loss.item() * bs
        totals["rec"]    += loss_rec.item() * bs
        totals["vq"]     += loss_vq.item() * bs
        totals["q_mse"]  += quant_mse.item() * bs
        totals["perplex"]+= perplex.item() * bs
        totals["usage"]  += usage.item() * bs
        totals["dead"]   += dead_pct.item() * bs
        totals["n"]      += bs

    for k in ["loss","rec","vq","q_mse","perplex","usage","dead"]:
        totals[k] /= max(1, totals["n"])

    # embedding norms
    embed = model.quant.embed.to(device)
    norms = torch.linalg.norm(embed, dim=1)
    en_mean = norms.mean().item()
    en_min  = norms.min().item()
    en_max  = norms.max().item()

    return {
        "loss": totals["loss"],
        "rec": totals["rec"],
        "vq": totals["vq"],
        "q_mse": totals["q_mse"],
        "perplex": totals["perplex"],
        "usage": totals["usage"],
        "dead": totals["dead"],
        "embed_norm_mean": en_mean,
        "embed_norm_min": en_min,
        "embed_norm_max": en_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--ckpt", type=str, default=os.path.join("outputs", "checkpoints", "ckpt_best.pt"))
    ap.add_argument("--split", type=str, choices=["train","test"], default="test")
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = VQVAE(
        in_channels=cfg["model"]["in_channels"],
        z_channels=cfg["model"]["z_channels"],
        hidden=cfg["model"]["hidden"],
        n_res_blocks=cfg["model"]["n_res_blocks"],
        n_codes=cfg["model"]["n_codes"],
        beta=cfg["model"]["beta"],
        ema_decay=cfg["model"]["ema_decay"],
        ema_eps=cfg["model"]["ema_eps"],
    ).to(device)

    # Load checkpoint
    if os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # Data
    loader = build_loaders(cfg, split=args.split)

    # Metrics
    metrics = compute_codebook_metrics(model, loader, device, n_codes=cfg["model"]["n_codes"])

    # Print
    print(f"Split: {args.split}")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    # Save CSV
    out_csv = os.path.join("outputs", f"codebook_eval_{args.split}.csv")
    os.makedirs("outputs", exist_ok=True)
    header = ["split","loss","rec","vq","q_mse","perplex","usage","dead","embed_norm_mean","embed_norm_min","embed_norm_max"]
    write_header = not os.path.isfile(out_csv)
    with open(out_csv, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        row = [
            args.split,
            metrics["loss"],
            metrics["rec"],
            metrics["vq"],
            metrics["q_mse"],
            metrics["perplex"],
            metrics["usage"],
            metrics["dead"],
            metrics["embed_norm_mean"],
            metrics["embed_norm_min"],
            metrics["embed_norm_max"],
        ]
        f.write(",".join(f"{x}" for x in row) + "\n")


if __name__ == "__main__":
    main()


