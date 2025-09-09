import os, argparse, yaml, math, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from contextlib import nullcontext
from torch import amp

from models.vqvae import VQVAE
from utils import set_seed, save_grid, CSVLogger

def get_dataloaders(cfg):
    mean, std = cfg["data"]["normalize_mean"], cfg["data"]["normalize_std"]
    tfm = transforms.Compose([
        transforms.Resize(cfg["data"]["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root=cfg["data"]["root"], train=True, download=True, transform=tfm)
    test_set  = datasets.CIFAR10(root=cfg["data"]["root"], train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=False)
    return train_loader, test_loader

def train_one_epoch(model, loader, opt, scaler, device, grad_clip, n_codes, sample_bank, max_bank, epoch=None, epochs=None):
    model.train()
    running = {"loss": 0.0, "rec": 0.0, "vq": 0.0, "q_mse": 0.0, "perplex": 0.0, "usage": 0.0, "dead": 0.0, "n": 0}
    desc = "train" if epoch is None or epochs is None else f"train ep {epoch}/{epochs}"
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, _ in pbar:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        amp_ctx = amp.autocast(device_type="cuda", enabled=(scaler is not None)) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            x_rec, loss_vq, idx, z_q, z_e = model(x)
            loss_rec = F.l1_loss(x_rec, x)  # L1 for sharper reconstructions
            loss = loss_rec + loss_vq
        # update latent sample bank with encoder latents (z_e)
        with torch.no_grad():
            flat = z_e.detach().permute(0,2,3,1).contiguous().view(-1, z_e.size(1))
            take = min(256, flat.size(0))
            sel = flat[torch.randperm(flat.size(0), device=flat.device)[:take]]
            if sample_bank is None:
                sample_bank = sel
            else:
                sample_bank = torch.cat([sample_bank, sel], dim=0)
                if sample_bank.size(0) > max_bank:
                    sample_bank = sample_bank[-max_bank:]
        # codebook metrics
        quant_mse = F.mse_loss(z_q.detach(), z_e.detach())
        hist = torch.bincount(idx.view(-1), minlength=n_codes).float().to(x.device)
        usage = (hist > 0).float().mean()
        p = hist / hist.sum().clamp_min(1.0)
        perplex = torch.exp(-(p * (p + 1e-12).log()).sum())
        dead_pct = 1.0 - usage
        # NaN/Inf guard
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        bs = x.size(0)
        running["loss"] += loss.item() * bs
        running["rec"]  += loss_rec.item() * bs
        running["vq"]   += loss_vq.item() * bs
        running["q_mse"]   += quant_mse.item() * bs
        running["perplex"] += perplex.item() * bs
        running["usage"]   += usage.item() * bs
        running["dead"]    += dead_pct.item() * bs
        running["n"]    += bs
        pbar.set_postfix(loss=running["loss"]/running["n"], rec=running["rec"]/running["n"], vq=running["vq"]/running["n"], q_mse=running["q_mse"]/running["n"], ppl=running["perplex"]/running["n"])
    for k in ["loss","rec","vq","q_mse","perplex","usage","dead"]:
        running[k] /= max(1, running["n"])
    return running, sample_bank

@torch.no_grad()
def evaluate(model, loader, device, n_codes):
    model.eval()
    tot = {"loss": 0.0, "rec": 0.0, "vq": 0.0, "q_mse": 0.0, "perplex": 0.0, "usage": 0.0, "dead": 0.0, "n": 0}
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        amp_ctx = amp.autocast(device_type="cuda", enabled=(device.type == "cuda")) if device.type == "cuda" else nullcontext()
        with amp_ctx:
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
        tot["loss"] += loss.item() * bs
        tot["rec"]  += loss_rec.item() * bs
        tot["vq"]   += loss_vq.item() * bs
        tot["q_mse"]   += quant_mse.item() * bs
        tot["perplex"] += perplex.item() * bs
        tot["usage"]   += usage.item() * bs
        tot["dead"]    += dead_pct.item() * bs
        tot["n"]    += bs
    for k in ["loss","rec","vq","q_mse","perplex","usage","dead"]:
        tot[k] /= max(1, tot["n"])
    return tot

def save_samples(model, loader, device, outdir, epoch):
    os.makedirs(outdir, exist_ok=True)
    x, _ = next(iter(loader))
    x = x.to(device)[:32]
    x_rec, _, _, _, _ = model(x)
    save_grid(x_rec, os.path.join(outdir, f"recon_epoch{epoch:04d}.png"), nrow=8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--n_codes", type=int, default=None)
    ap.add_argument("--ema_decay", type=float, default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # override da CLI
    if args.epochs is not None: cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None: cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None: cfg["train"]["lr"] = args.lr
    if args.beta is not None: cfg["model"]["beta"] = args.beta
    if args.n_codes is not None: cfg["model"]["n_codes"] = args.n_codes
    if args.ema_decay is not None: cfg["model"]["ema_decay"] = args.ema_decay

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(cfg)

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

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    if cfg["train"]["amp"] and device.type == "cuda":
        scaler = amp.GradScaler(enabled=True)
    else:
        scaler = None


    ckpt_dir = os.path.join("outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = CSVLogger(
        os.path.join("outputs", "log.csv"),
        header=["epoch","split","loss","rec","vq","q_mse","perplex","usage","dead","embed_norm_mean","embed_norm_min","embed_norm_max"]
    )

    best_loss = float("inf")
    total_start = time.perf_counter()
    # latent sample bank for reseeding dead codes
    sample_bank = None
    max_bank = 8192

    for ep in range(1, cfg["train"]["epochs"]+1):
        epoch_start = time.perf_counter()
        n_codes = cfg["model"]["n_codes"]
        tr, sample_bank = train_one_epoch(model, train_loader, opt, scaler, device, cfg["train"]["grad_clip"], n_codes, sample_bank, max_bank, epoch=ep, epochs=cfg["train"]["epochs"])
        te = evaluate(model, test_loader, device, n_codes)

        # reseed dead/low-usage codes at the end of epoch
        num_reseeded = model.quant.reseed_dead_codes(min_count=5, sample_bank=sample_bank)
        if num_reseeded > 0:
            print(f"[epoch {ep}] reseeded {num_reseeded} codes")

        with torch.no_grad():
            embed = model.quant.embed.to(device)
            norms = torch.linalg.norm(embed, dim=1)
            en_mean = norms.mean().item()
            en_min  = norms.min().item()
            en_max  = norms.max().item()

        logger.log([ep, "train", tr["loss"], tr["rec"], tr["vq"], tr["q_mse"], tr["perplex"], tr["usage"], tr["dead"], en_mean, en_min, en_max])
        logger.log([ep, "val",   te["loss"], te["rec"], te["vq"], te["q_mse"], te["perplex"], te["usage"], te["dead"], en_mean, en_min, en_max])

        # console summary for this epoch
        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {ep}/{cfg['train']['epochs']} | train loss: {tr['loss']:.4f}, rec: {tr['rec']:.4f}, vq: {tr['vq']:.4f} | val loss: {te['loss']:.4f}, rec: {te['rec']:.4f}, vq: {te['vq']:.4f} | time: {epoch_time:.2f}s")

        # samples
        if (ep % cfg["log"]["samples_every"]) == 0:
            save_samples(model, test_loader, device, outdir="outputs", epoch=ep)

        # save last
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "epoch": ep},
                   os.path.join(ckpt_dir, "ckpt_last.pt"))
        # save best
        if cfg["log"]["save_best"] and te["loss"] < best_loss:
            best_loss = te["loss"]
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "epoch": ep},
                       os.path.join(ckpt_dir, "ckpt_best.pt"))

    logger.close()
    total_time = time.perf_counter() - total_start
    print(f"Training finished in {total_time/60:.2f} min. Check outputs/ for results.")

if __name__ == "__main__":
    main()
