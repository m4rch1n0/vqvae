from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.transformer import GPT, GPTConfig
from src.generation.sampling import generate_autoregressive
from src.models.vae import VAE


def load_gpt(checkpoint: Path, config_path: Path, device: torch.device) -> GPT:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model_cfg = GPTConfig(
        vocab_size=int(cfg['model']['vocab_size']) if 'vocab_size' in cfg['model'] else 65536,
        seq_len=int(cfg['model']['seq_len']),
        embed_dim=int(cfg['model']['embed_dim']),
        num_layers=int(cfg['model']['num_layers']),
        num_heads=int(cfg['model']['num_heads']),
        attn_dropout=float(cfg['model']['attn_dropout']),
        resid_dropout=float(cfg['model']['resid_dropout']),
        mlp_ratio=float(cfg['model']['mlp_ratio']),
        class_conditional=bool(cfg['model'].get('class_conditional', False)),
        num_classes=int(cfg['model'].get('num_classes', 0)),
    )
    model = GPT(model_cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model


def load_vae(checkpoint: Path, device: torch.device) -> VAE:
    with open(Path("configs/vae.yaml"), "r") as f:
        vae_cfg = yaml.safe_load(f) or {}
    model = VAE(
        in_channels=int(vae_cfg.get("in_channels", 1)),
        enc_channels=vae_cfg.get("enc_channels", [32, 64, 128]),
        dec_channels=vae_cfg.get("dec_channels", [128, 64, 32]),
        latent_dim=int(vae_cfg.get("latent_dim", 16)),
        recon_loss=str(vae_cfg.get("recon_loss", "bce")),
        output_image_size=int(vae_cfg.get("output_image_size", 28)),
        norm_type=str(vae_cfg.get("norm_type", "none")),
        mse_use_sigmoid=bool(vae_cfg.get("mse_use_sigmoid", True)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def decode_codes_to_images(decoder: torch.nn.Module, codes_2d: np.ndarray, codebook_path: Path, device: torch.device) -> torch.Tensor:
    codebook = torch.load(codebook_path, map_location='cpu')
    z_medoid = codebook['z_medoid'].float().to(device)  # (K,D)
    B, H, W = codes_2d.shape
    zq = z_medoid[codes_2d.reshape(B, -1).astype(np.int64)]  # (B, H*W, D)
    # Average latent across positions (simple baseline); more advanced decoders may map codes->latent grid
    zq_agg = zq.mean(dim=1)
    x_logits = decoder(zq_agg)
    return torch.sigmoid(x_logits)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Generate images from transformer codes")
    p.add_argument('--transformer_ckpt', type=str, required=True)
    p.add_argument('--transformer_cfg', type=str, required=True)
    p.add_argument('--vae_ckpt', type=str, required=True)
    p.add_argument('--codebook_pt', type=str, required=True)
    p.add_argument('--H', type=int, default=32)
    p.add_argument('--W', type=int, default=32)
    p.add_argument('--num_samples', type=int, default=16)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--top_p', type=float, default=0.0)
    p.add_argument('--class_id', type=int, default=-1)
    p.add_argument('--out', type=str, default='generated.png')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_gpt(Path(args.transformer_ckpt), Path(args.transformer_cfg), device)
    vae = load_vae(Path(args.vae_ckpt), device)

    seq_len = model.config.seq_len
    start = torch.zeros((args.num_samples, 1), dtype=torch.long, device=device)
    classes = None
    if model.config.class_conditional and args.class_id >= 0:
        classes = torch.full((args.num_samples,), int(args.class_id), dtype=torch.long, device=device)

    generated = generate_autoregressive(
        model, start, max_new_tokens=seq_len - 1, temperature=args.temperature,
        top_k=(args.top_k if args.top_k > 0 else None), top_p=(args.top_p if args.top_p > 0 else None), classes=classes,
    )

    codes = generated[:, : (args.H * args.W)].view(args.num_samples, args.H, args.W).detach().cpu().numpy()
    imgs = decode_codes_to_images(vae.decoder, codes, Path(args.codebook_pt), device)

    from torchvision.utils import save_image, make_grid
    grid = make_grid(imgs.cpu(), nrow=int(max(1, int(args.num_samples**0.5))))
    save_image(grid, args.out)
    print("Saved:", args.out)


if __name__ == '__main__':
    main()


