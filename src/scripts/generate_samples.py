import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import yaml

from src.models import Transformer, SpatialVAE


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model: Transformer, x: torch.Tensor, steps: int, temperature: float = 1.0, top_k: int = None):
    """Autoregressive sampling from the Transformer model."""
    model.eval()
    for k in range(steps):
        logits = model(x)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)
    return x


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer
    transformer_cfg = config['transformer']
    transformer = Transformer(
        num_classes=transformer_cfg['num_classes'],
        num_tokens=transformer_cfg['num_tokens'],
        embed_dim=transformer_cfg['embed_dim'],
        n_layers=transformer_cfg['n_layers'],
        n_head=transformer_cfg['n_head'],
        max_seq_len=transformer_cfg['max_seq_len'],
        dropout=transformer_cfg['dropout']
    ).to(device)
    transformer.load_state_dict(torch.load(config['transformer_ckpt_path']))

    # Load VAE
    vae_cfg = config['vae']
    vae = SpatialVAE(
        in_channels=vae_cfg['in_channels'],
        enc_channels=tuple(vae_cfg['enc_channels']),
        dec_channels=tuple(vae_cfg['dec_channels']),
        latent_dim=vae_cfg['latent_dim'],
        recon_loss=vae_cfg['recon_loss'],
        output_image_size=vae_cfg['output_image_size'],
        norm_type=vae_cfg['norm_type']
    ).to(device)
    vae.load_state_dict(torch.load(config['vae_ckpt_path'])['model_state_dict'])

    # Load Codebook
    codebook = torch.load(config['codebook_path'], weights_only=False)
    codebook_vectors = codebook['z_medoid'].to(device)

    # Sample
    num_samples = config['sampling']['num_samples']
    context = torch.zeros(num_samples, 1, dtype=torch.long, device=device)
    codes = sample(transformer, context, steps=transformer_cfg['max_seq_len'] - 1,
                   temperature=config['sampling']['temperature'], top_k=config['sampling']['top_k'])
    
    # Decode
    codes_quantized = codebook_vectors[codes]
    codes_quantized = codes_quantized.permute(0, 2, 1).view(num_samples, vae_cfg['latent_dim'], 4, 4)
    
    reconstructions = vae.decoder(codes_quantized).sigmoid()

    save_image(reconstructions, config['out_path'], nrow=int(num_samples**0.5))
    print(f"Saved generated images to {config['out_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the sampling config file.")
    args = parser.parse_args()
    main(args.config)
