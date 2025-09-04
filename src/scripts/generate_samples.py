import argparse
from typing import Optional

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
def sample(model: Transformer, x: torch.Tensor, steps: int, temperature: float = 1.0, top_k: int = None, y: Optional[torch.Tensor] = None):
    """Autoregressive sampling from the Transformer model."""
    model.eval()
    for k in range(steps):
        logits = model(x, y=y)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)
    return x


def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and codebook
    transformer_cfg = cfg['transformer']
    transformer = Transformer(**transformer_cfg).to(device)
    transformer.load_state_dict(torch.load(cfg['transformer_ckpt_path']))

    vae_cfg = cfg['vae']
    vae = SpatialVAE(**vae_cfg).to(device)
    vae.load_state_dict(torch.load(cfg['vae_ckpt_path'])['model_state_dict'])

    codebook = torch.load(cfg['codebook_path'], weights_only=False)
    codebook_vectors = codebook['z_medoid'].to(device)

    # Multi-class sampling 
    sampling_cfg = cfg['sampling']
    class_labels = sampling_cfg.get("class_labels", [None]) # Default to unconditional
    samples_per_class = sampling_cfg.get("samples_per_class", 8)
    
    all_reconstructions = []

    for class_label in class_labels:
        print(f"Generating {samples_per_class} samples for class: {class_label if class_label is not None else 'Unconditional'}")
        context = torch.zeros(samples_per_class, 1, dtype=torch.long, device=device)
        
        y = None
        if class_label is not None:
            y = torch.tensor([class_label] * samples_per_class, dtype=torch.long, device=device)

        codes = sample(transformer, context, steps=transformer_cfg['max_seq_len'] - 1,
                       temperature=sampling_cfg['temperature'], top_k=sampling_cfg['top_k'], y=y)
        
        codes_quantized = codebook_vectors[codes]
        codes_quantized = codes_quantized.permute(0, 2, 1).view(samples_per_class, vae_cfg['latent_dim'], 4, 4)
        
        reconstructions = vae.decoder(codes_quantized).sigmoid()
        all_reconstructions.append(reconstructions)

    final_grid = torch.cat(all_reconstructions, dim=0)
    save_image(final_grid, cfg['out_path'], nrow=samples_per_class)
    print(f"Saved generated images to {cfg['out_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the sampling config file.")
    args = parser.parse_args()
    main(args.config)
