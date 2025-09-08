import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import yaml

from src.models import Transformer, SpatialVAE, VAE


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
    is_vanilla_vae = cfg.get('vanilla_vae', False)
    
    if is_vanilla_vae:
        vae = VAE(**vae_cfg).to(device)
    else:
        vae = SpatialVAE(**vae_cfg).to(device)
    
    vae.load_state_dict(torch.load(cfg['vae_ckpt_path'])['model_state_dict'])

    codebook = torch.load(cfg['codebook_path'], weights_only=False)
    codebook_vectors = codebook['z_medoid'].to(device)

    # Multi-class sampling 
    class_labels = cfg.get("class_labels", [None])  # Default to unconditional
    samples_per_class = cfg.get("samples_per_class", 8)
    temperature = cfg.get("temperature", 1.0)
    top_k = cfg.get("top_k", None)
    
    all_reconstructions = []

    for class_label in class_labels:
        print(f"Generating {samples_per_class} samples for class: {class_label if class_label is not None else 'Unconditional'}")
        
        y = None
        if class_label is not None:
            y = torch.tensor([class_label] * samples_per_class, dtype=torch.long, device=device)

        if is_vanilla_vae:
            # Vanilla VAE: use BOS token approach
            bos_token = transformer_cfg['num_tokens'] - 1
            context = torch.full((samples_per_class, 1), bos_token, dtype=torch.long, device=device)
            codes = sample(transformer, context, steps=transformer_cfg['max_seq_len'] - 1,
                           temperature=temperature, top_k=top_k, y=y)
            # Remove the initial context token (BOS)
            codes = codes[:, 1:]
        else:
            # Spatial VAE: generate full sequence without BOS token
            # Start with a random first token, then generate the rest autoregressively
            first_token = torch.randint(0, transformer_cfg['num_tokens'], (samples_per_class, 1), device=device)
            codes = sample(transformer, first_token, steps=transformer_cfg['max_seq_len'] - 1,
                           temperature=temperature, top_k=top_k, y=y)
        
        if is_vanilla_vae:
            # Vanilla VAE: all codes in sequence should be the same
            single_codes = codes[:, 0]  # Shape: (samples_per_class,)
            codes_quantized = codebook_vectors[single_codes]  # Shape: (samples_per_class, latent_dim)
        else:
            # For spatial VAE: reshape to 4x4 grid
            codes_quantized = codebook_vectors[codes]
            codes_quantized = codes_quantized.permute(0, 2, 1).view(samples_per_class, vae_cfg['latent_dim'], 4, 4)
        
        reconstructions = vae.decoder(codes_quantized).sigmoid()
        all_reconstructions.append(reconstructions)

    final_grid = torch.cat(all_reconstructions, dim=0)
    
    # Create output directory and save
    import os
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cfg['output_filename'])
    
    save_image(final_grid, output_path, nrow=samples_per_class)
    print(f"Saved generated images to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the sampling config file.")
    args = parser.parse_args()
    main(args.config)
