# Sanity check: Compare Riemannian vs Euclidean distances on k-NN edges
import os, sys, numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.geo.knn_graph_optimized import build_knn_graph
from src.geo.riemannian_metric import edge_lengths_riemannian
from src.utils.checkpoint_utils import get_vae_decoder

import argparse

# Dataset configurations
DATASET_CONFIGS = {
    'mnist': {
        'latents_path': "experiments/vae_mnist/latents_val/z.pt",
        'checkpoint_path': "experiments/vae_mnist/checkpoints/best.pt"
    },
    'cifar10': {
        'latents_path': "experiments/cifar10/vanilla/euclidean/vae/latents_val/z.pt",
        'checkpoint_path': "experiments/cifar10/vanilla/euclidean/vae/checkpoints/best.pt"
    },
    'fashionmnist': {
        'latents_path': "experiments/fashionmnist/vanilla/euclidean/vae/latents_val/z.pt", 
        'checkpoint_path': "experiments/fashionmnist/vanilla/euclidean/vae/checkpoints/best.pt"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Riemannian vs Euclidean distance sanity check")
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'fashionmnist'], 
                       default='mnist', help='Dataset to use for analysis')
    return parser.parse_args()

def load_latents(latent_path):
    """Load latent vectors from file"""
    pth = Path(latent_path)
    if pth.exists():
        obj = torch.load(pth, map_location="cpu")
        if isinstance(obj, dict) and "z" in obj: 
            return obj["z"].float()
        if torch.is_tensor(obj): 
            return obj.float()
    raise FileNotFoundError(f"Latents not found at: {latent_path}")


def run_experiment(args):
    """Run Riemannian vs Euclidean distance comparison experiment"""
    
    # Get paths from dataset config
    latents_path = DATASET_CONFIGS[args.dataset]['latents_path']
    checkpoint_path = DATASET_CONFIGS[args.dataset]['checkpoint_path']
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "experiments" / "geo" / "riemann_sanity" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)
    
    # Fixed parameters
    k_neighbors = 10
    max_edges = 2000
    
    # Load latent vectors
    print(f"Loading latents from: {latents_path}")
    z = load_latents(latents_path)
    z = z.cpu()
    N, D = z.shape[0], z.shape[1]
    print(f"Loaded {N} latent vectors of dimension {D}")
    
    # Build k-NN graph and sample edges
    print(f"Building k-NN graph with k={k_neighbors}")
    W, _ = build_knn_graph(z.numpy(), k=k_neighbors, metric="euclidean", mode="distance", sym="mutual")
    rows, cols = W.nonzero()
    
    # Sample random edges
    n_edges = min(max_edges, len(rows))
    rng = np.random.RandomState(0)
    indices = rng.choice(len(rows), n_edges, replace=False)
    i, j = rows[indices], cols[indices]
    print(f"Sampled {n_edges} edges from k-NN graph")

    # Compute edge lengths
    ze = z.numpy()
    de = np.linalg.norm(ze[j] - ze[i], axis=1)  # Euclidean
    
    # Load decoder using utility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = get_vae_decoder(checkpoint_path, latent_dim=D, device=device)
    if decoder is None:
        print("Cannot load decoder. Exiting.")
        return

    zi, zj = z[i].to(device), z[j].to(device)
    with torch.no_grad():
        dr = edge_lengths_riemannian(decoder, zi, zj, batch_size=256).cpu().numpy()  # Riemannian

    # Compute and save statistics
    ratios = dr / (de + 1e-8)
    corr = np.corrcoef(de, dr)[0,1]
    mean_ratio = np.mean(ratios)
    
    np.savez(
        os.path.join(output_dir, f"sanity_stats_{args.dataset}.npz"),
        corr=corr, ratio=mean_ratio, de=de, dr=dr, 
        dataset=args.dataset, decoder_type=f"real_VAE_{args.dataset.upper()}"
    )
    
    print(f"Results: correlation={corr:.3f}, mean_ratio={mean_ratio:.3f}")

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.scatter(de, dr, s=6, alpha=0.6)
    ax1.set_xlabel("Euclidean edge length")
    ax1.set_ylabel("Riemannian edge length")
    ax1.set_title(f"{args.dataset.upper()} - Riemannian vs Euclidean (k={k_neighbors})")
    
    ax2.hist(ratios, bins=50)
    ax2.set_xlabel("Ratio Riemannian / Euclidean")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of length ratios")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"riemann_analysis_{args.dataset}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Analysis complete! Plots saved to: {plot_path}")


if __name__ == "__main__":
    args = parse_args()
    print(f"Running Riemann sanity check on {args.dataset.upper()} dataset")
    run_experiment(args)
