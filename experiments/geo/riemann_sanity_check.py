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
from src.models.vae import VAE

# Configuration - modify these values as needed
LATENTS_PATH = "experiments/vae_mnist/latents_val/z.pt"
CHECKPOINT_PATH = "experiments/vae_mnist/checkpoints/best.pt"
OUTPUT_DIR = "experiments/geo/riemann_sanity"
K_NEIGHBORS = 10
MAX_EDGES = 2000
LATENT_DIM = 16
IMAGE_SHAPE = (1, 28, 28)

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

def load_decoder(checkpoint_path, latent_dim, device="cpu"):
    """Load VAE decoder from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        vae = VAE(latent_dim=latent_dim)
        vae.load_state_dict(state_dict)
        decoder = vae.decoder.eval()
        print(f"Decoder loaded from: {checkpoint_path}")
        return decoder.to(device)
    except Exception as e:
        print(f"Error loading decoder: {e}")
        return None

def run_experiment():
    """Run Riemannian vs Euclidean distance comparison experiment"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load latent vectors
    print(f"Loading latents from: {LATENTS_PATH}")
    z = load_latents(LATENTS_PATH)
    z = z.cpu()
    N, D = z.shape[0], z.shape[1]
    print(f"Loaded {N} latent vectors of dimension {D}")
    
    # Build k-NN graph
    print(f"Building k-NN graph with k={K_NEIGHBORS}")
    W, info = build_knn_graph(z.numpy(), k=K_NEIGHBORS, metric="euclidean", mode="distance", sym="mutual")
    rows, cols = W.nonzero()
    M = min(MAX_EDGES, len(rows))
    
    # Sample random edges
    rng = np.random.RandomState(0)
    sel = rng.choice(len(rows), M, replace=False)
    i, j = rows[sel], cols[sel]
    print(f"Sampled {M} edges from k-NN graph")

    # Compute Euclidean edge lengths
    ze = z.numpy()
    de = np.linalg.norm(ze[j] - ze[i], axis=1)

    # Load decoder and compute Riemannian edge lengths
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = load_decoder(CHECKPOINT_PATH, latent_dim=D, device=device)
    
    if decoder is None:
        print("Cannot load decoder. Exiting.")
        return

    zi = z[i].to(device)
    zj = z[j].to(device)
    with torch.no_grad():
        dr = edge_lengths_riemannian(decoder, zi, zj, batch_size=256).cpu().numpy()

    # Compute statistics
    corr = np.corrcoef(de, dr)[0,1]
    ratio = np.mean(dr / (de + 1e-8))
    ratios = dr / (de + 1e-8)

    # Save results
    np.savez(
        os.path.join(OUTPUT_DIR, "sanity_stats.npz"),
        corr=corr,
        ratio=ratio,
        de=de,
        dr=dr,
        decoder_type="real_VAE_MNIST"
    )

    print(f"Results: correlation={corr:.3f}, mean_ratio={ratio:.3f}")

    # Create plots
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(de, dr, s=6, alpha=0.6)
    plt.xlabel("Euclidean edge length")
    plt.ylabel("Riemannian edge length")
    plt.title(f"Riemannian vs Euclidean (k={K_NEIGHBORS})")
    
    plt.subplot(1, 2, 2)
    plt.hist(ratios, bins=50)
    plt.xlabel("Ratio Riemannian / Euclidean")
    plt.ylabel("Count")
    plt.title("Distribution of length ratios")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "riemann_analysis.png"), dpi=150)
    print(f"Plots saved to: {OUTPUT_DIR}/riemann_analysis.png")


if __name__ == "__main__":
    run_experiment()
