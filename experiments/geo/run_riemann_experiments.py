# Effect of replacing Euclidean edge weights with Riemannian lengths in k-NN graph
import os, sys, numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scipy.sparse.csgraph import connected_components, dijkstra
from scipy import sparse
from src.geo.knn_graph_optimized import build_knn_graph, largest_connected_component
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
    parser = argparse.ArgumentParser(description="Riemannian graph effects analysis")
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

# Auto-detection functions now imported from src.utils.checkpoint_utils

def mean_shortest_path(W, sources_idx):
    """Compute mean shortest path distance from sources"""
    D = np.asarray(dijkstra(W, directed=False, indices=sources_idx))
    valid_paths = D[(np.isfinite(D) & (D > 0))]
    return float(valid_paths.mean()) if valid_paths.size > 0 else float("inf")

def pick_sources_from_lcc(W, num_sources, rng):
    """Pick random sources from largest connected component"""
    lcc_nodes = np.where(largest_connected_component(W))[0]
    return rng.choice(lcc_nodes, size=min(num_sources, len(lcc_nodes)), replace=False)

def run_experiment(args):
    """Run graph reweighting effects comparison experiment"""
    
    # Get paths from dataset config
    latents_path = DATASET_CONFIGS[args.dataset]['latents_path']
    checkpoint_path = DATASET_CONFIGS[args.dataset]['checkpoint_path']
    
    # Auto-generate output directory (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "experiments" / "geo" / "riemann_graph_effects" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)
    
    # Fixed parameters
    k_neighbors = 10
    reweight_mode = "subset"
    sample_edges = 5000
    num_bins = 5
    num_sources = 8
    rng = np.random.RandomState(0)

    # Load latent vectors
    print(f"Loading latents from: {latents_path}")
    z = load_latents(latents_path)
    z = z.cpu()
    N, D = z.shape[0], z.shape[1]
    print(f"Loaded {N} latent vectors of dimension {D}")
    
    # Load decoder using utility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = get_vae_decoder(checkpoint_path, latent_dim=D, device=device)
    
    if decoder is None:
        print("Cannot load decoder. Exiting.")
        return

    # Build k-NN graph
    print(f"Building k-NN graph with k={k_neighbors}")
    W_euc, _ = build_knn_graph(z.numpy(), k=k_neighbors, metric="euclidean", mode="distance", sym="mutual")

    # Analyze baseline connectivity
    ncomp_euc, labels_euc = connected_components(W_euc, directed=False)
    lcc_size_euc = int(np.bincount(labels_euc).max())
    src = pick_sources_from_lcc(W_euc, num_sources, rng)
    mean_sp_euc = mean_shortest_path(W_euc, src)
    print(f"[Euclidean] components={ncomp_euc}, LCC size={lcc_size_euc}, mean_sp={mean_sp_euc:.4f}")

    # Get unique edges (i<j only) for reweighting
    rows, cols = W_euc.nonzero()
    mask = rows < cols
    i_all, j_all = rows[mask], cols[mask]
    
    if reweight_mode == "full":
        i_sel, j_sel = i_all, j_all
        print(f"Re-weighting ALL {len(i_sel)} edges")
    else:
        # Stratified sampling by Euclidean distance
        ze = z.numpy()
        distances = np.linalg.norm(ze[j_all] - ze[i_all], axis=1)
        quantiles = np.quantile(distances, np.linspace(0, 1, num_bins + 1)[1:-1])
        bins = np.digitize(distances, quantiles)
        
        # Sample from each bin
        n_per_bin = max(1, sample_edges // num_bins)
        selected = []
        for b in range(num_bins):
            candidates = np.where(bins == b)[0]
            if len(candidates) > 0:
                n_take = min(n_per_bin, len(candidates))
                selected.extend(rng.choice(candidates, n_take, replace=False))
        
        i_sel, j_sel = i_all[selected], j_all[selected]
        print(f"Re-weighting {len(i_sel)} edges (stratified sampling)")

    # Compute Riemannian edge lengths and update graph
    zi, zj = z[i_sel].to(device), z[j_sel].to(device)
    with torch.no_grad():
        riem_lengths = edge_lengths_riemannian(decoder, zi, zj, batch_size=256).cpu().numpy()

    # Create Riemannian-weighted graph
    W_riem = W_euc.tolil()
    W_riem[i_sel, j_sel] = W_riem[j_sel, i_sel] = riem_lengths  # Symmetric update
    W_riem = W_riem.tocsr()

    # Compare connectivity metrics
    ncomp_r, labels_r = connected_components(W_riem, directed=False)
    lcc_size_r = int(np.bincount(labels_r).max())
    mean_sp_r = mean_shortest_path(W_riem, src)
    ratio_sp = mean_sp_r / mean_sp_euc if np.isfinite(mean_sp_euc) else np.inf

    print(f"[Riemann]  components={ncomp_r}, LCC size={lcc_size_r}, mean_sp={mean_sp_r:.4f}")
    print(f"[Effect]   mean shortest-path ratio (Riem/Eucl) = {ratio_sp:.3f}")

    # Save results
    out_npz = os.path.join(output_dir, f"graph_effects_{args.dataset}.npz")
    np.savez(out_npz,
        ncomp_euc=ncomp_euc, lcc_size_euc=lcc_size_euc, mean_sp_euc=mean_sp_euc,
        ncomp_riem=ncomp_r, lcc_size_riem=lcc_size_r, mean_sp_riem=mean_sp_r,
        ratio_sp=ratio_sp, reweight_mode=reweight_mode,
        sample_edges=len(i_sel), k=k_neighbors, num_sources=len(src), 
        dataset=args.dataset
    )
    print(f"Saved metrics to: {out_npz}")

    # Create plot
    plt.figure(figsize=(5,4))
    plt.bar(["Euclidean","Riemann"], [mean_sp_euc, mean_sp_r])
    plt.ylabel("Mean shortest-path distance")
    plt.title(f"{args.dataset.upper()} - k={k_neighbors}, mode={reweight_mode}, edges={len(i_sel)}")
    plt.tight_layout()
    out_png = os.path.join(output_dir, f"graph_effects_{args.dataset}.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")

if __name__ == "__main__":
    args = parse_args()
    print(f"Running Riemann graph effects analysis on {args.dataset.upper()} dataset")
    run_experiment(args)
