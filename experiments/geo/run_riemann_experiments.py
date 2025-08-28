# Effect of replacing Euclidean edge weights with Riemannian lengths in k-NN graph
import os, sys, numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scipy.sparse.csgraph import connected_components, dijkstra
from src.geo.knn_graph_optimized import build_knn_graph, largest_connected_component
from src.geo.riemannian_metric import edge_lengths_riemannian
from src.models.vae import VAE

# Configuration - modify these values as needed
LATENTS_PATH = "experiments/vae_mnist/latents_val/z.pt"
CHECKPOINT_PATH = "experiments/vae_mnist/checkpoints/best.pt"
OUTPUT_DIR = "experiments/geo/riemann_graph_effects"
K_NEIGHBORS = 10
REWEIGHT_MODE = "subset"  # "subset" or "full"
SAMPLE_EDGES = 5000
NUM_BINS = 5
NUM_SOURCES = 8

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

def mean_shortest_path(W, sources_idx):
    """Compute mean shortest path distance from sources"""
    D = dijkstra(W, directed=False, indices=sources_idx)
    D = np.asarray(D)
    mask = np.isfinite(D) & (D > 0)
    return float(D[mask].mean()) if mask.any() else float("inf")

def pick_sources_from_lcc(W, num_sources, rng):
    """Pick random sources from largest connected component"""
    lcc_mask = largest_connected_component(W)
    lcc_nodes = np.where(lcc_mask)[0]
    k = min(num_sources, len(lcc_nodes))
    return rng.choice(lcc_nodes, size=k, replace=False)

def run_experiment():
    """Run graph reweighting effects comparison experiment"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.RandomState(0)

    # Load latent vectors
    print(f"Loading latents from: {LATENTS_PATH}")
    z = load_latents(LATENTS_PATH)
    z = z.cpu()
    N, D = z.shape[0], z.shape[1]
    print(f"Loaded {N} latent vectors of dimension {D}")
    
    # Load decoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = load_decoder(CHECKPOINT_PATH, latent_dim=D, device=device)
    
    if decoder is None:
        print("Cannot load decoder. Exiting.")
        return

    # Build k-NN graph
    print(f"Building k-NN graph with k={K_NEIGHBORS}")
    W_euc, _ = build_knn_graph(z.numpy(), k=K_NEIGHBORS, metric="euclidean", mode="distance", sym="mutual")

    # Analyze baseline connectivity
    ncomp_euc, labels_euc = connected_components(W_euc, directed=False)
    lcc_size_euc = int(np.bincount(labels_euc).max())
    src = pick_sources_from_lcc(W_euc, NUM_SOURCES, rng)
    mean_sp_euc = mean_shortest_path(W_euc, src)
    print(f"[Euclidean] components={ncomp_euc}, LCC size={lcc_size_euc}, mean_sp={mean_sp_euc:.4f}")

    # Select edges to reweight (deduplicated: i<j)
    rows, cols = W_euc.nonzero()
    mask_triu = rows < cols
    i_all, j_all = rows[mask_triu], cols[mask_triu]
    if i_all.size == 0:
        raise RuntimeError("No edges found in the k-NN graph.")
    
    # Compute Euclidean distances for stratification
    ze = z.numpy()
    de_all = np.linalg.norm(ze[j_all] - ze[i_all], axis=1)

    if REWEIGHT_MODE == "full":
        i_sel, j_sel = i_all, j_all
        print(f"Re-weighting ALL edges: {len(i_sel)}")
    else:
        m = min(SAMPLE_EDGES, len(i_all))
        # Stratify by quantiles of Euclidean distance
        qs = np.quantile(de_all, np.linspace(0, 1, NUM_BINS + 1)[1:-1])
        bins = np.digitize(de_all, qs)
        per_bin = max(1, m // NUM_BINS)
        idx_sel = []
        for b in range(NUM_BINS):
            cand = np.where(bins == b)[0]
            if cand.size == 0: 
                continue
            take = min(per_bin, cand.size)
            idx_sel.append(rng.choice(cand, size=take, replace=False))
        if len(idx_sel) == 0:
            raise RuntimeError("No edges selected; check SAMPLE_EDGES/NUM_BINS.")
        sel = np.concatenate(idx_sel)
        i_sel, j_sel = i_all[sel], j_all[sel]
        print(f"Re-weighting {len(i_sel)} edges (stratified by Euclidean length)")

    # Compute Riemannian edge lengths
    zi = z[i_sel].to(device)
    zj = z[j_sel].to(device)
    with torch.no_grad():
        L_riem = edge_lengths_riemannian(decoder, zi, zj, batch_size=256).cpu().numpy()

    # Update weights symmetrically
    from scipy import sparse
    W_riem = W_euc.tolil()
    W_riem[i_sel, j_sel] = L_riem
    W_riem[j_sel, i_sel] = L_riem
    W_riem = W_riem.tocsr()

    # Analyze post-reweight connectivity
    ncomp_r, labels_r = connected_components(W_riem, directed=False)
    lcc_size_r = int(np.bincount(labels_r).max())
    mean_sp_r = mean_shortest_path(W_riem, src)  # same sources for comparison
    ratio_sp = mean_sp_r / mean_sp_euc if np.isfinite(mean_sp_euc) else np.inf

    print(f"[Riemann]  components={ncomp_r}, LCC size={lcc_size_r}, mean_sp={mean_sp_r:.4f}")
    print(f"[Effect]   mean shortest-path ratio (Riem/Eucl) = {ratio_sp:.3f}")

    # Save results
    out_npz = os.path.join(OUTPUT_DIR, "graph_effects.npz")
    np.savez(out_npz,
        ncomp_euc=ncomp_euc, lcc_size_euc=lcc_size_euc, mean_sp_euc=mean_sp_euc,
        ncomp_riem=ncomp_r, lcc_size_riem=lcc_size_r, mean_sp_riem=mean_sp_r,
        ratio_sp=ratio_sp, reweight_mode=REWEIGHT_MODE,
        sample_edges=len(i_sel), k=K_NEIGHBORS, num_sources=len(src)
    )
    print(f"Saved metrics to: {out_npz}")

    # Create plot
    plt.figure(figsize=(5,4))
    plt.bar(["Euclidean","Riemann"], [mean_sp_euc, mean_sp_r])
    plt.ylabel("Mean shortest-path distance")
    plt.title(f"k={K_NEIGHBORS}, mode={REWEIGHT_MODE}, edges={len(i_sel)}")
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "graph_effects.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")

if __name__ == "__main__":
    run_experiment()
