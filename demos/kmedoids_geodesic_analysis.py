"""
Geodesic K-medoids analysis on a latent space.

Outputs:
- Elbow: geodesic QE vs K
- PCA-like scatter with assignments and medoids
- Code usage histogram with perplexity

If labels are provided, purity/NMI/ARI are also reported.
"""
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


from src.geo.knn_graph_optimized import build_knn_graph
from src.geo.geo_shortest_paths import dijkstra_multi_source
from src.geo.kmeans_optimized import fit_kmedoids_optimized


def load_latents(latent_path: Path) -> np.ndarray:
    """Load latent matrix z from a torch file."""
    if not latent_path.exists():
        raise FileNotFoundError(f"Latents not found at: {latent_path}")
    obj = torch.load(latent_path, map_location="cpu")
    if isinstance(obj, dict) and "z" in obj:
        z = obj["z"].float().numpy()
    elif torch.is_tensor(obj):
        z = obj.float().numpy()
    else:
        raise ValueError("Unsupported latent file format")
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (N,D). Got shape={z.shape}")
    return z


def load_labels(label_path: Path) -> Optional[np.ndarray]:
    """Load labels if available; return None otherwise."""
    if not label_path.exists():
        return None
    obj = torch.load(label_path, map_location="cpu")
    if torch.is_tensor(obj):
        return obj.numpy()
    return None


def compute_purity(assign: np.ndarray, labels: np.ndarray, num_clusters: int) -> float:
    """Cluster purity given assignments and integer labels."""
    total = len(assign)
    if total == 0:
        return float('nan')
    purity_sum = 0
    for k in range(num_clusters):
        members = labels[assign == k]
        if members.size == 0:
            continue
        counts = np.bincount(members)
        purity_sum += counts.max()
    return float(purity_sum / total)


def compute_perplexity(assign: np.ndarray, num_clusters: int) -> float:
    """Code usage perplexity from assignment distribution."""
    if assign.size == 0:
        return 0.0
    counts = np.bincount(assign, minlength=num_clusters).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    entropy = -np.sum(nz * np.log(nz + 1e-12))
    return float(np.exp(entropy))

def evaluate_setup(
    W, K_values: List[int], inits: List[str], seed: int,
    labels: Optional[np.ndarray], out_dir: Path, tag: str,
) -> List[Dict]:
    """Run k-medoids and collect metrics and plots."""
    metrics: List[Dict] = []
    N = W.shape[0]

    for K in K_values:
        for init in inits:
            medoids, assign, qe_geo = fit_kmedoids_optimized(W, K=K, init=init, seed=seed)
            D = dijkstra_multi_source(W, medoids)  # (K, N)
            dmin = D[assign, np.arange(N)]
            finite_mask = np.isfinite(dmin)
            
            qe_geo_finite = qe_geo if np.isfinite(qe_geo) else float('inf')
            finite_fraction = float(finite_mask.mean())

            purity = float('nan')
            nmi = float('nan')
            ari = float('nan')
            if labels is not None:
                purity = compute_purity(assign, labels, K)
                nmi = float(normalized_mutual_info_score(labels, assign))
                ari = float(adjusted_rand_score(labels, assign))

            ppl = compute_perplexity(assign, K)

            metrics.append({
                "graph": tag,
                "K": int(K),
                "init": init,
                "seed": int(seed),
                "qe_geo_finite": qe_geo_finite,
                "finite_fraction": finite_fraction,
                "purity": purity,
                "nmi": nmi,
                "ari": ari,
                "perplexity": ppl,
            })

            if K == K_values[0] and init == inits[0]:
                plot_pca_with_clusters(W, assign, medoids, out_dir / f"pca_clusters_{tag}_K{K}_{init}.png")
                plot_code_usage(assign, K, out_dir / f"code_usage_{tag}_K{K}_{init}.png")

    return metrics


def plot_pca_with_clusters(W, assign: np.ndarray, medoids: np.ndarray, out_path: Path) -> None:
    """PCA on distance-to-medoids features and scatter with medoids highlighted."""
    D = dijkstra_multi_source(W, medoids)  # (K, N)
    X = D.T  # (N, K)
    X = np.where(np.isfinite(X), X, np.nan)
    col_max = np.nanmax(X, axis=0)
    col_max[col_max == 0] = 1.0
    X = np.where(np.isnan(X), (col_max * 1.1)[None, :], X)

    pca = PCA(n_components=2, random_state=0)
    Z2 = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(Z2[:, 0], Z2[:, 1], c=assign, cmap="tab20", s=8, alpha=0.8, linewidths=0)
    # Highlight medoids
    plt.scatter(Z2[medoids, 0], Z2[medoids, 1], c='black', s=60, marker='*', label='Medoids')
    plt.legend(loc='best')
    plt.title("PCA of distance-to-medoids representation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_code_usage(assign: np.ndarray, K: int, out_path: Path) -> None:
    counts = np.bincount(assign, minlength=K)
    probs = counts / max(1, counts.sum())
    nz = probs[probs > 0]
    entropy = -np.sum(nz * np.log(nz + 1e-12))
    perplexity = np.exp(entropy)

    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(K), counts, width=0.9)
    plt.xlabel("Code index")
    plt.ylabel("Count")
    plt.title(f"Code usage (perplexity={perplexity:.2f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_elbow(metrics: List[Dict], out_path: Path, tag: str) -> None:
    """Elbow plot: qe_geo_finite vs K (one line per init)."""
    inits = sorted(set(m["init"] for m in metrics if m["graph"] == tag))
    K_values = sorted(set(int(m["K"]) for m in metrics if m["graph"] == tag))

    plt.figure(figsize=(6, 4))
    for init in inits:
        series = [np.mean([m["qe_geo_finite"] for m in metrics if m["graph"] == tag and m["init"] == init and m["K"] == K]) for K in K_values]
        plt.plot(K_values, series, marker='o', label=f"{init}")

    plt.xlabel("K (number of codes)")
    plt.ylabel("Geodesic QE (finite nodes)")
    plt.title(f"Elbow ({tag})")
    plt.legend(title="init")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def auto_detect_paths(experiment_dir: Path) -> dict:
    """Auto-detect latents and labels paths from experiment directory."""
    exp_dir = Path(experiment_dir)
    
    # Find VAE directory
    vae_dir = exp_dir / "vae"
    if not vae_dir.exists():
        raise FileNotFoundError(f"VAE directory not found: {vae_dir}")
    
    # Look for validation latents in VAE directory structure  
    latents_paths = list(vae_dir.rglob("latents_val/z.pt"))
    if not latents_paths:
        raise FileNotFoundError(f"Validation latents not found in: {vae_dir}")
    latents_path = latents_paths[0]
    
    # Look for labels (optional)
    labels_paths = list(vae_dir.rglob("latents_val/y.pt"))
    labels_path = labels_paths[0] if labels_paths else None
    
    return {
        "latents_path": latents_path,
        "labels_path": labels_path,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Geodesic K-medoids clustering analysis")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (e.g., experiments/fashionmnist/vanilla/euclidean)"
    )
    parser.add_argument(
        "--k_graph", 
        type=int, 
        default=10, 
        help="k-NN graph connectivity (default: 10)"
    )
    parser.add_argument(
        "--graph_sym", 
        type=str, 
        choices=["mutual", "union"],
        default="mutual", 
        help="Graph symmetrization (default: mutual)"
    )
    parser.add_argument(
        "--K_values", 
        type=str, 
        default="32,64,128", 
        help="Comma-separated codebook sizes (default: 32,64,128)"
    )
    parser.add_argument(
        "--inits", 
        type=str, 
        default="kpp,random", 
        help="Comma-separated initialization methods (default: kpp,random)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main() -> Path:
    """Run the geodesic k-medoids demo with command line config."""
    args = parse_args()
    
    # Auto-detect paths from experiment directory
    try:
        paths = auto_detect_paths(args.experiment_dir)
        print(f"Auto-detected paths:")
        print(f"  Latents: {paths['latents_path']}")
        if paths['labels_path']:
            print(f"  Labels: {paths['labels_path']}")
        else:
            print(f"  Labels: Not found (will skip label-based metrics)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return Path(".")
    
    LATENTS_PATH = paths["latents_path"]
    LABELS_PATH = paths["labels_path"]
    
    K_GRAPH = args.k_graph
    GRAPH_SYM = args.graph_sym
    K_VALUES = [int(x.strip()) for x in args.K_values.split(",")]
    INITS = [x.strip() for x in args.inits.split(",")]
    SEED = args.seed

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"demo_outputs/kmedoids_geodesic_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[demo] K-medoids Geodesic Demo")
    print("[demo] Loading latents...")
    z = load_latents(LATENTS_PATH)
    y = load_labels(LABELS_PATH) if LABELS_PATH else None
    N, D = z.shape
    print(f"[demo] Loaded {N} vectors (dim={D}), labels={'yes' if y is not None else 'no'}")

    print(f"[demo] Building k-NN graph: k={K_GRAPH}, sym={GRAPH_SYM}")
    W_euc, _ = build_knn_graph(z, k=K_GRAPH, metric="euclidean", mode="distance", sym=GRAPH_SYM)

    print("[demo] Evaluating geodesic k-medoids on Euclidean-weight graph...")
    metrics_all = evaluate_setup(W_euc, K_VALUES, INITS, seed=SEED, labels=y, out_dir=out_dir, tag="euclidean")

    import csv, json
    csv_path = out_dir / "metrics.csv"
    if metrics_all:
        keys = list(metrics_all[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in metrics_all:
                writer.writerow(row)
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_all, f, indent=2)

    plot_elbow(metrics_all, out_dir / "elbow.png", tag="euclidean")

    print(f"[demo] Done. Outputs saved to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()


