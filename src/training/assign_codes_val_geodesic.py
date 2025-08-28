"""Assign validation samples to geodesic codebook using graph distances."""
import argparse
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from src.geo.knn_graph_optimized import build_knn_graph
from src.geo.geo_shortest_paths import dijkstra_single_source


def load_tensor(path):
    """Load tensor with compatibility for different PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_latents(path):
    """Load latent vectors from file (handles both dict and tensor formats)."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        latents = data.get("z", data)
    else:
        latents = data
    return latents.float().cpu().numpy()


def find_nearest_train_indices(train_latents, medoid_latents, batch_size=8192):
    """Map each medoid to its nearest neighbor in the training set."""
    train_tensor = torch.from_numpy(train_latents)
    medoid_tensor = torch.from_numpy(medoid_latents)
    
    nearest_indices = []
    for i in range(0, len(medoid_tensor), batch_size):
        batch = medoid_tensor[i:i+batch_size]
        distances = torch.cdist(batch, train_tensor)
        nearest_indices.append(distances.argmin(dim=1))
    
    return torch.cat(nearest_indices).numpy()

def main():
    parser = argparse.ArgumentParser(description="Assign validation samples to geodesic codebook")
    parser.add_argument("--latents_train", required=True, help="Training latents file")
    parser.add_argument("--latents_val", required=True, help="Validation latents file") 
    parser.add_argument("--codebook", required=True, help="Codebook file (must contain 'z_medoid')")
    parser.add_argument("--out_codes_val", required=True, help="Output validation codes file")
    parser.add_argument("--k", type=int, default=20, help="k-NN graph parameter")
    parser.add_argument("--sym", type=str, default="union", help="Graph symmetry (union/mutual)")
    args = parser.parse_args()

    # Load data
    train_latents = load_latents(args.latents_train)
    val_latents = load_latents(args.latents_val)
    all_latents = np.concatenate([train_latents, val_latents], axis=0)
    
    num_train, num_val = len(train_latents), len(val_latents)
    print(f"Loaded {num_train} training and {num_val} validation samples")

    # Load codebook
    codebook = load_tensor(args.codebook)
    if "z_medoid" not in codebook:
        raise KeyError("Codebook must contain 'z_medoid' key")
    medoid_latents = codebook["z_medoid"].float().cpu().numpy()
    print(f"Loaded codebook with {len(medoid_latents)} medoids")

    # Map medoids to training indices for graph construction
    medoid_train_indices = find_nearest_train_indices(train_latents, medoid_latents)
    unique_sources, inverse_mapping = np.unique(medoid_train_indices, return_inverse=True)
    print(f"Using {len(unique_sources)} unique source nodes for geodesic distances")

    # Build joint k-NN graph (training + validation)
    graph, _ = build_knn_graph(all_latents, k=args.k, sym=args.sym, metric="euclidean", mode="distance")
    print(f"Built k-NN graph: {graph.shape[0]} nodes, {graph.nnz//2} edges")

    # Analyze graph connectivity
    num_components, component_labels = connected_components(graph, directed=False)
    print(f"Graph has {num_components} connected components")

    # Check which components contain source nodes
    source_components = component_labels[unique_sources]
    components_with_sources = np.zeros(num_components, dtype=bool)
    components_with_sources[np.unique(source_components)] = True

    # Compute geodesic distances from each source to all validation nodes
    distances_to_val = np.full((len(unique_sources), num_val), np.inf, dtype=np.float32)
    
    for i, source_idx in enumerate(unique_sources):
        all_distances = dijkstra_single_source(graph, source=source_idx)
        distances_to_val[i] = all_distances[num_train:]  # Extract validation distances

    # Handle validation nodes in components without sources (Euclidean fallback)
    val_component_labels = component_labels[num_train:]
    nodes_without_sources = ~components_with_sources[val_component_labels]
    
    if nodes_without_sources.any():
        num_fallback = nodes_without_sources.sum()
        print(f"Warning: {num_fallback} validation nodes need Euclidean fallback")
        
        # Compute Euclidean distances for fallback
        isolated_val = val_latents[nodes_without_sources]
        euclidean_assignments = torch.cdist(
            torch.from_numpy(isolated_val),
            torch.from_numpy(medoid_latents)
        ).argmin(dim=1).numpy()

    # Assign each validation node to nearest medoid
    geodesic_assignments = distances_to_val.argmin(axis=0)
    
    # Map back to original medoid indices
    source_to_medoid = np.array([np.where(inverse_mapping == j)[0][0] 
                                for j in range(len(unique_sources))], dtype=int)
    final_codes = source_to_medoid[geodesic_assignments]

    # Apply Euclidean fallback where needed
    if nodes_without_sources.any():
        final_codes[nodes_without_sources] = euclidean_assignments

    # Save results
    np.save(args.out_codes_val, final_codes.astype(np.int32))
    print(f"Saved validation codes to {args.out_codes_val}")


if __name__ == "__main__":
    main()
