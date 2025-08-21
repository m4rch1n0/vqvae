"""
Geodesic Distance Analysis for different k values
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from src.geo.knn_graph import build_knn_graph
from src.geo.geo_shortest_paths import dijkstra_multi_source


# Analyze geodesic distances for 3 k values
k_values = [4, 5,6]

def random_latents(N=150, D=2, scale=2.0, seed=42):
    """Generate random latent vectors for visualization."""
    r = np.random.RandomState(seed)
    return (r.randn(N, D) * scale).astype(np.float32)


def plot_geodesic_distances_vs_k(z, k_values, source_idx=0, output_path=None):
    """Show geodesic distances for different k values."""
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 5))
    if len(k_values) == 1:
        axes = [axes]
    
    print(f"Source node: index={source_idx}, coordinates=({z[source_idx, 0]:.2f}, {z[source_idx, 1]:.2f})")
    
    for i, k in enumerate(k_values):
        # Build graph and analyze connectivity
        W, _ = build_knn_graph(z, k=k, mode="distance")
        n_components, labels = sparse.csgraph.connected_components(W, directed=False)
        
        # Compute geodesic distances from source
        geodesic_distances = dijkstra_multi_source(W, sources=[source_idx])[0]
        
        # Analyze connectivity
        finite_mask = np.isfinite(geodesic_distances)
        n_connected = finite_mask.sum()
        connectivity_pct = (n_connected / len(z)) * 100
        
        # Distance statistics for connected nodes
        connected_distances = geodesic_distances[finite_mask]
        if len(connected_distances) > 0:
            min_dist = connected_distances.min()
            max_dist = connected_distances.max()
            mean_dist = connected_distances.mean()
            # Exclude source node (distance=0) for more meaningful stats
            non_source_distances = connected_distances[connected_distances > 1e-10]
            if len(non_source_distances) > 0:
                min_nonzero = non_source_distances.min()
                mean_nonzero = non_source_distances.mean()
            else:
                min_nonzero = mean_nonzero = 0
        else:
            min_dist = max_dist = mean_dist = min_nonzero = mean_nonzero = 0
        
        # Print detailed statistics
        print(f"k={k}:")
        print(f"  Connected nodes: {n_connected}/{len(z)} ({connectivity_pct:.1f}%)")
        print(f"  Graph components: {n_components}")
        print(f"  Geodesic distances - min: {min_dist:.2f} (source to itself), max: {max_dist:.2f}")
        print(f"                     - min to others: {min_nonzero:.2f}, mean: {mean_nonzero:.2f}")
        print()
        
        # Plot disconnected nodes in black
        disconnected_mask = ~finite_mask
        if disconnected_mask.any():
            axes[i].scatter(z[disconnected_mask, 0], z[disconnected_mask, 1], 
                          c='black', s=30, alpha=0.8, label='Unreachable')
        
        # Plot connected nodes with geodesic distance colors
        if finite_mask.any():
            connected_distances = geodesic_distances[finite_mask]
            scatter = axes[i].scatter(z[finite_mask, 0], z[finite_mask, 1], 
                                    c=connected_distances, cmap='viridis', s=30)
            plt.colorbar(scatter, ax=axes[i])
        
        # Plot source node
        axes[i].scatter(z[source_idx, 0], z[source_idx, 1], 
                       c='red', s=100, marker='*', label='Source')
        
        # Add legend and title
        axes[i].set_title(f'k={k}\n{n_connected}/{len(z)} connected ({connectivity_pct:.0f}%)')
        axes[i].set_aspect('equal')
        if disconnected_mask.any():
            axes[i].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def main():
    """Analyze geodesic distances for different k values."""
    print("Analyzing geodesic distances vs k...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'demo_outputs', 'knn_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random 2D data
    z = random_latents()
    print(f"Generated {z.shape[0]} points in {z.shape[1]}D space")
    
    # Choose source node (first point by default, you can change this)
    source_idx = 0  # Using first point as source for geodesic distance calculation
    print(f"Using point {source_idx} as source (red star in plots)")
    
    
    print(f"\nAnalyzing geodesic distances for k = {k_values}")
    print("Note: Geodesic distance = shortest path on k-NN graph using Dijkstra\n")
    
    fig = plot_geodesic_distances_vs_k(z, k_values, source_idx=source_idx,
                                       output_path=os.path.join(output_dir, 'geodesic_vs_k.png'))
    
    plt.show()
    print(f"Visualization saved in: {output_dir}")


if __name__ == "__main__":
    main()
