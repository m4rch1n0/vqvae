"""
Interactive VAE k-NN Geodesic Visualization Demo

Interactive exploration tool for k-NN graphs and geodesic distances in VAE latent spaces.
Generates synthetic data and launches an interactive visualization.
"""
import numpy as np

from visualizations.interactive_knn_viz import create_interactive_demo


def create_synthetic_latents():
    """Generate synthetic 2D latent data that simulates MNIST digit distributions."""
    np.random.seed(42)
    n_points_per_digit = 30
    
    centers = []
    for i in range(10):
        angle = 2 * np.pi * i / 10
        cx = 3 * np.cos(angle)
        cy = 3 * np.sin(angle)
        centers.append((cx, cy))
    
    latents_list = []
    labels_list = []
    
    for digit, (cx, cy) in enumerate(centers):
        cluster_points = np.random.multivariate_normal(
            [cx, cy], 
            [[0.64, 0.24], [0.24, 0.64]],
            n_points_per_digit
        )
        latents_list.append(cluster_points)
        labels_list.extend([digit] * n_points_per_digit)
    
    latents_2d = np.vstack(latents_list)
    labels = np.array(labels_list)
    
    return latents_2d, labels


def main():
    print("Interactive VAE k-NN Geodesic Visualization")
    
    # Generate synthetic data
    print("Generating synthetic latent data...")
    latents_2d, labels = create_synthetic_latents()
    print(f"Generated {len(latents_2d)} points with {len(np.unique(labels))} classes")
    
    # Launch interactive visualization
    viz = create_interactive_demo(latents_2d, labels)
    viz.show()

if __name__ == '__main__':
    main()