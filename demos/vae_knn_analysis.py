"""
VAE k-NN Geodesic Distance Analysis Demo

This demo trains a VAE on MNIST and analyzes the learned latent space structure
using k-NN graphs and geodesic distances. It helps understand how well the VAE
preserves local neighborhood relationships between different digit classes.

The analysis visualizes:
- How digits are distributed in the 2D latent space
- Connectivity patterns for different k values in k-NN graphs  
- Geodesic distances from each digit's centroid to all other points

This is useful for evaluating the quality of latent representations and
understanding how manifold structure emerges in VAE latent spaces.
"""
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.mnist import get_mnist_loaders
from src.models.vae import VAE
from src.utils.system import set_seed, get_device
from src.training.engine import TrainingEngine
from src.geo.knn_graph import build_knn_graph
from src.geo.geo_shortest_paths import dijkstra_multi_source


def get_config():
    return {
        'seed': 42,
        'device': 'auto',
        'max_epochs': 5,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'early_stop': False,
        'max_samples': 300,
        'k_values': [3, 4, 5, 6]
    }


def create_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_dir = Path(f"demo_outputs/vae_mnist_geodesic_{timestamp}")
    demo_dir.mkdir(parents=True, exist_ok=True)
    return demo_dir


def setup_model_and_data(config):
    set_seed(config['seed'])
    device = get_device(config['device'])
    
    data_cfg = OmegaConf.load('configs/mnist/data.yaml')
    vae_cfg = OmegaConf.load('configs/mnist/vae.yaml')
    
    # Data
    train_loader, val_loader = get_mnist_loaders(
        root=to_absolute_path(data_cfg.root),
        batch_size=data_cfg.batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        augment=data_cfg.augment,
    )
    
    # Model
    model = VAE(
        in_channels=getattr(vae_cfg, 'in_channels', 1),
        enc_channels=vae_cfg.enc_channels,
        dec_channels=vae_cfg.dec_channels,
        latent_dim=vae_cfg.latent_dim,
        recon_loss=vae_cfg.recon_loss,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    return model, train_loader, val_loader, optimizer, device


def train_model(model, train_loader, val_loader, optimizer, device, config):
    print("Training VAE on MNIST...")
    
    engine = TrainingEngine(model=model, optimizer=optimizer, device=device)
    
    engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['max_epochs'],
        early_stop=config['early_stop'],
        checkpoint_dir=None,
        logger=None,
        output_dir=None,
        save_latents_flag=False
    )
    
    return model


def extract_latents(model, data_loader, device, max_samples):
    model.eval()
    latents, labels = [], []
    
    with torch.no_grad():
        for data, target in data_loader:
            if len(latents) * data.size(0) >= max_samples:
                break
                
            data = data.to(device)
            mu, logvar = model.encoder(data)
            z = model.reparameterize(mu, logvar)
            
            latents.append(z.cpu().numpy())
            labels.append(target.numpy())
    
    return np.vstack(latents)[:max_samples], np.hstack(labels)[:max_samples]


def reduce_to_2d(latents):
    if latents.shape[1] <= 2:
        return latents
    
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    return latents_2d


def find_class_centroids(latents, labels):
    centroid_indices = {}
    
    for digit in range(10):
        mask = labels == digit
        if mask.any():
            class_latents = latents[mask]
            centroid = class_latents.mean(axis=0)
            

            distances = np.linalg.norm(class_latents - centroid, axis=1)
            closest_idx_in_class = distances.argmin()
            global_indices = np.where(mask)[0]
            centroid_indices[digit] = global_indices[closest_idx_in_class]
    
    return centroid_indices


def plot_distribution(latents_2d, labels, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        mask = labels == digit
        if mask.any():
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                      c=[colors[digit]], label=f'Digit {digit}', 
                      s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax.set_title('MNIST Digits in Latent Space', fontsize=16, fontweight='bold')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = output_dir / 'latent_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Distribution saved: {output_path}")


def plot_geodesic_for_k(latents_2d, k, centroid_indices, output_dir):
    from scipy import sparse
    
    available_digits = list(centroid_indices.keys())
    n_digits = len(available_digits)
    

    rows = 2
    cols = (n_digits + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n_digits > 1 else [axes]
    
    for i, digit in enumerate(available_digits):
        source_idx = centroid_indices[digit]
        

        W, _ = build_knn_graph(latents_2d, k=k, mode="distance")
        

        geodesic_distances = dijkstra_multi_source(W, sources=[source_idx])[0]
        

        finite_mask = np.isfinite(geodesic_distances)
        disconnected_mask = ~finite_mask
        
        if disconnected_mask.any():
            axes[i].scatter(latents_2d[disconnected_mask, 0], latents_2d[disconnected_mask, 1], 
                          c='black', s=30, alpha=0.8, label='Unreachable')
        
        if finite_mask.any():
            connected_distances = geodesic_distances[finite_mask]
            scatter = axes[i].scatter(latents_2d[finite_mask, 0], latents_2d[finite_mask, 1], 
                                    c=connected_distances, cmap='viridis', s=30)
            plt.colorbar(scatter, ax=axes[i])
        

        axes[i].scatter(latents_2d[source_idx, 0], latents_2d[source_idx, 1], 
                       c='red', s=100, marker='*', label='Source')
        

        n_connected = finite_mask.sum()
        connectivity_pct = (n_connected / len(latents_2d)) * 100
        axes[i].set_title(f'Digit {digit}\n{n_connected}/{len(latents_2d)} connected ({connectivity_pct:.0f}%)')
        axes[i].set_aspect('equal')
        
        if disconnected_mask.any():
            axes[i].legend(loc='upper right', fontsize=8)
    

    for j in range(n_digits, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle(f'Geodesic Distances - k={k}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_path = output_dir / f'geodesic_analysis_k{k}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def analyze_geodesics(latents, labels, k_values, output_dir):
    latents_2d = reduce_to_2d(latents)
    centroid_indices = find_class_centroids(latents_2d, labels)
    
    print(f"Found centroids for digits: {list(centroid_indices.keys())}")
    

    plot_distribution(latents_2d, labels, output_dir)
    

    for k in k_values:
        print(f"Analyzing k={k}...")
        plot_geodesic_for_k(latents_2d, k, centroid_indices, output_dir)
    
    return latents_2d, centroid_indices


def main():
    print("VAE k-NN Geodesic Analysis Demo")
    
    config = get_config()
    output_dir = create_output_dir()
    
    # Setup and train
    model, train_loader, val_loader, optimizer, device = setup_model_and_data(config)
    model = train_model(model, train_loader, val_loader, optimizer, device, config)
    
    if model is None:
        print("Training failed")
        return
    
    # Extract and analyze
    latents, labels = extract_latents(model, val_loader, device, config['max_samples'])
    print(f"Extracted {len(latents)} vectors (dim={latents.shape[1]})")
    
    analyze_geodesics(latents, labels, config['k_values'], output_dir)
    
    print(f"Done. Plots saved in: {output_dir}")
    return output_dir


if __name__ == '__main__':
    main()
