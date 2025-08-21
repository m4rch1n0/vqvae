"""
Shared plotting utilities for VQ-VAE project visualizations.

Common styling, color schemes, and helper functions for consistent plot generation
across demos, experiments, and interactive visualizations.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Dict, Any


def setup_matplotlib_style():
    """Configure matplotlib with project-wide styling preferences."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3
    })


def get_digit_colors(n_digits: int = 10) -> List[str]:
    """
    Get consistent color scheme for MNIST digit visualization.
    
    **Arguments:**
    - n_digits: number of digit classes (default: 10)
    
    **Returns:**
    - List of hex color codes for digit classes
    """
    # Use tab10 colormap for consistency
    cmap = plt.cm.tab10
    return [cmap(i) for i in range(n_digits)]


def plot_latent_scatter(latents: np.ndarray, labels: np.ndarray, 
                       title: str = "Latent Space Distribution",
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create scatter plot of 2D latent representations colored by digit class.
    
    **Arguments:**
    - latents: 2D latent points (N, 2)
    - labels: digit labels (N,)
    - title: plot title
    - save_path: optional file path for saving
    - figsize: figure dimensions
    
    **Returns:**
    - matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_digit_colors()
    
    for digit in range(10):
        mask = labels == digit
        if np.any(mask):
            ax.scatter(latents[mask, 0], latents[mask, 1], 
                      c=[colors[digit]], label=f'Digit {digit}', 
                      alpha=0.7, s=20)
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_distance_heatmap(distances: np.ndarray, labels: np.ndarray,
                         source_digit: int, k_value: int,
                         title_prefix: str = "Geodesic Distances",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create heatmap visualization of distances from source digit to all points.
    
    **Arguments:**
    - distances: distance array (N,)
    - labels: digit labels (N,) 
    - source_digit: digit class used as source
    - k_value: k-NN parameter for title
    - title_prefix: prefix for plot title
    - save_path: optional save path
    - figsize: figure dimensions
    
    **Returns:**
    - matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Distance histogram by digit class
    colors = get_digit_colors()
    for digit in range(10):
        mask = labels == digit
        if np.any(mask):
            ax1.hist(distances[mask], bins=30, alpha=0.6, 
                    color=colors[digit], label=f'Digit {digit}', density=True)
    
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distance Distribution (Source: Digit {source_digit})')
    ax1.legend()
    
    # Distance vs digit class boxplot
    digit_distances = [distances[labels == digit] for digit in range(10)]
    bp = ax2.boxplot(digit_distances, labels=[str(i) for i in range(10)], 
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Target Digit Class')
    ax2.set_ylabel('Distance')
    ax2.set_title(f'Distance by Target Class (k={k_value})')
    
    plt.suptitle(f'{title_prefix} from Digit {source_digit} Centroid', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_connectivity_comparison(metrics_euc: Dict[str, Any], 
                               metrics_riem: Dict[str, Any],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Compare connectivity metrics between Euclidean and Riemannian graphs.
    
    **Arguments:**
    - metrics_euc: Euclidean graph metrics dictionary
    - metrics_riem: Riemannian graph metrics dictionary  
    - save_path: optional save path
    - figsize: figure dimensions
    
    **Returns:**
    - matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Connectivity metrics comparison
    metrics = ['ncomp', 'lcc_size']
    euc_vals = [metrics_euc.get(m, 0) for m in metrics]
    riem_vals = [metrics_riem.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, euc_vals, width, label='Euclidean', alpha=0.8)
    ax1.bar(x + width/2, riem_vals, width, label='Riemannian', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Count')
    ax1.set_title('Graph Connectivity Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Components', 'LCC Size'])
    ax1.legend()
    
    # Distance metrics comparison
    if 'mean_sp_euc' in metrics_euc and 'mean_sp_riem' in metrics_riem:
        dist_vals = [metrics_euc['mean_sp_euc'], metrics_riem['mean_sp_riem']]
        ax2.bar(['Euclidean', 'Riemannian'], dist_vals, alpha=0.8, 
               color=['steelblue', 'orange'])
        ax2.set_ylabel('Average Shortest Path')
        ax2.set_title('Path Length Comparison')
        
        # Add ratio annotation
        ratio = metrics_riem['mean_sp_riem'] / metrics_euc['mean_sp_euc']
        ax2.text(0.5, max(dist_vals) * 0.9, f'Ratio: {ratio:.3f}', 
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_plot_with_timestamp(fig: plt.Figure, base_name: str, 
                           output_dir: str = "outputs") -> str:
    """
    Save plot with timestamp to avoid overwriting.
    
    **Arguments:**
    - fig: matplotlib Figure to save
    - base_name: base filename without extension
    - output_dir: output directory
    
    **Returns:**
    - str: full path where plot was saved
    """
    from datetime import datetime
    from pathlib import Path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"{base_name}_{timestamp}.png"
    full_path = output_path / filename
    
    fig.savefig(full_path, dpi=150, bbox_inches='tight')
    return str(full_path)


# Initialize style on import
setup_matplotlib_style()
