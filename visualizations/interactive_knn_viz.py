"""
Interactive k-NN Geodesic Visualization

Interactive tool for exploring k-NN graphs and geodesic distances in VAE latent spaces.
Click on points to select source nodes and change k values using control buttons.
"""
import numpy as np
import matplotlib
# Configure matplotlib backend to avoid Qt issues on Linux
try:
    matplotlib.use('TkAgg')  # Try TkAgg as most stable backend
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # Fallback to Qt5Agg
    except ImportError:
        matplotlib.use('Agg')  # Last resort (non-interactive)
        print("Warning: Using non-interactive backend. Charts may not display properly.")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
from pathlib import Path

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.geo.knn_graph import build_knn_graph
from src.geo.geo_shortest_paths import dijkstra_multi_source


class InteractiveKNNVisualizer:
    """Interactive visualization for exploring k-NN graphs and geodesic distances."""
    
    def __init__(self, latents_2d, labels=None):
        """Initialize the interactive visualizer."""
        self.latents_2d = latents_2d
        self.labels = labels
        self.current_k = 3
        self.min_k = 1
        self.max_k = 15
        self.selected_node = 0
        self._knn_cache = {}
        self._colorbar = None
        
        self._setup_figure()
        self._setup_controls()
        self.update_plot()
        
    def _setup_figure(self):
        """Configure the matplotlib figure."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        plt.subplots_adjust(bottom=0.15, right=0.85)
        
    def _setup_controls(self):
        """Configure the control buttons."""
        ax_prev = plt.axes([0.15, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.26, 0.02, 0.1, 0.04])
        ax_reset = plt.axes([0.4, 0.02, 0.1, 0.04])
        
        self.btn_prev = Button(ax_prev, 'k -1')
        self.btn_next = Button(ax_next, 'k +1')
        self.btn_reset = Button(ax_reset, 'Reset')
        
        self.btn_prev.on_clicked(self._prev_k)
        self.btn_next.on_clicked(self._next_k)
        self.btn_reset.on_clicked(self._reset_view)
        
    def _on_click(self, event):
        """Handle mouse click to select a node."""
        if event.inaxes != self.ax:
            return
        
        distances = np.sqrt((self.latents_2d[:, 0] - event.xdata)**2 + 
                           (self.latents_2d[:, 1] - event.ydata)**2)
        self.selected_node = np.argmin(distances)
        self.update_plot()
        
    def _prev_k(self, event):
        """Decrease the k value."""
        self.current_k = max(self.min_k, self.current_k - 1)
        self.update_plot()
        
    def _next_k(self, event):
        """Increase the k value."""
        self.current_k = min(self.max_k, self.current_k + 1)
        self.update_plot()
        
    def _reset_view(self, event):
        """Reset view to first node and initial k value."""
        self.selected_node = 0
        self.current_k = 3
        self.update_plot()
        
    def _get_knn_graph(self, k):
        """Get k-NN graph with caching."""
        if k not in self._knn_cache:
            W, neighbors_dict = build_knn_graph(self.latents_2d, k=k, mode="distance")
            self._knn_cache[k] = (W, neighbors_dict)
        return self._knn_cache[k]
        
    def _compute_geodesic_distances(self, k, source_idx):
        """Compute geodesic distances from source node."""
        W, _ = self._get_knn_graph(k)
        geodesic_distances = dijkstra_multi_source(W, sources=[source_idx])[0]
        return geodesic_distances
        
    def _draw_knn_edges(self, k, source_idx):
        """Draw k-NN edges from the selected node."""
        _, neighbors_dict = self._get_knn_graph(k)
        neighbors_indices = neighbors_dict["indices"]
        
        for neighbor_idx in neighbors_indices[source_idx]:
            self.ax.plot([self.latents_2d[source_idx, 0], self.latents_2d[neighbor_idx, 0]], 
                        [self.latents_2d[source_idx, 1], self.latents_2d[neighbor_idx, 1]], 
                        'red', alpha=0.8, linewidth=2, zorder=1)
                        
    def _draw_points_by_distance(self, geodesic_distances):
        """Draw points colored by geodesic distance."""
        finite_mask = np.isfinite(geodesic_distances)
        unreachable_mask = ~finite_mask
        
        if finite_mask.any():
            connected_distances = geodesic_distances[finite_mask]
            scatter = self.ax.scatter(self.latents_2d[finite_mask, 0], 
                                    self.latents_2d[finite_mask, 1], 
                                    c=connected_distances, 
                                    cmap='viridis', 
                                    s=50, 
                                    alpha=0.8, 
                                    zorder=2)
            
            if self._colorbar is None:
                cbar_ax = self.fig.add_axes([0.87, 0.15, 0.03, 0.7])
                self._colorbar = self.fig.colorbar(scatter, cax=cbar_ax)
                self._colorbar.set_label('Geodesic Distance')
            else:
                self._colorbar.update_normal(scatter)
        
        if unreachable_mask.any():
            self.ax.scatter(self.latents_2d[unreachable_mask, 0], 
                           self.latents_2d[unreachable_mask, 1], 
                           c='black', 
                           s=50, 
                           alpha=0.5, 
                           label='Unreachable',
                           zorder=2)
                           
    def _draw_source_node(self, source_idx):
        """Highlight the selected source node."""
        self.ax.scatter(self.latents_2d[source_idx, 0], 
                       self.latents_2d[source_idx, 1], 
                       c='red', 
                       s=150, 
                       marker='*', 
                       edgecolors='black', 
                       linewidth=2, 
                       label='Selected node',
                       zorder=3)
                       
    def _compute_connectivity_stats(self, geodesic_distances):
        """Compute connectivity statistics."""
        finite_mask = np.isfinite(geodesic_distances)
        n_connected = finite_mask.sum()
        connectivity_pct = (n_connected / len(self.latents_2d)) * 100
        
        if n_connected > 1:
            mean_distance = geodesic_distances[finite_mask].mean()
        else:
            mean_distance = 0.0
            
        return n_connected, connectivity_pct, mean_distance
        
    def update_plot(self):
        """Update the visualization with current parameters."""
        self.ax.clear()
        
        k = self.current_k
        source_idx = self.selected_node
        
        # Compute geodesic distances
        geodesic_distances = self._compute_geodesic_distances(k, source_idx)
        
        # Draw components
        self._draw_knn_edges(k, source_idx)
        self._draw_points_by_distance(geodesic_distances)
        self._draw_source_node(source_idx)
        
        # Compute and display statistics
        n_connected, connectivity_pct, mean_distance = self._compute_connectivity_stats(geodesic_distances)
        
        # Configure title and labels
        title = (f'k={k} | Node: {source_idx} | '
                f'Connected: {n_connected}/{len(self.latents_2d)} ({connectivity_pct:.1f}%) | '
                f'Mean dist: {mean_distance:.2f}')
        
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Latent Dimension 1')
        self.ax.set_ylabel('Latent Dimension 2')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        if np.any(~np.isfinite(geodesic_distances)):
            self.ax.legend(loc='upper right', fontsize=8)
        
        plt.draw()
        
    def show(self):
        """Display the interactive visualization."""
        plt.show()


def create_interactive_demo(latents_2d, labels=None):
    """Create an interactive k-NN visualization demo."""
    return InteractiveKNNVisualizer(latents_2d, labels)