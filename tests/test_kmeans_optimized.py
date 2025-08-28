import numpy as np
import pytest
from scipy import sparse

from src.geo.kmeans_optimized import (
    kpp_initialization_graph, 
    fit_kmedoids_optimized,
    assign_points_to_medoids,
    compute_quantization_error
)
from src.geo.knn_graph_optimized import build_knn_graph


def complete_graph(N: int, w: float = 1.0):
    """Create complete graph with N nodes."""
    rows, cols, data = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            rows += [i, j]
            cols += [j, i]
            data += [w, w]
    return sparse.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)


def triangle_graph():
    """Create simple triangle graph."""
    rows = [0, 1, 1, 2, 0, 2]
    cols = [1, 0, 2, 1, 2, 0]
    data = [1.0] * 6
    return sparse.csr_matrix((data, (rows, cols)), shape=(3, 3), dtype=np.float32)


def random_latents(N=200, D=16, seed=0):
    """Generate random latent vectors for testing."""
    r = np.random.RandomState(seed)
    return r.randn(N, D).astype(np.float32)


class TestKppInitializationGraph:
    """Tests for kpp_initialization_graph function."""
    
    def test_basic_functionality(self):
        """Test basic k-means++ initialization."""
        W = complete_graph(10)
        K = 3
        centers = kpp_initialization_graph(W, K, seed=42)
        
        assert len(centers) == K
        assert all(0 <= c < 10 for c in centers)
        assert len(set(centers)) == K  # No duplicates
    
    def test_single_center(self):
        """Test with K=1."""
        W = triangle_graph()
        centers = kpp_initialization_graph(W, K=1, seed=42)
        
        assert len(centers) == 1
        assert 0 <= centers[0] < 3
    
    def test_deterministic_with_seed(self):
        """Test that same seed gives same results."""
        W = complete_graph(8)
        K = 3
        
        centers1 = kpp_initialization_graph(W, K, seed=42)
        centers2 = kpp_initialization_graph(W, K, seed=42)
        
        assert centers1 == centers2
    
    def test_disconnected_graph_handling(self):
        """Test behavior with disconnected graph."""
        # Create two disconnected triangles
        W1 = triangle_graph()
        W2 = triangle_graph()
        W = sparse.block_diag((W1, W2), format="csr", dtype=np.float32)
        
        centers = kpp_initialization_graph(W, K=2, seed=42)
        assert len(centers) <= 2  # May select fewer if graph is disconnected
        assert all(0 <= c < 6 for c in centers)


class TestFitKmedoidsOptimized:
    """Tests for fit_kmedoids_optimized function."""
    
    def test_basic_functionality(self):
        """Test basic k-medoids functionality."""
        W = complete_graph(10)
        K = 3
        
        medoids, assign, qe = fit_kmedoids_optimized(W, K=K, init="kpp", seed=42)
        
        # Check output shapes and types
        assert medoids.shape == (K,)
        assert assign.shape == (10,)
        assert medoids.dtype == int
        assert assign.dtype == int
        assert isinstance(qe, float)
        
        # Check medoids are valid indices
        assert all(0 <= m < 10 for m in medoids)
        assert len(set(medoids)) == K  # No duplicate medoids
        
        # Check assignments are valid
        assert all(0 <= a < K for a in assign)
        
        # Check quantization error is non-negative
        assert qe >= 0
    
    def test_both_initialization_methods(self):
        """Test both random and kpp initialization work."""
        W = complete_graph(15)
        K = 4
        
        medoids_kpp, assign_kpp, qe_kpp = fit_kmedoids_optimized(W, K=K, init="kpp", seed=42)
        medoids_rand, assign_rand, qe_rand = fit_kmedoids_optimized(W, K=K, init="random", seed=42)
        
        # Both should have correct shapes
        assert medoids_kpp.shape == (K,)
        assert medoids_rand.shape == (K,)
        assert assign_kpp.shape == (15,)
        assert assign_rand.shape == (15,)
        
        # Both should have valid quantization errors
        assert qe_kpp >= 0 and qe_rand >= 0
    
    def test_single_cluster(self):
        """Test with K=1."""
        W = triangle_graph()
        medoids, assign, qe = fit_kmedoids_optimized(W, K=1, init="kpp", seed=42)
        
        assert medoids.shape == (1,)
        assert assign.shape == (3,)
        assert 0 <= medoids[0] < 3
        assert all(a == 0 for a in assign)  # All assigned to cluster 0
        assert qe >= 0
    
    def test_deterministic_with_seed(self):
        """Test that same seed gives same results."""
        W = complete_graph(12)
        K = 3
        
        medoids1, assign1, qe1 = fit_kmedoids_optimized(W, K=K, init="kpp", seed=42)
        medoids2, assign2, qe2 = fit_kmedoids_optimized(W, K=K, init="kpp", seed=42)
        
        np.testing.assert_array_equal(medoids1, medoids2)
        np.testing.assert_array_equal(assign1, assign2)
        assert abs(qe1 - qe2) < 1e-6
    
    def test_real_knn_graph_integration(self):
        """Test on real k-NN graph from latent vectors."""
        z = random_latents(N=50, D=8, seed=1)
        W, _ = build_knn_graph(z, k=5, mode="distance", sym="mutual")
        
        medoids, assign, qe = fit_kmedoids_optimized(W, K=5, init="kpp", seed=42)
        
        assert len(medoids) <= 5  # May be fewer due to disconnected components
        assert assign.shape == (W.shape[0],)
        assert qe >= 0
        
        # Test that all assignments are valid
        valid_assignments = set(range(len(medoids)))
        assert all(a in valid_assignments for a in assign)
    
    def test_invalid_initialization_raises(self):
        """Test that invalid initialization method raises ValueError."""
        W = triangle_graph()
        
        with pytest.raises(ValueError):
            fit_kmedoids_optimized(W, K=2, init="invalid", seed=42)


class TestComponentFunctions:
    """Test individual component functions."""
    
    def test_assign_points_to_medoids(self):
        """Test point assignment function."""
        W = complete_graph(6)
        medoids = np.array([0, 2, 4], dtype=int)
        
        assign = assign_points_to_medoids(W, medoids)
        
        assert assign.shape == (6,)
        assert assign.dtype == int
        assert all(0 <= a < len(medoids) for a in assign)
        
        # Each medoid should be assigned to itself
        for i, medoid in enumerate(medoids):
            assert assign[medoid] == i
    
    def test_compute_quantization_error(self):
        """Test quantization error computation."""
        W = complete_graph(6)
        medoids = np.array([0, 2, 4], dtype=int)
        assign = np.array([0, 0, 1, 1, 2, 2], dtype=int)
        
        qe = compute_quantization_error(W, medoids, assign)
        
        assert isinstance(qe, float)
        assert qe >= 0
    
    def test_disconnected_graph_error_handling(self):
        """Test behavior with disconnected graphs."""
        # Create two disconnected components
        W1 = complete_graph(3, w=1.0)
        W2 = complete_graph(3, w=1.0) 
        W = sparse.block_diag((W1, W2), format="csr", dtype=np.float32)
        
        # This should handle disconnected components gracefully
        medoids, assign, qe = fit_kmedoids_optimized(W, K=2, init="kpp", seed=42)
        
        assert len(medoids) <= 2
        assert assign.shape == (6,)
        # Quantization error may be infinite for disconnected components
        assert qe >= 0 or np.isinf(qe)


if __name__ == "__main__":
    pytest.main([__file__])
