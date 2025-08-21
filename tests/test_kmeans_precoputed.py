import numpy as np
import pytest
from scipy import sparse

from src.geo.kmeans_precoputed import _kpp_init_on_graph, fit_kmedoids_graph


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


class TestKppInitOnGraph:
    """Tests for _kpp_init_on_graph function."""
    
    def test_basic_functionality(self):
        """Test basic k-means++ initialization."""
        W = complete_graph(10)
        K = 3
        centers = _kpp_init_on_graph(W, K, seed=42)
        
        assert len(centers) == K
        assert all(0 <= c < 10 for c in centers)
        assert len(set(centers)) == K  # No duplicates
    
    def test_single_center(self):
        """Test with K=1."""
        W = triangle_graph()
        centers = _kpp_init_on_graph(W, K=1, seed=42)
        
        assert len(centers) == 1
        assert 0 <= centers[0] < 3
    
    def test_deterministic_with_seed(self):
        """Test that same seed gives same results."""
        W = complete_graph(8)
        K = 3
        
        centers1 = _kpp_init_on_graph(W, K, seed=42)
        centers2 = _kpp_init_on_graph(W, K, seed=42)
        
        assert centers1 == centers2
    
    def test_invalid_k_raises_error(self):
        """Test that invalid K values raise assertions."""
        W = triangle_graph()
        
        with pytest.raises(AssertionError):
            _kpp_init_on_graph(W, K=0, seed=42)
        
        with pytest.raises(AssertionError):
            _kpp_init_on_graph(W, K=5, seed=42)  # K > N


class TestFitKmedoidsGraph:
    """Tests for fit_kmedoids_graph function."""
    
    def test_basic_functionality(self):
        """Test basic k-medoids functionality."""
        W = complete_graph(10)
        K = 3
        
        medoids, assign = fit_kmedoids_graph(W, K=K, init="kpp", seed=42)
        
        # Check output shapes and types
        assert medoids.shape == (K,)
        assert assign.shape == (10,)
        assert medoids.dtype == int
        assert assign.dtype == int
        
        # Check medoids are valid indices
        assert all(0 <= m < 10 for m in medoids)
        assert len(set(medoids)) == K  # No duplicate medoids
        
        # Check assignments are valid
        assert all(0 <= a < K for a in assign)
    
    def test_both_initialization_methods(self):
        """Test both random and kpp initialization work."""
        W = complete_graph(15)
        K = 4
        
        medoids_kpp, assign_kpp = fit_kmedoids_graph(W, K=K, init="kpp", seed=42)
        medoids_rand, assign_rand = fit_kmedoids_graph(W, K=K, init="random", seed=42)
        
        # Both should have correct shapes
        assert medoids_kpp.shape == (K,)
        assert medoids_rand.shape == (K,)
        assert assign_kpp.shape == (15,)
        assert assign_rand.shape == (15,)
    
    def test_single_cluster(self):
        """Test with K=1."""
        W = triangle_graph()
        medoids, assign = fit_kmedoids_graph(W, K=1, init="kpp", seed=42)
        
        assert medoids.shape == (1,)
        assert assign.shape == (3,)
        assert 0 <= medoids[0] < 3
        assert all(a == 0 for a in assign)  # All assigned to cluster 0
    
    def test_deterministic_with_seed(self):
        """Test that same seed gives same results."""
        W = complete_graph(12)
        K = 3
        
        medoids1, assign1 = fit_kmedoids_graph(W, K=K, init="kpp", seed=42)
        medoids2, assign2 = fit_kmedoids_graph(W, K=K, init="kpp", seed=42)
        
        np.testing.assert_array_equal(medoids1, medoids2)
        np.testing.assert_array_equal(assign1, assign2)
    
    def test_invalid_inputs_raise_errors(self):
        """Test error handling for invalid inputs."""
        W = triangle_graph()
        
        # Invalid K values
        with pytest.raises(AssertionError):
            fit_kmedoids_graph(W, K=0, init="kpp", seed=42)
        
        with pytest.raises(AssertionError):
            fit_kmedoids_graph(W, K=5, init="kpp", seed=42)  # K > N
        
        # Invalid initialization method
        with pytest.raises(ValueError):
            fit_kmedoids_graph(W, K=2, init="invalid", seed=42)
        
        # Non-sparse matrix
        W_dense = np.array([[0, 1], [1, 0]])
        with pytest.raises(AssertionError):
            fit_kmedoids_graph(W_dense, K=1, init="kpp", seed=42)
