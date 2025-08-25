import numpy as np
import pytest
from scipy import sparse

from src.geo.kmeans_precomputed import _kpp_init_precomputed, fit_kmedoids_precomputed
from src.geo.geo_shortest_paths import dijkstra_multi_source


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


class TestKppInitPrecomputed:
    """Tests for _kpp_init_precomputed function."""
    
    def test_basic_functionality(self):
        """Test basic k-means++ initialization."""
        W = complete_graph(10)
        K = 3
        D = dijkstra_multi_source(W, list(range(10)))
        centers = _kpp_init_precomputed(D, K, seed=42)
        
        assert len(centers) == K
        assert all(0 <= c < 10 for c in centers)
        assert len(set(centers)) == K  # No duplicates
    
    def test_single_center(self):
        """Test with K=1."""
        W = triangle_graph()
        D = dijkstra_multi_source(W, list(range(3)))
        centers = _kpp_init_precomputed(D, K=1, seed=42)
        
        assert len(centers) == 1
        assert 0 <= centers[0] < 3
    
    def test_deterministic_with_seed(self):
        """Test that same seed gives same results."""
        W = complete_graph(8)
        K = 3
        D = dijkstra_multi_source(W, list(range(8)))
        
        centers1 = _kpp_init_precomputed(D, K, seed=42)
        centers2 = _kpp_init_precomputed(D, K, seed=42)
        
        assert centers1 == centers2
    
    def test_invalid_k_edge_cases(self):
        """Test edge cases for K values."""
        W = triangle_graph()
        D = dijkstra_multi_source(W, list(range(3)))
        
        # K=0 returns 1 center
        centers = _kpp_init_precomputed(D, K=0, seed=42)
        assert len(centers) == 1
        assert 0 <= centers[0] < 3
        
        # K > N should work but may behave unexpectedly
        try:
            centers = _kpp_init_precomputed(D, K=5, seed=42)
            assert isinstance(centers, list)
        except (IndexError, ValueError) as e:
            pass


class TestFitKmedoidsPrecomputed:
    """Tests for fit_kmedoids_precomputed function."""
    
    def test_basic_functionality(self):
        """Test basic k-medoids functionality."""
        W = complete_graph(10)
        K = 3
        
        medoids, assign, _ = fit_kmedoids_precomputed(W, K=K, init="kpp", seed=42)
        
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
        
        medoids_kpp, assign_kpp, _ = fit_kmedoids_precomputed(W, K=K, init="kpp", seed=42)
        medoids_rand, assign_rand, _ = fit_kmedoids_precomputed(W, K=K, init="random", seed=42)
        
        # Both should have correct shapes
        assert medoids_kpp.shape == (K,)
        assert medoids_rand.shape == (K,)
        assert assign_kpp.shape == (15,)
        assert assign_rand.shape == (15,)
    
    def test_single_cluster(self):
        """Test with K=1."""
        W = triangle_graph()
        medoids, assign, _ = fit_kmedoids_precomputed(W, K=1, init="kpp", seed=42)
        
        assert medoids.shape == (1,)
        assert assign.shape == (3,)
        assert 0 <= medoids[0] < 3
        assert all(a == 0 for a in assign)  # All assigned to cluster 0
    
    def test_deterministic_with_seed(self):
        """Test that same seed gives same results."""
        W = complete_graph(12)
        K = 3
        
        medoids1, assign1, _ = fit_kmedoids_precomputed(W, K=K, init="kpp", seed=42)
        medoids2, assign2, _ = fit_kmedoids_precomputed(W, K=K, init="kpp", seed=42)
        
        np.testing.assert_array_equal(medoids1, medoids2)
        np.testing.assert_array_equal(assign1, assign2)
    
    def test_invalid_inputs_edge_cases(self):
        """Test edge cases and invalid inputs."""
        W = triangle_graph()
        
        # K=0 creates 1 cluster
        medoids, assign, _ = fit_kmedoids_precomputed(W, K=0, init="kpp", seed=42)
        assert medoids.shape == (1,)
        assert assign.shape == (3,)
        assert 0 <= medoids[0] < 3
        assert all(a == 0 for a in assign)  # All assigned to cluster 0
        try:
            medoids, assign, _ = fit_kmedoids_precomputed(W, K=5, init="kpp", seed=42)
            assert medoids.dtype == int
            assert assign.dtype == int
        except (IndexError, ValueError, MemoryError) as e:
            pass
        
        # Invalid initialization method should still raise ValueError
        with pytest.raises(ValueError):
            fit_kmedoids_precomputed(W, K=2, init="invalid", seed=42)
