import numpy as np
from src.geo.knn_graph import build_knn_graph, largest_connected_component
from src.geo.geo_shortest_paths import dijkstra_multi_source

def test_knn_geo_end_to_end():
    rs = np.random.RandomState(0)
    z = rs.randn(100, 8).astype(np.float32)
    W, _ = build_knn_graph(z, k=5, mode="distance", sym="mutual")
    lcc = largest_connected_component(W)
    W_lcc = W[lcc][:, lcc]
    D = dijkstra_multi_source(W_lcc, sources=[0, W_lcc.shape[0]//2])
    # all finite in LCC
    assert np.isfinite(D).all()
