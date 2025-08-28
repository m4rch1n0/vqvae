import argparse, numpy as np, torch
from scipy.sparse.csgraph import connected_components
from src.geo.knn_graph_optimized import build_knn_graph
from src.geo.geo_shortest_paths import dijkstra_single_source

def _tload(path):
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")

def _load_latents(path):
    z = torch.load(path, map_location="cpu")
    return z.get("z", z).float().cpu().numpy() if isinstance(z, dict) else z.float().cpu().numpy()

def _map_medoids_to_train_indices(z_train_np, z_medoid_np, batch=8192):
    # Mappa ogni medoid al NN in z_train per evitare mismatch di indici
    zt = torch.from_numpy(z_train_np)
    zm = torch.from_numpy(z_medoid_np)
    out = []
    for i in range(0, len(zm), batch):
        d = torch.cdist(zm[i:i+batch], zt)      # (b, Ntrain)
        out.append(d.argmin(dim=1).cpu())
    return torch.cat(out, 0).numpy()            # (K,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_train", required=True)
    ap.add_argument("--latents_val", required=True)
    ap.add_argument("--codebook", required=True)          # must contain 'z_medoid'
    ap.add_argument("--out_codes_val", required=True)
    ap.add_argument("--k", type=int, default=20)          # ↑ default 20
    ap.add_argument("--sym", type=str, default="union")   # union > mutual per connettività
    args = ap.parse_args()

    # 1) Load
    z_train = _load_latents(args.latents_train)   # (Ntr,D)
    z_val   = _load_latents(args.latents_val)     # (Nva,D)
    z_all   = np.concatenate([z_train, z_val], 0) # (Ntr+Nva,D)
    Ntr, Nva = len(z_train), len(z_val)

    cb = _tload(args.codebook)
    if "z_medoid" not in cb:
        raise KeyError("codebook.pt must contain key 'z_medoid' (K,D).")
    z_medoid = cb["z_medoid"].float().cpu().numpy()  # (K,D)

    # 2) Map medoids to train indices (avoid relying on medoid_indices)
    src_train_idx = _map_medoids_to_train_indices(z_train, z_medoid)  # (K,)
    # de-duplicate (più medoid potrebbero mappare allo stesso train idx)
    uniq_src, inv = np.unique(src_train_idx, return_inverse=True)
    print(f"[info] sources in train: {len(uniq_src)} unique nodes (from K={len(z_medoid)})")

    # 3) Build kNN graph with distance weights, union symmetry for connectivity
    W, _ = build_knn_graph(z_all, k=args.k, sym=args.sym, metric="euclidean", mode="distance")
    print(f"[info] joint graph: {W.shape[0]} nodes, {W.nnz//2} undirected edges (approx)")

    # 4) Connected components
    n_comp, labels = connected_components(W, directed=False)
    print(f"[info] connected components: {n_comp}")

    # Component coverage: per componente, c'è almeno una sorgente?
    comp_has_src = np.zeros(n_comp, dtype=bool)
    comp_ids_src = labels[uniq_src]             # sorgenti sono in blocco train (0..Ntr-1)
    comp_has_src[np.unique(comp_ids_src)] = True

    # 5) Geodesic assignment: per ogni sorgente unica, run Dijkstra singola sorgente
    INF = np.inf
    D_val_all = np.full((len(uniq_src), Nva), INF, dtype=np.float32)
    for i, s in enumerate(uniq_src.tolist()):
        d = dijkstra_single_source(W, source=s)     # (Ntr+Nva,)
        D_val_all[i] = d[Ntr:]                      # solo nodi val

    # 6) Per nodi val in componenti senza sorgenti -> fallback euclideo
    comp_ids_val = labels[Ntr:]                     # (Nva,)
    mask_no_src = ~comp_has_src[comp_ids_val]       # (Nva,)
    d_euc = None
    if mask_no_src.any():
        print(f"[warn] {mask_no_src.sum()} val nodes in components with NO source. Using EUCLIDEAN fallback for them.")
        d_euc = torch.cdist(torch.from_numpy(z_val[mask_no_src]),
                            torch.from_numpy(z_medoid)).argmin(dim=1).cpu().numpy()

    # 7) Geodesic argmin per i nodi coperti
    geo_argmin = D_val_all.argmin(axis=0)           # (Nva,) indice nella lista uniq_src

    # 8) Costruisci vettore finale dei codici (0..K-1) allineati a z_medoid
    uniq_to_medoid = np.array([np.where(inv==j)[0][0] for j in range(len(uniq_src))], dtype=np.int64)
    codes_val = uniq_to_medoid[geo_argmin]

    # 9) Inserisci fallback euclideo dove non c'è copertura
    if mask_no_src.any():
        codes_val[mask_no_src] = d_euc.astype(np.int64)

    np.save(args.out_codes_val, codes_val.astype(np.int32))
    print(f"[done] Saved geodesic assignments for {Nva} val samples -> {args.out_codes_val}")

if __name__ == "__main__":
    main()
