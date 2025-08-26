import argparse, torch, numpy as np
from pathlib import Path

def _tload(path, ml="cpu"):
    try: return torch.load(path, map_location=ml, weights_only=False)
    except TypeError: return torch.load(path, map_location=ml)

def knnprop_assign(z_train, codes_train, z_val, k=10, bs=4096):
    # Assegna ad ogni z_val il medoid più votato dai suoi k NN in z_train (euclideo per trovare i vicini).
    zt = z_train.float().cpu()
    zv = z_val.float().cpu()
    ct = torch.as_tensor(codes_train, dtype=torch.long)
    out = torch.empty(len(zv), dtype=torch.long)
    for i in range(0, len(zv), bs):
        q = zv[i:i+bs]                            # (b,D)
        # cdist (b, Ntrain): fai in chunk per memoria se serve
        # pick top-k NN indici
        d = torch.cdist(q, zt)                    # (b, Ntrain)
        nn_idx = d.topk(k, largest=False).indices # (b,k)
        votes = ct[nn_idx]                        # (b,k)
        # moda per riga; se tie, prendi il primo NN
        for r in range(votes.size(0)):
            vals, counts = votes[r].unique(return_counts=True)
            mode = vals[counts.argmax()]
            out[i+r] = mode
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_train", required=True)     # .../latents_train/z.pt
    ap.add_argument("--latents_val",   required=True)     # .../latents_val/z.pt
    ap.add_argument("--codes_train",   required=True)     # geodesic assignments for train (codes_train.npy)
    ap.add_argument("--out_codes_val", required=True)     # out: codes_val_knnprop.npy
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    ztr = torch.load(args.latents_train, map_location="cpu")
    if isinstance(ztr, dict): ztr = ztr.get("z", ztr)
    zva = torch.load(args.latents_val, map_location="cpu")
    if isinstance(zva, dict): zva = zva.get("z", zva)

    ctr = np.load(args.codes_train)
    cva = knnprop_assign(ztr, ctr, zva, k=args.k)
    np.save(args.out_codes_val, cva.numpy())
    print("Saved:", args.out_codes_val)


