import argparse, torch, numpy as np
from pathlib import Path

def load_codebook(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def assign_euclidean(latents_path, codebook_pt, out_path, bs=4096):
    z_all = torch.load(latents_path, map_location="cpu")
    if isinstance(z_all, dict): z_all = z_all.get("z", z_all)
    z_all = z_all.float().cpu()
    cb = load_codebook(codebook_pt)
    z_med = cb["z_medoid"].float().cpu()
    # compute assignments
    out = torch.empty(len(z_all), dtype=torch.long)
    for i in range(0, len(z_all), bs):
        d = torch.cdist(z_all[i:i+bs], z_med)
        out[i:i+bs] = d.argmin(dim=1)
    np.save(out_path, out.numpy())
    print("Saved assignments to:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=4096)
    args = ap.parse_args()
    assign_euclidean(args.latents, args.codebook, args.out, args.bs)


