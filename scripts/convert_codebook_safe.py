#!/usr/bin/env python
import argparse, torch
import numpy as np

def load_any(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    args = ap.parse_args()

    cb = load_any(args.in_pt)
    z = cb.get("z_medoid", None)
    if z is None:
        raise KeyError("z_medoid non trovato nel codebook.")
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
    elif not isinstance(z, torch.Tensor):
        z = torch.tensor(z)

    safe = {"z_medoid": z.float()}  # solo tensori
    torch.save(safe, args.out_pt)
    print(f"Salvato codebook 'safe' in: {args.out_pt}")

if __name__ == "__main__":
    main()


