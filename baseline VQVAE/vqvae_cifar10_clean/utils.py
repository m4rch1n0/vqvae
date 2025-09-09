import os, random, csv
import torch
import numpy as np
from torchvision.utils import save_image

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_img_range(x):
    # x in [-1,1] -> [0,1] clamped
    return (x.clamp(-1,1) + 1) / 2

def save_grid(tensor, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(to_img_range(tensor), path, nrow=nrow)

class CSVLogger:
    def __init__(self, path, header):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        self.file = open(path, "a", newline="")
        self.writer = csv.writer(self.file)
        if write_header:
            self.writer.writerow(header)
            self.file.flush()
    def log(self, row):
        self.writer.writerow(row)
        self.file.flush()
    def close(self):
        self.file.close()
