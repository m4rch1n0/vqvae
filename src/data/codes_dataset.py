from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CodesDataset(Dataset):
    """A dataset for loading sequence-- Bormalized quantized codes."""

    def __init__(self, codes_path: str, labels_path: Optional[str] = None):
        codes = np.load(codes_path)
        labels = torch.load(labels_path) if labels_path else None
        
        # Filter out sequences containing -1
        valid_mask = ~(codes == -1).any(axis=(1, 2))
        self.codes = codes[valid_mask]
        if labels is not None:
            self.labels = labels[valid_mask]
        else:
            self.labels = None

        # Flatten the spatial codes into sequences
        N, H, W = self.codes.shape
        self.codes = self.codes.reshape(N, H * W)
        
        self.seq_len = H * W

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Input sequence (all but last token)
        x = torch.from_numpy(self.codes[idx, :-1]).long()
        
        # Target sequence (all but first token)
        y = torch.from_numpy(self.codes[idx, 1:]).long()
        
        if self.labels is not None:
            return x, y, self.labels[idx]
        else:
            return x, y


class VanillaCodesDataset(Dataset):
    """
    A dataset for the legacy Vanilla VAE.

    It loads a single code per image and prepends a BOS token.
    The task is to predict the code given the BOS token and a class label.
    Sequence format: [BOS, code]
    """
    def __init__(self, codes_path: str, labels_path: Optional[str] = None, num_tokens: int = 512):
        codes = np.load(codes_path) # Shape: (N,)
        labels = torch.load(labels_path) if labels_path else None
        
        # The BOS token is the last valid index in the embedding table
        self.bos_token = num_tokens - 1

        # Filter out invalid codes (-1)
        valid_mask = (codes != -1)
        self.codes = codes[valid_mask]
        if labels is not None:
            self.labels = labels[valid_mask]
        else:
            self.labels = None
        
        self.seq_len = 2  # [BOS, code]

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Input sequence is just the BOS token
        x = torch.tensor([self.bos_token]).long()
        
        # Target sequence is just the code
        y = torch.tensor([self.codes[idx]]).long()
        
        if self.labels is not None:
            return x, y, self.labels[idx]
        else:
            return x, y
