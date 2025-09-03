from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = {
    'PAD': 0,
    'BOS': 1,
    'EOS': 2,
    'CLS': 3,
}


@dataclass
class SequenceDataConfig:
    codes_path: str  # path to torch file with codes [N, T] or list of tensors
    labels_path: Optional[str] = None  # optional class labels [N]
    add_bos: bool = True
    add_eos: bool = True
    add_cls: bool = False
    pad_token_id: int = SPECIAL_TOKENS['PAD']
    bos_token_id: int = SPECIAL_TOKENS['BOS']
    eos_token_id: int = SPECIAL_TOKENS['EOS']
    cls_token_id: int = SPECIAL_TOKENS['CLS']


class SequenceDataset(Dataset):
    def __init__(self, cfg: SequenceDataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        codes_file = Path(cfg.codes_path)
        if not codes_file.exists():
            raise FileNotFoundError(f"Codes file not found: {codes_file}")
        obj = torch.load(codes_file)
        if isinstance(obj, list):
            self.codes = obj
        else:
            self.codes = [row for row in obj]

        if cfg.labels_path is not None and Path(cfg.labels_path).exists():
            labels = torch.load(cfg.labels_path)
            self.labels = labels.tolist() if torch.is_tensor(labels) else labels
        else:
            self.labels = [None] * len(self.codes)

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int):
        seq = self.codes[idx]
        if torch.is_tensor(seq):
            seq = seq.clone().to(dtype=torch.long)
        else:
            seq = torch.tensor(seq, dtype=torch.long)

        pieces: List[torch.Tensor] = []
        if self.cfg.add_cls:
            pieces.append(torch.tensor([self.cfg.cls_token_id], dtype=torch.long))
        if self.cfg.add_bos:
            pieces.append(torch.tensor([self.cfg.bos_token_id], dtype=torch.long))
        pieces.append(seq)
        if self.cfg.add_eos:
            pieces.append(torch.tensor([self.cfg.eos_token_id], dtype=torch.long))
        x = torch.cat(pieces, dim=0)

        y = self.labels[idx]
        return x, (y if y is None else int(y))


def pad_collate_fn(batch, pad_token_id: int = SPECIAL_TOKENS['PAD']):
    seqs, labels = zip(*batch)
    lengths = [s.size(0) for s in seqs]
    max_len = max(lengths)
    out = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
        attn[i, : s.size(0)] = 1
    labels_tensor = None
    if labels[0] is not None:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    return out, attn, labels_tensor

