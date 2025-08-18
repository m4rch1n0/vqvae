from __future__ import annotations

from typing import Optional

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility across Python, NumPy and PyTorch.

    This disables cuDNN nondeterministic algorithms for full determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    """Return the appropriate torch.device based on the user argument.

    device_arg: "auto" to select CUDA if available, otherwise CPU; or an
    explicit device string like "cuda", "cuda:0" or "cpu".
    """
    if device_arg == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device_arg
    return torch.device(resolved)


