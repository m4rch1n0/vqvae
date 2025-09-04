"""Get data loaders for datasets."""

from typing import Tuple, Optional
import logging

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .codes_dataset import CodesDataset

_DATA_ROOT = "data/"
_VALID_NAMES = {"MNIST", "FashionMNIST", "CIFAR10"}

"""
Data loading and preprocessing utilities.
"""

from .mnist import get_mnist_loaders
from .factory import get_data_loaders

__all__ = ["get_mnist_loaders", "get_data_loaders"]

def get_code_loaders(
    codes_path: str,
    labels_path: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Creates train/val DataLoaders for quantized codes."""
    
    # For now, we'll use the same data for train and val as an example.
    # A proper train/val split should be done on the codes.
    dataset = CodesDataset(codes_path=codes_path, labels_path=labels_path)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shuffle=True,
    )
    
    # Placeholder for a validation loader
    val_loader = DataLoader(
        dataset, # WARNING: Using same data for validation
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shuffle=False,
    )

    return loader, val_loader
