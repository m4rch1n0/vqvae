from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(root: str, batch_size: int, num_workers: int = 4,
                      pin_memory: bool = True, persistent_workers: bool = True,
                      augment: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Return train/val DataLoaders for MNIST.

    If augment is True, apply a light RandomRotation.
    """
    tfms = [transforms.ToTensor()]
    if augment:
        tfms = [transforms.RandomRotation(10)] + tfms
    transform = transforms.Compose(tfms)

    train_ds = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent_workers)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=persistent_workers)
    return train_loader, val_loader