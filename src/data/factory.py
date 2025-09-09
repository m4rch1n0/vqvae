from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def _get_mnist_loaders(root: str, batch_size: int, num_workers: int = 4,
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


def _get_fashion_mnist_loaders(root: str, batch_size: int, num_workers: int = 4,
                               pin_memory: bool = True, persistent_workers: bool = True,
                               augment: bool = False) -> Tuple[DataLoader, DataLoader]:
    tfms = [transforms.ToTensor()]
    if augment:
        tfms = [transforms.RandomRotation(10)] + tfms
    transform = transforms.Compose(tfms)

    train_ds = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root, train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def _get_cifar10_loaders(root: str, batch_size: int, num_workers: int = 4,
                         pin_memory: bool = True, persistent_workers: bool = True,
                         augment: bool = False) -> Tuple[DataLoader, DataLoader]:
    # CIFAR-10 per-channel normalization
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2430, 0.2610)
    norm = transforms.Normalize(mean=cifar_mean, std=cifar_std)
    tfms_train = [transforms.ToTensor(), norm]
    if augment:
        tfms_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + tfms_train
    transform_train = transforms.Compose(tfms_train)
    transform_test = transforms.Compose([transforms.ToTensor(), norm])

    train_ds = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def get_data_loaders(name: str, root: str, batch_size: int, num_workers: int = 4,
                     pin_memory: bool = True, persistent_workers: bool = True,
                     augment: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Generic dataset factory returning train/val DataLoaders.

    Supported names: "MNIST", "FashionMNIST", "CIFAR10" (case-insensitive).
    Defaults to MNIST if the name is unrecognized.
    """
    if not isinstance(name, str):
        name = str(name)
    key = name.strip().lower()

    if key == "mnist":
        return _get_mnist_loaders(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            augment=augment,
        )
    if key in {"fashionmnist", "fashion-mnist", "fashion_mnist"}:
        return _get_fashion_mnist_loaders(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            augment=augment,
        )
    if key == "cifar10":
        return _get_cifar10_loaders(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            augment=augment,
        )

    # Fallback
    return _get_mnist_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        augment=augment,
    )


