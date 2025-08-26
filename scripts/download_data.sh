#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/download_data.sh [mnist|fashion|cifar10]
DS=${1:-mnist}
export DS

python3 - <<'PY'
import os
import sys
from torchvision import datasets, transforms

ds = os.environ.get('DS', 'mnist').lower()
root = './data'

if ds == 'mnist':
    datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    print('MNIST downloaded to', root)
elif ds in ('fashion', 'fashionmnist', 'fashion-mnist'):
    datasets.FashionMNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    datasets.FashionMNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    print('FashionMNIST downloaded to', root)
elif ds == 'cifar10':
    datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
    datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    print('CIFAR10 downloaded to', root)
else:
    print('Unknown dataset:', ds)
    sys.exit(1)
PY