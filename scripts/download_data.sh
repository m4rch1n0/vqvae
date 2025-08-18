#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from torchvision import datasets
from torchvision import transforms

root = './data'
_ = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
_ = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
print('MNIST scaricato in', root)
PY