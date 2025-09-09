#!/usr/bin/env bash
set -euo pipefail

echo "Installing PyTorch..."

# Watch out for the version of ROCm (mine)
if /opt/rocm/bin/rocminfo >/dev/null 2>&1; then
    echo "Installing PyTorch with ROCm support..."
    pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch torchvision
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "Installing PyTorch with CUDA support..."
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
else
    echo "Installing PyTorch CPU-only version..."
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi

# install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# install project in development mode
echo "Installing project in development mode..."
pip install -e .

# verify installation
python3 - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available (ROCm counts as CUDA):', torch.cuda.is_available())
print('torch.version.hip:', getattr(torch.version, 'hip', None))
if torch.cuda.is_available():
    print('Device count:', torch.cuda.device_count())
    print('Device name  :', torch.cuda.get_device_name(0))
PY