#!/usr/bin/env bash
set -euo pipefail

# Verifica GPU e ROCm
/opt/rocm/bin/rocminfo >/dev/null 2>&1 && echo "ROCm OK" || echo "ROCm non trovato (ok se stai usando container con ROCm)."

python3 - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available (ROCm counts as CUDA):', torch.cuda.is_available())
print('torch.version.hip:', getattr(torch.version, 'hip', None))
if torch.cuda.is_available():
    print('Device count:', torch.cuda.device_count())
    print('Device name  :', torch.cuda.get_device_name(0))
PY

# Installazione pacchetti (assumi Python già presente)
# Per ROCm 6.4 su Ubuntu 24, se non hai già torch:
# pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch torchvision
pip install -r requirements.txt