#!/usr/bin/env bash
set -euo pipefail
TRANSFER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# First-wave work is CPU only. Future GPU contracts must set and verify 1 or 2.
export CUDA_VISIBLE_DEVICES="${TRANSFER_AUG_GPU:-}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR="$TRANSFER_ROOT/transfer_aug/.cache/matplotlib"
export XDG_CACHE_HOME="$TRANSFER_ROOT/transfer_aug/.cache"
export TORCH_HOME="$TRANSFER_ROOT/transfer_aug/.cache/torch"
exec "$TRANSFER_ROOT/.conda/eda/bin/python" "$@"
