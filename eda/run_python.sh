#!/usr/bin/env bash
set -euo pipefail
EDA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR="$EDA_ROOT/.cache/matplotlib"
export TORCH_HOME="$EDA_ROOT/.cache/torch"
export XDG_CACHE_HOME="$EDA_ROOT/.cache"
exec "$EDA_ROOT/.conda/eda/bin/python" "$@"
