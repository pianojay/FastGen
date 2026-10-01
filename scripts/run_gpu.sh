#!/usr/bin/env bash
set -euo pipefail
cd /home/user/jw_soflow/FastGen-SiD-DiT
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME="$PWD/artifacts/torch_cache"
export XDG_CACHE_HOME="$PWD/artifacts/cache"
export MPLCONFIGDIR="$PWD/artifacts/matplotlib"
export PYTHONPATH="$PWD"
exec /home/user/miniconda3/envs/fastgen_sid_dit/bin/python -u "$@"
