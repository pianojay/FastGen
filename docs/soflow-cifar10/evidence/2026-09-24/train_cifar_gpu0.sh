#!/usr/bin/env bash
set -euo pipefail
cd /home/user/jw_soflow/SoFlow
run_dir=runs/cifar10_gpu0
mkdir -p "$run_dir"
exec 9>"$run_dir/launch.lock"
flock -n 9 || exit 1
echo $$ > "$run_dir/pid"
printf "running\n" > "$run_dir/exit_status"
date -Is
trap 'code=$?; echo "$code" > "$run_dir/exit_status"' EXIT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
python_bin=/home/user/miniconda3/envs/soflow/bin/python
"$python_bin" -c 'from importlib.metadata import distributions; print("\n".join(sorted({d.metadata["Name"] + "==" + d.version for d in distributions() if d.metadata["Name"]})))'  > "$run_dir/environment.txt"
"$python_bin" -c 'import sys, torch, torchvision; print(sys.executable); print(torch.__version__, torch.__file__); print(torchvision.__version__, torchvision.__file__)' > "$run_dir/interpreter.txt"
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node=1 train.py --config cifar_gpu0.yaml
