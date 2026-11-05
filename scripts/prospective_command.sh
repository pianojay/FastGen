#!/usr/bin/env bash
# Template for one separately authorized fresh campaign.
set -euo pipefail
cd /home/jw/FastGen-master/workers/FastGen/soflow-a100
py=/home/jw/.local/share/fastgen-soflow-a100/venv/bin/python
config=${1:-configs/soflow_a100_s3_60k.json}
read -r run steps budget eval_reserve < <("$py" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["run_dir"],c["total_steps"],c["budget_seconds"],c["final_eval_reserve_seconds"])' "$config")
cutoff=$(date -ud '2026-11-07 06:59:00Z' +%s)
[[ $((cutoff - $(date +%s))) -ge $((budget + 60)) ]]
export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
exec 9>runs/gpu_0.lock
flock -n 9
exec 8>/home/jw/.local/share/fastgen-soflow-a100/gpu-5b21247d.lock
flock -n 8
[[ $(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader) == GPU-5b21247d-a82b-11f1-bf7c-810566ab0ea6 ]]
[[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
[[ ! -e "$run" && -z "$(git status --porcelain)" ]]
timeout --kill-after=30s "$((budget - eval_reserve))s" "$py" -u train.py --config "$config" --skip-eval
"$py" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["reason"] == "schedule_complete" and s["step"] == int(sys.argv[2])' "$run/status.json" "$steps"
printf -v step '%06d' "$steps"
checkpoint="$run/checkpoints/step_$step.pt"
for nfe in 1 4; do
    timeout --kill-after=15s 22m "$py" -m scripts.evaluate_s3_a100 --config "$config" \
        --checkpoint "$checkpoint" --run-dir "$run/paired_dev" --nfe "$nfe" \
        --seeds 4101 4102 4103 4104
done
timeout --kill-after=15s 6m "$py" -m scripts.export --checkpoint "$checkpoint" --output "$run/export" --few-nfe 4
timeout --kill-after=15s 6m "$py" -m scripts.verify_export --package "$run/export"
