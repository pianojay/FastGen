"""Measure actual CIFAR training updates, retaining the 800,000-step schedule."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('sizes', type=int, nargs='+')
args = parser.parse_args()
os.chdir(ROOT)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONNOUSERSITE='1',
           PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
results_path = ROOT / 'runs/cifar10_throughput/results.json'
results = json.loads(results_path.read_text()) if results_path.exists() else []
for size in args.sizes:
    output = ROOT / f'runs/cifar10_throughput/b{size}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'ckpts').exists():
        raise SystemExit(f'{output} already contains a run; use a fresh directory for a comparable measurement')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1',
               'train.py', '--config', f'runs/cifar10_throughput/b{size}.yaml', '--run-steps', '5']
    print(f'Measuring microbatch {size}: 2 warmup + 3 timed optimizer updates', flush=True)
    peak_mib = 0
    with (output / 'console.log').open('w') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            memory = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True)
            if memory.returncode == 0:
                peak_mib = max(peak_mib, int(memory.stdout.strip()))
            time.sleep(1)
    log = (output / 'console.log').read_text()
    times = [float(x) for x in re.findall(r'Time = ([0-9.]+) seconds', log)]
    peaks = [float(x) for x in re.findall(r'Mem = ([0-9.]+) GB', log)]
    result = {'microbatch': size, 'effective_batch': 1024, 'exit_code': process.returncode,
              'update_seconds': times, 'peak_device_mib': peak_mib,
              'peak_allocated_gib': max(peaks, default=None), 'command': command}
    if process.returncode == 0 and len(times) == 5:
        median = statistics.median(times[2:])
        result.update(median_seconds=median, images_per_second=1024/median,
                      projected_training_days=800000*median/86400)
    else:
        result['error_tail'] = log[-3000:]
    results.append(result)
    results_path.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if process.returncode != 0:
        break
