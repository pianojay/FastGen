"""CPU-only evidence retention for the nine fixed runs; no model execution.

The trainer retains best/latest states. This watcher additionally preserves each
predeclared evaluation's EMA weights before latest can advance. It never changes
training, selection, RNG, checkpoints used by the trainer, or source identity.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from flow_sid.common import ROOT, atomic_json, digest


def archive():
    config = json.loads((ROOT / 'configs/campaign.json').read_text())
    changes = []
    for run in ['T', 'S1', 'A1', 'S2', 'A2', 'S3', 'A3', 'S4', 'A4']:
        folder = ROOT / 'runs/campaign' / run
        status_file = folder / 'status.json'
        if not status_file.exists():
            continue
        status = json.loads(status_file.read_text())
        step = status['step']
        settings = config['teacher'] if run == 'T' else config['student']
        target = ROOT / 'artifacts/evaluated_weights' / run
        target.mkdir(parents=True, exist_ok=True)
        path = target / f'{step:06d}.pt'
        if step in settings['evaluate_at'] and status['phase'] in ['evaluation', 'complete', 'confirmation'] and not path.exists():
            source_path = folder / 'latest.pt'
            state = torch.load(source_path, map_location='cpu', weights_only=False)
            if state['step'] != step:
                del state
                continue
            hasher = hashlib.sha256()
            for value in state['ema'].values():
                hasher.update(value.contiguous().numpy().tobytes())
            payload = {'weights': state['ema'], 'step': step, 'run': run,
                       'teacher_sha256': state['teacher_sha256'], 'source': state['identity'],
                       'config_sha256': state['config_sha256'], 'partition_sha256': state['partition_sha256']}
            torch.save(payload, path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
            atomic_json(path.with_suffix('.json'), {'path': str(path.relative_to(ROOT)),
                        'sha256': digest(path), 'ema_tensor_sha256': hasher.hexdigest(),
                        'step': step, 'run': run, 'saved_at': time.time(),
                        'method': 'CPU extraction of EMA from complete-update recovery checkpoint'})
            changes.append({'run': run, 'step': step, 'ema_tensor_sha256': hasher.hexdigest()})
            del payload, state
        for step in settings['evaluate_at']:
            meta = target / f'{step:06d}.json'
            metrics = folder / f'eval_{step:06d}' / 'metrics.json'
            if meta.exists() and metrics.exists():
                value = json.loads(meta.read_text())
                if 'evaluation' not in value:
                    value['evaluation'] = json.loads(metrics.read_text())
                    atomic_json(meta, value)
    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'This evidence utility is CPU-only'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    deadline = json.loads((ROOT / 'runs/campaign/allowance.json').read_text())['started_at'] + 547200
    while True:
        for event in archive():
            print(json.dumps(event), flush=True)
        final = ROOT / 'runs/campaign/A4/status.json'
        if not args.watch or time.time() >= deadline or (final.exists() and json.loads(final.read_text())['phase'] == 'complete'):
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
