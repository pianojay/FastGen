"""Measure the real S3 update loop with disposable weights and no checkpoints."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from soflow_fastgen.common import atomic_json, digest, gpu_telemetry, seed_all, source_provenance
from soflow_fastgen.data import datasets, loader
from soflow_fastgen.paired_evaluation import environment_identity
from soflow_fastgen.training import Trainer


def profile(args):
    config = json.loads(Path(args.config).read_text())
    config['microbatch_size'] = args.microbatch
    if (config['batch_size'] != 128 or not config['augmentation']
            or config['batch_size'] % args.microbatch):
        raise ValueError('Expected augmented effective batch 128 with equal microbatches')
    torch.set_num_threads(config['torch_threads'])
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., config['memory_target_bytes'] / capacity))
    seed_all(config['seed'])
    train_set, _, _ = datasets(config)
    trainer = Trainer(config)
    parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    if parameters != 89_140_419 or trainer.optimizer.state:
        raise ValueError('Expected the S3 architecture with a fresh optimizer')

    result = {
        'state': 'running', 'pid': os.getpid(), 'source': source_provenance(),
        'config_sha256': digest(args.config), 'partition_sha256': digest(config['partition']),
        'parameters': parameters, 'microbatch': args.microbatch,
        'effective_batch': config['batch_size'], 'num_workers': config['num_workers'],
        'environment': environment_identity(), 'cuda_runtime': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'initialization': 'scratch', 'data': 'real S3 Pokémon optimization partition',
        'gpu_at_start': gpu_telemetry(), 'updates': [],
    }
    stream = iter(())
    epoch = 0
    torch.cuda.reset_peak_memory_stats()
    try:
        for phase, count in (('warmup', args.warmup), ('measured', args.measured),
                             ('terminal', args.terminal)):
            if phase == 'terminal':
                # Advance only the curriculum. Weights and optimizer are disposable.
                trainer.step = config['total_steps'] - count
            for index in range(count):
                started = time.monotonic()
                batch = next(stream, None)
                if batch is None:
                    stream = iter(loader(train_set, config, epoch, 0))
                    epoch += 1
                    batch = next(stream)
                loaded = time.monotonic()
                row = trainer.update(*batch)
                row.update(phase=phase, epoch=epoch - 1, loader_seconds=loaded - started,
                           loop_seconds=time.monotonic() - started)
                result['updates'].append(row)
                if (index + 1) % 10 == 0 or index + 1 == count:
                    print(json.dumps(row, allow_nan=False), flush=True)
                    atomic_json(args.output, result)
    except torch.cuda.OutOfMemoryError as error:
        result.update(state='oom_at_18gb_allocator_target', error=str(error))
    else:
        result['state'] = 'complete'
        result['timing'] = {}
        for phase in ('measured', 'terminal'):
            rows = [row for row in result['updates'] if row['phase'] == phase]
            result['timing'][phase] = {
                'updates': len(rows),
                'mean_update_seconds': statistics.mean(row['seconds'] for row in rows),
                'median_update_seconds': statistics.median(row['seconds'] for row in rows),
                'mean_loop_seconds': statistics.mean(row['loop_seconds'] for row in rows),
                'median_loop_seconds': statistics.median(row['loop_seconds'] for row in rows),
                'max_loop_seconds': max(row['loop_seconds'] for row in rows),
            }
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    result['eligible'] = (result['state'] == 'complete'
                          and result['peak_allocated_bytes'] < config['memory_target_bytes'])
    result['gpu_at_end'] = gpu_telemetry()
    atomic_json(args.output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research_s3_profile_train.json')
    parser.add_argument('--microbatch', type=int, choices=(16, 32, 64), required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--measured', type=int, default=7)
    parser.add_argument('--terminal', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.measured, args.terminal) <= 0:
        parser.error('All three phases require at least one update')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('PYTHONNOUSERSITE') != '1':
        parser.error('Set CUDA_VISIBLE_DEVICES=0 and PYTHONNOUSERSITE=1')
    profile(args)
