"""Bounded scratch qualification before allocating a changed model's full run."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch

from scripts.launch import run_gpu_job, used_seconds
from scripts.training_study import process_identity
from soflow_fastgen.common import (
    ROOT, atomic_json, digest, gpu_telemetry, process_tree_memory, seed_all,
    source_provenance, verify_protected,
)
from soflow_fastgen.data import datasets, loader
from soflow_fastgen.paired_evaluation import PairedAdapter, environment_identity, fingerprint
from soflow_fastgen.training import Trainer


def batches(dataset, config):
    epoch = 0
    while True:
        yield from loader(dataset, config, epoch, 0)
        epoch += 1


def probe(protocol, microbatch):
    verify_protected()
    config = json.loads(Path(protocol['training_config']).read_text())
    config['microbatch_size'] = microbatch
    torch.set_num_threads(config['torch_threads'])
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(config['memory_target_bytes'] / capacity)
    seed_all(config['seed'])
    train_set, _, _ = datasets(config)
    stream = batches(train_set, config)
    trainer = Trainer(config)
    assert trainer.step == trainer.epoch == trainer.cursor == 0
    assert not trainer.optimizer.state
    parameters = sum(p.numel() for p in trainer.model.parameters())
    if parameters != protocol['expected_parameters'] or parameters >= 100_000_000:
        raise ValueError('Changed model violates the declared parameter count')
    torch.cuda.reset_peak_memory_stats()
    rows = []
    total = protocol['warmup_updates'] + protocol['measured_updates']
    for index in range(total):
        row = trainer.update(*next(stream))
        row['phase'] = 'warmup' if index < protocol['warmup_updates'] else 'measured'
        if (index + 1) % 10 == 0:
            row.update(process_tree_memory())
            atomic_json(Path(protocol['run_dir'])/'probe_status.json',
                        dict(row, microbatch=microbatch, planned_early_updates=total))
            print(json.dumps({'phase': row['phase'], 'update': index + 1,
                              'seconds': row['seconds'],
                              'allocated_bytes': row['allocated_bytes']}), flush=True)
        rows.append(row)
    # This exercises the terminal loss/gap path, not a trained 40k optimizer state.
    trainer.step = config['total_steps'] - protocol['terminal_updates']
    terminal = []
    for _ in range(protocol['terminal_updates']):
        terminal.append(trainer.update(*next(stream)))
    terminal[-1].update(process_tree_memory())
    measured = rows[protocol['warmup_updates']:]
    all_rows = rows + terminal
    peak = max(row['allocated_bytes'] for row in all_rows)
    result = {
        'initialization': 'scratch', 'initial_step': 0, 'initial_optimizer_entries': 0,
        'checkpoint_saved': False, 'parameters': parameters,
        'microbatch': microbatch, 'effective_batch': config['batch_size'],
        'eligible': peak < config['memory_target_bytes'],
        'mean_update_seconds': statistics.mean(row['seconds'] for row in measured),
        'median_update_seconds': statistics.median(row['seconds'] for row in measured),
        'first_half_mean_seconds': statistics.mean(row['seconds'] for row in measured[:len(measured)//2]),
        'last_half_mean_seconds': statistics.mean(row['seconds'] for row in measured[len(measured)//2:]),
        'terminal_mean_update_seconds': statistics.mean(row['seconds'] for row in terminal),
        'peak_training_allocated_bytes': peak,
        'peak_training_reserved_bytes': max(row['reserved_bytes'] for row in all_rows),
        'peak_host_rss_bytes': max(row['host_peak_rss_bytes'] for row in all_rows),
        'peak_sampled_process_tree_pss_bytes': max(row.get('process_tree_pss_bytes', 0) for row in all_rows),
        'updates': rows, 'terminal_gap_updates': terminal,
        'terminal_limitation': 'Step is advanced only to exercise the declared terminal loss path; weights and optimizer have only the recorded disposable updates.',
        'training_config_sha256': digest(protocol['training_config']),
        'environment': environment_identity(), 'gpu': gpu_telemetry(),
    }
    backbone = trainer.ema
    del trainer, stream
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampling = {}
    with torch.no_grad():
        labels = torch.arange(32, device='cuda', dtype=torch.long)
        for nfe in (1, 4):
            adapter = PairedAdapter(backbone, nfe, [1-i/nfe for i in range(1, nfe)], 5999)
            times = []
            for index in range(protocol['sampling_warmup_batches'] + protocol['sampling_measured_batches']):
                torch.cuda.synchronize()
                started = time.monotonic()
                output = adapter.sample((32, 3, 64, 64), device='cuda', category=labels)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                if index >= protocol['sampling_warmup_batches']:
                    times.append(elapsed)
                del output
            expected = nfe * (protocol['sampling_warmup_batches'] + protocol['sampling_measured_batches'])
            assert adapter.network_calls == expected
            sampling[str(nfe)] = {'batch_size': 32, 'measured_batches': len(times),
                                 'mean_batch_seconds': statistics.mean(times),
                                 'max_batch_seconds': max(times), 'network_calls': expected,
                                 'seconds_for_95_full_batches_estimate': 95 * statistics.mean(times),
                                 'limitation': 'Disposable weights; excludes PNG writing and FID; no quality score.'}
    result['sampling_timing'] = sampling
    result['peak_sampling_allocated_bytes'] = torch.cuda.max_memory_allocated()
    return result


def supervise(path, protocol):
    run = Path(protocol['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    with (run/'workflow.lock').open('a') as own_lock, (ROOT/'runs/gpu_1.lock').open('a') as gpu_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run/'record.json').exists() or used_seconds(run):
            raise ValueError('Existing qualification requires accounting reconciliation; no automatic allowance renewal')
        verify_protected()
        config = json.loads(Path(protocol['training_config']).read_text())
        if config['batch_size'] != 128 or digest(config['partition']) != protocol['partition_sha256']:
            raise ValueError('Effective batch or data partition changed')
        active = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError(f'GPU 1 has an active compute process: {active}')
        provenance = source_provenance()
        record = {'state': 'running', 'protocol': protocol, 'protocol_sha256': fingerprint(protocol),
                  'source': provenance, 'environment': environment_identity(),
                  'worker': process_identity(), 'attempts': [], 'gpu_job_seconds': 0.,
                  'authorization_reference': protocol['authorization_reference']}
        atomic_json(run/'record.json', record)
        stopped = False

        def stop_worker(signum, frame):
            nonlocal stopped
            stopped = True
            active_path = run/'active_job.json'
            if active_path.exists():
                child = json.loads(active_path.read_text())
                pid = child.get('pid')
                if pid and Path(f'/proc/{pid}/stat').exists():
                    ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                    if ticks == child['proc_start_ticks']:
                        os.killpg(pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop_worker)
        signal.signal(signal.SIGINT, stop_worker)

        def job(stage, command):
            if stopped:
                raise RuntimeError('Qualification stopped by signal')
            if source_provenance()['source_sha256'] != provenance['source_sha256']:
                raise ValueError('Qualification source/configuration changed')
            attempt = {'stage': stage, 'command': command, 'started_unix': time.time(),
                       'used_seconds_before': used_seconds(run)}
            record['active_stage'] = stage
            record['attempts'].append(attempt)
            atomic_json(run/'record.json', record)
            with (run/f'{stage}.log').open('w') as output:
                result = run_gpu_job(command, {'budget_seconds': protocol['budget_seconds']-30},
                                     run, stdout=output, stderr=subprocess.STDOUT)
            attempt.update(returncode=result, ended_unix=time.time(), used_seconds_after=used_seconds(run))
            record['gpu_job_seconds'] = used_seconds(run)
            atomic_json(run/'record.json', record)
            if result != 0 or stopped:
                raise RuntimeError(f'Qualification stage {stage} failed: {run/stage}.log')

        try:
            job('contracts', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
            for microbatch in protocol['microbatch_candidates']:
                job(f'microbatch_{microbatch}', [sys.executable, '-m', 'scripts.qualify_training',
                    '--config', path, '--probe', str(microbatch)])
                result = json.loads((run/f'microbatch_{microbatch}.json').read_text())
                if result['eligible']:
                    atomic_json(run/'selected.json', result)
                    record.update(state='qualified', selected_microbatch=microbatch,
                                  completed_unix=time.time(), active_stage=None)
                    atomic_json(run/'record.json', record)
                    print(json.dumps({'state': record['state'], 'microbatch': microbatch,
                                      'mean_update_seconds': result['mean_update_seconds'],
                                      'peak_training_allocated_bytes': result['peak_training_allocated_bytes'],
                                      'gpu_job_seconds': used_seconds(run)}), flush=True)
                    return
            raise RuntimeError('No declared microbatch satisfies the qualification memory target')
        except BaseException as error:
            record.update(state='stopped' if stopped else 'failed', error=str(error),
                          gpu_job_seconds=used_seconds(run))
            atomic_json(run/'record.json', record)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe', type=int)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('Use physical GPU 1 with PYTHONNOUSERSITE=1')
    protocol = json.loads(Path(args.config).read_text())
    if args.probe is not None:
        if args.probe not in protocol['microbatch_candidates']:
            raise ValueError('Undeclared microbatch probe')
        try:
            result = probe(protocol, args.probe)
        except torch.cuda.OutOfMemoryError as error:
            result = {'eligible': False, 'microbatch': args.probe, 'error': str(error)}
        atomic_json(Path(protocol['run_dir'])/f'microbatch_{args.probe}.json', result)
    else:
        supervise(args.config, protocol)
