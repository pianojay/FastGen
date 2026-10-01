"""Disposable engineering check, not a quality trial or training restart."""
import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from afm_fastgen.common import (ROOT, GpuLease, append_json, atomic_json, configure,
                               memory, process_identity, verify_protected)
from afm_fastgen.training import Trainer


def equal_state(a, b, path=''):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError(f'Recovery tensor mismatch {path}: {float((a.cpu()-b.cpu()).abs().max())}')
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal_state(a[key], b[key], path+'/'+str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            equal_state(x, y, path+'/'+str(i))
    else:
        import numpy as np
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), path
        else:
            assert a == b, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/qualification.json')
    parser.add_argument('--out', default='artifacts/qualification')
    parser.add_argument('--cycles', type=int, default=12)
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if args.microbatch:
        c['microbatch'] = args.microbatch
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    configure(c)
    with GpuLease():
        start = time.monotonic()
        atomic_json(out/'identity.json', {'process': process_identity(), 'config': c, 'protected': verify_protected()})
        # Separate process is necessary: same-process flock semantics can hide bugs.
        p = subprocess.run([sys.executable, '-c', 'from afm_fastgen.common import GpuLease; GpuLease().__enter__()'],
                           capture_output=True, text=True)
        assert p.returncode and 'Duplicate GPU 0 job refused' in p.stderr
        trainer = Trainer(c)
        torch.cuda.reset_peak_memory_stats()
        sample = torch.randn(2, 3, 64, 64, device='cuda')
        with torch.no_grad():
            a = trainer.g(sample, torch.tensor([0, 1], device='cuda'))
            b = trainer.g(sample, torch.tensor([1, 0], device='cuda'))
            assert float((a-b).abs().max()) > 0
        trainer.export(out/'untrained.pt')
        records = []
        for _ in range(args.cycles):
            row = trainer.cycle(); append_json(out/'metrics.jsonl', row); records.append(row)
            print(json.dumps(row), flush=True)
        mem = memory()
        t_save = time.monotonic(); trainer.save(out/'resume.pt'); checkpoint_seconds = time.monotonic()-t_save
        expected = trainer.cycle()
        trainer.save(out/'expected_next.pt')
        del trainer; gc.collect(); torch.cuda.empty_cache()
        recovered = Trainer(c, resume=out/'resume.pt')
        actual = recovered.cycle()
        for key in expected:
            if 'seconds' not in key:
                assert expected[key] == actual[key], (key, expected[key], actual[key])
        expected_state = torch.load(out/'expected_next.pt', map_location='cpu', weights_only=False)
        actual_state = recovered.state_dict()
        expected_state.pop('optimizer_seconds'); actual_state.pop('optimizer_seconds')
        equal_state(expected_state, actual_state)
        params = recovered.parameters
        del recovered, expected_state, actual_state; gc.collect(); torch.cuda.empty_cache()
        report = {'config': c, 'parameters': params, 'warm_cycle_seconds': [r['cycle_seconds'] for r in records[2:]],
                  'mean_warm_cycle_seconds': sum(r['cycle_seconds'] for r in records[2:])/len(records[2:]),
                  'checkpoint_seconds': checkpoint_seconds, 'memory': mem,
                  'exact_next_cycle_recovery': True, 'duplicate_job_refused': True,
                  'hard_crash_recovery_tested': False, 'disposable_cycles_executed': args.cycles+2,
                  'g_updates_executed': args.cycles+2, 'd_updates_executed': args.cycles+2,
                  'real_presentations_executed': (args.cycles+2)*c['batch_size']*2,
                  'seconds_before_eval': time.monotonic()-start}
        atomic_json(out/'qualification.json', report)
        if args.smoke:
            from afm_fastgen.evaluation import evaluate, export_bundle
            report['loader_check'] = export_bundle(out/'untrained.pt', out/'export_bundle')
            report['evaluation'] = evaluate(out/'untrained.pt', c, out/'evaluation_smoke', [901], smoke=True)
        report['total_seconds'] = time.monotonic()-start
        atomic_json(out/'qualification.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
