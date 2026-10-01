"""Disposable full workload profiling under the physical GPU 2 lease."""
import argparse
import copy
import gc
import json
import os
import signal
import subprocess
import sys
import time
from flow_sid.common import ROOT, StageCost, atomic_json, source_identity, verify_sources
from flow_sid.gpu import GPU2Lease


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['hardware', 'profile', 'recovery'], required=True)
    parser.add_argument('--config', default='configs/probe_recipe.json')
    parser.add_argument('--updates', type=int, default=4)
    parser.add_argument('--full-rate', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    signal.alarm(7200)
    with StageCost('qualification_' + args.stage), GPU2Lease() as lease:
        import torch
        from flow_sid.engine import TrainingSystem, cpu_copy, memory_metrics, setup_torch, tensor_hash
        from flow_sid.networks import FlowNet
        setup_torch()
        def probe_update(system):
            ledger_path = ROOT / 'runs/qualification_allowance.json'
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempted_updates': 0}
            if ledger['attempted_updates'] >= 96:
                raise RuntimeError('Finite disposable probe update allowance exhausted')
            ledger['attempted_updates'] += 1
            atomic_json(ledger_path, ledger)
            return system.update()
        if args.stage == 'hardware':
            modules = {}
            for name in ['torch', 'torchvision', 'numpy', 'scipy', 'PIL', 'cleanfid', 'kagglehub', 'tqdm']:
                module = __import__(name)
                location = str(module.__file__)
                assert location.startswith('/home/user/miniconda3/envs/fastgen_sid_dit/')
                modules[name] = location
            test = subprocess.run([sys.executable, '-c',
                'from flow_sid.gpu import GPU2Lease; GPU2Lease().__enter__()'],
                cwd=ROOT, capture_output=True, text=True)
            assert test.returncode != 0 and 'lease occupied' in test.stderr
            record = {'device': str(torch.cuda.get_device_properties(0)),
                      'bf16_native': torch.cuda.is_bf16_supported(including_emulation=False),
                      'python': sys.executable, 'packages': modules, 'duplicate_lease_refused': True,
                      'sources': verify_sources(), 'torch': torch.__version__, 'cuda': torch.version.cuda}
            atomic_json(ROOT / 'runs/hardware.json', record)
            print(json.dumps(record), flush=True)
            return
        torch.manual_seed(cfg['teacher_seed'])
        probe_teacher = cpu_copy(FlowNet().state_dict())
        output = {'config': cfg, 'source': source_identity(), 'stages': {}}
        if args.stage == 'profile':
            for run in ['T', 'A4']:
                torch.cuda.reset_peak_memory_stats()
                config = copy.deepcopy(cfg)
                config['student']['gan_warmup'] = 0
                config['student']['gan_ramp'] = 1
                config['probe_full_rollout'] = True
                if args.full_rate:
                    config['teacher']['lr_warmup'] = 1
                    config['student']['lr_warmup'] = 1
                system = TrainingSystem(config, run, probe_teacher, 'disposable-scratch-probe')
                results = []
                for _ in range(args.updates):
                    result = probe_update(system)
                    print(json.dumps({'run': run, **result}), flush=True)
                    results.append(result)
                output['stages'][run] = {'updates': results, 'memory': memory_metrics(),
                                         'counts': system.counts}
                del system
                gc.collect()
                torch.cuda.empty_cache()
        else:
            config = copy.deepcopy(cfg)
            config['effective_batch'] = config['microbatch'] * 2
            config['student']['gan_warmup'] = 0
            config['student']['gan_ramp'] = 1
            for run in ['T', 'A4']:
                system = TrainingSystem(config, run, probe_teacher, 'disposable-scratch-probe')
                probe_update(system)
                saved = system.snapshot()
                path = ROOT / 'artifacts/qualification_recovery.pt'
                torch.save(saved, path)
                del saved
                expected_metrics = probe_update(system)
                expected = system.snapshot()
                del system
                gc.collect()
                torch.cuda.empty_cache()
                system = TrainingSystem(config, run, probe_teacher, 'disposable-scratch-probe')
                system.restore(torch.load(path, map_location='cpu', weights_only=False))
                actual_metrics = probe_update(system)
                actual = system.snapshot()
                def equal(a, b, path='state'):
                    if torch.is_tensor(a):
                        assert torch.equal(a, b), path
                    elif isinstance(a, np.ndarray):
                        assert np.array_equal(a, b), path
                    elif isinstance(a, dict):
                        assert a.keys() == b.keys(), path
                        for k in a: equal(a[k], b[k], path + '/' + str(k))
                    elif isinstance(a, (tuple, list)):
                        assert len(a) == len(b)
                        for i, (v, w) in enumerate(zip(a, b)): equal(v, w, path + '/' + str(i))
                    else:
                        assert a == b, path
                import numpy as np
                equal(expected, actual)
                for key in expected_metrics:
                    if key != 'seconds': equal(expected_metrics[key], actual_metrics[key], key)
                output['stages'][run] = {'next_complete_update_bitwise_equal': True,
                                         'interruption': 'save, destroy objects, reload, next complete update',
                                         'memory': memory_metrics()}
                print(json.dumps({'recovery': run, 'exact': True}), flush=True)
                del system, expected, actual
                path.unlink()
                gc.collect()
                torch.cuda.empty_cache()
            # Full G/fake/optimizer/EMA zero-adversarial equivalence.
            config['student']['d_weight'] = 0.
            config['student']['g_gan_weight'] = 0.
            expected = None
            for run in ['S1', 'A1']:
                system = TrainingSystem(config, run, probe_teacher, 'disposable-scratch-probe')
                metrics = probe_update(system)
                state = system.snapshot()
                state.pop('run')
                if expected is None:
                    expected = state
                else:
                    equal(expected, state)
                del system
                gc.collect()
                torch.cuda.empty_cache()
            output['zero_adversarial_equivalence'] = True
        lease.verify_ownership()
        atomic_json(ROOT / ('runs/qualification_' + args.stage + '.json'), output)


if __name__ == '__main__':
    main()
