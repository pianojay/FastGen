"""One finite baseline; no selection, second seed, extension or recipe changes."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
import traceback
import torch
from afm_fastgen.common import (ROOT, GpuLease, append_json, atomic_json, configure,
                               digest, memory, process_identity, source_identity, verify_protected)
from afm_fastgen.training import Trainer
from afm_fastgen.reporting import fixed_panel, nearest_gallery, loss_plots


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); config_path = Path(args.config); c = json.loads(config_path.read_text())
    configure(c)
    root = ROOT/c['run_dir']; root.mkdir(parents=True, exist_ok=True)
    with GpuLease():
        record_path = root/'record.json'; prior = json.loads(record_path.read_text()) if record_path.exists() else None
        if prior and not args.resume:
            raise RuntimeError('Existing baseline refuses fresh restart')
        if args.resume and not prior:
            raise RuntimeError('No baseline to resume')
        if prior and prior['state'] == 'completed':
            raise RuntimeError('Completed baseline refuses extension')
        identity = {'config_sha256': digest(config_path), 'source': source_identity(), 'protected': verify_protected()}
        if prior:
            assert identity == prior['identity'], 'Exact recovery identity mismatch'
            assert prior['recovery_attempts'] < c['max_recoveries']
        record = prior or {'identity': identity, 'created_unix': time.time(), 'attempts': [], 'recovery_attempts': 0}
        if prior:
            record['recovery_attempts'] += 1
        attempt = {'process': process_identity(), 'start_unix': time.time(), 'resume': args.resume}
        record['attempts'].append(attempt); record['state'] = 'starting'
        atomic_json(record_path, record)
        remaining = c['watchdog_seconds']-(time.time()-record['created_unix'])
        if remaining <= 0:
            raise RuntimeError('Finite baseline watchdog allowance exhausted')
        def timeout(signum, frame):
            raise TimeoutError('Finite baseline watchdog expired')
        signal.signal(signal.SIGALRM, timeout); signal.alarm(int(remaining))
        start = time.monotonic(); metrics = root/'metrics.jsonl'
        try:
            trainer = Trainer(c, resume=root/'checkpoints/latest.pt' if args.resume else None)
            atomic_json(root/'parameters.json', trainer.parameters)
            torch.cuda.reset_peak_memory_stats(); record['state'] = 'training'; atomic_json(record_path, record)
            checkpoint_seconds = 0.; sample_seconds = 0.
            # Fixed fresh-state checkpoint makes an early exact recovery possible.
            if trainer.step == 0:
                trainer.save(root/'checkpoints/latest.pt')
            while trainer.step < c['horizon_g_updates']:
                row = trainer.cycle(); row['attempt'] = len(record['attempts']); row['unix_time'] = time.time()
                append_json(metrics, row)
                if trainer.step % c['status_interval'] == 0 or trainer.step == 1:
                    status = {'state': 'training', **row, 'memory': memory(),
                              'attempt_seconds': time.monotonic()-start,
                              'elapsed_watchdog_seconds': time.time()-record['created_unix']}
                    atomic_json(root/'status.json', status); print(json.dumps(status), flush=True)
                if trainer.step % c['checkpoint_interval'] == 0 or trainer.step == c['horizon_g_updates']:
                    t = time.monotonic(); trainer.save(root/'checkpoints/latest.pt')
                    checkpoint_seconds += time.monotonic()-t
                if trainer.step % c['sample_interval'] == 0 or trainer.step == c['horizon_g_updates']:
                    t = time.monotonic(); fixed_panel(trainer.ema, c, root/f'panels/step{trainer.step:06d}.png', trainer.step)
                    sample_seconds += time.monotonic()-t
            # Preserve the exact final artifact across evaluation-only recovery.
            if not (root/'one_nfe.pt').exists():
                trainer.export(root/'one_nfe.pt')
            training = {'completed_g_updates': trainer.g_updates, 'completed_d_updates': trainer.d_updates,
                        'g_real_presentations': trainer.gs.presentations, 'd_real_presentations': trainer.ds.presentations,
                        'optimizer_seconds': trainer.seconds, 'attempt_seconds': time.monotonic()-start,
                        'checkpoint_seconds_this_attempt': checkpoint_seconds, 'sample_seconds_this_attempt': sample_seconds,
                        'memory': memory(), 'parameters': trainer.parameters,
                        'checkpoint_sha256': digest(root/'one_nfe.pt'), 'recovery_attempts': record['recovery_attempts']}
            atomic_json(root/'training.json', training)
            del trainer; gc.collect(); torch.cuda.empty_cache()
            record['state'] = 'evaluating'; atomic_json(record_path, record)
            atomic_json(root/'status.json', {'state': 'evaluating', 'step': c['horizon_g_updates']})
            from afm_fastgen.evaluation import evaluate, export_bundle
            bundle = root/'export_bundle'
            if (bundle/'loader_check.json').exists():
                assert digest(bundle/'one_nfe.pt') == digest(root/'one_nfe.pt')
                loader = json.loads((bundle/'loader_check.json').read_text())
            else:
                if bundle.exists():
                    bundle.rename(root/f'export_bundle_interrupted_{time.time_ns()}')
                loader = export_bundle(root/'one_nfe.pt', bundle)
            scores = evaluate(root/'one_nfe.pt', c, root/'evaluations/final_ema', c['eval_seeds'])
            nearest_gallery(c, root/f'evaluations/final_ema/seed{c["eval_seeds"][0]}', root/'nearest_optimization.png')
            loss_plots(metrics, root/'training_curves.svg')
            record.update({'state': 'completed', 'training': training, 'evaluations': scores,
                           'loader_check': loader, 'completed_unix': time.time()})
            attempt['seconds'] = time.monotonic()-start; attempt['outcome'] = 'completed'
            atomic_json(record_path, record); atomic_json(root/'status.json', {'state': 'completed', 'record': str(record_path)})
        except BaseException as error:
            attempt['seconds'] = time.monotonic()-start; attempt['outcome'] = 'failed'
            record.update({'state': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
            atomic_json(record_path, record); atomic_json(root/'status.json', {'state': 'failed', 'error': repr(error)})
            raise
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    main()
