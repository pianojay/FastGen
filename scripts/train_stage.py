"""One of the nine predeclared runs; no searches, extra arms or implicit restarts."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from flow_sid.common import (ROOT, StageCost, atomic_json, append_json, digest,
                             object_hash, source_identity, verify_sources)
from flow_sid.gpu import GPU2Lease


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', choices=['T', 'S1', 'A1', 'S2', 'A2', 'S3', 'A3', 'S4', 'A4'], required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/campaign.json').read_text())
    assert not config.get('probe_full_rollout')
    verify_sources()
    identity = source_identity()
    frozen = json.loads((ROOT / 'resources/campaign_source.json').read_text())
    assert identity['sha256'] == frozen['sha256'], 'Training source changed after protocol freeze'
    identity = frozen
    run = args.run
    folder = ROOT / 'runs/campaign' / run
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / 'latest.pt'
    start = time.monotonic()
    with StageCost('run_' + run), GPU2Lease() as lease:
        import torch
        from flow_sid.engine import TrainingSystem, cpu_copy, save_checkpoint, setup_torch, memory_metrics
        from flow_sid.evaluation import TeacherSampler, evaluate_model, exported_state
        from model import ModelOneNFE, ModelFewNFE
        setup_torch()
        teacher_state = teacher_hash = None
        if run != 'T':
            selected = json.loads((ROOT / 'runs/campaign/T/selected.json').read_text())
            qualification = json.loads((ROOT / 'runs/campaign/T/qualified.json').read_text())
            assert qualification['teacher_checkpoint_sha256'] == selected['checkpoint_sha256']
            assert qualification['eligible_for_distillation']
            teacher_path = ROOT / selected['checkpoint']
            teacher_hash = digest(teacher_path)
            assert teacher_hash == selected['checkpoint_sha256']
            teacher_state = torch.load(teacher_path, map_location='cpu', weights_only=False)['weights']
            pair_config = json.loads((ROOT / 'resources/frozen_students.json').read_text())
            assert pair_config['config_sha256'] == object_hash(config)
            assert pair_config['teacher_checkpoint_sha256'] == teacher_hash
        system = TrainingSystem(config, run, teacher_state, teacher_hash)
        if latest.exists():
            system.restore(torch.load(latest, map_location='cpu', weights_only=False), identity=identity)
        selection_path = folder / 'selected.json'
        selected = json.loads(selection_path.read_text()) if selection_path.exists() else None
        settings = config['teacher'] if run == 'T' else config['student']
        if run != 'T':
            settings = {**settings, **config['nfe_runs'][str(system.nfe)]}
        events = folder / 'metrics.jsonl'
        atomic_json(folder / 'identity.json', {'run': run, 'teacher_sha256': teacher_hash,
                    'initial_weights_sha256': system.initial_hash, 'source': identity,
                    'config_sha256': object_hash(config), 'counts': system.counts})

        def status(phase, **extra):
            data = {'run': run, 'phase': phase, 'step': system.step,
                    'horizon': settings['updates'], 'time': time.time(),
                    'pid': os.getpid(), 'exposures': system.exposures.copy(), **extra}
            atomic_json(folder / 'status.json', data)
            atomic_json(ROOT / 'runs/current_status.json', data)

        def checkpoint():
            status('checkpoint')
            save_checkpoint(latest, system.snapshot(identity))

        def evaluate_selection():
            nonlocal selected
            target = folder / f'eval_{system.step:06d}'
            status('evaluation', seed=config['evaluation']['selection_seed'])
            system.optimizer.zero_grad(set_to_none=True)
            if system.fake_optimizer:
                system.fake_optimizer.zero_grad(set_to_none=True)
            if run == 'T':
                original = cpu_copy(system.g.state_dict())
                system.g.load_state_dict(system.ema)
                system.g.eval()
                model = TeacherSampler(system.g, settings['solver_calls'])
            else:
                model = (ModelOneNFE() if system.nfe == 1 else ModelFewNFE()).cuda()
                model.load_state_dict(exported_state(system.ema, system.nfe)['state_dict'])
            metrics = evaluate_model(model, target, config['evaluation']['selection_seed'],
                                     config['evaluation']['batch_size'])
            if run == 'T':
                system.g.load_state_dict(original)
                del original
            del model
            torch.cuda.empty_cache()
            metrics.update({'run': run, 'step': system.step})
            append_json(folder / 'evaluations.jsonl', metrics)
            print(json.dumps({'event': 'evaluation', **metrics}), flush=True)
            if selected is None or metrics['fid'] < selected['fid']:
                best = folder / 'best.pt'
                save_checkpoint(best, {'weights': cpu_copy(system.ema), 'run': run, 'step': system.step,
                                      'teacher_sha256': teacher_hash, 'source': identity,
                                      'config_sha256': object_hash(config)})
                selected = {**metrics, 'checkpoint': str(best.relative_to(ROOT)),
                            'checkpoint_sha256': digest(best), 'teacher_sha256': teacher_hash,
                            'gallery_directory': str(target.relative_to(ROOT))}
                atomic_json(selection_path, selected)
            if torch.cuda.max_memory_allocated() >= 20_000_000_000:
                raise RuntimeError('Evaluation exceeded memory limit')

        # If interrupted after checkpoint but before its evaluation, replay that evaluation.
        evaluated = set()
        if not latest.exists():
            checkpoint()
        if (folder / 'evaluations.jsonl').exists():
            evaluated = {json.loads(line)['step'] for line in (folder / 'evaluations.jsonl').read_text().splitlines()}
        if system.step in settings['evaluate_at'] and system.step not in evaluated:
            evaluate_selection()
        while system.step < settings['updates']:
            status('training')
            metrics = system.update()
            append_json(events, metrics)
            if system.step % config['log_every'] == 0 or system.step == 1:
                lease.verify_ownership()
                memory = memory_metrics()
                append_json(folder / 'memory.jsonl', {'step': system.step, **memory})
                print(json.dumps({'event': 'progress', 'run': run, **metrics,
                                  'peak_allocated_GB': memory['peak_allocated_bytes'] / 1e9}), flush=True)
                status('training', metrics=metrics, memory=memory)
            if system.step % config['checkpoint_every'] == 0 or system.step in settings['evaluate_at']:
                checkpoint()
            if system.step in settings['evaluate_at']:
                evaluate_selection()
        checkpoint()
        assert selected is not None
        if run != 'T':
            weights = torch.load(ROOT / selected['checkpoint'], map_location='cpu', weights_only=False)['weights']
            export = ROOT / 'artifacts/exports' / (run + '.ckpt')
            save_checkpoint(export, exported_state(weights, system.nfe))
            model = (ModelOneNFE() if system.nfe == 1 else ModelFewNFE()).cuda()
            model.load_state_dict(exported_state(weights, system.nfe)['state_dict'])
            status('confirmation', seed=config['evaluation']['confirmation_seed'])
            confirmation = evaluate_model(model, folder / 'confirmation', config['evaluation']['confirmation_seed'],
                                          config['evaluation']['batch_size'])
            atomic_json(folder / 'export.json', {'path': str(export.relative_to(ROOT)), 'sha256': digest(export),
                'selected_step': selected['step'], 'nfe': system.nfe, 'parameters': system.counts['student'],
                'teacher_sha256': teacher_hash, 'ladder': list(__import__('flow_sid.flow', fromlist=['ladder']).ladder(system.nfe)),
                'confirmation': confirmation})
        memory = memory_metrics()
        status('complete', selected=selected, memory=memory, attempt_seconds=time.monotonic() - start)
        print(json.dumps({'event': 'complete', 'run': run, 'selected': selected, 'memory': memory}), flush=True)


if __name__ == '__main__':
    main()
