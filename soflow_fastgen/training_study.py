"""Frozen protocol and paired evaluation for scratch-training comparisons."""
import gc
import json
import math
from pathlib import Path
import statistics

import torch

from .common import ROOT, atomic_json, digest, source_provenance, verify_protected
from .evaluation import prepare_reference
from .models import Soflow
from .paired_evaluation import (
    NOISE_PROTOCOL, environment_identity, evaluate_schedule, fingerprint, image_manifest,
)


def validate_protocol(protocol, training):
    if protocol['initialization'] != 'scratch' or protocol['starting_checkpoint'] is not None:
        raise ValueError('New experiments must initialize from scratch')
    run = (ROOT / protocol['run_dir']).resolve()
    run.relative_to((ROOT / 'runs').resolve())
    if run != (ROOT / training['run_dir']).resolve():
        raise ValueError('Training and experiment run directories differ')
    if run == (ROOT / protocol['reference_checkpoint']).resolve().parent.parent:
        raise ValueError('The completed reference run is immutable')
    if (training['total_steps'] != protocol['expected_steps']
            or protocol['curve_steps'][-1] != training['total_steps']
            or sorted(set(protocol['curve_steps'])) != protocol['curve_steps']
            or min(protocol['curve_steps']) <= 0
            or training['checkpoint_retain_steps'] != protocol['curve_steps'][:-1]):
        raise ValueError('Full horizon and retained curve checkpoints must match')
    if training['eval_steps'] or training['budget_seconds'] != protocol['training_budget_seconds']:
        raise ValueError('Evaluation must have separate cumulative accounting')
    if (training['batch_size'] != 128 or not 0 < training['microbatch_size'] <= 128
            or training['memory_limit_bytes'] != 20_000_000_000):
        raise ValueError('Invalid effective batch or training memory limit')
    if (protocol['noise_protocol'] != NOISE_PROTOCOL or protocol['nfes'] != [1, 4]
            or protocol['eval_batch_size'] != 32 or protocol['samples_per_category'] != 20
            or protocol['weight_source'] != 'ema'):
        raise ValueError('Invalid frozen evaluation conditions')
    selection, confirmation = protocol['selection_seeds'], protocol['confirmation_seeds']
    seeds = selection + confirmation
    if (len(selection) != 2 or len(confirmation) != 4 or len(set(seeds)) != 6
            or not all(type(s) is int and 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Selection and confirmation seeds must be distinct')
    if (protocol['max_retries'] != 1 or protocol['shutdown_reserve_seconds'] < 30
            or any(protocol[k] <= protocol['shutdown_reserve_seconds'] for k in
                   ('qualification_budget_seconds', 'training_budget_seconds', 'evaluation_budget_seconds'))):
        raise ValueError('Invalid finite job/recovery allowances')
    return run


def load_study(path):
    protocol = json.loads(Path(path).read_text())
    training = json.loads(Path(protocol['training_config']).read_text())
    validate_protocol(protocol, training)
    return protocol, training


def check_inputs(protocol, training):
    verify_protected()
    if digest(training['partition']) != protocol['partition_sha256']:
        raise ValueError('Optimization/development partition changed')
    if digest(protocol['reference_checkpoint']) != protocol['reference_checkpoint_sha256']:
        raise ValueError('Historical reference checkpoint changed')
    if fingerprint(image_manifest(protocol['reference_images'])) != protocol['reference_sha256']:
        raise ValueError('Historical development reference pixels changed')
    with torch.device('meta'):
        backbone = Soflow(**training['architecture'])
    count = sum(p.numel() for p in backbone.parameters())
    if count != protocol['expected_parameters'] or count >= 100_000_000:
        raise ValueError('Loaded inference parameter count violates the protocol')
    return count


def check_prepared(protocol, record):
    if (record['protocol_sha256'] != fingerprint(protocol)
            or record['training_config_sha256'] != digest(protocol['training_config'])
            or record['source_sha256'] != fingerprint(source_provenance()['source_sha256'])
            or record['environment'] != environment_identity()):
        raise ValueError('Prepared source, environment or configuration changed')


def validate_pairing(rows):
    draws, initial, identities = {}, {}, set()
    for row in rows:
        key = (row['checkpoint_label'], row['nfe'], row['seed'])
        if key in identities or not math.isfinite(row['fid']):
            raise ValueError('Duplicate or nonfinite evaluation row')
        identities.add(key)
        pair = (row['nfe'], row['seed'])
        if pair in draws and draws[pair] != row['noise_sha256']:
            raise ValueError('Checkpoint comparisons require identical actual noise')
        draws[pair] = row['noise_sha256']
        if row['seed'] in initial and initial[row['seed']] != row['noise_sha256'][0]:
            raise ValueError('Initial noise differs across NFEs')
        initial[row['seed']] = row['noise_sha256'][0]


def decide(rows, protocol, stage, nfes=None):
    validate_pairing(rows)
    scores = {(r['checkpoint_label'], r['nfe'], r['seed']): r['fid'] for r in rows}
    candidate = f"step_{protocol['expected_steps']:06d}"
    decisions = {}
    for nfe in protocol['nfes'] if nfes is None else nfes:
        seeds = protocol[f'{stage}_seeds']
        reference = [scores[('reference', nfe, seed)] for seed in seeds]
        final = [scores[(candidate, nfe, seed)] for seed in seeds]
        deltas = [a-b for a, b in zip(final, reference)]
        better = all(delta < 0 for delta in deltas)
        decisions[str(nfe)] = {
            'nfe': nfe, 'seeds': seeds, 'reference_fids': reference, 'candidate_fids': final,
            'reference_mean': statistics.mean(reference), 'candidate_mean': statistics.mean(final),
            'paired_deltas': deltas, 'mean_delta': statistics.mean(deltas),
            'eligible': better if stage == 'selection' else None,
            'outcome': ('provisional' if better else 'no_candidate_for_confirmation')
                       if stage == 'selection' else
                       ('confirmed_under_protocol' if better else 'inconclusive'),
            'limitation': 'Generation-seed comparison of one training seed on reused development data.',
        }
    return decisions


def evaluate_study(protocol, stage, prepared):
    training = json.loads(Path(protocol['training_config']).read_text())
    check_prepared(protocol, prepared)
    check_inputs(protocol, training)
    torch.set_num_threads(training['torch_threads'])
    reference = prepare_reference(training)
    if fingerprint(image_manifest(reference)) != protocol['reference_sha256']:
        raise ValueError('Study reference pixels do not match EXP-1')
    run = Path(protocol['run_dir'])
    nfes = protocol['nfes']
    if stage == 'control':
        labels = [('reference', Path(protocol['reference_checkpoint']), 20000)]
        seeds = protocol['selection_seeds']
    elif stage == 'selection':
        labels = [('reference', Path(protocol['reference_checkpoint']), 20000)] + [
            (f'step_{step:06d}', run/'checkpoints'/f'step_{step:06d}.pt', step)
            for step in protocol['curve_steps']]
        seeds = protocol['selection_seeds']
    elif stage == 'confirmation':
        selected = json.loads((run/'selection.json').read_text())['decisions']
        nfes = [n for n in nfes if selected[str(n)]['eligible']]
        labels = [('reference', Path(protocol['reference_checkpoint']), 20000),
                  (f"step_{protocol['expected_steps']:06d}",
                   run/'checkpoints'/f"step_{protocol['expected_steps']:06d}.pt",
                   protocol['expected_steps'])]
        seeds = protocol['confirmation_seeds']
    else:
        raise ValueError('Unknown evaluation stage')
    rows = []
    for label, checkpoint, step in labels if nfes else []:
        checkpoint_hash = digest(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['step'] != step or state['extra']['partition_sha256'] != protocol['partition_sha256']:
            raise ValueError('Checkpoint step or partition changed')
        if label != 'reference':
            if (state['config'] != training
                    or fingerprint(state['extra']['source_sha256']) != prepared['source_sha256']):
                raise ValueError('Candidate checkpoint does not belong to this frozen scratch run')
        backbone = Soflow(**state['config']['architecture']).eval().requires_grad_(False)
        backbone.load_state_dict(state['ema'])
        parameters = sum(p.numel() for p in backbone.parameters())
        if parameters >= 100_000_000:
            raise ValueError('Inference parameter limit exceeded')
        del state
        backbone.cuda()
        for nfe in nfes:
            evaluation = dict(protocol, checkpoint_sha256=checkpoint_hash, nfe=nfe)
            schedule = {'name': 'uniform', 'intermediate_times': [1-i/nfe for i in range(1, nfe)]}
            for seed in seeds:
                row = dict(evaluate_schedule(backbone, evaluation, schedule, seed, reference,
                                            prepared['source_sha256'], protocol['reference_sha256']),
                           checkpoint_label=label, checkpoint_step=step, parameters=parameters)
                rows.append(row)
        del backbone
        gc.collect()
        torch.cuda.empty_cache()
    validate_pairing(rows)
    decisions = decide(rows, protocol, stage, nfes) if stage != 'control' else {}
    atomic_json(run/f'{stage}.json', {
        'stage': stage, 'protocol_sha256': fingerprint(protocol), 'decisions': decisions,
        'results': rows, 'outcome': 'not_run_no_eligible_candidate' if not nfes else 'completed',
        'official_score': False,
    })
    return decisions
