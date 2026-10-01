"""Read-only campaign audit; this never generates extra evaluation samples."""
import argparse
import json
import math
from pathlib import Path
from flow_sid.common import ROOT, atomic_json, digest, object_hash, source_identity, verify_sources


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--through-nfe', type=int, choices=[2, 4], required=True)
    args = parser.parse_args()
    cfg = read(ROOT / 'configs/campaign.json')
    frozen = read(ROOT / 'resources/campaign_source.json')
    assert source_identity()['sha256'] == frozen['sha256']
    source_check = verify_sources()
    config_hash = object_hash(cfg)
    teacher = read(ROOT / 'runs/campaign/T/selected.json')
    teacher_hash = teacher['checkpoint_sha256']
    assert digest(ROOT / teacher['checkpoint']) == teacher_hash
    student_protocol = read(ROOT / 'resources/frozen_students.json')
    assert student_protocol['teacher_checkpoint_sha256'] == teacher_hash
    ledger = read(ROOT / 'runs/campaign/allowance.json')
    runs = ['T'] + [method + str(n) for n in range(1, args.through_nfe + 1) for method in ['S', 'A']]
    output = {'source_check': source_check, 'source_sha256': frozen['sha256'],
              'config_sha256': config_hash, 'teacher_checkpoint_sha256': teacher_hash,
              'through_nfe': args.through_nfe, 'runs': {}, 'pairs': {}}
    histories = {}
    identities = {}
    for run in runs:
        folder = ROOT / 'runs/campaign' / run
        status, identity, selected = [read(folder / f'{name}.json') for name in ['status', 'identity', 'selected']]
        assert status['phase'] == 'complete', (run, status['phase'])
        settings = cfg['teacher'] if run == 'T' else cfg['nfe_runs'][run[1:]]
        assert status['step'] == settings['updates']
        assert identity['config_sha256'] == config_hash and identity['source'] == frozen
        assert selected['checkpoint_sha256'] == digest(ROOT / selected['checkpoint'])
        if run != 'T':
            assert identity['teacher_sha256'] == teacher_hash
            assert selected['teacher_sha256'] == teacher_hash
            assert identity['initial_weights_sha256'] == student_protocol['teacher_tensor_sha256']
        for key in ['teacher', 'student', 'fake_score_discriminator']:
            if identity['counts'][key] is not None:
                assert identity['counts'][key] == 61_823_875 <= 100_000_000
        assert identity['counts']['discriminator_head'] == 0
        recorded = rows(folder / 'metrics.jsonl')
        history = {}
        for row in recorded:
            assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float))), (run, row['step'])
            if row['step'] in history:
                # Exact replay must retain every algorithmic metric and draw.
                assert {k: v for k, v in row.items() if k != 'seconds'} == {
                    k: v for k, v in history[row['step']].items() if k != 'seconds'}
            history[row['step']] = row
        assert set(history) == set(range(1, settings['updates'] + 1))
        assert history[status['step']]['exposures'] == status['exposures']
        evaluations = rows(folder / 'evaluations.jsonl')
        assert set(e['step'] for e in evaluations) == set(settings['evaluate_at'])
        for e in evaluations:
            assert e['samples'] == 3020 and e['seed'] == cfg['evaluation']['selection_seed']
            assert math.isfinite(e['fid']) and e['fid'] >= 0
        best = min(evaluations, key=lambda e: (e['fid'], e['step']))
        assert (selected['fid'], selected['step']) == (best['fid'], best['step'])
        memories = rows(folder / 'memory.jsonl') + [status['memory']]
        allocated = max(m['peak_allocated_bytes'] for m in memories)
        reserved = max(m['peak_reserved_bytes'] for m in memories)
        assert allocated < 20_000_000_000
        consumed = ledger['runs'][run]['consumed_seconds']
        attempts = ledger['runs'][run]['attempts']
        assert all('end' in a for a in attempts)
        assert len(attempts) <= 1 + cfg['watchdogs']['max_exact_recoveries_per_run']
        assert math.isclose(consumed, sum(a['seconds'] for a in attempts), abs_tol=.001)
        record = {'counts': identity['counts'], 'selected_step': selected['step'],
                  'selection_fid': selected['fid'], 'selected_checkpoint_sha256': selected['checkpoint_sha256'],
                  'exposures': status['exposures'], 'attempts': len(attempts),
                  'recorded_complete_update_attempts': len(recorded), 'stage_seconds': consumed,
                  'complete_update_seconds': sum(r['seconds'] for r in recorded),
                  'selection_generation_and_fid_seconds': sum(e['seconds'] for e in evaluations),
                  'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
                  'maximum_sampled_device_MiB': max(int(m['device']['used_MiB']) for m in memories),
                  'maximum_sampled_host_RSS_bytes': max(m['host']['Rss_bytes'] for m in memories),
                  'maximum_sampled_host_PSS_bytes': max(m['host']['Pss_bytes'] for m in memories),
                  'host_high_water_bytes': max(m['host']['VmHWM_bytes'] for m in memories)}
        if run != 'T':
            export = read(folder / 'export.json')
            assert export['sha256'] == digest(ROOT / export['path'])
            assert export['nfe'] == int(run[1:])
            assert export['selected_step'] == selected['step']
            assert export['teacher_sha256'] == teacher_hash
            assert export['confirmation']['seed'] == cfg['evaluation']['confirmation_seed']
            assert export['confirmation']['samples'] == 3020
            assert math.isfinite(export['confirmation']['fid'])
            record.update({'export_sha256': export['sha256'], 'confirmation_fid': export['confirmation']['fid'],
                           'confirmation_generation_and_fid_seconds': export['confirmation']['seconds']})
            batch = cfg['effective_batch']
            expected_images = batch * settings['updates']
            assert status['exposures']['g_images'] == expected_images
            assert status['exposures']['fake_images'] == expected_images
            assert status['exposures']['d_real_images'] == (expected_images if run.startswith('A') else 0)
        output['runs'][run] = record
        histories[run], identities[run] = history, identity
    for n in range(1, args.through_nfe + 1):
        s, a = 'S' + str(n), 'A' + str(n)
        assert identities[s]['initial_weights_sha256'] == identities[a]['initial_weights_sha256']
        for step, rs in histories[s].items():
            ra = histories[a][step]
            for key in ['draw_g', 'draw_fake', 'step_g', 'step_fake', 'lr', 'loss_scale']:
                assert rs[key] == ra[key], (s, a, step, key)
        for checkpoint in cfg['nfe_runs'][str(n)]['evaluate_at']:
            ss, aa = [read(ROOT / 'runs/campaign' / run / f'eval_{checkpoint:06d}/metrics.json') for run in [s, a]]
            assert ss['sampling_draws_sha256'] == aa['sampling_draws_sha256']
        ss, aa = [read(ROOT / 'runs/campaign' / run / 'confirmation/metrics.json') for run in [s, a]]
        assert ss['sampling_draws_sha256'] == aa['sampling_draws_sha256']
        output['pairs'][str(n)] = {'common_training_draws_equal': True,
                                  'common_evaluation_draws_equal': True,
                                  'common_initialization_sha256': identities[s]['initial_weights_sha256'],
                                  'matched_complete_updates': len(histories[s])}
    output['charged_run_seconds_teacher_once'] = sum(r['stage_seconds'] for r in output['runs'].values())
    path = ROOT / f'resources/audit_through_nfe{args.through_nfe}.json'
    atomic_json(path, output)
    print(json.dumps({'audit': 'passed', 'runs': runs, 'path': str(path)}))


if __name__ == '__main__':
    main()
