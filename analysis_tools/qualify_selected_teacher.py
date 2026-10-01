"""Three disposable full-path A4 updates from the selected scratch teacher.

This uses the frozen training implementation, never writes campaign model state,
and spends the already declared engineering-probe allowance. It is a numerical
and memory check, with no FID evaluation or quality selection.
"""
import copy
import json
import signal
from flow_sid.common import (ROOT, StageCost, atomic_json, digest, object_hash,
                             source_identity, verify_sources)
from flow_sid.gpu import GPU2Lease


def main():
    signal.alarm(600)
    cfg = json.loads((ROOT / 'configs/campaign.json').read_text())
    frozen = json.loads((ROOT / 'resources/campaign_source.json').read_text())
    assert source_identity()['sha256'] == frozen['sha256']
    verify_sources()
    selected = json.loads((ROOT / 'runs/campaign/T/selected.json').read_text())
    status = json.loads((ROOT / 'runs/campaign/T/status.json').read_text())
    assert status['phase'] == 'complete'
    teacher_path = ROOT / selected['checkpoint']
    teacher_hash = digest(teacher_path)
    assert teacher_hash == selected['checkpoint_sha256']
    result_path = ROOT / 'runs/qualification_selected_teacher.json'
    assert not result_path.exists(), 'No duplicate trained-teacher qualification'
    with StageCost('qualification_selected_teacher'), GPU2Lease() as lease:
        import torch
        from flow_sid.engine import TrainingSystem, memory_metrics, setup_torch, tensor_hash
        setup_torch()
        weights = torch.load(teacher_path, map_location='cpu', weights_only=False)['weights']
        probe = copy.deepcopy(cfg)
        probe['probe_full_rollout'] = True
        probe['student']['lr_warmup'] = 1
        probe['student']['gan_warmup'] = 0
        probe['student']['gan_ramp'] = 1
        system = TrainingSystem(probe, 'A4', weights, teacher_hash)
        original_teacher = tensor_hash(list(system.teacher.state_dict().values()))
        del weights
        results = []
        for _ in range(3):
            ledger_path = ROOT / 'runs/qualification_allowance.json'
            ledger = json.loads(ledger_path.read_text())
            assert ledger['attempted_updates'] < 96
            ledger['attempted_updates'] += 1
            ledger['selected_teacher_attempted_updates'] = ledger.get('selected_teacher_attempted_updates', 0) + 1
            atomic_json(ledger_path, ledger)
            metrics = system.update()
            results.append(metrics)
            print(json.dumps(metrics), flush=True)
        assert tensor_hash(list(system.teacher.state_dict().values())) == original_teacher
        assert all(p.grad is None for p in system.teacher.parameters())
        assert results[-1]['gan_coefficient'] == cfg['student']['g_gan_weight']
        assert all(m['step_g'] == 3 and m['step_fake'] == 3 for m in results)
        memory = memory_metrics()
        assert memory['peak_allocated_bytes'] < 20_000_000_000
        lease.verify_ownership()
        record = {'purpose': __doc__, 'teacher_checkpoint_sha256': teacher_hash,
                  'teacher_tensor_sha256': original_teacher,
                  'campaign_config_sha256': object_hash(cfg), 'probe_config': probe,
                  'source': frozen, 'updates': results, 'counts': system.counts,
                  'memory': memory, 'frozen_teacher_unchanged': True,
                  'disposable_weights_not_saved': True}
        atomic_json(result_path, record)
        print(json.dumps({'qualification': 'passed', 'memory': memory}), flush=True)


if __name__ == '__main__':
    main()
