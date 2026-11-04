"""Master-level contract and handoff checks, without redoing worker analyses."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    ledger = json.loads((ROOT / 'EXPERIMENT_LEDGER.json').read_text())
    cohort_dir = ROOT / 'shared/cohort_v1'
    assert sha(cohort_dir / 'cohort.json') == ledger['cohort_sha256']
    assert sha(cohort_dir / 'manifest.json') == ledger['bundle_manifest_sha256']
    assert sha(ROOT / ledger['common_protocol']) == ledger['common_protocol_sha256']
    cohort = json.loads((cohort_dir / 'cohort.json').read_text())
    assert len(cohort['rows']) == 604
    assert len(cohort['representative_review_ids']) == 16
    grouped = {}
    for row in cohort['rows']:
        grouped.setdefault(row['category'], set()).add(row['content_pixel_sha256'])
        assert row['content_pixel_sha256'] != row['intra_pixel_sha256']
        assert row['content_pixel_sha256'] != row['inter_pixel_sha256']
        assert row['category'] != row['inter_category']
        assert row['intra_path'].split('/')[1] == row['category']
    assert len(grouped) == 151 and all(len(v) == 4 for v in grouped.values())
    paths = []
    for name in ['A', 'B']:
        paths += (ROOT.parent / f'eda/resources/{name}.txt').read_text().splitlines()
    meta = json.loads((ROOT.parent / 'eda/features/metadata.json').read_text())
    assert hashlib.sha256('\n'.join(paths).encode()).hexdigest() == meta['paths_sha256']
    a1 = set((ROOT.parent / 'eda/resources/A1.txt').read_text().splitlines())
    assert all(row[role + '_path'] in a1 for row in cohort['rows'] for role in ['content', 'intra', 'inter'])
    accepted = []
    for entry in ledger['experiments']:
        contract = ROOT / entry['contract']
        assert sha(contract) == entry['contract_sha256']
        if not args.final:
            continue
        result_path = ROOT / entry['result']
        result = json.loads(result_path.read_text())
        assert result['experiment_id'] == entry['experiment_id']
        assert result['contract_revision'] == entry['contract_revision']
        for key in ['contract_sha256', 'common_protocol_sha256', 'cohort_sha256', 'bundle_manifest_sha256']:
            expected = entry[key] if key in entry else ledger[key]
            assert result[key] == expected, (entry['experiment_id'], key)
        assert result['status'] in ['completed', 'budget_limited', 'blocked', 'failed']
        assert result['training_applicable'] is False
        expected = {'TA-001': (1812, 48), 'TA-002': (14496, 384)}[entry['experiment_id']]
        if result['status'] == 'completed':
            for stem, count in [('transformed_output', expected[0]), ('control_output', 604),
                                ('output', expected[0] + 604), ('representative_review', expected[1]),
                                ('tail_review', 8), ('review', expected[1] + 8)]:
                assert result[f'expected_{stem}_count'] == count
                assert result[f'completed_{stem}_count'] == count
            for name, check in result['checks'].items():
                assert check['pass'] is True, (name, 'Completed handoff has failed check')
                assert (result_path.parent / check['proof']).is_file(), (name, 'Missing check proof')
        resource = result['resources']
        assert resource['device'] == 'cpu'
        assert resource['parameters'] == 0 and resource['gpu_seconds'] == 0
        assert resource['elapsed_seconds'] <= entry['wall_budget_seconds']
        summary = result_path.parent / 'SUMMARY.md'
        assert len(summary.read_text().split()) <= 500, 'Worker summary exceeds contract limit'
        evidence = result['evidence']
        assert isinstance(evidence, dict) and evidence
        for name, value in evidence.items():
            assert isinstance(value, str), (name, 'Expected relative path')
            target = (result_path.parent / value).resolve()
            assert target.is_relative_to(ROOT), (name, 'Evidence outside research directory')
            assert target.exists(), (name, target)
        manifest = result_path.parent / evidence['hashes']
        manifest_hash = result.get('artifact_manifest_sha256') or result.get('evidence_notes', {}).get('hash_inventory_sha256')
        assert sha(manifest) == manifest_hash, 'Artifact manifest digest differs from handoff'
        accepted.append({'experiment_id': entry['experiment_id'], 'status': result['status'],
                         'result_sha256': sha(result_path), 'summary_sha256': sha(summary),
                         'evidence_references_checked': len(evidence)})
    report = {'frozen_contracts_unchanged': True, 'cohort_partition_and_sampling_checks': True,
              'pixel_cache_path_alignment_verified': True, 'final_handoffs': accepted,
              'scope': 'Master checks compact contract/summary/evidence metadata. Detailed numerical/image/output verification is delegated to workers.'}
    (ROOT / 'shared' / ('handoff_acceptance.json' if args.final else 'dispatch_preflight.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
