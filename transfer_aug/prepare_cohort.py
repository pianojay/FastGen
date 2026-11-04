"""Freeze first-wave A1-only inputs; never load models or inspect B pixels."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'transfer_aug'
OUT = HERE / 'shared/cohort_v1'
SEED = 20261007


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def verify_integrity():
    p = json.loads((ROOT / 'eda/resources/provenance.json').read_text())
    files = dict(p['protected_sha256'])
    for name in ['A.txt', 'A1.txt', 'A2.txt', 'B.txt', 'development_split.json']:
        rel = 'eda/resources/' + name
        files[rel] = sha(ROOT / rel)
    for rel, digest in files.items():
        assert sha(ROOT / rel) == digest, rel
    assert files['eda/resources/development_split.json'] == p['partition_sha256']
    meta = json.loads((ROOT / 'eda/features/metadata.json').read_text())
    assert sha(ROOT / 'eda/features/pixels64.npy') == meta['pixels_sha256']
    assert sha(ROOT / 'eda/reports/inventory.csv') == meta['inventory_sha256']
    files['eda/features/pixels64.npy'] = meta['pixels_sha256']
    files['eda/reports/inventory.csv'] = meta['inventory_sha256']
    return files


def main():
    original_integrity = verify_integrity()
    a = (ROOT / 'eda/resources/A.txt').read_text().splitlines()
    a1 = (ROOT / 'eda/resources/A1.txt').read_text().splitlines()
    indices = {p: i for i, p in enumerate(a)}
    assert len(a1) == 14059 and set(a1) <= set(a)
    with (ROOT / 'eda/reports/inventory.csv').open() as f:
        inventory = {r['path']: r for r in csv.DictReader(f)}
    pixels = np.load(ROOT / 'eda/features/pixels64.npy', mmap_mode='r')
    # The validated cache is A followed by B. Access only A1 indexes here.
    categories = {}
    for path in a1:
        row = inventory[path]
        assert row['partition'] == 'A1'
        digest = hashlib.sha256(pixels[indices[path]].tobytes()).hexdigest()
        categories.setdefault(row['category'], {}).setdefault(digest, path)
    names = sorted(categories)
    assert len(names) == 151
    rng = np.random.default_rng(SEED)
    cohort = []
    for category in names:
        choices = sorted(categories[category].values())
        selected = rng.choice(choices, 4, replace=False).tolist()
        for ordinal, content in enumerate(selected):
            eligible = [p for p in choices if p != content]
            intra = str(rng.choice(eligible))
            other_class = str(rng.choice([n for n in names if n != category]))
            content_hash = hashlib.sha256(pixels[indices[content]].tobytes()).hexdigest()
            eligible_other = [p for h, p in sorted(categories[other_class].items()) if h != content_hash]
            assert eligible_other, other_class
            inter = str(rng.choice(eligible_other))
            entry = {'sample_id': f'{int(inventory[content]["category_id"]):03d}_{ordinal}',
                     'category': category, 'category_id': int(inventory[content]['category_id']),
                     'ordinal': ordinal, 'inter_category': other_class}
            for role, path in [('content', content), ('intra', intra), ('inter', inter)]:
                assert sha(ROOT / 'data/pokemon-generation-one-22k' / path) == inventory[path]['sha256'], path
                entry[role + '_path'] = path
                entry[role + '_file_sha256'] = inventory[path]['sha256']
                entry[role + '_pixel_sha256'] = hashlib.sha256(pixels[indices[path]].tobytes()).hexdigest()
            cohort.append(entry)
    assert len(cohort) == 604
    review_names = sorted(np.random.default_rng(SEED + 1).choice(names, 16, replace=False).tolist())
    review_ids = [r['sample_id'] for r in cohort if r['category'] in review_names and r['ordinal'] == 0]
    bundle = {'schema': 1, 'seed': SEED, 'partition': 'A1', 'count': 604,
              'category_count': 151, 'per_category': 4,
              'sampling': 'Uniform distinct cached-pixel hashes within class; intra donor differs from content; inter class uniformly sampled among other 150 classes then uniform distinct pixels, excluding content hash.',
              'preprocessing': 'Previously verified official dataset transform and uint8 export; use cached 64x64 RGB pixels unchanged.',
              'rows': cohort, 'representative_review_ids': review_ids,
              'representative_review_categories': review_names, 'review_seed': SEED + 1}
    encoded = json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if (OUT / 'cohort.json').exists():
        assert (OUT / 'cohort.json').read_text() == encoded, 'Frozen cohort differs'
        frozen = json.loads((OUT / 'manifest.json').read_text())
        for name, digest in frozen['files'].items(): assert sha(OUT / name) == digest, name
        assert frozen['protected_inputs'] == original_integrity
        print('Frozen cohort regenerated identically; all bundle and protected-input hashes verified.')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cohort.json').write_text(encoded)
    for role in ['content', 'intra', 'inter']:
        array = np.stack([pixels[indices[r[role + '_path']]] for r in cohort])
        np.save(OUT / f'{role}_uint8.npy', array, allow_pickle=False)
    # Metadata-only snapshot of every source file, for detecting unrelated edits
    # without reading B images. Worker/source content hashes remain A1-only.
    stat_snapshot = {}
    for path in sorted(inventory):
        st = (ROOT / 'data/pokemon-generation-one-22k' / path).stat()
        stat_snapshot[path] = [st.st_size, st.st_mtime_ns]
    write_json(OUT / 'source_stat_snapshot.json', stat_snapshot)
    names_to_hash = ['cohort.json', 'content_uint8.npy', 'intra_uint8.npy', 'inter_uint8.npy', 'source_stat_snapshot.json']
    manifest = {'schema': 1, 'cohort_id': 'A1-transfer-20261007-v1',
                'files': {n: sha(OUT / n) for n in names_to_hash},
                'protected_inputs': original_integrity,
                'builder_sha256': sha(__file__), 'source_count': len(stat_snapshot),
                'learned_models_loaded': False, 'gpu_used': False}
    write_json(OUT / 'manifest.json', manifest)
    assert verify_integrity() == original_integrity
    print(json.dumps({'count': len(cohort), 'review_ids': review_ids,
                      'cohort_sha256': manifest['files']['cohort.json'], 'manifest_sha256': sha(OUT / 'manifest.json')}, indent=2))


if __name__ == '__main__': main()
