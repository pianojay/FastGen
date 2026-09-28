"""Build an offline gallery from existing PNGs; no model, GPU or network needed."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run_dir):
    run = Path(run_dir).resolve()
    config = json.loads((run / 'config.json').read_text())
    partition_path = ROOT / config['partition']
    partition = json.loads(partition_path.read_text())
    split_root = ROOT / 'data/pokemon-generation-one-22k'
    mapping_path = split_root / 'category_to_id.json'
    train_path = split_root / 'train_split.txt'
    if (sha256(mapping_path) != partition['category_mapping_sha256'] or
            sha256(train_path) != partition['source_train_sha256']):
        raise ValueError('Mapping or training manifest differs from the run partition')
    mapping = json.loads(mapping_path.read_text())
    categories = [{'name': name, 'id': idx} for name, idx in sorted(mapping.items(), key=lambda x: x[1])]
    evaluations = []
    image_count = 0
    for path in sorted((run / 'evaluations').glob('step_*/nfe*/metrics.json')):
        metric = json.loads(path.read_text())
        if metric['partition_sha256'] != sha256(partition_path):
            raise ValueError(f'Partition mismatch: {path}')
        files = {p.name for p in path.parent.glob('*.png')}
        expected = {f'{c["id"]:04d}_{i:02d}.png' for c in categories for i in range(20)}
        if files != expected or len(files) != metric['count']:
            raise ValueError(f'Incomplete or unexpected samples: {path.parent}')
        evaluations.append(dict(step=int(path.parent.parent.name.removeprefix('step_')),
            nfe=metric['nfe'], seed=metric['seed'], fid=metric['fid'],
            path=path.parent.relative_to(run).as_posix()))
        image_count += len(files)
    if not evaluations:
        raise ValueError('No complete saved evaluations found')
    references = {str(c['id']): [] for c in categories}
    reference_manifest = json.loads((run / 'reference_dev/manifest.json').read_text())
    if reference_manifest['partition_sha256'] != sha256(partition_path):
        raise ValueError('Reference cache belongs to a different partition')
    paths = [line.strip() for line in train_path.read_text().splitlines() if line.strip()]
    for offset, index in enumerate(partition['development_indices']):
        source = Path(paths[index])
        category = source.parent.name if source.parent.name not in {'', 'PokemonData'} else source.stem.rsplit('-', 1)[0]
        relative = f'reference_dev/{offset:04d}.png'
        if not (run / relative).is_file():
            raise ValueError(f'Missing reference: {relative}')
        references[str(mapping[category])].append(relative)
    if any(len(items) != 20 for items in references.values()):
        raise ValueError('Expected 20 development references per category')
    panels = {}
    for path in sorted((run / 'samples').glob('*/nfe*.png')):
        panels.setdefault(str(int(path.parent.name)), {})[path.stem.removeprefix('nfe')] = path.relative_to(run).as_posix()
    data = dict(run=run.name, categories=categories, evaluations=evaluations,
        references=references, panels=panels, count=image_count)
    # Embed data so file:// works without fetch, a web server, or CDN dependencies.
    payload = json.dumps(data).replace('<', '\\u003c')
    template = Path(__file__).with_name('gallery.html').read_text()
    output = run / 'gallery.html'
    output.write_text(template.replace('__GALLERY_DATA__', payload))
    print(json.dumps({'gallery': str(output), 'categories': len(categories),
        'evaluation_passes': len(evaluations), 'generated_images': image_count,
        'reference_images': sum(map(len, references.values())), 'panel_checkpoints': len(panels)}, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='runs/pokemon_unet128_seed42')
    build(parser.parse_args().run_dir)
