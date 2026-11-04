"""Development partitions are indices into the unchanged official train split."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from dataset import PokemonDataset, read_split, category_from_filename
from .common import ROOT, atomic_json, digest, verify_protected


def prepare(data_root, output, seed=42):
    verify_protected()
    root = Path(data_root)
    split_root = ROOT / 'data/pokemon-generation-one-22k'
    paths = read_split(split_root / 'train_split.txt')
    validation = read_split(split_root / 'val_split.txt')
    mapping = json.loads((split_root / 'category_to_id.json').read_text())
    assert len(paths) == 17079 and len(validation) == 3020 and len(mapping) == 151
    assert len(set(paths)) == len(paths) and not set(paths) & set(validation)
    groups = defaultdict(list)
    hashes = defaultdict(list)
    for i, name in enumerate(paths):
        category = category_from_filename(name)
        assert category in mapping
        with Image.open(root / name) as image:
            image.verify()
        groups[category].append(i)
        hashes[digest(root / name)].append(name)
    development = []
    duplicate_paths = {p for group in hashes.values() if len(group) > 1 for p in group}
    for name, indices in sorted(groups.items()):
        indices = [i for i in indices if paths[i] not in duplicate_paths]
        indices = sorted(indices, key=lambda i: hashlib.sha256(f'{seed}:{paths[i]}'.encode()).hexdigest())
        assert len(indices) > 20
        development.extend(indices[:20])
    development = sorted(development)
    held_out = set(development)
    optimization = [i for i in range(len(paths)) if i not in held_out]
    manifest = {'seed': seed, 'selection': '20 globally byte-unique training files per category, SHA256(seed:path) ordering',
        'exact_duplicate_cross_partition_groups': 0,
        'source_train_sha256': digest(split_root / 'train_split.txt'),
        'category_mapping_sha256': digest(split_root / 'category_to_id.json'),
        'optimization_indices': optimization, 'development_indices': development,
        'optimization_count': len(optimization), 'development_count': len(development),
        'class_counts': dict(Counter(category_from_filename(p) for p in paths)),
        'exact_duplicate_groups': [v for v in hashes.values() if len(v) > 1]}
    atomic_json(output, manifest)
    return manifest


def datasets(config):
    verify_protected()
    manifest = json.loads(Path(config['partition']).read_text())
    split_root = ROOT / 'data/pokemon-generation-one-22k'
    if manifest['source_train_sha256'] != digest(split_root / 'train_split.txt'):
        raise ValueError('Partition belongs to a different official train manifest')
    if manifest['category_mapping_sha256'] != digest(split_root / 'category_to_id.json'):
        raise ValueError('Partition category mapping changed')
    mapping = json.loads((split_root / 'category_to_id.json').read_text())
    dataset = PokemonDataset(config['data_root'], split_root / 'train_split.txt',
                             return_category=True, category_to_id=mapping)
    train_indices, dev_indices = manifest['optimization_indices'], manifest['development_indices']
    if (set(train_indices) & set(dev_indices) or len(train_indices) != 14059 or
            len(dev_indices) != 3020 or set(train_indices + dev_indices) != set(range(len(dataset)))):
        raise ValueError('Invalid development partition')
    return (torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, dev_indices), mapping)


def loader(dataset, config, epoch, cursor):
    generator = torch.Generator().manual_seed(config['seed'] + epoch)
    indices = torch.randperm(len(dataset), generator=generator).tolist()[cursor:]
    return torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
        sampler=indices, num_workers=config['num_workers'], drop_last=True,
        pin_memory=True, generator=torch.Generator().manual_seed(config['seed'] + epoch + 100000),
        **({'prefetch_factor': 2} if config['num_workers'] else {}))
