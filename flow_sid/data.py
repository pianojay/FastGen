"""Read-only supplied images and copied partition; never invokes downloader.

Cache construction follows the AFM ee74486 data utility, adapted locally.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize
from .common import ROOT, PARTITION_SHA, atomic_json, digest, object_hash

IMAGE_ROOT = Path('/home/user/jw_soflow/FastGen-SoFlow/data/pokemon-generation-one-22k')


def partition(split):
    source = ROOT / 'data/pokemon-generation-one-22k'
    manifest = json.loads((ROOT / 'resources/development_split.json').read_text())
    assert digest(ROOT / 'resources/development_split.json') == PARTITION_SHA
    assert digest(source / 'train_split.txt') == manifest['source_train_sha256']
    assert digest(source / 'category_to_id.json') == manifest['category_mapping_sha256']
    assert not set(manifest['optimization_indices']) & set(manifest['development_indices'])
    assert sorted(manifest['optimization_indices'] + manifest['development_indices']) == list(range(17079))
    mapping = json.loads((source / 'category_to_id.json').read_text())
    paths = (source / 'train_split.txt').read_text().splitlines()
    chosen = [paths[i] for i in manifest[split + '_indices']]
    return chosen, [mapping[Path(p).parts[1]] for p in chosen]


def prepare_cache():
    folder = ROOT / 'artifacts/data_cache'
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for split in ['optimization', 'development']:
        paths, labels = partition(split)
        identity = {'partition_sha256': PARTITION_SHA, 'paths_sha256': object_hash(paths),
                    'transform': 'PIL RGB -> torchvision Resize([64,64]) bilinear -> uint8 CHW'}
        path = folder / (split + '.npy')
        meta = folder / (split + '.json')
        if path.exists() and meta.exists():
            rec = json.loads(meta.read_text())
            assert rec['identity'] == identity and rec['cache_sha256'] == digest(path)
        else:
            def read(p):
                full = IMAGE_ROOT / p
                sha = digest(full)
                with Image.open(full) as image:
                    pixel = np.asarray(resize(image.convert('RGB'), [64, 64]), dtype=np.uint8)
                return pixel.transpose(2, 0, 1), sha
            pixels = np.lib.format.open_memmap(str(path) + '.tmp', mode='w+', dtype=np.uint8,
                                               shape=(len(paths), 3, 64, 64))
            hashes = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                for i, (pixel, sha) in enumerate(pool.map(read, paths)):
                    pixels[i] = pixel
                    hashes.append(sha)
            pixels.flush()
            del pixels
            Path(str(path) + '.tmp').replace(path)
            rec = {'identity': identity, 'cache_sha256': digest(path), 'count': len(paths),
                   'source_hashes': dict(zip(paths, hashes))}
            atomic_json(meta, rec)
        records[split] = {k: v for k, v in rec.items() if k != 'source_hashes'}
    opt = json.loads((folder / 'optimization.json').read_text())['source_hashes']
    dev = json.loads((folder / 'development.json').read_text())['source_hashes']
    assert not set(opt.values()) & set(dev.values()), 'Exact cross-partition duplicates'
    return records


class BalancedData:
    """Uniform class, then uniform optimization image; explicit stream and counters."""
    def __init__(self):
        self.paths, labels = partition('optimization')
        self.pixels = np.load(ROOT / 'artifacts/data_cache/optimization.npy', mmap_mode='r')
        self.labels = torch.tensor(labels)
        self.by_class = [torch.where(self.labels == c)[0] for c in range(151)]
        self.presentations = 0

    def take(self, count, rng):
        labels = torch.randint(151, (count,), generator=rng)
        uniform = torch.rand(count, generator=rng)
        indices = torch.tensor([self.by_class[c][int(u * len(self.by_class[c]))]
                                for c, u in zip(labels.tolist(), uniform.tolist())])
        flips = torch.rand(count, generator=rng) < .5
        x = torch.from_numpy(np.array(self.pixels[indices.numpy()], copy=True)).float().div_(255).sub_(.5).div_(.5)
        x = torch.where(flips[:, None, None, None], x.flip(-1), x)
        self.presentations += count
        return x, labels, {'indices': indices, 'flip': flips}
