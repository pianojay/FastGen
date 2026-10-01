"""Read-only official preprocessing and fixed partition; no downloader calls."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from .common import ROOT, atomic_json, digest


class Images(torch.utils.data.Dataset):
    def __init__(self, config, split):
        self.root = Path(config['data_root'])
        manifest = json.loads((ROOT/config['partition']).read_text())
        source = ROOT/'data/pokemon-generation-one-22k'
        assert digest(source/'train_split.txt') == manifest['source_train_sha256']
        assert digest(source/'category_to_id.json') == manifest['category_mapping_sha256']
        mapping = json.loads((source/'category_to_id.json').read_text())
        all_paths = (source/'train_split.txt').read_text().splitlines()
        indices = manifest[split+'_indices']
        assert not set(manifest['optimization_indices']) & set(manifest['development_indices'])
        assert sorted(manifest['optimization_indices']+manifest['development_indices']) == list(range(17079))
        self.paths = [all_paths[i] for i in indices]
        self.labels = [mapping[Path(p).parts[1]] for p in self.paths]
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                            transforms.Normalize((.5,)*3, (.5,)*3)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root/self.paths[index]) as image:
            x = self.transform(image.convert('RGB'))
        return x, self.labels[index]


def prepare_cache(config):
    """Lossless uint8 resize cache in AFM; saves shared disk/CPU work each epoch."""
    dataset = Images(config, 'optimization')
    folder = ROOT/'artifacts/data_cache'; folder.mkdir(parents=True, exist_ok=True)
    path = folder/'optimization.npy'; manifest_path = folder/'manifest.json'
    identity = {'partition_sha256': digest(ROOT/config['partition']), 'count': len(dataset),
                'transform': 'PIL RGB -> torchvision Resize((64,64)) bilinear -> uint8 CHW',
                'paths_sha256': hashlib.sha256('\n'.join(dataset.paths).encode()).hexdigest()}
    if path.exists() and manifest_path.exists():
        record = json.loads(manifest_path.read_text())
        assert record['identity'] == identity and record['cache_sha256'] == digest(path)
        return path, record
    def read(p):
        full = dataset.root/p
        file_hash = digest(full)
        with Image.open(full) as im:
            arr = np.array(transforms.functional.resize(im.convert('RGB'), [64, 64]), dtype=np.uint8).transpose(2, 0, 1)
        return arr, file_hash
    hashes = []
    arr = np.lib.format.open_memmap(str(path)+'.tmp', mode='w+', dtype=np.uint8, shape=(len(dataset), 3, 64, 64))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, (pixels, h) in enumerate(pool.map(read, dataset.paths)):
            arr[i] = pixels; hashes.append(h)
    arr.flush(); del arr
    Path(str(path)+'.tmp').replace(path)
    record = {'identity': identity, 'source_hashes': dict(zip(dataset.paths, hashes)), 'cache_sha256': digest(path)}
    atomic_json(manifest_path, record)
    return path, record


class DataStream:
    """Independent shuffled, lossless wraparound streams; explicit cursor/RNG."""
    def __init__(self, pixels, labels, seed):
        self.pixels = pixels
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.rng = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(labels), generator=self.rng)
        self.cursor = 0; self.epochs = 0; self.presentations = 0

    def take(self, count):
        ids = []
        while count:
            n = min(count, len(self.order)-self.cursor)
            ids.append(self.order[self.cursor:self.cursor+n]); self.cursor += n; count -= n
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.order), generator=self.rng)
                self.cursor = 0; self.epochs += 1
        ids = torch.cat(ids); self.presentations += len(ids)
        x = torch.from_numpy(np.array(self.pixels[ids.numpy()], copy=True)).to('cuda').float().div_(255).sub_(.5).div_(.5)
        y = self.labels[ids].to('cuda')
        flips = torch.rand(len(ids), device='cuda') < .5
        x = torch.where(flips[:, None, None, None], x.flip(-1), x)
        return x, y

    def state_dict(self):
        return {'rng': self.rng.get_state(), 'order': self.order, 'cursor': self.cursor,
                'epochs': self.epochs, 'presentations': self.presentations}

    def load_state_dict(self, state):
        self.rng.set_state(state['rng'])
        for name in ['order', 'cursor', 'epochs', 'presentations']:
            setattr(self, name, state[name])
