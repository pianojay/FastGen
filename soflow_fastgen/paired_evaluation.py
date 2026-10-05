"""Paired evaluation primitives retained from B1 commit 94f2915."""


import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import torch

from .common import atomic_json, digest, process_tree_memory, resources
from .sampling import query_times, sample

NOISE_PROTOCOL = 'cpu-per-draw-stream-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def environment_identity():
    return {'python': sys.version, 'executable': sys.executable,
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'torchvision', 'clean-fid', 'numpy', 'Pillow')}}


class PairedAdapter(torch.nn.Module):
    """Independent noise streams per call preserve the same draws across schedules.

    Draws are made on CPU, then passed explicitly to the sampler. Batch size and
    category order are fixed by the protocol. Each stream hashes its actual draws.
    """
    def __init__(self, backbone, nfe, intermediate_times, seed):
        super().__init__()
        self.backbone, self.nfe = backbone, nfe
        self.intermediate_times = intermediate_times
        self.generators, self.hashes = [], []
        self.network_calls = 0
        self.batches = 0
        for slot in range(nfe):
            key = f'{NOISE_PROTOCOL}:{seed}:{slot}'.encode()
            value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63)
            self.generators.append(torch.Generator(device='cpu').manual_seed(value))
            self.hashes.append(hashlib.sha256())

    def sample(self, shape, *, device='cuda', category=None):
        draws = []
        for generator, hasher in zip(self.generators, self.hashes):
            draw = torch.randn(shape, generator=generator, dtype=torch.float32, device='cpu')
            hasher.update(draw.numpy().tobytes())
            draws.append(draw.to(device))
        calls = []
        hook = self.backbone.network.register_forward_pre_hook(lambda module, args: calls.append(1))
        try:
            result = sample(self.backbone, shape, device=device, category=category,
                            nfe=self.nfe, intermediate_times=self.intermediate_times, noise=draws)
        finally:
            hook.remove()
        if len(calls) != self.nfe:
            raise RuntimeError(f'Expected {self.nfe} actual network calls, got {len(calls)}')
        self.network_calls += len(calls)
        self.batches += 1
        return result

    def noise_sha256(self):
        return [hasher.hexdigest() for hasher in self.hashes]


def evaluation_identity(protocol, schedule, seed, source_hash, reference_hash):
    return {
        'protocol_sha256': fingerprint(protocol), 'source_sha256': source_hash,
        'environment_sha256': fingerprint(environment_identity()),
        'checkpoint_sha256': protocol['checkpoint_sha256'], 'weight_source': protocol.get('weight_source', 'ema'),
        'partition_sha256': protocol['partition_sha256'], 'reference_sha256': reference_hash,
        'nfe': protocol['nfe'], 'query_times': list(query_times(protocol['nfe'], schedule['intermediate_times'])),
        'seed': seed, 'noise_protocol': NOISE_PROTOCOL, 'batch_size': protocol['eval_batch_size'],
        'category_order': '0..150, each repeated 20 times', 'count': 3020,
        'metric': 'development_clean_fid', 'official_score': False,
    }


def image_manifest(directory):
    return {p.name: digest(p) for p in sorted(Path(directory).glob('*.png'))}


def evaluate_schedule(backbone, protocol, schedule, seed, reference, source_hash, reference_hash):
    from evaluate import generate_samples, compute_fid

    identity = evaluation_identity(protocol, schedule, seed, source_hash, reference_hash)
    root = Path(protocol['run_dir']) / 'evaluations'
    output = root / f'{schedule["name"]}_seed{seed}_{fingerprint(identity)[:16]}'
    report = output / 'metrics.json'
    expected = {f'{category:04d}_{index:02d}.png' for category in range(151) for index in range(20)}
    if report.exists():
        record = json.loads(report.read_text())
        images = image_manifest(output)
        if (record['identity'] != identity or not math.isfinite(record['fid'])
                or set(images) != expected or fingerprint(images) != record['images_sha256']
                or record['network_calls'] != protocol['nfe'] * math.ceil(3020/protocol['eval_batch_size'])):
            raise ValueError('Evaluation cache identity or image integrity changed')
        return record
    if output.exists():
        interrupted = root / 'interrupted'
        interrupted.mkdir(parents=True, exist_ok=True)
        output.rename(interrupted / f'{output.name}_{time.time_ns()}')
    output.mkdir(parents=True)
    atomic_json(output / 'identity.json', identity)
    adapter = PairedAdapter(backbone, protocol['nfe'], schedule['intermediate_times'], seed)
    categories = torch.arange(151, dtype=torch.long).repeat_interleave(20)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    generate_samples(adapter, output, categories, batch_size=protocol['eval_batch_size'])
    manifest = image_manifest(output)
    if set(manifest) != expected:
        raise RuntimeError('Expected exactly 3,020 images, 20 per category')
    score = compute_fid(output, reference, batch_size=protocol['eval_batch_size'])
    if not math.isfinite(score):
        raise FloatingPointError('Non-finite FID')
    if torch.cuda.max_memory_allocated() >= 20_000_000_000:
        raise RuntimeError('Evaluation exceeded the allocation limit')
    record = {'identity': identity, 'schedule': schedule['name'], 'seed': seed, 'fid': score,
              'noise_sha256': adapter.noise_sha256(), 'images_sha256': fingerprint(manifest),
              'network_calls': adapter.network_calls, 'batches': adapter.batches,
              'nfe': protocol['nfe'], 'seconds': time.monotonic()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'output_dir': str(output),
              **resources(), **process_tree_memory()}
    atomic_json(output / 'images.json', manifest)
    atomic_json(report, record)
    print(json.dumps(record), flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return record
