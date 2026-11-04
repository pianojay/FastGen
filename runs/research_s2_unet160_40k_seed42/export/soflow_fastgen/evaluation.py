import gc
import json
import math
from pathlib import Path
import time

import torch
from torchvision.transforms.functional import to_pil_image
from evaluate import generate_samples, compute_fid
from .common import atomic_json, digest, seed_all
from .data import datasets
from .models import Soflow
from .sampling import sample


class SamplingAdapter(torch.nn.Module):
    def __init__(self, backbone, nfe):
        super().__init__()
        self.backbone, self.nfe = backbone, nfe

    def sample(self, shape, *, device='cuda', category=None):
        return sample(self.backbone, shape, device=device, category=category, nfe=self.nfe)


def prepare_reference(config):
    _, development, _ = datasets(config)
    fingerprint = digest(config['partition'])
    path = Path(config['run_dir']) / 'reference_dev'
    metadata = path / 'manifest.json'
    if metadata.exists():
        record = json.loads(metadata.read_text())
        if record['partition_sha256'] != fingerprint or len(list(path.glob('*.png'))) != len(development):
            raise ValueError('Development reference cache mismatch')
        return path
    path.mkdir(parents=True, exist_ok=True)
    if any(path.glob('*.png')):
        raise ValueError('Incomplete reference directory; preserve it for inspection and use a fresh run')
    for i, (image, _) in enumerate(development):
        to_pil_image(((image + 1)/2).clamp(0,1)).convert('RGB').save(path/f'{i:04d}.png')
    atomic_json(metadata, {'partition_sha256':fingerprint, 'count':len(development),
                          'source':'held-out official TRAIN images; unaugmented'})
    return path


def evaluate_checkpoint(checkpoint, config, nfes, seeds, budget=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    backbone = Soflow(**state['config']['architecture']).cuda().eval().requires_grad_(False)
    backbone.load_state_dict(state['ema'])
    del state
    reference = prepare_reference(config)
    root = Path(config['run_dir']) / 'evaluations' / Path(checkpoint).stem
    categories = torch.arange(151, dtype=torch.long).repeat_interleave(20)
    results = []
    checkpoint_hash = digest(checkpoint)
    for nfe in nfes:
        for seed in seeds:
            output = root / f'nfe{nfe}_seed{seed}'
            report = output / 'metrics.json'
            if report.exists():
                record = json.loads(report.read_text())
                if (record['checkpoint_sha256'] != checkpoint_hash or
                        record['partition_sha256'] != digest(config['partition'])):
                    raise ValueError('Evaluation checkpoint changed')
                results.append(record)
                continue
            if budget and budget.remaining() < 900:
                return results
            if output.exists() and any(output.glob('*.png')):
                # Keep interrupted work for inspection; regenerate from the recorded seed.
                interrupted = root/'interrupted'
                interrupted.mkdir(exist_ok=True)
                output.rename(interrupted/f'{output.name}_{time.time_ns()}')
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            seed_all(seed)
            generate_samples(SamplingAdapter(backbone,nfe), output, categories,
                             batch_size=config['eval_batch_size'])
            if len(list(output.glob('*.png'))) != 3020:
                raise RuntimeError('Incorrect evaluation image count')
            score = compute_fid(output, reference, batch_size=config['eval_batch_size'])
            if not math.isfinite(score):
                raise FloatingPointError('Non-finite FID')
            record = {'metric':'development_clean_fid', 'fid':score, 'nfe':nfe, 'seed':seed,
                'count':3020, 'checkpoint_sha256':checkpoint_hash,
                'partition_sha256':digest(config['partition']), 'official_score':False,
                'seconds':time.monotonic()-started,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            atomic_json(report, record)
            results.append(record)
            print(json.dumps(record), flush=True)
            if budget: budget.save()
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(root/'summary.json', results)
    return results
