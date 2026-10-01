"""Official clean-FID/preprocessing on the fixed train-derived development set."""
import contextlib
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import to_pil_image
from .common import ROOT, PARTITION_SHA, atomic_json, digest
from .data import partition
from .engine import cpu_copy
from .flow import teacher_sample, ladder


def development_reference():
    folder = ROOT / 'artifacts/development_reference'
    folder.mkdir(parents=True, exist_ok=True)
    pixels = np.load(ROOT / 'artifacts/data_cache/development.npy', mmap_mode='r')
    if len(list(folder.glob('*.png'))) != 3020:
        # Match official prepare_reference_set float normalization and PNG quantization.
        for i, image in enumerate(pixels):
            normalized = torch.tensor(np.array(image, copy=True)).float().div(255).sub(.5).div(.5)
            to_pil_image(((normalized + 1) / 2).clamp(0, 1)).convert('RGB').save(folder / f'{i:04d}.png')
    assert len(list(folder.glob('*.png'))) == 3020
    return folder


def exported_state(weights, nfe):
    state = {'backbone.' + k: v.detach().cpu() for k, v in weights.items()}
    state['sampler_nfe'] = torch.tensor(nfe, dtype=torch.long)
    state['sampler_ladder'] = torch.tensor(list(ladder(nfe)) + [0.] * (4 - nfe))
    return {'state_dict': state}


class TeacherSampler(torch.nn.Module):
    def __init__(self, net, steps):
        super().__init__()
        self.net, self.steps = net, steps

    @torch.no_grad()
    def sample(self, shape, *, device='cuda', category=None, **kwargs):
        noise = kwargs.get('noises', [None])[0]
        if noise is None:
            noise = torch.randn(shape, device=device)
        with torch.autocast('cuda', dtype=torch.float16):
            return teacher_sample(self.net, noise, category, self.steps)


def diagnostics(folder, save_folder):
    """Class-level diversity, duplicates, saturation, nearest optimization images.

    Neighbors use pixels at 16x16; they are an audit, not a perceptual training loss.
    No extra learned network or classifier enters training or sampling.
    """
    files = sorted(Path(folder).glob('*.png'))
    arrays = np.stack([np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8) for p in files])
    generated_labels = np.array([int(p.stem.split('_')[0]) for p in files])
    reference = np.load(ROOT / 'artifacts/data_cache/development.npy', mmap_mode='r')
    _, ref_labels = partition('development')
    _, opt_labels = partition('optimization')
    opt_paths, _ = partition('optimization')
    optimization = np.load(ROOT / 'artifacts/data_cache/optimization.npy', mmap_mode='r')
    def features(x):
        # 4x4 area averaging: finite transparent pixel-space structural summary.
        return x.astype(np.float32).reshape(len(x), 16, 4, 16, 4, 3).mean((2, 4)).reshape(len(x), -1) / 255
    feature = features(arrays)
    real_feature = features(reference.transpose(0, 2, 3, 1))
    opt_feature = features(optimization.transpose(0, 2, 3, 1))
    classes = []
    nearest = {}
    for c in range(151):
        rows = np.where(generated_labels == c)[0]
        if not len(rows):
            continue
        real_rows = np.where(np.array(ref_labels) == c)[0]
        f, rf = feature[rows], real_feature[real_rows]
        distances = ((f[:, None] - f[None, :])**2).mean(2)
        upper = distances[np.triu_indices(len(f), 1)]
        real_dist = ((rf[:, None] - rf[None, :])**2).mean(2)
        real_mean = float(real_dist[np.triu_indices(len(rf), 1)].mean())
        mean = float(upper.mean()) if len(upper) else 0.
        unique = len({arrays[i].tobytes() for i in rows})
        candidates = np.where(np.array(opt_labels) == c)[0]
        ds = ((f[:, None] - opt_feature[candidates][None, :])**2).mean(2)
        chosen = candidates[ds.argmin(1)]
        for row, idx, mse in zip(rows, chosen, ds.min(1)):
            nearest[files[row].name] = {'optimization_path': opt_paths[idx], 'mse_16': float(mse),
                                        'optimization_index': int(idx)}
        classes.append({'class': c, 'unique': unique, 'samples': len(rows),
            'pairwise_mse_16': mean, 'reference_pairwise_mse_16': real_mean,
            'relative_diversity': mean / max(real_mean, 1e-12),
            'near_duplicate_pairs_mse_below_0.0001': int((upper < .0001).sum()),
            'potential_collapse': bool(mean / max(real_mean, 1e-12) < .05 or unique < len(rows) * .5),
            'saturation_fraction': float(((arrays[rows] == 0) | (arrays[rows] == 255)).mean())})
    record = {'classes': classes, 'potential_collapse_classes': [c['class'] for c in classes if c['potential_collapse']],
              'exact_unique_fraction': sum(c['unique'] for c in classes) / len(arrays),
              'median_relative_diversity': float(np.median([c['relative_diversity'] for c in classes])),
              'metric_limit': 'Pixel diagnostics flag duplication; recognition requires gallery review.'}
    save_folder = Path(save_folder)
    atomic_json(save_folder / 'diversity.json', record)
    atomic_json(save_folder / 'neighbors.json', nearest)
    mapping = json.loads((ROOT / 'data/pokemon-generation-one-22k/category_to_id.json').read_text())
    names = {v: k for k, v in mapping.items()}
    # Fixed classes spanning the label IDs, plus every flagged class in separate sheets.
    class_ids = list(range(0, 151, 10))
    if 150 not in class_ids: class_ids.append(150)
    for title, selected in [('fixed', class_ids), ('flagged', record['potential_collapse_classes'])]:
        for page in range(0, len(selected), 16):
            chosen_classes = selected[page:page + 16]
            canvas = Image.new('RGB', (128 + 64 * 10, 24 + 78 * len(chosen_classes)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((2, 2), 'Class | 2 development references | 6 fixed-noise generations | nearest optimization to first', fill='black')
            for r, c in enumerate(chosen_classes):
                y = 24 + r * 78
                draw.text((2, y + 15), names[c], fill='black')
                refs = np.where(np.array(ref_labels) == c)[0][:2]
                rows = np.where(generated_labels == c)[0][:6]
                panels = [reference[idx].transpose(1, 2, 0) for idx in refs] + [arrays[idx] for idx in rows]
                if len(rows):
                    idx = nearest[files[rows[0]].name]['optimization_index']
                    panels.append(optimization[idx].transpose(1, 2, 0))
                for j, panel in enumerate(panels):
                    canvas.paste(Image.fromarray(panel), (128 + 64 * j, y))
            canvas.save(save_folder / f'gallery_{title}_{page // 16}.png')
    return {k: v for k, v in record.items() if k != 'classes'}


@torch.no_grad()
def evaluate_model(model, output, seed, batch_size=16, diagnostics_enabled=True):
    from evaluate import generate_samples, compute_fid
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    hasher = hashlib.sha256()
    class AuditSampler(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def sample(self, shape, *, device, category):
            n = int(self.inner.sampler_nfe.item()) if hasattr(self.inner, 'sampler_nfe') else 1
            noises = [torch.randn(shape, device=device) for _ in range(n)]
            for tensor in [category, *noises]:
                hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            output = self.inner.sample(shape, device=device, category=category, noises=noises)
            if not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite generated image; evaluation rejected')
            return output
    # Fork protects all global training RNG states during selection/evaluation.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        generated = generate_samples(AuditSampler(model), output / 'images',
            torch.arange(151).repeat_interleave(20), batch_size=batch_size, device='cuda:0')
        assert len(list(generated.glob('*.png'))) == 3020
        fid = compute_fid(generated, development_reference(), device='cuda:0', batch_size=batch_size)
    info = {'fid': fid, 'seed': seed, 'samples': 3020, 'partition_sha256': PARTITION_SHA,
            'sampling_draws_sha256': hasher.hexdigest(),
            'official_evaluate_sha256': digest(ROOT / 'evaluate.py'),
            'seconds': time.monotonic() - start}
    if diagnostics_enabled:
        info['diversity'] = diagnostics(generated, output)
    atomic_json(output / 'metrics.json', info)
    return info
