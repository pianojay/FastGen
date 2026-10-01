"""Fixed development reference; unchanged official PNG conversion and clean FID."""
import collections
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from evaluate import generate_samples, compute_fid, prepare_reference_set
from model import Model
from .common import ROOT, atomic_json, digest, memory, seed_all, verify_protected
from .data import Images


def reference(config):
    folder = ROOT/'artifacts/reference_dev'
    dataset = Images(config, 'development')
    identity = {'partition_sha256': digest(ROOT/config['partition']), 'count': len(dataset),
                'source': 'fixed development indices into official TRAIN manifest',
                'official_validation_used': False,
                'source_hashes': {p: digest(dataset.root/p) for p in dataset.paths}}
    manifest = folder/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == identity
        assert len(list(folder.glob('*.png'))) == 3020
        return folder
    if folder.exists() and any(folder.glob('*.png')):
        raise RuntimeError('Incomplete reference; inspect before rebuilding')
    dm = SimpleNamespace(val_dataloader=lambda: DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False))
    prepare_reference_set(dm, folder)
    assert len(list(folder.glob('*.png'))) == 3020
    atomic_json(manifest, identity)
    return folder


def export_bundle(checkpoint, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint, directory/'one_nfe.pt')
    for name in ['model.py', 'requirements.txt']:
        shutil.copy2(ROOT/name, directory/name)
    for dirname, names in {'src': ['__init__.py', 'utils.py'],
                           'afm_fastgen': ['__init__.py', 'models.py', 'unet.py']}.items():
        (directory/dirname).mkdir()
        for name in names:
            shutil.copy2(ROOT/dirname/name, directory/dirname/name)
    shutil.copy2(ROOT/'scripts/check_export.py', directory/'check_export.py')
    shutil.copytree(ROOT/'licenses', directory/'licenses')
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    p = subprocess.run([sys.executable, 'check_export.py'], cwd=directory, env=env,
                       capture_output=True, text=True, timeout=180)
    (directory/'loader_check.log').write_text(p.stdout+p.stderr)
    if p.returncode:
        raise RuntimeError(p.stdout+p.stderr)
    atomic_json(directory/'files.json', {str(p.relative_to(directory)): digest(p)
                                        for p in directory.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    return json.loads((directory/'loader_check.json').read_text())


def evaluate(checkpoint, config, output_root, seeds, smoke=False):
    verify_protected()
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    ref = reference(config)
    model = Model.load_checkpoint(str(checkpoint), evaluate_mode='one_nfe', device='cuda')
    categories = torch.arange(151, dtype=torch.long).repeat_interleave(20)
    scores = []
    for seed in seeds:
        folder = output_root/f'seed{seed}'
        if (folder/'metrics.json').exists():
            rec = json.loads((folder/'metrics.json').read_text())
            assert rec['checkpoint_sha256'] == digest(checkpoint)
            scores.append(rec); continue
        if folder.exists():
            # Exact seeded evaluation recovery: preserve partial evidence and
            # regenerate the same declared pass, never select a different seed.
            folder.rename(output_root/f'seed{seed}_interrupted_{time.time_ns()}')
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        start = time.monotonic(); seed_all(seed)
        generate_samples(model, folder, categories, batch_size=config['eval_batch_size'])
        counts = collections.Counter(p.stem.split('_')[0] for p in folder.glob('*.png'))
        assert counts == {f'{i:04d}': 20 for i in range(151)}
        # clean-fid 0.1.35 hardcodes /tmp for this weight. Redirect storage only;
        # retain its exact extractor, preprocessing, official wrapper and metric.
        from cleanfid.inception_torchscript import InceptionV3W
        cache = ROOT/'artifacts/cache/clean-fid'; cache.mkdir(parents=True, exist_ok=True)
        def local_extractor(_path, **kwargs):
            return InceptionV3W(str(cache), **kwargs)
        with patch('cleanfid.features.InceptionV3W', local_extractor):
            score = compute_fid(folder, ref, batch_size=config['eval_batch_size'])
        rec = {'fid': score, 'metric': 'development_clean_fid', 'official_score': False,
               'seed': seed, 'nfe': 1, 'count': 3020, 'samples_per_class': 20,
               'checkpoint_sha256': digest(checkpoint), 'partition_sha256': digest(ROOT/config['partition']),
               'reference_manifest_sha256': digest(ref/'manifest.json'),
               'fid_extractor_sha256': digest(cache/'inception-2015-12-05.pt'),
               'eval_batch_size': config['eval_batch_size'], 'seconds': time.monotonic()-start,
               'smoke_only': smoke, 'noise_protocol': 'official generate_samples; CUDA randn each batch',
               'memory': memory()}
        atomic_json(folder/'metrics.json', rec); print(json.dumps(rec), flush=True); scores.append(rec)
    del model; gc.collect(); torch.cuda.empty_cache()
    atomic_json(output_root/'summary.json', scores)
    return scores
