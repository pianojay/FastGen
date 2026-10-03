"""Package minimal inference sources and independently load each actual export."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import torch
from flow_sid.common import ROOT, StageCost, atomic_json, digest, object_hash


SOURCES = ['model.py', 'src/__init__.py', 'src/utils.py', 'flow_sid/__init__.py',
           'flow_sid/unet.py', 'flow_sid/networks.py', 'flow_sid/flow.py']


def copy_sources(destination):
    for relative in SOURCES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--through-nfe', type=int, choices=[2, 4], required=True)
    args = parser.parse_args()
    source_hash = object_hash({p: digest(ROOT / p) for p in SOURCES})
    prior_path = ROOT / 'resources/export_verification_through_nfe2.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() and args.through_nfe == 4 else {}
    output = {'purpose': 'CPU loading and exact call-count test of selected actual exports',
              'inference_sources_sha256': source_hash, 'runs': {}}
    with StageCost(f'export_verification_through_nfe{args.through_nfe}'):
        for nfe in range(1, args.through_nfe + 1):
            for method in ['S', 'A']:
                run = method + str(nfe)
                folder = ROOT / 'runs/campaign' / run
                status = json.loads((folder / 'status.json').read_text())
                assert status['phase'] == 'complete'
                export = json.loads((folder / 'export.json').read_text())
                selected = json.loads((folder / 'selected.json').read_text())
                checkpoint = ROOT / export['path']
                assert digest(checkpoint) == export['sha256']
                previous = prior.get('runs', {}).get(run)
                if (previous and prior.get('inference_sources_sha256') == source_hash
                        and previous['checkpoint_sha256'] == export['sha256']
                        and previous.get('selected_checkpoint_sha256') == selected['checkpoint_sha256']
                        and previous.get('selected_weights_bitwise_equal')):
                    output['runs'][run] = {**previous, 'verification_reused_from': str(prior_path.relative_to(ROOT))}
                    continue
                assert digest(ROOT / selected['checkpoint']) == selected['checkpoint_sha256']
                weights = torch.load(ROOT / selected['checkpoint'], map_location='cpu', weights_only=False)['weights']
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                assert set(payload) == {'state_dict'}
                state = payload['state_dict']
                assert set(state) == {'backbone.' + k for k in weights} | {'sampler_nfe', 'sampler_ladder'}
                for key, value in weights.items():
                    exported = state['backbone.' + key]
                    assert exported.dtype == value.dtype and torch.equal(exported, value), (run, key)
                del weights, payload, state, value, exported
                with tempfile.TemporaryDirectory(prefix=run + '_isolated_', dir=ROOT / 'artifacts') as temp:
                    temp = Path(temp)
                    copy_sources(temp)
                    shutil.copyfile(ROOT / 'scripts/check_export.py', temp / 'check_export.py')
                    shutil.copyfile(checkpoint, temp / 'student.ckpt')
                    env = os.environ.copy()
                    env.update({'PYTHONPATH': str(temp), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1',
                                'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
                    result = subprocess.run([sys.executable, 'check_export.py', 'student.ckpt', '--nfe', str(nfe)],
                                            cwd=temp, env=env, capture_output=True, text=True, timeout=180)
                    assert result.returncode == 0, (run, result.stderr)
                    record = json.loads(result.stdout)
                    record['checkpoint_sha256'] = export['sha256']
                    record['selected_checkpoint_sha256'] = selected['checkpoint_sha256']
                    record['selected_weights_bitwise_equal'] = True
                    output['runs'][run] = record
                    print(json.dumps({'run': run, **record}), flush=True)
        destination = ROOT / 'artifacts/exports/inference'
        destination.mkdir(parents=True, exist_ok=True)
        copy_sources(destination)
        shutil.copyfile(ROOT / 'data/pokemon-generation-one-22k/category_to_id.json', destination / 'category_to_id.json')
        manifest = {'files': {p: digest(ROOT / p) for p in SOURCES},
                    'exports': {run: {'path': '../' + run + '.ckpt', 'sha256': r['checkpoint_sha256'],
                                      'parameters': r['parameters'], 'nfe': r['nfe']}
                                for run, r in output['runs'].items()},
                    'required_runtime': {'python': '3.10', 'torch': '2.6.0', 'numpy': '2.2.6'},
                    'training_networks_in_export': False}
        atomic_json(destination / 'manifest.json', manifest)
        (destination / 'README.md').write_text(
            '# Selected SiD/SiDA students\n\n'
            'Run Python from this directory. Each adjacent checkpoint contains one selected student '
            'and its NFE/ladder buffers. No teacher or fake-score network is required. '
            'Tested runtime: Python 3.10, PyTorch 2.6.0 and NumPy 2.2.6.\n\n'
            '```python\nimport torch\nfrom model import Model\n'
            'model = Model.load_checkpoint("../S2.ckpt", evaluate_mode="few_nfe", device="cpu")\n'
            'images = model.sample((1, 3, 64, 64), category=torch.tensor([25]), device="cpu")\n```\n\n'
            'Use `one_nfe` for S1/A1; use `few_nfe` for S2/A2/S3/A3/S4/A4. '
            'The checkpoint fixes the exact call count. Output is RGB in the model’s normalized '
            'pixel coordinates; the official image writer applies `(images + 1) / 2` and clips to [0,1]. '
            'Category IDs are listed in `category_to_id.json`. These are local development results, '
            'with one training realization per arm. NVIDIA notices remain in the copied U-Net source.\n')
        atomic_json(ROOT / f'resources/export_verification_through_nfe{args.through_nfe}.json', output)


if __name__ == '__main__':
    main()
