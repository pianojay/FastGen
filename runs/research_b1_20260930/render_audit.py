"""Render the brief's fixed sample/reference panels without modifying source PNGs."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from dataset import category_from_filename, read_split
from soflow_fastgen.common import atomic_json, digest

run = Path(__file__).resolve().parent
p = json.loads((run / 'protocol.json').read_text())
selection = json.loads((run / 'selection.json').read_text())
selected = selection['decision']['selected']
stage = 'confirmation' if selected else 'selection'
seed = p[f'{stage}_seeds'][0]
rows = json.loads((run / f'{stage}.json').read_text())['results']
names = ['uniform', selected] if selected else [s['name'] for s in p['schedules']]
directories = {r['schedule']: Path(r['output_dir']) for r in rows if r['seed'] == seed}
mapping = json.loads((ROOT / 'data/pokemon-generation-one-22k/category_to_id.json').read_text())
labels = {value:key for key,value in mapping.items()}
config = json.loads((ROOT / p['baseline_config']).read_text())
partition = json.loads((ROOT / config['partition']).read_text())
paths = read_split(ROOT / 'data/pokemon-generation-one-22k/train_split.txt')
references = {category: [] for category in p['diagnostic_categories']}
for position, index in enumerate(partition['development_indices']):
    category = mapping[category_from_filename(paths[index])]
    if category in references:
        references[category].append(run / 'reference_dev' / f'{position:04d}.png')

output = run / 'audit'
output.mkdir(exist_ok=True)
manifest = {'seed': seed, 'stage': stage, 'schedule_names': names,
            'renderer_sha256': digest(__file__), 'panels': []}
for category in p['diagnostic_categories']:
    sources = [('Development references', references[category][:4])]
    sources.extend((name, [directories[name] / f'{category:04d}_{i:02d}.png'
                           for i in p['diagnostic_sample_indices']]) for name in names)
    fig, axes = plt.subplots(len(sources), 4, figsize=(8, 1.8 * len(sources) + .7), dpi=140)
    cells = []
    for row, (name, images) in enumerate(sources):
        for col, path in enumerate(images):
            with Image.open(path) as im:
                axes[row, col].imshow(im.convert('RGB'), interpolation='nearest')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_title(f'{name}\n{col if row else "ref " + str(col)}', fontsize=8)
            cells.append({'row': name, 'index': col, 'path': str(path.relative_to(ROOT))
                          if path.is_absolute() else str(path), 'sha256': digest(path)})
    fig.suptitle(f'{category}: {labels[category]} | NFE 4 | generated seed {seed}\n'
                 'Generated rows share noise; references are independent real images.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92))
    path = output / f'{category:03d}_{labels[category]}.png'
    fig.savefig(path)
    plt.close(fig)
    manifest['panels'].append({'category': category, 'name': labels[category],
                               'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'cells': cells})
atomic_json(output / 'manifest.json', manifest)
print(json.dumps({'panels': len(manifest['panels']), 'output': str(output), 'seed': seed}))
