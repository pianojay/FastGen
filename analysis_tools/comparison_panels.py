"""Compose existing evaluation pixels; no model calls or generated edits.

Broad review uses classes 0..150 and sample indices 00/01 without selection.
Three larger example panels use fixed classes Eevee, Gyarados and Metapod.
They were chosen before the first adversarial result to include recognizable
and structurally difficult classes already visible in the SiD/teacher galleries.
"""
import argparse
import json
import math
from PIL import Image, ImageDraw
from flow_sid.common import ROOT, StageCost, atomic_json, digest
from flow_sid.data import partition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--through-nfe', type=int, choices=[1, 2, 3, 4], required=True)
    args = parser.parse_args()
    names = {v: k for k, v in json.loads((ROOT / 'data/pokemon-generation-one-22k/category_to_id.json').read_text()).items()}
    _, labels = partition('development')
    refs = {c: [i for i, label in enumerate(labels) if label == c] for c in range(151)}
    selected = {}
    manifest = {'purpose': __doc__, 'through_nfe': args.through_nfe, 'sources': {}, 'panels': []}
    destination = ROOT / 'artifacts/report/comparisons'
    destination.mkdir(parents=True, exist_ok=True)
    runs = ['T'] + [m + str(n) for n in range(1, args.through_nfe + 1) for m in ['S', 'A']]
    for run in runs:
        folder = ROOT / 'runs/campaign' / run
        assert json.loads((folder / 'status.json').read_text())['phase'] == 'complete'
        selected[run] = json.loads((folder / 'selected.json').read_text())
    def pixels(path):
        relative = str(path.relative_to(ROOT))
        manifest['sources'][relative] = digest(path)
        with Image.open(path) as opened:
            im = opened.convert('RGB')
            assert im.size == (64, 64)
            return im.copy()
    def sample(run, c, i):
        return ROOT / selected[run]['gallery_directory'] / 'images' / f'{c:04d}_{i:02d}.png'
    def reference(c, i=0):
        return ROOT / 'artifacts/development_reference' / f'{refs[c][i]:04d}.png'
    for n in range(1, args.through_nfe + 1):
        s, a = 'S'+str(n), 'A'+str(n)
        assert selected[s]['sampling_draws_sha256'] == selected[a]['sampling_draws_sha256']
        for start in range(0, 151, 26):
            classes = list(range(start, min(start+26, 151)))
            canvas = Image.new('RGB', (692, 40 + 86 * math.ceil(len(classes)/2)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 4), f'{n} NFE, selected checkpoints: reference | {s} samples 00,01 | {a} samples 00,01', fill='black')
            draw.text((4, 19), 'All classes in ID order; original 64x64 pixels. S/A noise matches within this pair.', fill='black')
            for index, c in enumerate(classes):
                col, row = index % 2, index // 2
                x, y = 4 + col * 346, 40 + row * 86
                draw.text((x, y), f'{c}: {names[c]}', fill='black')
                sources = [reference(c), sample(s,c,0), sample(s,c,1), sample(a,c,0), sample(a,c,1)]
                for j, source in enumerate(sources):
                    canvas.paste(pixels(source), (x+66*j, y+14))
            path = destination / f'broad_pair_{n}_{start//26}.png'
            canvas.save(path)
            manifest['panels'].append(str(path.relative_to(ROOT)))
    for c in [30, 50, 80]:
        assert names[c] == {30:'Eevee', 50:'Gyarados', 80:'Metapod'}[c]
        canvas = Image.new('RGB', (534, 42+80*(len(runs)+1)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f'{names[c]}: 6 fixed-index samples for every selected model', fill='black')
        draw.text((4, 19), 'Same noise within each SiD/SiDA pair; noise differs across NFE budgets.', fill='black')
        draw.text((4, 55), 'Development refs', fill='black')
        for j in range(6):
            canvas.paste(pixels(reference(c,j)), (136+66*j,42))
        for row, run in enumerate(runs):
            y=122+80*row
            draw.text((4,y+15), f'{run}, step {selected[run]["step"]}', fill='black')
            for j in range(6):
                canvas.paste(pixels(sample(run,c,j)), (136+66*j,y))
        path = destination / f'class_{names[c]}_through_nfe{args.through_nfe}.png'
        canvas.save(path)
        manifest['panels'].append(str(path.relative_to(ROOT)))
    atomic_json(destination / f'manifest_through_nfe{args.through_nfe}.json',manifest)
    print(json.dumps({'composed_existing_samples': True, 'panels':manifest['panels']}))


if __name__ == '__main__':
    with StageCost('analysis_comparison_panels'):
        main()
