"""Local Pillow panels and lightweight SVG time series, without extra dependencies."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms.functional import to_pil_image
from .common import ROOT, atomic_json
from .data import Images

CLASSES = ['Bulbasaur', 'Charmander', 'Squirtle', 'Pikachu', 'Eevee', 'Jigglypuff',
           'Psyduck', 'Gengar', 'Snorlax', 'Magikarp', 'Dragonite', 'Mewtwo']


def font(size=17):
    path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


@torch.no_grad()
def fixed_panel(generator, config, output, step):
    mapping = json.loads((ROOT/'data/pokemon-generation-one-22k/category_to_id.json').read_text())
    classes = [name for name in CLASSES if name in mapping]
    labels = torch.tensor([mapping[name] for name in classes for _ in range(8)], device='cuda')
    rng = torch.Generator(device='cuda').manual_seed(941)
    noise = torch.randn((len(labels), 3, 64, 64), generator=rng, device='cuda')
    pixels = []
    for start in range(0, len(labels), 16):
        image = generator(noise[start:start+16], labels[start:start+16]).clamp(-1, 1).cpu()
        pixels.extend(to_pil_image((x+1)/2).convert('RGB') for x in image)
    dataset = Images(config, 'optimization')
    canvas = Image.new('RGB', (1150, 50+100*len(classes)), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'AFM EMA step {step:,} | two optimization references + eight fixed-noise samples', fill='black', font=font())
    meta = []
    for row, name in enumerate(classes):
        y = 48+row*100; draw.text((8, y+28), name, fill='black', font=font(14))
        ids = [i for i, label in enumerate(dataset.labels) if label == mapping[name]][:2]
        for col, index in enumerate(ids):
            x, _ = dataset[index]
            im = to_pil_image((x+1)/2).convert('RGB').resize((88, 88))
            canvas.paste(im, (130+col*92, y))
        for col in range(8):
            canvas.paste(pixels[row*8+col].resize((88, 88)), (350+col*98, y))
        meta.append({'class': name, 'category': mapping[name], 'reference_paths': [dataset.paths[i] for i in ids]})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output)
    atomic_json(output.with_suffix('.json'), {'seed': 941, 'step': step, 'classes': meta, 'samples_per_class': 8})


def nearest_gallery(config, generated_dir, output):
    dataset = Images(config, 'optimization')
    mapping = json.loads((ROOT/'data/pokemon-generation-one-22k/category_to_id.json').read_text())
    pixels = np.load(ROOT/'artifacts/data_cache/optimization.npy', mmap_mode='r')
    classes = [x for x in CLASSES if x in mapping]
    canvas = Image.new('RGB', (1050, 55+len(classes)*115), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'Four generated images per class, each beside its nearest same-class optimization image (pixel MSE)', font=font(15), fill='black')
    rows = []
    for row, name in enumerate(classes):
        label = mapping[name]; ids = np.where(np.array(dataset.labels) == label)[0]
        train = np.array(pixels[ids], dtype=np.float32)/255
        y = 52+row*115; draw.text((8, y+35), name, font=font(14), fill='black')
        row_record = {'class': name, 'examples': []}
        for j in range(4):
            path = Path(generated_dir)/f'{label:04d}_{j:02d}.png'
            gen = Image.open(path).convert('RGB'); a = np.array(gen, dtype=np.float32).transpose(2, 0, 1)/255
            distances = np.mean((train-a[None])**2, axis=(1, 2, 3))
            nearest = int(np.argmin(distances)); index = int(ids[nearest])
            real = Image.fromarray(np.array(pixels[index]).transpose(1, 2, 0))
            x = 130+j*230; canvas.paste(gen.resize((92, 92)), (x, y)); canvas.paste(real.resize((92, 92)), (x+100, y))
            draw.text((x, y+94), f'gen | train MSE {distances[nearest]:.4f}', font=font(11), fill='black')
            row_record['examples'].append({'generated': str(path), 'nearest_optimization_path': dataset.paths[index], 'pixel_mse': float(distances[nearest])})
        rows.append(row_record)
    output = Path(output); canvas.save(output)
    atomic_json(output.with_suffix('.json'), {'diagnostic': 'same-class pixel MSE only; not perceptual or proof against memorization', 'rows': rows})


def loss_plots(metrics, output):
    rows = [json.loads(line) for line in Path(metrics).read_text().splitlines()]
    panels = [(['g_adv', 'g_ot'], 'Generator: adversarial and weighted OT'),
              (['d_adv', 'd_r1', 'd_r2', 'd_center'], 'Discriminator: adversarial and weighted penalties'),
              (['d_real_logit', 'd_fake_logit', 'g_fake_logit'], 'Discriminator logits'),
              (['g_gradient_norm', 'd_gradient_norm'], 'Gradient norms before clipping'),
              (['cycle_seconds', 'd_seconds', 'g_seconds'], 'Update timing (TITAN RTX seconds)')]
    colors = ['#1264a3', '#c94916', '#287a43', '#934aad']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1500" viewBox="0 0 1080 1500"><rect width="1080" height="1500" fill="white"/>']
    max_step = max(r['step'] for r in rows)
    # Plot binned raw min/max and moving mean; full unaggregated JSONL retained.
    for panel, (keys, title) in enumerate(panels):
        top = panel*300+30; low = min(0., min(r[k] for r in rows for k in keys)); high = max(r[k] for r in rows for k in keys)
        high = max(high, low+1e-8)
        svg.append(f'<text x="70" y="{top}" font-family="sans-serif" font-size="18">{html.escape(title)}</text>')
        svg.append(f'<path d="M70 {top+15}V{top+230}H1040" fill="none" stroke="#777"/>')
        for ki, key in enumerate(keys):
            pts=[]; bars=[]; bins=max(1, len(rows)//800)
            for i in range(0, len(rows), bins):
                group = rows[i:i+bins]; x=70+970*group[-1]['step']/max_step
                values = [r[key] for r in group]; sy=lambda v: top+230-210*(v-low)/(high-low)
                bars.append(f'M{x:.1f} {sy(min(values)):.1f}V{sy(max(values)):.1f}')
                start=max(0, i-99); trend=sum(r[key] for r in rows[start:i+len(group)])/(i+len(group)-start)
                pts.append(f'{x:.1f},{sy(trend):.1f}')
            color=colors[ki]
            svg.append(f'<path d="{" ".join(bars)}" stroke="{color}" opacity="0.20"/>')
            svg.append(f'<polyline points="{" ".join(pts)}" stroke="{color}" stroke-width="1.6" fill="none"/>')
            svg.append(f'<text x="{90+ki*230}" y="{top+255}" font-family="sans-serif" font-size="12" fill="{color}">{key}</text>')
        for value in [low, (high+low)/2, high]:
            y=top+230-210*(value-low)/(high-low)
            svg.append(f'<text x="4" y="{y:.1f}" font-family="sans-serif" font-size="11">{value:.3g}</text>')
        svg.append(f'<text x="780" y="{top+280}" font-family="sans-serif" font-size="12">0 to {max_step:,} G updates; raw range + trend</text>')
    svg.append('</svg>'); Path(output).write_text('\n'.join(svg))
