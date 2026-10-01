"""Compact campaign tables and unaltered loss records with readable SVG trends."""
import csv
import html
import json
from pathlib import Path
import statistics
from flow_sid.common import ROOT

RUNS = ['T', 'S1', 'A1', 'S2', 'A2', 'S3', 'A3', 'S4', 'A4']


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def curve(rows, key, path):
    values = [(r['step'], r[key]) for r in rows if key in r]
    if not values: return
    width, height, pad = 800, 300, 42
    xs, ys = zip(*values)
    low, high = min(ys), max(ys)
    span = max(high - low, 1e-12)
    def xy(x, y):
        return f'{pad+(width-2*pad)*x/max(xs):.1f},{height-pad-(height-2*pad)*(y-low)/span:.1f}'
    stride = max(1, len(values) // 1600)
    points = ' '.join(xy(x,y) for x,y in values[::stride])
    smooth = []
    for i in range(0, len(values), max(1, len(values)//200)):
        chunk = values[max(0, i-49):i+1]
        smooth.append((values[i][0], statistics.mean(y for x,y in chunk)))
    trend = ' '.join(xy(x,y) for x,y in smooth)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="42" y="22" font-family="sans-serif" font-size="15">{html.escape(key)}: raw (gray), trailing 50-update mean (blue)</text>
<path d="M42 42 V258 H758" fill="none" stroke="black"/>
<polyline points="{points}" fill="none" stroke="#c0c0c0" stroke-width=".7"/>
<polyline points="{trend}" fill="none" stroke="#1668b2" stroke-width="2"/>
<text x="3" y="48" font-size="10">{high:.4g}</text><text x="3" y="258" font-size="10">{low:.4g}</text>
<text x="680" y="285" font-size="12">{max(xs)} updates</text></svg>'''
    path.write_text(svg)


def main():
    ledger = read(ROOT/'runs/campaign/allowance.json', {'runs':{}})
    text = ['# Flow SiD/SiDA campaign results', '',
            'Local development evaluation only: 3,020 train-derived reference images, 20 generated images per class. '
            'One teacher and one training realization per arm. Sampling confirmations are not training replications.', '',
            '| Run | State / updates | NFE | Selection FID | Confirmation FID | Flagged classes | Charged stage hours |',
            '|---|---|---:|---:|---:|---:|---:|']
    results = {}
    for run in RUNS:
        folder = ROOT/'runs/campaign'/run
        status = read(folder/'status.json', {})
        selected = read(folder/'selected.json', {})
        export = read(folder/'export.json', {})
        confirmation = export.get('confirmation',{})
        hours = ledger['runs'].get(run,{}).get('consumed_seconds',0)/3600
        fmt = lambda x: f'{x:.4f}' if isinstance(x,(float,int)) else '—'
        flags = selected.get('diversity',{}).get('potential_collapse_classes')
        text.append(f'| {run} | {status.get("phase","not started")} / {status.get("step",0)} | '
            f'{32 if run=="T" else run[1:]} | {fmt(selected.get("fid"))} | {fmt(confirmation.get("fid"))} | '
            f'{len(flags) if flags is not None else "—"} | {hours:.3f} |')
        results[run] = {'selected':selected, 'confirmation':confirmation, 'hours':hours}
        if (folder/'metrics.jsonl').exists():
            rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            for key in ['fm_loss','sid_loss','fake_fm_loss','d_bce_loss','g_gan_loss','g_grad_norm','fake_D_grad_norm']:
                curve(rows,key,folder/(key+'.svg'))
    text += ['', '| NFE | Paired selection ΔFID (A−S) | Paired confirmation ΔFID (A−S) |', '|---:|---:|---:|']
    for nfe in range(1,5):
        s,a = results['S'+str(nfe)],results['A'+str(nfe)]
        def difference(key):
            return f'{a[key]["fid"]-s[key]["fid"]:+.4f}' if 'fid' in a[key] and 'fid' in s[key] else '—'
        text.append(f'| {nfe} | {difference("selected")} | {difference("confirmation")} |')
    total = sum(r['hours'] for r in results.values())
    text += ['', f'Training/evaluation stage hours charged so far: {total:.3f}; teacher charged once. '
             'Setup, preprocessing and qualification are recorded separately in `runs/cost.jsonl`. '
             'Each stage includes checkpoint I/O, evaluation, failed work and replay, where applicable.', '',
             'Checkpoint/source/teacher identities, memory snapshots, raw per-update curves, galleries, '
             'all-class diversity and optimization-neighbor audits are retained in each `runs/campaign/<run>/` directory. '
             'Potential-collapse flags are pixel diagnostics and require visual interpretation.', '']
    (ROOT/'docs/RESULTS.md').write_text('\n'.join(text))
    print('\n'.join(text))


if __name__ == '__main__':
    main()
