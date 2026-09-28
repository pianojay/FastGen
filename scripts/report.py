"""Render local progress; never turns an unfinished run into a success claim."""
import argparse
import json
from pathlib import Path
import html


def report(run_dir):
    run=Path(run_dir)
    metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()] if (run/'metrics.jsonl').exists() else []
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'state':'not_started'}
    evaluations=[]
    for path in sorted((run/'evaluations').glob('*/nfe*/metrics.json')):
        evaluations.append(json.loads(path.read_text()))
    content=['# Pokémon SoFlow local experiment', '',
        'Environment: local TITAN RTX; KCloud qualification and fresh training are separate.',
        '', 'Status: '+json.dumps(status), '',f'Logged updates: {len(metrics)}',
        '', 'Development FID (official training-derived holdout; not TA scores):',
        '', '| Checkpoint | NFE | Seed | FID |','|---|---:|---:|---:|']
    for path in sorted((run/'evaluations').glob('*/nfe*/metrics.json')):
        row=json.loads(path.read_text())
        content.append(f'| {path.parent.parent.name} | {row["nfe"]} | {row["seed"]} | {row["fid"]:.4f} |')
    if metrics:
        content += ['',f'Peak training allocation: {max(x["allocated_bytes"] for x in metrics)/1e9:.3f} GB.',
                    f'Peak parent RSS: {max(x["host_peak_rss_bytes"] for x in metrics)/1e9:.3f} GB.']
        memory=[row for row in metrics if 'process_tree_pss_bytes' in row]
        if memory:
            content += [f'Peak sampled process-tree PSS (including loader workers): {max(x["process_tree_pss_bytes"] for x in memory)/1e9:.3f} GB.']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(1,3,figsize=(13,3.5))
        for ax,key,label in zip(axes,['consistency_mse','grad_norm','seconds'],['Consistency MSE','Gradient norm','Seconds/update']):
            ax.plot([m['step'] for m in metrics],[m[key] for m in metrics],linewidth=.7)
            ax.set(xlabel='Optimizer update',ylabel=label)
        fig.tight_layout();fig.savefig(run/'curves.png',dpi=150);plt.close(fig)
        content+=['','![Training curves](curves.png)']
    samples=sorted((run/'samples').glob('*/nfe*.png'))
    content+=['','## Latest labeled grids']
    for path in samples[-4:]:content+=['',f'![{path.parent.name}, {path.stem}]({path.relative_to(run)})']
    (run/'REPORT.md').write_text('\n'.join(content)+'\n')
    images=['curves.png']+[str(p.relative_to(run)) for p in samples[-4:]]
    (run/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Pokémon SoFlow</title>'
        '<style>body{font-family:system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Pokémon SoFlow — local development</h1>'+
        ('<p><a href="gallery.html">Browse the sample gallery →</a></p>' if (run/'gallery.html').exists() else '')+
        '<pre>'+html.escape('\n'.join(content[:len(content)-len(samples[-4:])*2]))+'</pre>'+
        ''.join(f'<p><img src="{html.escape(p)}"></p>' for p in images))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='runs/pokemon_unet128_seed42');a=p.parse_args();report(a.run_dir)
