"""Render local progress; never turns an unfinished run into a success claim."""
import argparse
import json
from pathlib import Path
import html


def render_curves(metrics, run, smooth_window):
    """Show every raw observation alongside robust, centered local summaries."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    window = min(smooth_window, len(metrics))
    if window % 2 == 0:
        window -= 1
    steps = np.array([m['step'] for m in metrics])
    series = []
    for key in ('consistency_mse', 'grad_norm', 'seconds'):
        values = np.array([m[key] for m in metrics], dtype=float)
        padded = np.pad(values, window // 2, constant_values=np.nan)
        local = np.lib.stride_tricks.sliding_window_view(padded, window)
        lower, median, upper = np.nanquantile(local, [.1, .5, .9], axis=1)
        series.append((values, lower, median, upper, np.nanmean(local, axis=1)))

    logarithmic = bool(np.all(series[0][0] > 0))
    for filename, log_mse in [('curves.png', logarithmic), ('curves_linear.png', False)]:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
        for index, (ax, label, values) in enumerate(zip(axes,
                ['Consistency MSE', 'Gradient norm', 'Seconds/update'], series)):
            raw, lower, median, upper, mean = values
            ax.plot(steps, raw, color='#718096', linewidth=.45, alpha=.32,
                    label='Raw updates', zorder=1)
            ax.fill_between(steps, lower, upper, color='#268a75', alpha=.20,
                    linewidth=0, label='Local 10th–90th percentiles', zorder=2)
            ax.plot(steps, median, color='#08644f', linewidth=2,
                    label=f'Centered rolling median ({window:,} records)', zorder=4)
            if index == 0:
                ax.plot(steps, mean, color='#b45125', linewidth=1.3,
                        linestyle='--', label='Rolling mean (same window)', zorder=3)
                if log_mse:
                    ax.set_yscale('log')
            scale = 'logarithmic' if index == 0 and log_mse else 'linear'
            ax.set(title=f'{label} · {scale} scale', xlabel='Optimizer update', ylabel=label)
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.grid(True, which='major', color='#dce3e1', linewidth=.6)
            ax.set_axisbelow(True)
            if len(steps) > 1:
                ax.set_xlim(steps[0], steps[-1])
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(loc='upper right', fontsize=8, framealpha=.95)
        fig.savefig(run / filename, dpi=150)
        plt.close(fig)

    scale_note = ('Consistency MSE uses a logarithmic y-axis so both its typical level and large spikes remain visible.'
                  if logarithmic else 'Consistency MSE uses a linear y-axis because nonpositive values are present.')
    return (f'Trends use a centered {window}-record rolling window (one record per logged optimizer update); '
            'the window shortens at the edges. Green: median; shaded band: local 10th–90th percentiles '
            '(observed spread, not a confidence interval). Gray: every raw value, with no clipping or outlier removal. '
            'The dashed orange MSE line is the rolling mean, which is more sensitive to spikes. ' + scale_note +
            ' The alternative linear-scale figure retains the full raw range. '
            'Consistency MSE is a training diagnostic under a changing consistency-gap curriculum, '
            'not a fixed validation loss; an increase alone does not establish worsening generation quality.')


def report(run_dir, smooth_window=201):
    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError('smooth_window must be a positive odd integer')
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
        plot_note = render_curves(metrics, run, smooth_window)
        content+=['', '## Training curves', '', plot_note,
                  '', '![Training curves](curves.png)',
                  '', '[Open full-range linear-scale curves](curves_linear.png)']
    summary = '\n'.join(content)
    samples=sorted((run/'samples').glob('*/nfe*.png'))
    content+=['','## Latest labeled grids']
    for path in samples[-4:]:content+=['',f'![{path.parent.name}, {path.stem}]({path.relative_to(run)})']
    (run/'REPORT.md').write_text('\n'.join(content)+'\n')
    images=[str(p.relative_to(run)) for p in samples[-4:]]
    (run/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Pokémon SoFlow</title>'
        '<style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:0 16px}img{max-width:100%;height:auto}pre{white-space:pre-wrap}figure{margin:24px 0}figcaption{color:#42554d;font-size:14px;line-height:1.6}</style>'
        '<h1>Pokémon SoFlow — local development</h1>'+
        ('<p><a href="gallery.html">Browse the sample gallery →</a></p>' if (run/'gallery.html').exists() else '')+
        '<pre>'+html.escape(summary.split('\n## Training curves')[0])+'</pre>'+
        ('<h2>Training curves</h2><figure><img src="curves.png" alt="Three stacked training plots with raw values, rolling medians and percentile bands">'
         '<figcaption>'+html.escape(plot_note)+'</figcaption></figure>'
         '<p><a href="curves_linear.png">Open full-range linear-scale curves</a></p>' if metrics else '')+
        '<h2>Latest labeled grids</h2>'+
        ''.join(f'<p><img src="{html.escape(p)}"></p>' for p in images))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir', default='runs/pokemon_unet128_seed42')
    p.add_argument('--smooth-window', type=int, default=201,
                   help='Positive odd rolling-window length in logged updates (default: 201)')
    a=p.parse_args()
    report(a.run_dir, a.smooth_window)
