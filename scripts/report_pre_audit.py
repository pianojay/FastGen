"""Render completed fixed-audit evidence; never trains or selects a model."""
import argparse
import json
from pathlib import Path
import statistics
import torch

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from soflow_fastgen.common import ROOT, atomic_json, digest
from soflow_fastgen.audit_support import ClassSampler


def render(run):
    load=lambda name:json.loads((run/name).read_text())
    ledger=load('ledger.json'); manifest=load('frozen.json')
    sat=load('saturation/result.json');ready=load('readiness/result.json')
    image_metrics=load('saturation/image_metrics.json')
    mean_overshoot=statistics.mean(row['mean_overshoot'] for row in image_metrics)
    conditional_overshoot=mean_overshoot/sat['mean_clip_fraction']
    exact_max_abs=max(row['max_abs'] for row in image_metrics)
    ds={seed:load(f'd_seed{seed}/result.json') for seed in manifest['training_seeds']}
    assert all(j['state']=='completed' for j in ledger['jobs'])
    assert all(d['state']=='completed' and d['d_steps']==512 and d['g_steps']==0 for d in ds.values())
    exposure={}
    for seed in ds:
        sampler=ClassSampler(manifest['fit_groups'],seed+1000);seen=set();rows=[]
        held=set(manifest['hold_positions'])
        for step in range(513):
            if step in manifest['checkpoints']:
                saved=torch.load(run/f'd_seed{seed}/d_step{step:04d}.pt',map_location='cpu',weights_only=False)
                assert saved['sampler']==sampler.state_dict()
                del saved
                positions=manifest['fit_diagnostic_positions']
                is_seen=np.array([p in seen for p in positions])
                scores=load(f'd_seed{seed}/diagnostic_scores_step{step:04d}_seed6201.json')
                fit=np.array(scores['fit'])
                rows.append(dict(d_step=step,unique_fit_images=len(seen),fit_diagnostic_seen=int(is_seen.sum()),
                    seen_fit_mean_score=float(fit[is_seen].mean()) if is_seen.any() else None,
                    unseen_fit_mean_score=float(fit[~is_seen].mean()) if (~is_seen).any() else None,
                    held_mean_score=float(np.mean(scores['held'])),sampler_reconstruction_exact=True))
            if step<512:
                positions,_=sampler.draw(32)
                assert not held.intersection(positions)
                seen.update(positions)
        exposure[str(seed)]=rows
    figures=run/'figures';figures.mkdir(exist_ok=True)
    categories=sat['categories']
    real=np.array([r['real_exact_boundary'] for r in categories]);fake=np.array([r['clip_mean'] for r in categories])
    fig,axes=plt.subplots(2,1,figsize=(11,9),constrained_layout=True)
    axes[0].hist(fake*100,bins=20,color='#23788d');axes[0].set(xlabel='Generated raw coordinates outside [-1,1], per class (%)',ylabel='Classes')
    axes[1].scatter(real*100,fake*100,alpha=.65,color='#23788d');axes[1].set(xlabel='Real channel values exactly at -1 or 1, per class (%)',ylabel='Generated raw overshoot (%)',title='Different statistics; association does not establish a cause')
    fig.savefig(figures/'saturation.png',dpi=150);plt.close(fig)
    keys=[('held_pair_accuracy','Held-out real/fake pair accuracy'),('masked_image_gradient_energy','Masked image-gradient energy fraction'),('actual_vs_unmasked_cosine','Actual vs diagnostic-unmasked parameter-gradient cosine'),('correct_minus_wrong_real_score','Correct minus wrong-label real score')]
    fig,axes=plt.subplots(4,1,figsize=(12,15),constrained_layout=True)
    for ax,(key,label) in zip(axes,keys):
        for seed,d in ds.items():
            vals=[statistics.mean(row[key] for row in d['diagnostics'] if row['d_step']==s) for s in manifest['checkpoints']]
            ax.plot(manifest['checkpoints'],vals,'o-',label=f'D seed {seed}')
            if key=='held_pair_accuracy':
                vals=[statistics.mean(row['fit_pool_pair_accuracy'] for row in d['diagnostics'] if row['d_step']==s) for s in manifest['checkpoints']]
                ax.plot(manifest['checkpoints'],vals,'x--',alpha=.65,label=f'Fit-pool, {seed}')
        ax.set(xlabel='D updates (G frozen)',ylabel=label);ax.grid(alpha=.2);ax.legend()
    fig.savefig(figures/'diagnostics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,12),constrained_layout=True)
    for seed in ds:
        updates=[json.loads(line) for line in (run/f'd_seed{seed}/updates.jsonl').read_text().splitlines()]
        for ax,key in zip(axes,['adversarial','r1','r2']):
            y=np.array([r[key] for r in updates]);x=np.arange(1,len(y)+1)
            median=np.array([np.median(y[max(0,i-15):min(len(y),i+16)]) for i in range(len(y))])
            line,=ax.plot(x,median,lw=2,label=f'D seed {seed}, 31-update median')
            ax.plot(x,y,lw=.5,alpha=.23,color=line.get_color());ax.set(xlabel='D update',ylabel=key);ax.grid(alpha=.2);ax.legend()
    fig.savefig(figures/'d_training.png',dpi=150);plt.close(fig)
    timings={}
    for size in [8,32]:
        timings[str(size)]={}
        for kind in ['d','g']:
            values=[r['seconds'] for r in ready['updates'] if r['batch']==size and r['kind']==kind and (kind=='d' or r['rho']>0)]
            timings[str(size)][kind]=dict(mean=statistics.mean(values),min=min(values),max=max(values),count=len(values))
    summary=dict(audit='pre_program_20261004',completed_date='2026-10-05',manifest_sha256=digest(run/'frozen.json'),
        fixed_protocol={k:v for k,v in manifest.items() if k not in ['fit_groups','hold_positions','fit_diagnostic_positions','diagnostic_labels','hold_original_indices','hold_paths','category_names','source']},
        source_sha256=manifest['source']['source_sha256'],source_checkpoint_sha256=manifest['source_sha256'],
        partition_sha256=manifest['partition_sha256'],ledger=ledger,contracts_passed=38,
        saturation={k:v for k,v in sat.items() if k!='categories'},
        class_balanced_real_boundary=float(real.mean()),class_boundary_correlation=float(np.corrcoef(real,fake)[0,1]),
        mean_raw_overshoot=mean_overshoot,conditional_raw_overshoot=conditional_overshoot,exact_max_abs_raw=exact_max_abs,
        class_clip_min=float(fake.min()),class_clip_max=float(fake.max()),readiness=ready,timings=timings,discriminators=ds,exposure=exposure,
        limitations=['No generator learning pilot or FID measurement','No trained range-penalty arm','D holdout was seen during S3 pretraining',
            'Fit-pool diagnostic images may not all have appeared in D updates','A3 gradients average 604 images per draw set, not training batch 32',
            'Diagnostic unmasked vector is not the gradient of the deployed clamped objective'])
    atomic_json(run/'compact_evidence.json',summary)
    table=[]
    for seed,d in ds.items():
        for step in manifest['checkpoints']:
            rr=[r for r in d['diagnostics'] if r['d_step']==step]
            avg=lambda key:statistics.mean(r[key] for r in rr)
            table.append(f"| {seed} | {step} | {avg('held_pair_accuracy'):.3f} / {avg('fit_pool_pair_accuracy'):.3f} | {avg('masked_image_gradient_energy')*100:.2f}% | {avg('actual_parameter_grad_norm'):.4f} / {avg('blocked_parameter_grad_norm'):.4f} | {avg('actual_vs_unmasked_cosine'):.4f} | {avg('actual_vs_consistency_cosine'):.4f} | {avg('correct_minus_wrong_real_score'):.4f} |")
    rows='\n'.join(table)
    markdown=f'''# Fixed SoFlow pre-program audit — October 4–5, 2026

**Completed.** A0–A4 finished within the 12 aggregate GPU-job-hour ceiling.
No sustained generator fine-tuning, FID-improvement pilot, range-penalty training,
or autoresearch run occurred. S3 remains the incumbent.

| Measurement | Result |
|---|---:|
| Contract tests | 38 passed |
| Generator / discriminator parameters | 89,140,419 / 28,930,048 |
| Frozen S3 samples | 6,040, 40/class |
| Real optimization images inspected | 14,059 |
| D diagnostic holdout | 604 images, four/class |
| Independent D runs | 2 × 512 updates, B32, 16,384 presentations/run |
| A4 disposable D/G attempts including replay | {ready['attempted_d']} / {ready['attempted_g']} |
| Total aggregate GPU-job time | {ledger['used_seconds']/3600:.4f} hours |
| Failed GPU stages / automatic retries | 0 / 0 |
| Highest worker CUDA allocation | {max([sat['resources']['allocated_bytes'],ready['resources']['allocated_bytes']]+[d['resources']['allocated_bytes'] for d in ds.values()])/1e9:.3f} GB |

Raw evidence and galleries: `runs/{run.name}/`.
[Gallery](../../runs/{run.name}/saturation/gallery.html) ·
[Diagnostic plots](../../runs/{run.name}/report.html) ·
[Compact committed evidence](pre-program-evidence-20261005.json) ·
[Frozen source/input manifest](pre-program-manifest-20261004.json).

## Saturation

Generated raw channel values outside [-1,1]: **{sat['mean_clip_fraction']*100:.2f}%** overall;
class means range **{fake.min()*100:.2f}%–{fake.max()*100:.2f}%**. Real exact-boundary
channel values average **{real.mean()*100:.2f}%** under the same equal-class weighting.
These are different quantities, not a matched measure of error. Across-class
Pearson correlation is {np.corrcoef(real,fake)[0,1]:.3f}; this is descriptive and
does not establish why the generator overshoots. Mean overshoot is
{mean_overshoot:.5f} across all coordinates, or {conditional_overshoot:.5f}
conditional on clipping, in the [-1,1] scale. The exact maximum absolute raw
coordinate across these samples is {exact_max_abs:.4f}. The raw pixel quantiles in
the evidence use every 97th flattened coordinate, not exhaustive order statistics.

## Discriminator and parameter-gradient trajectories

Each cell averages the two fixed diagnostic noise sets; full per-draw values
remain in the evidence. G and EMA state hashes were unchanged at all checkpoints.
Wrong labels use a fixed 151-class permutation, not training augmentation.

| D seed | D updates | Pair accuracy: held / fit pool | Masked image energy | Parameter norm: actual / blocked | Actual vs unmasked cosine | Actual vs SCM cosine | Correct−wrong real score |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

The actual vector includes the clamp derivative. The blocked contribution is
a diagnostic vector–Jacobian product only; its sum with the actual vector is
not the gradient of the deployed clamped objective. Parameter contributions can
reinforce or cancel, so these norm ratios are not a percentage of learning lost.
All A3 vectors average 604 examples per noise set; they are not measurements of
the stochastic B32 training-update distribution.

SCM and squared-overshoot gradients were computed without updates and cached
because G was frozen. Their norms and cosines are in the evidence. The optional
range penalty passed analytic and finite-difference-related contracts, but its
effect on learning, image quality, and diversity remains untested.

## Readiness and cost

Phase transitions 0→.05→.1 and exact next-update replay passed at B8 and B32,
including sampler position, optimizers, EMA, RNG and counters. The original
35 contracts plus three audit-specific contracts passed. GPU-stage startup,
checkpointing and all GPU child stages are included once in the shared ledger; parallel jobs
are summed separately. Host RSS measures workers, not an aggregate host peak.

| Effective batch | D mean seconds/update | G mean seconds/update, rho>0 | D+G sum |
|---|---:|---:|---:|
| 8 | {timings['8']['d']['mean']:.3f} | {timings['8']['g']['mean']:.3f} | {timings['8']['d']['mean']+timings['8']['g']['mean']:.3f} |
| 32 | {timings['32']['d']['mean']:.3f} | {timings['32']['g']['mean']:.3f} | {timings['32']['d']['mean']+timings['32']['g']['mean']:.3f} |

These are short fixture measurements with data loading, exact penalties,
accumulation, optimizer, EMA and diagnostic update logging live. They exclude
checkpoint serialization and A3 analysis time. They establish execution cost,
not the number of updates required for useful learning. The native FP32 exact-
regularization path remains the qualified reference; no kernel optimization
or lazy-regularization substitution was introduced.

## Interpretation boundaries

- The D holdout was excluded from D updates, but S3 previously trained on those
  optimization images. It is not a generator memorization holdout.
- Fit-pool diagnostic images are sampled from the D-fit pool, not guaranteed
  to have appeared in its updates. The evidence reconstructs actual exposure
  and checks it against saved sampler states at every checkpoint. Pair accuracy
  is not semantic image quality.
- There is no range-penalty learning arm, generator response curve, FID claim,
  or evidence establishing the ideal warmup, ramp, optimizer or training horizon.
- A4's shared fixture sampler verifies exact recovery, not treatment/control
  draw matching. A future paired pilot must use separate G/D data samplers and
  declare GAN noise indexing across zero/nonzero-rho phases so extra D updates
  cannot shift the control's G data or common stochastic draws.
- Results from two D seeds describe this fixture; they are not independent
  generator-training replications. The tentative program remains inactive.
'''
    (run/'REPORT.md').write_text(markdown)
    (run/'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>SoFlow pre-program audit</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto;background:#f6f7f8;color:#17212b}img{width:100%}a{color:#146685}</style><h1>Fixed SoFlow audit</h1><p>Generator stayed frozen. Discriminator trajectories are diagnostics, not image-quality results.</p><p><a href="saturation/gallery.html">Browse all 151 classes</a> · <a href="REPORT.md">Detailed report</a></p><img src="figures/saturation.png"><img src="figures/diagnostics.png"><img src="figures/d_training.png">''')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    render(parser.parse_args().run_dir)
