"""Render the archived training metrics; performs no model execution."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'training_metrics.csv').open() as f:
    rows = list(csv.DictReader(f))
steps = [int(r['step']) for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
for ax, key, label in zip(axes, ['consistency_mse', 'gradient_norm', 'seconds_per_update'],
                         ['Training consistency MSE', 'Gradient norm', 'Seconds per optimizer update']):
    ax.plot(steps, [float(r[key]) for r in rows], marker='o', markersize=3, color='#2459a6')
    ax.set(xlabel='Optimizer update', ylabel=label)
    ax.grid(alpha=.25)
axes[2].axhline(13.603, color='#9b4335', linestyle='--', linewidth=1, label='Updates 21–130 mean: 13.603 s')
axes[2].legend(fontsize=8)
fig.suptitle('SoFlow CIFAR-10: local TITAN RTX run, 2026-09-24\nTraining statistics only; image quality was not evaluated', fontsize=12)
fig.savefig(root / 'training-observations.png', dpi=180)
fig.savefig(root / 'training-observations.pdf')
