# Pokémon SoFlow local experiment

Environment: local TITAN RTX; KCloud qualification and fresh training are separate.

Status: {"state": "training_complete", "step": 20000, "schedule_steps": 20000, "reason": "schedule_complete", "elapsed_budget_seconds": 36243.82595201023}

Logged updates: 20000

Development FID (official training-derived holdout; not TA scores):

| Checkpoint | NFE | Seed | FID |
|---|---:|---:|---:|
| step_005000 | 1 | 42 | 326.9586 |
| step_005000 | 4 | 42 | 103.8197 |
| step_010000 | 1 | 42 | 174.6829 |
| step_010000 | 4 | 42 | 57.4169 |
| step_020000 | 1 | 42 | 114.6036 |
| step_020000 | 1 | 43 | 113.3066 |
| step_020000 | 2 | 42 | 46.4976 |
| step_020000 | 2 | 43 | 46.6981 |
| step_020000 | 3 | 42 | 40.2009 |
| step_020000 | 3 | 43 | 40.3537 |
| step_020000 | 4 | 42 | 38.3041 |
| step_020000 | 4 | 43 | 38.2675 |

Peak training allocation: 8.613 GB.
Peak parent RSS: 2.094 GB.
Peak sampled process-tree PSS (including loader workers): 2.120 GB.

## Training curves

Trends use a centered 201-record rolling window (one record per logged optimizer update); the window shortens at the edges. Green: median; shaded band: local 10th–90th percentiles (observed spread, not a confidence interval). Gray: every raw value, with no clipping or outlier removal. The dashed orange MSE line is the rolling mean, which is more sensitive to spikes. Consistency MSE uses a logarithmic y-axis so both its typical level and large spikes remain visible. The alternative linear-scale figure retains the full raw range. Consistency MSE is a training diagnostic under a changing consistency-gap curriculum, not a fixed validation loss; an increase alone does not establish worsening generation quality.

![Training curves](curves.png)

[Open full-range linear-scale curves](curves_linear.png)

## Latest labeled grids

![020000, nfe1](samples/020000/nfe1.png)

![020000, nfe2](samples/020000/nfe2.png)

![020000, nfe3](samples/020000/nfe3.png)

![020000, nfe4](samples/020000/nfe4.png)
