# Pokémon SoFlow local experiment

Environment: local TITAN RTX; KCloud qualification and fresh training are separate.

Status: {"state": "running", "loss_surrogate": -0.0018871875945478678, "consistency_mse": 1.4809244871139526, "weight": 3.5761232376098633, "step": 40000, "grad_norm": 0.529146671295166, "seconds": 1.7165843807160854, "ema_decay": 0.9997750506136119, "host_rss_bytes": 1726586880, "host_peak_rss_bytes": 2111864832, "allocated_bytes": 8610886144, "reserved_bytes": 9726590976, "epoch": 366, "cursor": 13568, "elapsed_budget_seconds": 71261.5824514255, "process_tree_rss_bytes": 6943739904, "process_tree_pss_bytes": 1894683648, "process_tree_count": 5, "gpu": "NVIDIA TITAN RTX, 70, 1845, 6500, 9494, 99"}

Logged updates: 40000

Development FID (official training-derived holdout; not TA scores):

| Checkpoint | NFE | Seed | FID |
|---|---:|---:|---:|

Peak training allocation: 8.611 GB.
Peak parent RSS: 2.112 GB.
Peak sampled process-tree PSS (including loader workers): 2.053 GB.

## Training curves

Trends use a centered 201-record rolling window (one record per logged optimizer update); the window shortens at the edges. Green: median; shaded band: local 10th–90th percentiles (observed spread, not a confidence interval). Gray: every raw value, with no clipping or outlier removal. The dashed orange MSE line is the rolling mean, which is more sensitive to spikes. Consistency MSE uses a logarithmic y-axis so both its typical level and large spikes remain visible. The alternative linear-scale figure retains the full raw range. Consistency MSE is a training diagnostic under a changing consistency-gap curriculum, not a fixed validation loss; an increase alone does not establish worsening generation quality.

![Training curves](curves.png)

[Open full-range linear-scale curves](curves_linear.png)

## Latest labeled grids

![040000, nfe1](samples/040000/nfe1.png)

![040000, nfe2](samples/040000/nfe2.png)

![040000, nfe3](samples/040000/nfe3.png)

![040000, nfe4](samples/040000/nfe4.png)
