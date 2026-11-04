# Pokémon SoFlow local experiment

Environment: local TITAN RTX; KCloud qualification and fresh training are separate.

Status: {"state": "running", "loss_surrogate": -0.005798631347715855, "consistency_mse": 2.139399766921997, "weight": 3.8284709453582764, "step": 40000, "grad_norm": 0.4410560429096222, "seconds": 3.1782660726457834, "ema_decay": 0.9997750506136119, "host_rss_bytes": 1641095168, "host_peak_rss_bytes": 2532368384, "allocated_bytes": 7700113408, "reserved_bytes": 8487174144, "epoch": 366, "cursor": 13568, "elapsed_budget_seconds": 129944.50122633576, "process_tree_rss_bytes": 6538137600, "process_tree_pss_bytes": 1858456576, "process_tree_count": 5, "gpu": "NVIDIA TITAN RTX, 71, 1830, 6500, 8312, 99"}

Logged updates: 40000

Development FID (official training-derived holdout; not TA scores):

| Checkpoint | NFE | Seed | FID |
|---|---:|---:|---:|

Peak training allocation: 7.701 GB.
Peak parent RSS: 2.532 GB.
Peak sampled process-tree PSS (including loader workers): 2.004 GB.

## Training curves

Trends use a centered 201-record rolling window (one record per logged optimizer update); the window shortens at the edges. Green: median; shaded band: local 10th–90th percentiles (observed spread, not a confidence interval). Gray: every raw value, with no clipping or outlier removal. The dashed orange MSE line is the rolling mean, which is more sensitive to spikes. Consistency MSE uses a logarithmic y-axis so both its typical level and large spikes remain visible. The alternative linear-scale figure retains the full raw range. Consistency MSE is a training diagnostic under a changing consistency-gap curriculum, not a fixed validation loss; an increase alone does not establish worsening generation quality.

![Training curves](curves.png)

[Open full-range linear-scale curves](curves_linear.png)

## Latest labeled grids

![040000, nfe1](samples/040000/nfe1.png)

![040000, nfe2](samples/040000/nfe2.png)

![040000, nfe3](samples/040000/nfe3.png)

![040000, nfe4](samples/040000/nfe4.png)
