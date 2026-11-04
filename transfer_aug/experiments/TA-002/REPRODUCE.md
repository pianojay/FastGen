# TA-002 evidence and reproduction

All commands run from `/home/user/jw_soflow/exploration` using the frozen CPU launcher. No installations, GPU, model imports, or training apply. The two-thread environment is in `environment.json`.

Initial execution:

```bash
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-002/run.py all
```

This verifies the protected inputs, all source-file metadata and selected A1 hashes; runs numerical acceptance; generates 14,496 transforms plus 604 controls; computes metrics; and creates 32 representative sheets plus two four-tail sheets. Raw arrays are float64, unbounded RGB. `raw/layout.json` declares dimensions and indexing; the row index matches the frozen cohort. PNG export clips, multiplies by 255, uses NumPy round-to-nearest (`rint`), then casts to uint8.

The independent deterministic replay does not overwrite successful arrays, PNGs, metrics, selections, sheets or gallery:

```bash
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-002/run.py recompute
```

`visual_judgments_manual.json` records the worker's actual image-tool inspection of every sheet and the per-condition categorical judgments. `review.py` expands these authored judgments, without consulting numeric metrics, and checks coverage:

```bash
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-002/review.py
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-002/run.py integrity_after
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-002/finalize.py
```

Visual reviews are unblinded AI judgments relative to the content image. They measure neither label correctness nor training utility. A visible background change describes appearance in that region; it does not assert donor-object copying. Anatomy/markings combines visible shape and characteristic surface-color detail, so recognizable silhouette can coexist with disrupted markings. Numeric boundary and spectral measurements are not visual artifact labels.

The adaptation retains the official pinned NumPy FFT-shift centered symmetric square, with integer radii 0, 1, 2, 4; introduces only the declared amplitude interpolation strengths; and transposes its channel-first convention to channel-last RGB. Numerical acceptance executes only the two audited NumPy function definitions parsed from the source AST; it never imports upstream torch. `alpha=1` exactly matches that reference for every source, donor arm and radius. Radius zero replaces only DC and therefore interpolates per-channel means before clipping.

`outside_mask_raw_l2` and `outside_mask_clipped_l2` are absolute complex-FFT L2 norms. Relative forms divide by the content outside-mask norm; exactly zero denominators are flagged and stored as null. Common metrics use clipped floating values before quantization. Per-condition summaries preserve the two donor arms. The chosen diagnostic tails follow clipping-descending, RMS-descending, then sample-ID/radius/alpha-ascending order, with four distinct content IDs per arm.

`hashes.json` hashes implementation and declared outputs, excluding mutable status, the hash inventory itself, and the final result to avoid recursive hashes. `result.json` is the handoff index. A failed exploratory command due to the unavailable bare `python` executable is retained in `failures/`; it generated no data. Full execution used the authorized launcher.
