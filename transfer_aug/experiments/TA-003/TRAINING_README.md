# Final contract execution

`CONTRACT_v1.md` supersedes the preparation-only restriction above. `runner.py`
uses the shared frozen NumPy RGB64 cache and a `FrozenSampler` adapter rather
than reading/re-exporting individual source images. It verifies the assigned
physical GPU UUID and existing read-only lease, accounts all lease wall time,
qualifies 20 full batch16 updates plus exact next-step recovery, then discards
that state and resets all models/optimizers/replay/RNG before production.

The explicitly authorized command and recovery command are recorded in
`run_configuration.json`; `gpu_ledger.json` persists cumulative accounting.
The runner enforces the original update/LR/batch/parameter/memory ceilings,
reserves 600 seconds for inference, stores atomic checkpoints every 1,000
updates and required milestones, and generates 604 final diagnostic
translations, 604 cycle controls and 604 identity controls. Review subset
regeneration uses the same batch context to test exact quantized equality.
`report.py` uses the frozen descriptor function and PCA, writes per-image EDA,
8 representative timeline sheets plus the 8-image diagnostic-tail sheet.
`finalize.py` requires a direct visual-review record for every sheet and
resource/reproduction checks before emitting `result.json`.

`checkpoint_history.json` records hashes at each save event. Its `latest.pt`
entries are historical hashes of an intentionally overwritten recovery file;
only the last such entry describes the current `latest.pt`. Retained initial,
1,000, 5,000, 10,000 and final checkpoints are separately hashed in `result.json`
and `output_hashes.json` after the GPU session closes.

Primary method reference (as cited by the pinned upstream README): Jun-Yan Zhu,
Taesung Park, Phillip Isola and Alexei A. Efros, *Unpaired Image-to-Image
Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017,
[paper](https://arxiv.org/pdf/1703.10593.pdf).
