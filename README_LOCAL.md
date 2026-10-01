# FastGen-AFM baseline

The single teacher-free, pixel-space, one-call U-Net AFM baseline was stopped by the user October 1, 2026 after 10,294 G/D cycles because of strongly suspected within-class mode collapse. GPU 0 is released; the full 10,000-update checkpoint and logs are preserved. Trained development FID was not measured. See the [brief experiment report](docs/EXPERIMENT_REPORT.md), [frozen 50,000-update recipe](configs/baseline.json), and [stage accounting](docs/BASELINE_PLAN.md).

Read [AGENTS.md](AGENTS.md), [dated handoff](docs/HANDOFF.md), [implementation mapping](docs/IMPLEMENTATION.md) and [source provenance](resources/provenance.json). The official base `Model`, evaluation code and data manifests remain protected. [model.py](model.py) implements the one-call EMA generator; the few-call track remains unimplemented.

The implementation uses the separate `/home/user/miniconda3/envs/fastgen_afm/bin/python`, `PYTHONNOUSERSITE=1`, `CUDA_VISIBLE_DEVICES=0`, and four CPU threads. All outputs, caches and model artifacts stay in ignored `artifacts/` and `runs/`. GPU 1 and both sibling repositories are untouched. A durable lock refuses duplicate jobs. Do not resume this stopped baseline or start a new quality trial without a new user instruction.

Private source remote: https://github.com/pianojay/fastgen-afm
