# FastGen-AFM baseline

The single teacher-free, pixel-space, one-call U-Net AFM baseline was explicitly started October 1, 2026. Implementation and disposable GPU qualification are complete. The frozen 50,000-update recipe is in [configs/baseline.json](configs/baseline.json); current status, measurements and final results belong in [docs/BASELINE_PLAN.md](docs/BASELINE_PLAN.md).

Read [AGENTS.md](AGENTS.md), [dated handoff](docs/HANDOFF.md), [implementation mapping](docs/IMPLEMENTATION.md) and [source provenance](resources/provenance.json). The official base `Model`, evaluation code and data manifests remain protected. [model.py](model.py) implements the one-call EMA generator; the few-call track remains unimplemented.

Use the separate `/home/user/miniconda3/envs/fastgen_afm/bin/python`, `PYTHONNOUSERSITE=1`, `CUDA_VISIBLE_DEVICES=0`, and four CPU threads. All outputs, caches and model artifacts stay in ignored `artifacts/` and `runs/`. GPU 1 and both sibling repositories are untouched. A durable lock refuses duplicate jobs. Do not relaunch a completed baseline or start a new quality trial from this README.

Private source remote: https://github.com/pianojay/fastgen-afm
