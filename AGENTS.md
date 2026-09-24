# FastGen SoFlow work instructions

- This is a local TITAN RTX development project, not the KCloud VM.
- The separate SoFlow checkout and GPU 0 belong to the CIFAR agent. Do not change its files, environments, or processes.
- Use GPU 1 locally. A visible CUDA device may appear as logical GPU 0 after CUDA_VISIBLE_DEVICES=1.
- Use PYTHONNOUSERSITE=1 and the fastgen_soflow environment. Use python -m pip; include pip in fresh Conda environments.
- Preserve the official base Model class and every file in configs/protected_files.json. Verify their hashes before runs and exports.
- Official validation images must not train models or select checkpoints, hyperparameters, or sampling schedules.
- All generative weights start from scratch on the training-derived optimization partition. Only the official FID Inception is pretrained.
- Keep the first experiment to 24 cumulative GPU job hours, including qualification/evaluation, with four hours reserved for terminal evaluation.
- Checkpoint loaded inference models must contain fewer than 100M parameters. Actual training peak allocation must remain below 20,000,000,000 bytes.
- Microbatch tuning must retain the effective batch. Compare timings with full augmentation, gradient accumulation, optimizer and EMA live.
- Data augmentation follows upstream SoFlow. Pass identical transform metadata to both consistency branches and zeros at inference.
- Update docs/STATUS.md with measured facts, commands, limitations, and next actions. Do not call a running job complete or a local FID official.
- Use python -m unittest discover -s tests -v for contract/numerical checks. Do not rerun expensive profiling without a relevant code or hardware change.
- Private GitHub carries source; SFTP carries artifacts. No Slack messages or collaborator invitations have been authorized.
- KCloud begins with a fresh run and repeats its own memory, throughput, and environment qualification.


## Shared SoFlow CIFAR-10 documentation

The canonical documents from the SoFlow CIFAR-10 work reside in this checkout
so Astra and the CIFAR agent can use the same records:

- [CIFAR documentation index](docs/soflow-cifar10/README.md).
- [Training scale, paper comparison, and runtime measurements](docs/soflow-cifar10/cifar10-training-cost.md).
- [Training run record, scripts, modifications, and restart history](docs/soflow-cifar10/TRAINING_RUN.md).

Read these before applying CIFAR throughput or training-budget conclusions
to FastGen. They describe the separate TITAN RTX GPU 0 run in `../SoFlow`,
not the Pokémon experiment. The local raw logs, checkpoints, and training
code remain in that checkout. The documents preserve key measured values
for use when the raw artifacts are unavailable.

New documentation from the CIFAR work belongs under `docs/soflow-cifar10/`;
keep its index and both repositories' AGENTS.md links current. The user's
shared-documentation instruction permits maintaining these documents and
references; it does not authorize changing the other experiment's training
processes, code, or environment.
