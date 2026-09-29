# FastGen SoFlow work instructions

- This is a local TITAN RTX development project, not the KCloud VM.
- The separate SoFlow checkout and GPU 0 belong to the CIFAR agent. Do not change its files, environments, or processes.
- Use GPU 1 locally. A visible CUDA device may appear as logical GPU 0 after CUDA_VISIBLE_DEVICES=1.
- Use PYTHONNOUSERSITE=1 and the fastgen_soflow environment. Use python -m pip; include pip in fresh Conda environments.
- Preserve the official base Model class and every file in configs/protected_files.json. Verify their hashes before runs and exports.
- Official validation images must not train models or select checkpoints, hyperparameters, or sampling schedules.
- All generative weights start from scratch on the training-derived optimization partition. Only the official FID Inception is pretrained.
- The completed baseline EXP-1 had a 24-hour GPU-job ceiling with four hours reserved for terminal evaluation. Future research uses a default 12-hour cumulative GPU-job cap per model-training run, including its failures and resumed segments; follow program.md. Budget evaluation and confirmation per approved experiment, without a fixed 10/2 split. Neither allowance restarts automatically.
- Checkpoint loaded inference models must contain fewer than 100M parameters. Actual training peak allocation must remain below 20,000,000,000 bytes.
- Microbatch tuning must retain the effective batch. Compare timings with full augmentation, gradient accumulation, optimizer and EMA live.
- Baseline augmentation follows upstream SoFlow; catalogued augmentation comparisons are explicit recipe branches. Pass identical transform metadata to both consistency branches and zeros at inference.
- Update docs/STATUS.md with measured facts, commands, limitations, and next actions. Do not call a running job complete or a local FID official.
- Use python -m unittest discover -s tests -v for contract/numerical checks. Do not rerun expensive profiling without a relevant code or hardware change.
- Private GitHub carries source; SFTP carries artifacts. No Slack messages or collaborator invitations have been authorized.
- KCloud begins with a fresh run and repeats its own memory, throughput, and environment qualification.

The [completed Pokémon baseline report](docs/POKEMON_BASELINE_RESULTS.md) records the 20,000-update result, final FIDs, resources and next-experiment questions. Read it before treating historical running snapshots as current. The baseline completed on 2026-09-25; no follow-up run has been launched.

The [consolidated FastGen strategy](docs/FASTGEN_STRATEGY.md) connects the CIFAR closure, literature audit and Pokémon experiment. Read it before proposing learned sampling schedules: it distinguishes backbone training, calibration and evaluation costs, and separates the one-call and few-call decisions. Its future experiments are proposals, not launched jobs or changes to the completed run.


## Question-driven autoresearch

[program.md](program.md) defines the future research loop: choose an uncertainty, write a bounded experiment brief, run a controlled comparison, and update both the quality evidence and the explanation. The [research agenda and intervention catalog](docs/AUTORESEARCH_DIRECTIONS.md) connects four broad directions to SoFlow-specific options and records their implementation prerequisites.

One-call and few-call quality have equal standing and separate best candidates. Parameter ranges, trial lengths and confirmation rules belong to individual experiment briefs; the catalog is not a mandatory sweep. These documents do not launch a research session. Prepare each substantial experiment for user assessment before launch. Execute its approved preparation, runs, evaluation, and bounded recovery without repeated permission; report results and await approval for the next substantial experiment. The completed baseline, exact-resume guard and official evaluation remain protected; a changed recipe needs an explicit branch.

## Shared SoFlow CIFAR-10 documentation

The canonical documents from the SoFlow CIFAR-10 work reside in this checkout
so Astra and the CIFAR agent can use the same records:

- [CIFAR documentation index](docs/soflow-cifar10/README.md).
- [Few-NFE training-cost literature audit](docs/soflow-cifar10/few-nfe-training-cost-audit.md): direct training, distillation, pretrained tuning, and LD3/BézierFlow sampler optimization. Its pretrained options do not change FastGen's scratch-initialization requirement or permitted losses.
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


### CIFAR experiment report and closure

The [formal CIFAR experiment report](docs/soflow-cifar10/EXPERIMENT_REPORT.md)
records the conditions, loss trajectory, microbatch measurements, paper
comparison, one-day budget assessment, and stop decision. The GPU 0 CIFAR
run was stopped on 2026-09-24 after demonstrating initial trainability;
its full schedule projected to about 126 days. Image quality was not
measured. Last logged step is 130, but the latest recoverable main
checkpoint is step 5. Consult the archived evidence before transferring
these conclusions to FastGen, and do not restart the closed CIFAR run
automatically.
