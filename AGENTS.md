# FastGen SoFlow work instructions

## Current scope: adversarial integration and smoke verification

User-authorized October 4, 2026: implement SoFlow + class-conditioned StyleGAN2
discriminator + RpGAN, then stop after bounded component/smoke verification.
Read [the integration contract](docs/adversarial/INTEGRATION.md). S1/S2/S3 are
complete; C3 was stopped. The historical autonomous research loop is inactive
on this branch. No training campaign or next experiment is authorized here.


- This is a local TITAN RTX development project, not the KCloud VM.
- Preserve the separate SoFlow, AFM and SiD checkouts and their environments/processes. The CIFAR experiment is closed. This branch uses only physical GPU 1.
- Use GPU 1 locally. A visible CUDA device may appear as logical GPU 0 after CUDA_VISIBLE_DEVICES=1.
- Use PYTHONNOUSERSITE=1 and the fastgen_soflow environment. Use python -m pip; include pip in fresh Conda environments.
- Preserve the official base Model class and every file in configs/protected_files.json. Verify their hashes before runs and exports.
- Official validation images must not train models or select checkpoints, hyperparameters, or sampling schedules.
- This integration may initialize online G and G-EMA from the challenge-trained S3 final EMA. Its optimizer state is fresh; D starts from scratch. U-Net128/160/192 loaders remain supported. No external pretrained training components are permitted.
- New component counters start at zero while the inherited SoFlow consistency gap remains fixed at its mature value. Exact recovery restores the same configuration, state and RNG streams.
- The authorized S3 smoke harness is limited to four D updates and three G updates including replay, with a 900-second supervisor watchdog. Small numerical test fixtures are separate disposable checks. No FID study, warmup campaign, schedule extension or autonomous successor run belongs to this scope.
- Generator and discriminator must each contain fewer than 100M parameters; exported inference contains only the generator. Actual training peak allocation must remain below 20,000,000,000 bytes.
- Microbatch tuning must retain the effective batch. Compare timings with full augmentation, gradient accumulation, optimizer and EMA live.
- Baseline augmentation follows upstream SoFlow; catalogued augmentation comparisons are explicit recipe branches. Pass identical transform metadata to both consistency branches and zeros at inference.
- Update docs/STATUS.md with measured facts, commands, limitations, and next actions. Do not call a running job complete or a local FID official.
- Use python -m unittest discover -s tests -v for contract/numerical checks. Do not rerun expensive profiling without a relevant code or hardware change.
- Private GitHub carries source; SFTP carries artifacts. No Slack messages or collaborator invitations have been authorized.
- KCloud begins with a fresh run and repeats its own memory, throughput, and environment qualification.

The [completed Pokémon baseline report](docs/POKEMON_BASELINE_RESULTS.md) records the 20,000-update result, final FIDs, resources and next-experiment questions. Read it before treating historical running snapshots as current. The baseline completed on 2026-09-25. Check current research records and relevant branches for subsequent experiments; this baseline summary is not a live campaign ledger.

The [consolidated FastGen strategy](docs/FASTGEN_STRATEGY.md) connects the CIFAR closure, literature audit and Pokémon experiment. Read it before proposing learned sampling schedules: it distinguishes backbone training, calibration and evaluation costs, and separates the one-call and few-call decisions. Its future experiments are proposals, not launched jobs or changes to the completed run.


## Historical research instructions

[program.md](program.md) preserves the prior autonomous scratch-training loop as
historical context. It is inactive on this integration branch. Its scratch-only
and consecutive-experiment rules do not override the current scope above.
The [research log](docs/RESEARCH_LOG.md), completed results and C3 stop evidence
remain historical facts. No additional agents without an explicit request.

## Shared SoFlow CIFAR-10 documentation

The canonical documents from the SoFlow CIFAR-10 work reside in this checkout
so Astra and the CIFAR agent can use the same records:

- [CIFAR documentation index](docs/soflow-cifar10/README.md).
- [Few-NFE training-cost literature audit](docs/soflow-cifar10/few-nfe-training-cost-audit.md): direct training, distillation, pretrained tuning, and LD3/BézierFlow sampler optimization. Its pretrained examples do not authorize external pretrained training components; the current local-checkpoint initialization permission is stated above.
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
