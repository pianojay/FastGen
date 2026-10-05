# FastGen SoFlow work instructions

## Current scope: completed diagnostic pre-program; research draft inactive

The fixed audit completed October 5, 2026. Read
[its final report](docs/adversarial/PRE_PROGRAM_RESULTS.md). Do not rerun it or
launch a successor automatically. The user clarified October 5 that the intended
assignment was to **write** the pre-program; the other agent also executed it.
Earlier prose calling that execution authorized was mistaken. Preserve the
completed execution facts, artifacts, frozen manifest and accounting as history.

Current authorization, October 5: the user instructed “Order subagent Soflow
matched pilot draft to carry out the pilot.” Implement, qualify and execute
[the fixed matched pilot](docs/adversarial/MATCHED_PILOT_DRAFT.md): A on GPU1,
C on GPU2, 512 G updates per arm, fixed protocol and 4.5 aggregate GPU-job-hour
ceiling. No intermediate approval waits for in-scope steps. This does not activate
[program.md](program.md), authorize a successor, or replenish the completed audit's
historical allowance. Preserve S1/S2/S3, C3 and all audit evidence. The
[integration contract](docs/adversarial/INTEGRATION.md) defines the unchanged math.


- This is a local TITAN RTX development project, not the KCloud VM.
- Preserve the separate SoFlow, AFM and SiD checkouts and their environments/processes. The CIFAR experiment is closed. The completed audit used physical GPUs 1 and 2; availability is not launch permission.
- Audit workers use CUDA_VISIBLE_DEVICES=1 or CUDA_VISIBLE_DEVICES=2 and local cuda:0. Verify physical UUID and acquire the existing device-specific lease. No fallback to GPUs 0/3, distributed training, or additional agents.
- Use PYTHONNOUSERSITE=1 and the fastgen_soflow environment. Use python -m pip; include pip in fresh Conda environments.
- Preserve the official base Model class and every file in configs/protected_files.json. Verify their hashes before runs and exports.
- Official validation images must not train models or select checkpoints, hyperparameters, or sampling schedules.
- This integration may initialize online G and G-EMA from the challenge-trained S3 final EMA. Its G optimizer state is fresh. D started from scratch for the completed integration/audit; the authorized pilot reuses D20261004 with its optimizer, D streams and sampler as declared. U-Net128/160/192 loaders remain supported. No external pretrained training components are permitted.
- New G counters start at zero; the pilot preserves inherited D=512 separately while the inherited SoFlow consistency gap remains fixed at its mature value. Exact recovery restores the same configuration, state and RNG streams.
- The completed integration smoke remains historical evidence. Historical audit limits are in pre_program.md: two frozen-G D runs up to 512 updates each; disposable A4 at most 16 D and 16 G attempts including replay. The separately authorized fixed matched pilot permits its declared G tuning/FID study; no autonomous successor is authorized.
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

[program.scratch-history.md](program.scratch-history.md) preserves the prior autonomous scratch-training loop as
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
