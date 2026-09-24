# CIFAR-10 GPU 0 training

**Status: stopped on 2026-09-24 at 21:15:59 KST by an authorized feasibility decision.** See the [formal experiment report](EXPERIMENT_REPORT.md). Last logged update: 130; last recoverable main checkpoint: 5. Earlier running-status entries below are historical.

Canonical document location: `FastGen-SoFlow/docs/soflow-cifar10/TRAINING_RUN.md`.
All commands and relative code/artifact paths below refer to `/home/user/jw_soflow/SoFlow`, unless explicitly stated otherwise. This record was moved into Astra's documentation tree on 2026-09-24; historical entries retain their original dates and settings.

## Settings used after throughput tuning

GPU 0 only; effective batch 1,024; microbatch **64**; **16** accumulation passes per update; target 800,000 updates. Progress logging is every **10** updates. Default sampling and checkpoint intervals remain in place. The launcher disables user-site imports with `PYTHONNOUSERSITE=1` and uses the repaired conda environment.

## Setup and reproduction

Run from `/home/user/jw_soflow/SoFlow`:

```bash
python scripts/launch_cifar_gpu0.py
```

The script pins GPU 0, uses the soflow conda Python and one torch distributed worker, records dependency versions and interpreter paths, and records PID and exit status. A file lock prevents duplicate launches. The environment agent has repaired and isolated the dependencies; the launcher explicitly disables the Python user site. The module launcher avoids the system-Python torchrun executable.

Dataset: torchvision downloads the official Python archive from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./CIFAR10 and verifies its checksums. Training uses the 50,000-image training split.

## Changes

- `train.py`: optional `microbatch_size`, currently supported for CIFAR-10 consistency-only training. Each CPU batch is split before GPU transfer. Sample-weighted gradients and statistics accumulate; optimizer, EMA and counters advance once per logical batch. Default behavior uses one full batch. Added non-finite statistics detection, first-update timing and a guard against training past an already completed checkpoint.
- `cifar_gpu0.yaml`: original CIFAR configuration with dedicated output directory and microbatch size 16. Effective batch remains 1,024, with 64 chunks per optimizer update. Training remains 800,000 updates with original optimizer, schedule, augmentation and seed.
- `scripts/train_cifar_gpu0.sh`: reproducible persistent launch and process records.
- Existing user changes in readme.md are preserved.

Sampling: 32 demo images every 1,000 updates; 50,000 images every 25,000 updates; checkpoints every 25,000 updates. Metric evaluation is deferred. No separate tests or tuning runs are requested or run.

## Monitoring and completion

Logs: runs/cifar10_gpu0/log.txt and console.log. Outputs: ckpts/, figs/, evals/. The saved config and environment records are in the same directory.

Completion requires step 800,000, final checkpoint and scheduled samples, and exit status 0. A running process alone is not completion. Reusing the launch command resumes the latest saved checkpoint; upstream resume does not preserve exact RNG or data-loader position. Accumulation preserves the effective batch but not bitwise equivalence to multi-GPU training.

## Launch record — 2026-09-24

- Detached session started with launcher PID 393443; torch distributed launcher PID 393494. Process inspection confirmed it remained alive after the launching shell exited.
- The initial shell-background launch did not persist; the independent-session launcher above resolves that.
- Environment capture initially attempted pip freeze, but this conda environment has no pip module. Replaced capture with importlib.metadata and successfully saved the package list. Training imports succeeded through the conda interpreter with user-site dependencies (torch 2.7.1+cu126, torchvision 0.22.1+cu126).
- CIFAR-10 is downloading from the official Toronto server. The download is slow; a candidate PyTorch mirror returned HTTP 404 and was not used. The active official download was left intact.
- Status at handoff: dataset download in progress; no optimizer update or full training completion claimed. The same process will verify/extract the archive and begin training automatically. Runtime estimation must wait for actual optimizer throughput.

## Environment-repair hold — 2026-09-24

At the user's request, checked the session named `Install pygobject torch dependencies` (01a0d2e8-574f-78a1-be93-55b649e5b7f2). Its latest reported status at 19:38 KST: soflow repaired and verified; soflow_eval installing, with dependency/GPU/evaluator validation pending. See AGENTS.md for the other agent's environment notes.

Paused launcher 393443, torchrun 393494, and worker 393541 using SIGSTOP while the CIFAR archive was still downloading (approximately 81%). No optimizer updates had been logged. The paused worker must not be reused for training: it imported user-site packages before the repair. After the repair is complete, retire these old processes and launch fresh with PYTHONNOUSERSITE=1 (already added to the launcher by the environment agent). Preserve the partial dataset archive where possible. Training remains on hold; earlier launch-status notes are historical.

## Fresh launch after repair

At the user's request, retired all three paused processes. Resuming the official dataset archive with curl --continue-at - from byte 139460608; torchvision will verify the full archive checksum before extraction. Fresh launch sets PYTHONNOUSERSITE=1 and records repaired environment paths/versions. Launcher now stops on setup failures and replaces stale exit-status markers with running.

### Training active — 2026-09-24 20:22 KST

- Resumed archive download completed successfully; torchvision verified and extracted the dataset, then initialized the 55.93M-parameter UNet.
- Fresh persistent launcher PID: 416230. Launch command: `python scripts/launch_cifar_gpu0.py`. Environment records now show torch and torchvision under `/home/user/miniconda3/envs/soflow/lib/python3.10/site-packages`.
- Initial checkpoint saved to `runs/cifar10_gpu0/ckpts/0.pt`.
- First actual optimizer update at 20:22:35: batch 1024, microbatch 16, consistency MSE 1.252576, gradient norm 2.731277, elapsed 13.16 seconds. GPU 0 observed at 93% utilization and 4188 MiB memory; GPUs 1–3 idle.
- Startup-based extrapolation: 800000 * 13.16 / 86400 = approximately 122 days, excluding sampling overhead. This is provisional; use the regular 100-update timing for a better estimate.
- Current state: full training continues in the background; not complete. No separate tests were run in this continuation. Log output normally appears every 100 optimizer updates, so the log can remain unchanged for many minutes while training is active.

Follow progress with `tail -f runs/cifar10_gpu0/log.txt`. Process state is recorded in `pid`; `exit_status` contains `running` while active and the numeric exit code when the launcher finishes.

## Throughput tuning — 2026-09-24

User authorized throughput measurements and a larger GPU batch. The original run had only checkpoint 0 (its first scheduled checkpoint was at 25,000 updates), so its unsaved progress was lost when it was stopped. The chosen measurement's completed five-update checkpoint was copied to the main run, which resumes from step 5 with its model, EMA, and optimizer state.

Measurement command:

```bash
PYTHONNOUSERSITE=1 /home/user/miniconda3/envs/soflow/bin/python scripts/measure_cifar_microbatch.py 16 64 128 256
```

Each candidate used the real training loop with the original 800,000-update schedule, effective batch 1,024 and seed 42. `--run-steps 5` limits only the invocation, then saves its checkpoint; it does not change `total_steps`. Timings below are medians of updates 3–5, after two warmup updates. Device memory was sampled with nvidia-smi once per second; torch peak allocation was logged by training. This is a short throughput comparison, not a convergence or quality test.

| Microbatch | Accumulation passes | Seconds/update | Images/second | Peak device memory | Peak torch allocation |
|---|---|---|---|---|---|
| 16 | 64 | 14.73 | 69.52 | 4.09 GiB | 3.34 GiB |
| 64 | 16 | 12.72 | 80.50 | 10.33 GiB | 9.51 GiB |
| 128 | 8 | 13.22 | 77.46 | 18.72 GiB | 17.74 GiB |
| 256 | 4 | Out of memory | — | 23.34 GiB | — |

Selected microbatch **64**, the fastest measured candidate. Throughput improved by approximately 15.8% over microbatch 16 (13.6% less time per update). Microbatch 128 used more memory but was about 3.9% slower than 64; 256 exceeded available memory. Short-run projection is approximately 118 days for 800,000 updates, excluding sampling and checkpoint overhead. This replaces the earlier estimate based only on the first update. Steady-state thermals and runtime overhead can change the estimate.

Additional changes: train.py accepts `--run-steps`, checkpoints at that invocation limit, logs the first resumed update, and divides partial logging intervals by the actual update count. cifar_gpu0.yaml sets microbatch_size=64 and logging_steps=10. Measurement configs, logs and checkpoints remain under runs/cifar10_throughput; results.json stores raw measurements and commands. No learning-rate, model, precision, effective-batch, or training-schedule changes were made.

Relaunch: `python scripts/launch_cifar_gpu0.py`; fresh launcher PID 423732. The benchmark checkpoint was copied from runs/cifar10_throughput/b64/ckpts/5.pt to runs/cifar10_gpu0/ckpts/5.pt. Upstream resume reinitializes RNG and starts the recorded epoch again, so it is not an exact continuation of the data order.

Resume verified: at 20:47:13, step 6 completed with microbatch 64, effective batch 1024, MSE 1.244692 and gradient norm 2.676620, in 11.70 seconds. Full training is active in the background; the throughput-tuning task is complete, while the 800,000-update training run is not yet complete.

## Training cost reference

The paper comparison, epoch arithmetic, timing estimates, and microbatch
measurements are consolidated in
[CIFAR-10 training scale and runtime](cifar10-training-cost.md).

## Closure — 2026-09-24 21:15:59 KST

Stopped by SIGTERM to verified torchrun PID 423798 under the user's conditional authorization after assessing the one-day quality budget. The launcher exited with status 1 due to intentional termination. No training failure was observed in the selected configuration. The main run logged update 130; at least 125 logged updates beyond checkpoint 5 were not saved because the running trainer had no checkpoint-on-signal support. No new terminal checkpoint, generated sample grid, or FID result was produced.

Updates 21–130 took 1496.33 seconds (13.603 seconds/update), projecting to 125.95 days plus overhead for the full schedule. A day would allow approximately 6352 updates, or 0.794% of the schedule. Training loss fell, but meaningful image-quality improvement within a day was unestablished. This is a feasibility decision, not a claim that quality improvement is impossible.

The [formal report](EXPERIMENT_REPORT.md) and its adjacent evidence archive preserve the configuration, logs, timings, source hashes/patch, commands, and limitations. Do not treat the closed run as completed training or restart it automatically from the earlier launch instructions.
