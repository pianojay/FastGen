# Local Pokémon/FastGen experiment status

Recorded 2026-09-24, Asia/Seoul. This experiment runs on the local TITAN RTX server; it is not the KCloud VM or an official submission.

## Decisions and boundaries

The approved starting point is a class-conditioned, pixel-space SoFlow U-Net trained from scratch. Keep the separate CIFAR-10 agent's GPU 0 run and environment intact; Pokémon uses GPU 1, checkout `FastGen-SoFlow`, and Conda environment `fastgen_soflow`. The [CIFAR records](soflow-cifar10/README.md) provide shared context, but their batch size and throughput are not Pokémon measurements.

Source repository: [pianojay/fastgen-soflow](https://github.com/pianojay/fastgen-soflow), private. [Notion experiment EXP-1](https://app.notion.com/p/3e5f3544609181f5a1d7e29d547ea7fc). Large weights, raw data and generated images stay outside Git. [Migration instructions](MIGRATION.md) describe checked archives and SFTP. KCloud's first comparison starts from scratch and repeats memory/throughput qualification on its A100 with less than 20 GB available.

The baseline uses 64×64 RGB, 151 categories, width 128 with channel multipliers [1,2,2,2], two blocks/level, attention at resolution 16 and dropout 0.1. The inference model has **39,667,843 parameters**. Both exported modes contain one EMA backbone, with no training-model duplicate, teacher or optimizer. Each sampling step calls the directly conditioned backbone once; the wrapper performs no classifier-free guidance. NFE 1, 2, 3 and 4 have actual call-count tests.

Augmentation retains SoFlow's horizontal flip (~50%), scale (12%), anisotropic scale (12%) and translation (12%). Both consistency branches receive identical transform metadata; inference uses six zeros. Original/augmented labeled examples are saved. A future control sets `augmentation=false` and bypasses resampling entirely. No augmentation/architecture sweep starts automatically. Widths 160 and 192 would contain 61,934,243 and 89,140,419 parameters, respectively, and require separate qualification.

## Data and leakage checks

The unchanged official downloader fetched Kaggle dataset version 5 and composited 1,476 transparent files on white. Official manifests contain 17,079 training images and 3,020 validation images. Development uses 3,020 held-out official TRAIN images (20/category), leaving 14,059 optimization images. Official validation images are not used for fitting, checkpoint selection or sampling selection.

The audit found 1,073 exact-byte duplicate groups in official training. The first candidate partition crossed 286 such groups; it was replaced before baseline training. The final holdout selects only globally byte-unique files, giving zero exact duplicate groups across the optimization/development boundary. This deliberately changes the holdout distribution; near-duplicates are not removed. Partition SHA-256: `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`.

Hardware profiling occurred before that partition correction, on the same dataset/model and split sizes. It was not used as trained initialization or for quality selection. Baseline training starts fresh from the final partition.

## Qualification evidence

The new environment was cloned from repaired `soflow`; only the clone received FastGen dependencies. Environment-local pip and packages, `PYTHONNOUSERSITE=1`, and dependency checks passed. Direct versions are pinned in `requirements-fastgen-lock.txt`; the full installed package inventory is in `artifacts/environment-freeze.txt`. The NVIDIA-specific wheels need a compatible driver on migration.

Eight contract/numerical tests passed on GPU 1. They cover protected upstream files/base class, parameter count, checkpoint loading, class-conditioning gradients, NFE call counts, shared augmentation metadata, finite gradients at the terminal small gap, uneven gradient accumulation, and next-update equivalence after restoring optimizer/EMA/RNG.

Real 64×64 Pokémon images, augmentation, optimizer and EMA were active during profiling. Effective batch stayed 128. The following timings are medians of five measured updates after two warmup updates, not sustained-run guarantees.

| Microbatch | Seconds/update | Images/second | Peak allocated GB | Result |
|---|---:|---:|---:|---|
| 16 | 1.7741 | 72.15 | 4.726 | Fits |
| **32** | **1.6895** | **75.76** | **8.609** | Selected |
| 64 | 1.6971 | 75.42 | 16.389 | Fits; effectively tied on short timings |
| 128 | — | — | — | OOM at the enforced 18 GB profiling allowance |

GB above means decimal bytes/1e9. Microbatch 32 offers much more headroom than 64 at similar throughput. Its peak reserved memory was 9.733 GB. Actual training checks a 20,000,000,000-byte allocation ceiling. CUDA context and libraries add device overhead; nvidia-smi telemetry is logged separately.

A full-size integration smoke completed three updates and resumed to update four, preserving schedule length 20,000. Saved EMA exports loaded in an independent working directory through the protected FastGen loader with no architecture arguments; each had 39,667,843 parameters and the configured call count. The real official clean-FID functions evaluated all 3,020 generated images against the training-derived holdout: **469.2080 at update 4, NFE 1, seed 42**. This essentially untrained score validates the pipeline, not sample quality or convergence. It is not an official validation score.

Host memory is reported as parent RSS plus sampled process-tree PSS including data-loader workers. In the workflow smoke's first update, five processes had 1.808 GB aggregate PSS; summing their RSS gives 6.172 GB because shared pages are counted repeatedly. Continued training reports sustained values.

## Baseline execution and monitoring

Full workflow qualification is in progress; the 20,000-update baseline has not yet started. Its configuration is `configs/pokemon_local.json`, with effective batch 128, microbatch 32, FP32 RAdam at 1e-4, EMA warmup capped at 0.9999, and the upstream consistency-only time distributions/objective. The shortened curriculum changes the training recipe; this is not a paper reproduction.

20,000 updates process 2,560,000 images, approximately 182 complete passes over the optimization partition. The short profile projects about 9.4 hours for optimizer updates, plus checkpointing, loading, sampling and evaluation. The approved ceiling is 24 cumulative GPU-job hours, including qualification; four hours are reserved for final evaluation. Do not infer convergence from that estimate.

The persistent workflow writes JSONL loss/gradient/timing/memory logs, checkpoints and labeled grids every 500 updates, and refreshed Markdown/HTML reports. Intermediate development FID runs at updates 5,000 and 10,000 with NFE 1 and 4, seed 42. Terminal evaluation compares all four NFEs with seeds 42 and 43; the few-call export uses the lowest mean development FID among modes with complete seed results. If evaluation cannot finish, the exported metadata records the incomplete selection and the workflow reports that limitation. It does not manufacture scores.

Checkpoint resume includes optimizer, EMA, random generators, data-order epoch/cursor and partition identity. Changing microbatch or GPU can change the numerical trajectory; resume does not promise bitwise agreement across hardware. A forced crash can lose updates after the last atomic checkpoint. The budget file is updated every 10 updates and workflow boundaries; up to the last logging interval can be unaccounted after a hard system crash, and a timeout permits 30 seconds for graceful shutdown.

Follow `runs/pokemon_unet128_seed42/console.log`, `status.json`, `active_job.json`, `REPORT.md` and `report.html`. `exit_status=completed` means training/evaluation/export workflow completion, not a target FID or TA acceptance. Report generation and inference export run on CPU after the GPU work. Terminal source/artifact archives go to `dist/pokemon_unet128_seed42`.

## Provenance

Official starter pinned at `e3f5120000cd91037017d86f23993492195101a5`; vendored SoFlow at `cc47661a0b0843f8f1b6cff3a17107a052d18b21`. See [attribution](../THIRD_PARTY.md). The seven protected files and the base `Model` class have hash checks before training and export. Current modifications preserve those hashes.

Raw evidence is deliberately outside Git: `artifacts/tests.log`, `artifacts/pokemon_profile/summary.json`, `artifacts/integration_smoke.log`, `artifacts/export_verification.json`, `artifacts/fid_smoke.log`, and the corresponding run folders. Migration archives carry these artifacts. Source control contains this measured summary so it remains useful without the local logs.
