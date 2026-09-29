# Local Pokémon/FastGen experiment status

Updated 2026-09-30, Asia/Seoul. This experiment ran on the local TITAN RTX server; it is not the KCloud VM or an official submission.

## Completed baseline — reviewed 2026-09-28

The full 20,000-update workflow completed on September 25, including all terminal FIDs, export verification and SFTP archives. [Final results and interpretation](POKEMON_BASELINE_RESULTS.md) supersede the historical running snapshots below. Mean development FID across sampling seeds 42/43 is **113.9551 / 46.5978 / 40.2773 / 38.2858** for NFE **1 / 2 / 3 / 4**. Four calls were selected. These are training-derived development scores, not official validation scores.

Recorded GPU-job budget use: **10.1978 hours**; peak training allocation: **8.613 GB**; peak sampled process-tree PSS: **2.120 GB**. Each exported inference model has **39,667,843 parameters** and verified 1/4 backbone calls. All 20,000 logged updates have finite statistics. Export and bundle hashes were rechecked on September 28. The final panels show recognizable category-dependent forms at four calls and substantial one-call artifacts; broader class/diversity/memorization checks remain open. This review started no additional GPU work.

## Decisions and boundaries

### September 30 promptware recovery rehearsal

Reconciled program.md and AGENTS.md with the user's per-training-run budget and approval between substantial experiments. Removed the fixed exploration/confirmation split. A clean agent recovered a real CPU-only dummy worker after orderly termination: logged step 8, saved checkpoint 5, correct final sum 210 at step 20. Seven checks passed; total worker time 23.19 seconds against a 60-second allowance across both attempts. A concurrent duplicate launch was rejected by the worker's lock. No model training or GPU work occurred. [Report, evidence, and limitations](PROMPTWARE_RECOVERY_REHEARSAL.md) distinguish this from hard-crash or production-trainer validation. Post-test instructions require reconciling stale status fields and unresolved accounting after unclean exits.

### September 29 autoresearch instructions and research catalog

[Research instructions](../program.md) and the [SoFlow research catalog](AUTORESEARCH_DIRECTIONS.md) now connect the four agreed directions—error audits, frozen-model sampling, one-call learning, and the baseline recipe—to bounded experiment briefs. The detailed SoFlow options are candidate interventions, not a fixed sweep. Each experiment records its decision, competing explanations, control, exact intervention, budget, evaluation/confirmation rule, and conditional next questions. Quality improvement and explanatory support are separate conclusions; one-call and few-call candidates are tracked separately.

Correction on September 30: the default is 12 cumulative GPU-job hours per model-training run, including failures and recovery; evaluation and confirmation are budgeted per approved experiment, without a fixed 10/2 split. Each substantial next experiment requires user assessment and approval. It favors matched checkpoint branches for probing and fresh-training confirmation for recipe claims. This documentation does not start that session or replenish the completed baseline's budget. No controller, model/training API change, new training, sampling, calibration, or GPU evaluation was introduced.

Readiness notes identify prerequisite work, including explicit recipe branching, configurable sampling times, matched randomness, schedule-aware evaluation records, and objective mixing. The existing exact-resume guard remains intact. Link/fact review and five written scenarios cover an informative audit, confirmed FID improvement, inconclusive short run, implementation failure, and exhausted budget. These are documentation checks, not new experimental evidence.

### September 28 training-curve readability

The local report now stacks consistency MSE, gradient norm and update time vertically. Every raw observation remains visible underneath a centered 201-record rolling median and local 10th–90th percentile band. MSE additionally shows the rolling mean to expose sensitivity to high values, with a log-scale default and a linked full-range linear alternative. No data are clipped, classified as outliers or removed; the band is observed spread, not a confidence interval. `--smooth-window` controls the positive odd window length, with shorter edge windows. Historical archived evidence images remain unchanged.

The full run was rendered and inspected. Between updates 9,001–10,000 and 19,001–20,000, median consistency MSE increased from 0.9424 to 1.4577, while the 90th percentile increased from 1.8966 to 3.4752. Thus the later increase is not solely a handful of isolated extremes. This remains a diagnostic under a changing consistency-gap curriculum, not evidence by itself of worse generation quality; measured FID improved. This visualization change uses CPU only.

### September 28 gallery and experiment-order clarification

Browse categories displays one real held-out reference inside a compact sticky toolbar, adjacent to the category controls. It uses the first indexed development reference for the selected class, fixed across NFE/checkpoint/seed changes, with enlargement and an accessible source label/tooltip. The comparison grid follows immediately; repeated category headings and sample category captions were removed, and explanations are collapsed under Comparison notes. Sample captions retain NFE and index so they remain identifiable after scrolling past the column headers. The toolbar and reference stay visible when scrolling the 20 sample rows. Desktop/mobile layout, filtering and enlargement were checked in Chromium. The existing expandable 20-reference collection remains available for pose/style variation. This supports recognizing obvious class mismatches; a single example is not a class-fidelity measurement. No model sampling or data-partition changes were needed.

`python -m scripts.gallery --run-dir runs/pokemon_unet128_seed42` builds `gallery.html` beside the original report. The offline gallery indexes 36,240 existing generated images from 12 evaluation passes, 3,020 development references and 41 saved panel checkpoints. It supports category search, NFE/checkpoint/seed comparisons, image enlargement and downloadable browser-local review notes. It does not generate images or perform nearest-neighbor analysis. Evaluation rows are not reliably noise-paired across NFE; the fixed training panels reset their initial noise and are labeled separately.

Validation: the generator checked all evaluation filenames/counts and reference mappings against the partition hashes. Headless Chromium loaded the gallery directly from disk and passed filter, NFE/seed/checkpoint, zoom, references, note persistence/download, progression and mobile overflow checks with no JavaScript errors. Selected PNGs decoded successfully and desktop/mobile screenshots were inspected. Protected upstream hashes still pass. No GPU tests or training were rerun for this visualization change.

Longer training is a priority candidate, not a last resort. Both one-call and four-call FID improved from 10k to 20k updates; this supports testing more training, without proving its benefit. A small frozen-backbone schedule comparison can answer a cheaper, complementary question first, but cannot repair everything the backbone has not learned, and its best settings may change after further training. A one-call objective redesign should follow a clear training-duration control rather than automatically precede it.

An explicit continuation at the terminal consistency-gap ratio and a fresh longer curriculum are different experiments. Changing `total_steps` from 20k to 40k halfway through changes the gap ratio at update 20k from approximately 1/3200 to 1/sqrt(3200), about 57 times larger; it is not a neutral duration extension. The existing resume guard correctly rejects such changes. A proposed +10k-update terminal-curriculum continuation would require a separate recorded branch/run and explicit EMA behavior, and costs approximately 4.78 optimizer hours at observed throughput, plus evaluation and overhead. No continuation or sampler experiment has been launched.

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

Full workflow qualification passed: one actual update, development FID for all four NFEs (one seed for this smoke), automatic selection, inference export, independent checkpoint verification and SHA-256-verified source/artifact archives. The fresh 20,000-update baseline **started at 21:18 KST on 2026-09-24 and is running, not complete**. Workflow PID 433329; initial training PID 433343. Its initial source revision is `38137db`; each run/checkpoint records source hashes and Git provenance. The one-time qualification charge was 586.38 seconds, including a conservative 120-second allowance for ancillary GPU checks. Its configuration is `configs/pokemon_local.json`, with effective batch 128, microbatch 32, FP32 RAdam at 1e-4, EMA warmup capped at 0.9999, and the upstream consistency-only time distributions/objective. The shortened curriculum changes the training recipe; this is not a paper reproduction.

At update 40, the live baseline reported consistency MSE 1.2876, gradient norm 2.9529, 1.705 seconds/update, 8.610 GB peak torch allocation and 1.809 GB process-tree PSS. These early values establish finite training and stable memory, not convergence. Labeled smoke samples are still noise, as expected after only a few updates. Read the live run files for subsequent progress rather than treating this dated snapshot as current.

20,000 updates process 2,560,000 images, approximately 182 complete passes over the optimization partition. The short profile projects about 9.4 hours for optimizer updates, plus checkpointing, loading, sampling and evaluation. The approved ceiling is 24 cumulative GPU-job hours, including qualification; four hours are reserved for final evaluation. Do not infer convergence from that estimate.

The persistent workflow writes JSONL loss/gradient/timing/memory logs, checkpoints and labeled grids every 500 updates, and refreshed Markdown/HTML reports. Intermediate development FID runs at updates 5,000 and 10,000 with NFE 1 and 4, seed 42. Terminal evaluation compares all four NFEs with seeds 42 and 43; the few-call export uses the lowest mean development FID among modes with complete seed results. If evaluation cannot finish, the exported metadata records the incomplete selection and the workflow reports that limitation. It does not manufacture scores.

Checkpoint resume includes optimizer, EMA, random generators, data-order epoch/cursor and partition identity. Changing microbatch or GPU can change the numerical trajectory; resume does not promise bitwise agreement across hardware. A forced crash can lose updates after the last atomic checkpoint. The budget file is updated every 10 updates and workflow boundaries; up to the last logging interval can be unaccounted after a hard system crash, and a timeout permits 30 seconds for graceful shutdown.

Follow `runs/pokemon_unet128_seed42/console.log`, `status.json`, `active_job.json`, `REPORT.md` and `report.html`. `exit_status=completed` means training/evaluation/export workflow completion, not a target FID or TA acceptance. Report generation and inference export run on CPU after the GPU work. Terminal source/artifact archives go to `dist/pokemon_unet128_seed42`.

## Provenance

Official starter pinned at `e3f5120000cd91037017d86f23993492195101a5`; vendored SoFlow at `cc47661a0b0843f8f1b6cff3a17107a052d18b21`. See [attribution](../THIRD_PARTY.md). The seven protected files and the base `Model` class have hash checks before training and export. Current modifications preserve those hashes.

Raw evidence is deliberately outside Git: `artifacts/tests.log`, `artifacts/pokemon_profile/summary.json`, `artifacts/integration_smoke.log`, `artifacts/export_verification.json`, `artifacts/fid_smoke.log`, and the corresponding run folders. Migration archives carry these artifacts. Source control contains this measured summary so it remains useful without the local logs.


## Shared CIFAR feasibility report — 2026-09-24

The separate GPU 0 CIFAR experiment is closed by an authorized feasibility decision.
Its [formal report](soflow-cifar10/EXPERIMENT_REPORT.md) documents 130 logged updates,
initial trainability, 13.603 seconds/update, and an approximately 126-day full-run
projection. No generation-quality evaluation was performed; the latest saved main
checkpoint is step 5. These findings describe the original CIFAR configuration and
are not Pokémon measurements. This status entry links the CIFAR report only; the
Pokémon experiment's measurements and status are documented above.

## Shared training-cost literature audit — 2026-09-24

The [few-NFE literature audit](soflow-cifar10/few-nfe-training-cost-audit.md)
compares prescribed training budgets for SoFlow, MeanFlow, Shortcut, iCT,
Consistency Distillation, Progressive Distillation, SFD, and ECT (seven
comparison methods plus the SoFlow reference). It separates pretrained
fine-tuning from total training cost and records a budget discrepancy in
ECT's script named `run_ecm_1hour.sh`. This was documentation-only work;
published measurements are not new local results. FastGen's requirement
to initialize generative weights from scratch remains applicable.

The audit now also covers LD3 and BézierFlow as sampler optimization methods.
Their reported minute-scale calibration uses already trained generators;
it does not measure backbone learning time. Inspection of the CIFAR checkout
found a finite-jump denoise-and-renoise sampler, so ODE-scheduler methods
would require adaptation. This follow-up adds literature and code analysis
only; no training, calibration, or evaluation was launched.

## Consolidated interpretation — 2026-09-24

[FastGen strategy](FASTGEN_STRATEGY.md) integrates the CIFAR experiment, paper-cost audit and Pokémon baseline. The Pokémon presentation budget is 320 times smaller than the full CIFAR recipe. At 22:06 KST, update 1,650 was running with finite statistics and stable memory. The saved update-1,500 four-call panel shows coarse shapes; one-call samples remain noisy and class fidelity is still uncertain. Scheduled development FID is pending. The strategy records criteria for later sampler calibration, one-call tuning or capacity experiments; this review changed no training settings and launched no additional GPU work.
