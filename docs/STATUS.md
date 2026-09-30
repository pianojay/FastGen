# Local Pokémon/FastGen experiment status

Updated 2026-10-01, Asia/Seoul. This experiment ran on the local TITAN RTX server; it is not the KCloud VM or an official submission.

## Completed baseline — reviewed 2026-09-28

The full 20,000-update workflow completed on September 25, including all terminal FIDs, export verification and SFTP archives. [Final results and interpretation](POKEMON_BASELINE_RESULTS.md) supersede the historical running snapshots below. Mean development FID across sampling seeds 42/43 is **113.9551 / 46.5978 / 40.2773 / 38.2858** for NFE **1 / 2 / 3 / 4**. Four calls were selected. These are training-derived development scores, not official validation scores.

Recorded GPU-job budget use: **10.1978 hours**; peak training allocation: **8.613 GB**; peak sampled process-tree PSS: **2.120 GB**. Each exported inference model has **39,667,843 parameters** and verified 1/4 backbone calls. All 20,000 logged updates have finite statistics. Export and bundle hashes were rechecked on September 28. The final panels show recognizable category-dependent forms at four calls and substantial one-call artifacts; broader class/diversity/memorization checks remain open. This review started no additional GPU work.

## Decisions and boundaries

### October 1 S2 running — current execution

The wider **61,934,243-parameter U-Net160** passed two bounded qualification probes on GPU 1; every probe used fresh disposable weights, full augmentation, optimizer/EMA and effective batch 128. Microbatch 16 is selected after measuring **2.5753 seconds/update** versus **4.1306** at microbatch 32, with stable last-half/terminal timings of 2.5826/2.5860 seconds. Its peak qualification allocation was **6,164,570,624 bytes**, sampled process-tree PSS **2,644,388,864 bytes**; all 18 GPU contract/numerical tests passed. Both probes cost **854.159 GPU-job seconds** and are charged to the S2 method/campaign. [Measurement and allocation evidence](research/evidence/s2-allocation-20261001.json).

The [production recipe](../configs/research_s2_train.json) starts again from scratch for **40,000 updates**, matching S1's full horizon, objective, optimizer, EMA, augmentation and effective batch. The selected full optimizer estimate is **28.614 hours**. Its [frozen protocol](../configs/research_s2_20261001.json) permits a **33-hour cumulative training watchdog**, **90 minutes for paired selection/confirmation**, and **five minutes for production contracts**, with one bounded recovery and no allowance renewal. Training/sampling implementation hashes match S1; width and execution microbatch differ. The longer watchdog includes measured overhead and uncertainty, without adding updates. Qualification used an 18GB allocation target; the actual production peak will be measured under the strict 20GB guard.

S2 launched at **04:48:27 KST**, workflow PID **2529344**, prepared revision `e3657b2`, on branch `research/s2-scratch-unet160-20261001`. Production contracts passed all 18 tests (8.296 GPU-job seconds), and all four S1 control passes reproduced the prior FIDs and generated-image hashes exactly (244.847 seconds). The trainer, PID **2530771**, started at **04:52:41 KST** without `--resume`. Its initial checkpoint was verified at update/epoch/cursor zero, with an empty optimizer, identical initial model/EMA weights and the full 40k horizon. [Launch evidence and process identities](research/evidence/s2-launch-20261001.json).

At **04:54:55 KST**, the first **47 consecutive finite updates** averaged **2.5828 seconds**, with peak training allocation so far **6,159,269,888 bytes** and sampled process-tree PSS **1,849,230,336 bytes**. These are early measurements, not final maxima. There is no S2 FID or retry; S1 remains both track incumbents. Campaign cost before production was **20.7092 GPU-job hours**, already including both pretraining qualification probes. The early running campaign estimate is 20.8167 hours, excluding inherited EXP-1; final stage accounting is pending.

Launch command: `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/user/miniconda3/envs/fastgen_soflow/bin/python -m scripts.training_study --config configs/research_s2_20261001.json` after `--prepare`. Live authority is `runs/research_s2_unet160_40k_seed42/record.json`, `status.json`, `active_job.json` and the stage budget files. Source/configuration hashes are frozen; only documentation may change while this workflow runs. Next: supervise the full schedule and retained 20k/30k checkpoints, then the predeclared final 40k evaluation and eligible confirmation.

### October 1 S1 completed — current research result

S1 finished at **04:15:55 KST on October 1**, including its fresh 40,000-update schedule and all 32 paired development FID passes. The final EMA checkpoint improves both tracks on both selection seeds and all four reserved confirmation seeds. Confirmation mean FID is **82.3062 / 29.4079** at one/four calls, against **113.0753 / 38.2838** for EXP-1 under identical actual noise: **27.2% / 23.2% lower**. S1 is now the separate one-call and four-call incumbent. This confirms generation-seed consistency for one fresh training seed; it is not a fresh-training replication or an official validation result. [Complete evidence, per-seed scores, hashes and costs](research/evidence/s1-complete-20261001.json).

All **40,000 updates** are finite and consecutive, covering 5.12M image presentations. The final checkpoint has **39,667,843 parameters**; actual peak training allocation was **8,611,038,720 bytes**, sampled process-tree PSS **2,053,035,008 bytes**, and host-process peak RSS **2,111,864,832 bytes**. Measured optimizer time was **19.1362 hours**, versus the pre-run estimate of 19.1124. Training GPU-job time, including startup/checkpoints/panels, was **19.8025 hours**; evaluation **0.5372 hours**; qualification **0.0023 hours**; total S1 **20.3420 hours**. Campaign cost including B1 is **20.4719 hours**, excluding the inherited EXP-1 cost of 10.1978 hours. No failure or recovery occurred; unused allowances did not become additional training.

| Checkpoint in S1's fixed 40k curriculum | Mean one-call FID | Mean four-call FID |
|---|---|---|
| 20k midpoint | 140.6670 | 41.0388 |
| 30k | 97.3934 | 32.2356 |
| Final 40k, predeclared primary | 82.6983 | 29.4154 |

These curve scores use selection seeds 3101/3102. The last 10k updates improved FID by 15.1% / 8.7%, with smaller absolute gains than the preceding interval. S1 changes the full horizon-dependent curriculum as well as exposure relative to EXP-1; its 20k midpoint is not the completed 20k recipe. The fixed final panels show recognizable category-dependent forms and cleaner four-call outputs, with visible distortions remaining; class accuracy, diversity and memorization have not been established. Two/three-call FID is unmeasured for S1.

The [training config](../configs/research_s1_train.json) and [protocol](../configs/research_s1_20260930.json) remained frozen, with scratch model/optimizer/EMA/RNG/data state, seed 42, effective batch 128 and microbatch 32. Prepared source revision `1a4853e`, final checkpoint SHA-256 `b2c48b278dbaf5aebbecff6033a7ef26c68b45cef677b34dbb02f70d2412f688`. Protected files/base Model and frozen source/configuration hashes passed; actual noise pairing, 3,020 images per pass, and 95/380 backbone calls passed verification. All 16 GPU contract/numerical tests passed before training. Artifacts and complete stage accounting remain in `runs/research_s1_unet128_40k_seed42/`; `record.json` and `exit_status` report completed, and GPU 1 is idle after workflow exit.

Command used: `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/user/miniconda3/envs/fastgen_soflow/bin/python -m scripts.training_study --config configs/research_s1_20260930.json` after its `--prepare` step. The [launch](research/evidence/s1-launch-20260930.json), [20k](research/evidence/s1-20000-20260930.json) and [30k](research/evidence/s1-30000-20260930.json) snapshots remain historical evidence.

The inference package at `runs/research_s1_unet128_40k_seed42/export/` passed checksum verification and the official loader in its own directory: both models contain 39,667,843 parameters, return finite bounded images, and make exactly 1/4 backbone calls. Verification used CPU only (18.56 seconds including export), with zero additional GPU-job time. [Export evidence](research/evidence/s1-export-20261001.json).

Next: qualify **fresh U-Net160 (61,934,243 parameters), 40k updates** on GPU 1 and assign its full optimizer hours from measured augmented updates. Match S1's exposure and complete schedules to answer the capacity question; do not extend S1 or initialize from its weights. S3 remains conditional. The [official course page](https://generative-models.kaist.ac.kr/fastgen/) was directly rechecked October 1: TA reference FIDs remain 39.15 / 30.90 and the parameter limit remains 100M. Our different development references prevent an official score comparison.

### September 30 longer-training authorization and scratch-only plan — current

The user removed the fixed training-hour limit while retaining fewer than 100M loaded inference parameters and peak training allocation below 20,000,000,000 bytes. [program.md](../program.md#resources-and-boundaries) and AGENTS.md now require finite optimizer-hour allocations justified by learning and throughput evidence, with separately accounted overhead, qualification, evaluation and recovery. Older 12-hour policies below are historical; no replacement universal time cap or new per-experiment approval gate was introduced.

The user's subsequent correction requires **every new training experiment to start from scratch**. The unlaunched 20k-checkpoint continuation proposal is superseded. The [revised plan](RESEARCH_LOG.md#september-30-revised-scaling-decision) now selects **fresh U-Net128 and U-Net160 runs, each from update zero to 40k**, with the full optimizer/EMA/curriculum policy declared at initialization. EXP-1 is an evaluation reference only. Each run gets fresh model, optimizer, EMA, RNG and data state; exact recovery is limited to the same interrupted experiment with its schedule unchanged. A later longer horizon also requires a new scratch run.

At the measured 1.7201 seconds/update, U-Net128 requires **19.11 hours of new optimizer work**, excluding overhead/evaluation; the old 9.56-hour incremental estimate no longer applies. Historical 10k → 20k FID reductions of 34.4% / 33.3% motivate a complete longer recipe, not a quality forecast. Setting `total_steps=40000` from initialization lengthens the gap curriculum; S1's 20k midpoint is not the completed 20k recipe. Current code has fixed-LR RAdam, with progress-dependent optimizer state, EMA warmup and gap curriculum. S2 matches S1's full schedule to compare capacity; its hours require actual wider-model timings. U-Net192 remains conditional. [Calculation and initialization record](research/evidence/scaling-20260930.json).

Validation uses `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 /home/user/miniconda3/envs/fastgen_soflow/bin/python` with standard-library-only checks of initialization records, arithmetic, protected file/base-Model hashes and documentation links, plus `git diff --check`. No training, profiling or FID evaluation was launched. Next work is fresh-run/evaluation configuration and changed-architecture qualification on GPU 1; no continuation implementation is needed. S1's proposed evaluation allowance becomes 50 minutes for at most 32 primary passes because its new 20k midpoint and historical EXP-1 are distinct checkpoints.

### September 30 duration and capacity review — initial proposal, superseded above

The user is considering longer training and heavier SoFlow models. Read program.md, the baseline, strategy, catalog, current code and retained research records. `git reflog -12 --date=iso` and `git show 5d3db1b:docs/RESEARCH_LOG.md` recovered the later direction-selection record even though its research branch is absent from the current branch list. That record proposed C3/C1/D2 recipe comparisons; it did not launch them. The local [B1 result](../runs/research_b1_20260930/result_summary.json) records a completed negative sampler comparison: uniform four-call FID 38.0398 versus 38.2193 / 38.4546, with both candidates worse on both selection seeds. These scores use a different RNG protocol/seeds from EXP-1 and do not change its incumbents. `git worktree list --porcelain`, `git ls-remote --heads origin` and a matching process-list inspection found no additional checked-out research worktree, remote research branch or live FastGen training/evaluation process at review time.

The baseline's seed-42 FIDs improved from 174.6829 / 57.4169 at 10k to 114.6036 / 38.3041 at 20k for one/four calls. This supports a duration comparison without establishing that capacity or duration is the bottleneck. Proposed order for the user's scaling question: a longer U-Net128 control, then a fresh U-Net160 at matched image presentations and an explicitly matched curriculum; consider U-Net192 if the result and resource measurements justify it. Existing parameter counts are 39,667,843 / 61,934,243 / 89,140,419, but wider-model training memory and throughput remain unmeasured. A duration gain would support further compute allocation; a width gain over the duration control would support capacity scaling. An unresolved or negative comparison would return attention to the previously proposed objective/time-coverage recipes rather than automatically widen again.

At that review, the 12-hour cumulative default still applied. The later authorization above removes it. The timing estimates remain valid: another 10k updates extrapolate to 4.78 optimizer hours; 30k / 40k total updates extrapolate to 14.33 / 19.11 optimizer hours, excluding other job overhead. Continuation must retain optimizer/EMA/RNG/data state and explicitly hold the terminal gap; changing `total_steps` alone alters the curriculum and is correctly rejected by exact resume. The initial review launched no GPU work.

### September 30 autonomous research policy — current

The user clarified that a started research loop should run consecutive experiments without approval between them, including meaningful hours-long training. This supersedes the earlier per-experiment assessment gate below. program.md now organizes interventions by their role in finding substantial FID gains: validity/targeted diagnosis, recipe design, recipe optimization/compute scaling, and checkpoint/sampler selection, with evidence-based exceptions. Cheap checks must inform a decision rather than displace recipe exploration indefinitely. The original 12-hour default was subsequently removed by the longer-training authorization above; any later explicit user campaign limit still applies.

[RESEARCH_LOG.md](RESEARCH_LOG.md) supplies compact planned/result tables and linked configurations. Full essays and fresh statistical protocols per trial are not required. Existing reports and CPU recovery evidence remain historical records. This policy edit starts no GPU work and does not assert that experiments on other research branches disappeared; import their relevant evidence before choosing the next run.

### September 30 promptware recovery rehearsal

Reconciled program.md and AGENTS.md with the user's per-training-run budget and approval between substantial experiments. Removed the fixed exploration/confirmation split. A clean agent recovered a real CPU-only dummy worker after orderly termination: logged step 8, saved checkpoint 5, correct final sum 210 at step 20. Seven checks passed; total worker time 23.19 seconds against a 60-second allowance across both attempts. A concurrent duplicate launch was rejected by the worker's lock. No model training or GPU work occurred. [Report, evidence, and limitations](PROMPTWARE_RECOVERY_REHEARSAL.md) distinguish this from hard-crash or production-trainer validation. Post-test instructions require reconciling stale status fields and unresolved accounting after unclean exits.

### September 29 autoresearch instructions and research catalog

[Research instructions](../program.md) and the [SoFlow research catalog](AUTORESEARCH_DIRECTIONS.md) now connect the four agreed directions—error audits, frozen-model sampling, one-call learning, and the baseline recipe—to bounded experiment briefs. The detailed SoFlow options are candidate interventions, not a fixed sweep. Each experiment records its decision, competing explanations, control, exact intervention, budget, evaluation/confirmation rule, and conditional next questions. Quality improvement and explanatory support are separate conclusions; one-call and few-call candidates are tracked separately.

Historical correction on September 30, superseded by the autonomy and longer-training policies above: the then-default was 12 cumulative GPU-job hours per model-training run, including failures and recovery, with per-experiment assessment. Evaluation and confirmation were budgeted separately without a fixed 10/2 split. The matched-branch and fresh-training distinctions remain useful. That documentation started no session and introduced no controller, model/training API change, training, sampling, calibration, or GPU evaluation.

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
