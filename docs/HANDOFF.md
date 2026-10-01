# Context and source map

Prepared October 1, 2026 (Asia/Seoul). This document is a dated handoff, not a live process monitor. Decisions come from the user's approved preparation plan. Root [README_LOCAL](../README_LOCAL.md) distinguishes available source from work still required.

## Why this experiment

The team seeks strong Pokémon generation at exactly one and at most four model calls. The next experiment asks whether direct adversarial transport learning can produce a competitive one-call generator without first training a diffusion teacher. It tests discrete AFM in pixels with a familiar moderate U-Net, not another SoFlow capacity sweep. A strong teacher-free result or a well-measured limitation is useful; success is not guaranteed by the paper.

The first baseline covers one call only. Few-step AFM, AYF-inspired SoFlow post-training, CAFM and new capacity experiments are subsequent proposals. U-Net128 is an initial design reference, not a frozen architecture specification. The implementation agent chooses and records the discriminator architecture and full recipe before its one substantive training run.

## Evidence available now

| Work | Evidence and scope | Cost/status |
|---|---|---|
| SoFlow EXP-1, U-Net128, 20k | Local development mean FID 113.9551 / 38.2858 at 1 / 4 calls, sampling seeds 42/43; 39,667,843 inference parameters | 10.1978 GPU-job hours; completed |
| SoFlow S1, fresh U-Net128, 40k | Local confirmation mean 82.3062 / 29.4079 at 1 / 4 calls; paired EXP-1 controls 113.0753 / 38.2838 | 19.1362 optimizer hours; 20.3420 total qualification/train/evaluation hours; completed |
| SoFlow S2, fresh U-Net160, 40k | 61,934,243 parameters; GPU 1. At pinned source milestone 10k checkpoint verified; no candidate FID yet | Projected 28.614 optimizer hours; running as of handoff; U-Net192 remains conditional |
| Sangwoo: EDM + SiDA | Reported FID 28.07 at 1 call, 22.79 at 2 calls; ~61.8M student; teacher holdout FID ~17.8 at 35 NFE | RTX PRO 6000 Blackwell: teacher ~11–14h, one-step distillation ~8h, two-step ~5h; reported distillation VRAM 18.4–18.9GB |
| Ryan: BigGAN + DiffAugment | Reported FID 55.60 / 55.30 from repeated one-pass evaluation, not a learned few-step sampler; G 13.7M, D 19.7M | 11.95h/100k steps, reportedly peaked around 45k; peak VRAM 11.73GB |

S1 changes the horizon-dependent curriculum as well as exposure. Its gain is not a duration-only ablation or proof that SoFlow cannot scale. Sampling-seed replication is not independent training replication. Our development set excludes byte-duplicate holdout candidates; colleagues' exact split/preprocessing/noise may differ. Do not rank these as a common official leaderboard or extrapolate Blackwell runtimes to TITAN.

Outstanding team questions: SiD versus SiDA using the same teacher; other attempted/rejected one-call methods; complete cost and evaluation identities. Sangwoo said experiments were ongoing. Ryan mentioned future GAN/diffusion refinement. Unreported parallel work is speculation.

Sources:

- [SoFlow S1 post](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790813565465619), [S2 post](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790814777573579).
- [SiDA thread and cost tables](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790834501987609), [unanswered comparisons](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790840632645159).
- [GAN results and one-pass clarification](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790830169903529).
- [Meeting plan](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790603380548339): Oct. 7 after class, identify workable candidates; Oct. 26 select representative models; Oct. 31 optional mid-evaluation; Nov. 7 final submission.
- [Meeting memo](https://app.notion.com/p/3e9f3544609181deb002cd5e50c0651a), [EXP-1](https://app.notion.com/p/3e5f3544609181f5a1d7e29d547ea7fc). These are context links, not copied Notion content.

## Sources and adaptation

Exact full revisions and checksums are in [provenance](../resources/provenance.json): FastGen `3ccf326`, AFM `ab60013`, SoFlow reference `3841b79`. The new FastGen starter differs from SoFlow's pinned starter only in README text; protected code, manifests and base Model match. The latest rules explicitly include every training model in the individual 100M limit. The starter remains untouched here.

Read [AFM paper](https://arxiv.org/html/2511.22475v2) and the pinned snapshot, not just its README. [Align Your Flow](https://research.nvidia.com/labs/toronto-ai/AlignYourFlow/) is flow-map distillation plus optional adversarial fine-tuning. [CAFM](https://arxiv.org/html/2604.11521v1) discriminates velocity directions through a learned potential's derivatives; its major SiT/JiT results retain many integration steps. Neither is the baseline requested here.

| AFM source in the unmodified snapshot | Read for |
|---|---|
| [One-call config](../vendor/adversarial-flow-models/configs/train/afm/train_afm_1nfe.yaml) | Reference optimizer, OT decay, gradient penalties, EMA and architecture settings |
| [Trainer](../vendor/adversarial-flow-models/train_adversarial_flow_imagenet.py) | `prepare_input`, `training_step`, `is_dis_step`, `optimizer_step`, `ema_step`, `augment`, scheduling and checkpoint state |
| [Gradient normalization](../vendor/adversarial-flow-models/grad_norm.py) | Custom backward and persistent running-square buffer; this is not ordinary gradient clipping |
| [Schedules](../vendor/adversarial-flow-models/common/schedules.py) | Schedule indexing and end behavior |
| [Generator](../vendor/adversarial-flow-models/models/afm/dit/generator.py), [discriminator](../vendor/adversarial-flow-models/models/afm/dit/discriminator.py) | Class conditioning, endpoint/time interfaces and scalar discrimination; architectures are references, not Pokémon implementations |
| [Dependencies](../vendor/adversarial-flow-models/requirements_afm.txt) | Dependency audit only; do not bulk install this into the course environment |

Important porting facts:

- For one-step `s=1,t=0`, G maps image-shaped Gaussian noise to an image. The OT loss penalizes displacement from the **source noise**, not MSE against a randomly paired real image. G uses relativistic adversarial loss plus OT; D uses adversarial, finite-difference R1/R2 and logit-centering terms. Trace sign, reduction and gradient routing to code.
- The reference config uses x-prediction, AdamW betas (0,0.9), lr 1e-4, weight decay .01, EMA .9999, OT .2→.005, GP .25 on .25 of the batch, epsilon .01 and centering .01. These are **ImageNet latent-space reference values**, not an approved pixel-space recipe.
- Its eight-GPU, five-million-loop-step configuration is not a sensible ready-made local allocation. One loop step is an alternating D/G microstep; generator updates, discriminator updates, accumulated microsteps and data presentations must be distinguished when porting schedules.
- Real-data x-flip probability is .5. Discriminator augmentation defaults to zero in this config; the optional path pairs real/fake transforms using translation and erasing. These are different from SoFlow's transform-conditioned consistency augmentation. Inspect Pokémon examples and explicitly justify augmentation choices; do not equate “GAN adaptation” with removing all augmentation.
- The trainer creates `aug_p_schedule` but `get_aug_p` checks `augment_p_schedule`; audit this mismatch if enabling a schedule. The optional EMA autoreload path references `self.dit`; it is disabled by default. Do not silently inherit untested optional paths.
- The snapshot includes a `dit.py` download placeholder, pretrained-VAE assumptions, distributed machinery and external evaluation assets. It is intentionally preserved as reference, **not runnable as a Pokémon experiment**. Pixel-space U-Net adaptation removes the VAE and supplies its own G/D implementations. Keep modifications outside the pinned snapshot.

## Reusing SoFlow knowledge without coupling repositories

Local checkout: `/home/user/jw_soflow/FastGen-SoFlow`. Inspect files with `git -C /home/user/jw_soflow/FastGen-SoFlow show 3841b79044b0c64c1af4e2ea925e2068f681b767:<path>`; the active checkout can change as its agent documents progress. Copy only justified components later, into AFM, with attribution and adapted tests. No cross-repository Python imports.

| Path at that revision | Reuse candidate / adaptation needed |
|---|---|
| `soflow_fastgen/unet.py`, `soflow_fastgen/models.py` | U-Net architecture, class mapping and wrapper. Preserve NVIDIA CC BY-NC-SA attribution if copied; remove irrelevant consistency/preconditioning assumptions explicitly. |
| `soflow_fastgen/data.py` | Fixed train-derived indices and deterministic data order; never regenerate the held-out split. |
| `soflow_fastgen/evaluation.py`, `scripts/evaluate_dev.py` | Official clean-FID functions and development references; replace SoFlow sampling/checkpoint assumptions. |
| `soflow_fastgen/paired_evaluation.py` | Actual paired noise streams and cache identities; sampler adapter is SoFlow-specific. |
| `soflow_fastgen/common.py`, `scripts/launch.py` | Atomic persistence, RNG, process identity, memory and cost accounting; do not copy GPU 1 defaults. |
| `soflow_fastgen/training.py`, `train.py` | Recovery/data-cursor patterns; AFM additionally needs two optimizers and gradient-normalization state. Its training objective is not reusable. |
| `scripts/report.py`, `scripts/gallery.py` | Vertical plots with raw data plus trends and class-reference gallery; adapt to AFM metrics, not a fake consistency-MSE graph. |
| `docs/POKEMON_BASELINE_RESULTS.md`, `docs/research/evidence/s1-complete-20261001.json`, `docs/STATUS.md` | Baseline, paired S1 result and dated S2 status; cite exact evidence rather than treating stale running prose as live state. |

Pinned online source root: https://github.com/pianojay/fastgen-soflow/tree/3841b79044b0c64c1af4e2ea925e2068f681b767

## Data, environment and migration

The [copied partition](../resources/development_split.json) has SHA-256 `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`: 14,059 optimization images, 3,020 development images (20/class). It indexes the unchanged official training manifest of 17,079 images. Official validation has a separate 3,020 images. The development selection prevents exact-byte duplicates across partitions; near-duplicates remain possible and the selection changes the holdout distribution.

Local image root is `/home/user/jw_soflow/FastGen-SoFlow/data/pokemon-generation-one-22k`; image paths in manifests begin `PokemonData/`. Read images there, keep AFM's own split manifests and output paths here. Data were already composited on white. Do not rerun the downloader on shared input. Future migration must replace the data-root setting and verify manifests/content; an absolute local path is not part of a self-contained submission.

No environment was created. The implementation agent should create `fastgen_afm` with Python and pip, set `PYTHONNOUSERSITE=1`, and install only audited dependencies using that environment's interpreter. The known local SoFlow stack uses torch 2.7.1+cu126 / torchvision .22.1+cu126; the official README suggests torch 2.6 / torchvision .21. Treat both as references, not proof of AFM compatibility. TITAN does not support native BF16: do not inherit upstream BF16 defaults. Start with an explicitly verified precision policy.

GPU 0 UUID observed at preparation: `GPU-e9d30df5-3b84-1a7e-c505-c6fff0a0ce7f`; recheck occupancy before launching. GPU 1 is owned by SoFlow. Source moves through this private GitHub repository; artifacts/data move separately through checked archives/SFTP when requested. KCloud needs fresh memory/throughput qualification; do not translate GPU timings directly.
