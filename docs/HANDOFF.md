# Context, mathematical conventions and source navigation

Prepared October 2, 2026 (Asia/Seoul), from the user's approved two-handoff plan. This is a dated research handoff, not live job status. [Experiment mandate](EXPERIMENT_PLAN.md) and [operational boundaries](../AGENTS.md) govern future work.

## Why this comparison

The motivating hypothesis is that teacher-informed distillation can make adversarial training less destructive to diversity. The assigned experiment tests the incremental effect of adding adversarial training **under a fixed flow teacher**. Without an adversarial-only arm, it cannot isolate the teacher's protective effect. Neither Gaussian score/velocity equivalence nor a favorable paper FID establishes that a flow recipe will beat Sangwoo's EDM/SiDA recipe.

SiD trains a generator using information about real and generated distributions; the final student is not simply regressing a teacher score or velocity field. A flow teacher supplies score-equivalent information through its velocity/denoiser conversion. This differs from AYF/CTM finite-transition or trajectory distillation. The new project adapts SiD-DiT's flow formulation to **class-conditioned pixel U-Nets**, not pretrained text-conditioned latent DiTs. Its published COCO/ImageNet numbers and our local Pokémon FIDs are not a common leaderboard.

| Existing evidence | Measurement scope | Interpretation |
|---|---|---|
| AFM baseline: 10,294 G/D cycles, G 39.66M, D 9.92M; 2.396 training-job hours | Stopped October 1; complete checkpoint at 10k; no trained FID | Strong within-class collapse in displayed classes; does not establish all AFM recipes fail |
| SoFlow U-Net128: local mean FID 113.9551 / 38.2858 at 1 / 4 calls, 20k updates | 14,059/3,020 train-derived partition; two sampling seeds | Completed local baseline; not an official score |
| SoFlow U-Net128, fresh 40k recipe: 82.3062 / 29.4079 at 1 / 4 calls | Four confirmation sampling seeds; ~20.342 total GPU-job hours | Improved longer recipe, including changed curriculum; not proof of a pure duration effect |
| SoFlow U-Net160, ~61.93M | Separate GPU 1 capacity experiment; consult its dated evidence at the pinned source | Preparation does not manage or assume live job progress |
| Sangwoo EDM/SiDA: reported 28.07 / 22.79 at 1 / 2 NFE; teacher holdout ~17.8 at 35 NFE | Reported ~61.8M DDPM++, 151 classes, 64×64; exact config/partition/checkpoint not supplied | Architecture-family reference and contextual comparison, not a controlled baseline |
| Ryan BigGAN + DiffAugment: reported 55.60 / 55.30 | Both one-pass evaluation draws; ~11.95h, G 13.7M, D 19.7M | GAN context; not a learned two-step comparison |

Sangwoo reports teacher x-flip, dropout .25 and optional geometric ADA; teacher-initialized student and fake-score/D; batch 128, lr 2e-5 and class-balanced sampling in a best variant. A late fake-score reset reportedly improved holdout quality. Do not copy that reset into just one arm or translate EDM sigma 2.5→.4 directly to flow times. His reported Blackwell teacher/distillation timings are not predictions for physical GPU 2. As of the reviewed thread, he had not disclosed a flow-SiDA trial and planned further summaries; absence of a post is not evidence of absence of work.

Context: [SiDA report/thread](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790834501987609), [same-teacher ablation question](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790840632645159), [Notion experiment](https://app.notion.com/p/3ecf35446091816e8e82e523cca6247c), [paper stash](https://app.notion.com/p/24ff9f816c1b4130b7eec1ac8ec2a64b). Relevant existing stash reviews: [DMD](https://app.notion.com/p/3e2f3544609181d9b229fd92299c04c7), [AYF](https://app.notion.com/p/3d7f35446091803d8fbac035a193a4c2), [CTM](https://app.notion.com/p/3d3f354460918108b399fda09190b800). These links do not authorize posts or page edits.

## Flow convention and losses to trace

Use clean data at t=0 and Gaussian noise at t=1, with class c:

\[
x_t=(1-t)x_0+t\epsilon,\qquad \epsilon\sim\mathcal N(0,I),\qquad
v^*(x_t,t,c)=\mathbb E[\epsilon-x_0\mid x_t,t,c].
\]

The scratch teacher learns conditional flow matching, with recorded time sampling and weighting:

\[
L_{\rm FM}=\mathbb E[w(t)\|v_\phi(x_t,t,c)-(\epsilon-x_0)\|^2].
\]

For a velocity predictor, the denoiser is \(D_\phi=x_t-t v_\phi\). For 0<t<1, its corresponding score estimate is

\[
s_\phi(x_t,t,c)=\frac{(1-t)D_\phi(x_t,t,c)-x_t}{t^2}
=-\frac{x_t+(1-t)v_\phi(x_t,t,c)}{t}.
\]

This follows from the Gaussian corruption formula. Avoid singular score evaluation at t=0; document interior loss-time bounds. The sampler can start at t=1 using the denoiser/velocity interface without dividing by 1-t. Native EDM preconditioning and noise scales must be adapted mathematically rather than renamed. Teacher inference integrates from 1 toward 0; validate the velocity sign with an analytic example.

SiD approximates Fisher-divergence-based distribution matching using the fixed real teacher and an online fake-score model. In the pinned flow code, on corrupted generated images let R and F be real-teacher and fake-model clean predictions and X the generated clean image. Its elementwise generator surrogate is

\[
(R-F)\big[(R-X)-\alpha(R-F)\big]w(t),
\]

with documented reductions/scales (at alpha=1 this becomes (R-F)(F-X)w). The fake model regresses velocity toward noise minus **detached** generated samples. The SiDA path adds discriminator BCE terms, sharing fake-model features. These equations identify the source path; **gradient routing is part of the algorithm**, not determined by writing the scalar expression alone. Frozen teacher/fake parameters during G updates do not automatically mean detaching their input derivatives. Trace the official functions and test the intended input-Jacobian paths; do not substitute DMD's stopped score-difference update.

For K-step sampling, the reference repeatedly denoises an input formed as `(1-t_k) * previous_clean + t_k * noise`, with fresh/fixed/DDIM variants. The training loop samples a step position and generates its preceding rollout; validate stopping gradients through earlier steps and training the chosen step. Record the ladder and noise coupling before training each matched pair. Each K student must execute exactly K learned calls at inference; no ODE teacher in its sampler.

Primary reading: [Score Distillation of Flow Matching Models](https://arxiv.org/html/2509.25127v1), especially §§2–3; [SiDA](https://arxiv.org/html/2410.14919v4), loss/algorithm and adversarial construction. Paper default hyperparameters are references, not a frozen local recipe.

## Immutable source map

Full hashes, omissions and pins: [provenance](../resources/provenance.json). Starter `3ccf326`; SiD-DiT `626b79f`; SiDA `29df86a` (`sida` branch). All included source files are byte-identical to their revisions. Only SiD-DiT's upstream generated teaser and prompt assets were omitted; no model implementation/config/license was omitted. Original vendor Markdown can therefore reference intentionally absent example assets. No external data/checkpoints were downloaded.

| Source | Read for |
|---|---|
| [Flow training loop](../vendor/ml-sid-dit/training/sid_dit_training_loop.py) | `training_loop`: teacher/G/fake initialization, fake velocity target, generator surrogate, `train_diffusiongan`, alpha, weighting schemes, adversarial warmup, reductions and gradient switches |
| [Flow utilities](../vendor/ml-sid-dit/training/sid_dit_util.py) | `sid_dit_generate`, `sid_dit_denoise`: velocity→clean conversion, corruption, step ladder, train/inference rollout branches, CFG branches to remove |
| [Flow feature wrapper](../vendor/ml-sid-dit/training/transformer_with_encoder.py) | Reference real/fake feature/logit interface; replace transformer/text-specific implementation with a U-Net adaptation |
| [SiDA losses](../vendor/sid-sida/sida_training/sida_loss.py) | `SIDA_EDMLoss`, generator/fake-score/shared-encoder adversarial objectives; EDM-specific sigma weighting is not flow-ready |
| [SiDA U-Net/D wrapper](../vendor/sid-sida/sida_training/sida_networks.py) | `SongUNet_EncoderDecoder`, `EDMPrecond_EncoderDecoder`, decoder/encoder flags and pooled discriminator outputs |
| [Public DDPM++ components](../vendor/sid-sida/training/networks.py) | `SongUNet`, embeddings, blocks, preconditioning; preserve source notices and define the new flow wrapper |
| [SiDA training loop](../vendor/sid-sida/sida_training/sida_training_loop.py), [SiD loop](../vendor/sid-sida/training/sid_training_loop.py) | Alternating phases, initialization, update/exposure accounting and original distillation-only comparator |
| [Augmentation](../vendor/sid-sida/training/augment.py) | Geometric/conditioning reference; separate data transforms from discriminator transforms, with matched distillation settings across arms |
| [SiD-DiT launcher](../vendor/ml-sid-dit/run_sid_dit.sh) | Reference only: auto-selects GPUs, assumes large pretrained/text/latent assets and contains a stale entrypoint; **do not execute** |

Specific adaptation hazards:

- Upstream flow SiDA switches from full SiD loss to `0.5 * SiD + 0.5 * GAN` after its warmup. A simple flag toggle therefore changes the distillation coefficient too. Our matched pair must preserve the same distillation coefficient and add a separately recorded adversarial term; document this deliberate adaptation and test the zero-adversarial equivalence.
- Freeze the same fake-score loss/update schedule and common randomness within a pair. Enabling real-data D training must not accidentally change student class/noise/time draws or consume its RNG stream. Use separate RNG streams; equal seed integers alone are not matched draws.
- Source code filters NaNs and has mixed-precision checkpoint helpers. The adaptation must surface/count nonfinite values and preserve exact recovery precision/state; do not quietly discard samples or claim bitwise recovery from lossy state saves.
- The public repository expects pretrained DiTs, text encoders, VAEs, FSDP and potentially BF16. It is an implementation reference, not a compatible Pokémon command or proof of single-GPU memory fit. No downloaded prompt corpus is needed for class labels.
- Distillation-only and adversarial variants share a fake-model backbone. Record the head and full framework size; do not call a small head the whole discriminator. Exact Sangwoo architecture/hyperparameters remain unavailable.

## Local reusable sources, data and migration

Use `git show <pin>:<path>` in the indicated sibling checkout. These commits are local references; availability on their GitHub default branches is not assumed. Do not runtime-import or edit the siblings. No utility code is copied during preparation.

| Checkout and fixed commit | Paths / purpose |
|---|---|
| FastGen-AFM `ee74486c4bc01328223419e29372ba77883688a5` | `afm_fastgen/data.py`, `common.py`, `evaluation.py`, `reporting.py`: fixed partition, state/measurement patterns, official FID, galleries; `scripts/check_export.py` / `qualify.py`: standalone loading and recovery checks |
| FastGen-SoFlow `a53eb644d9902e6d521af46fbda319d4e45ed158` | `soflow_fastgen/unet.py`, `data.py`, `evaluation.py`, `paired_evaluation.py`: U-Net/data/reference/paired-draw patterns; `scripts/report.py`, `scripts/gallery.py`: readable trends and class-reference galleries |

AFM's device constant and lease are GPU 0-specific; flow-SiDA must implement GPU 2 identity and a device-wide lock. AFM/SoFlow objectives and their time/augmentation wrappers are not inherited by this project. Preserve NVIDIA notices when copying U-Net code later.

The [copied partition](../resources/development_split.json) SHA-256 is `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`: 14,059 optimization and 3,020 development images (20/class), indexing the official 17,079 training files. Exact-byte duplicate cross-partition groups were excluded by construction; near-duplicates remain possible. Official validation is separate and never used for decisions here.

Read images from `/home/user/jw_soflow/FastGen-SoFlow/data/pokemon-generation-one-22k` (paths begin `PokemonData/`), already composited on white. Keep local manifests and outputs here. Future migration must remap image roots and verify data identity; private GitHub transports source, artifacts move separately when requested. GPU 2 hardware/UUID, compatible precision, memory and throughput remain to be qualified. No environment has been created. Runtime/memory from TITAN/Blackwell/A100 are not interchangeable.
