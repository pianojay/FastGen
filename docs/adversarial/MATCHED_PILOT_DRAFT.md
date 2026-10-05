# Matched SoFlow adversarial learning pilot — draft, October 5, 2026

> **AUTHORIZED FIXED PILOT — October 5, 2026.** Originally drafted without
> execution permission. The user subsequently instructed: “Order subagent Soflow
> matched pilot draft to carry out the pilot.” Implementation, targeted checks,
> both fixed runs and evaluation are now authorized without intermediate gates.
> This supersedes the original draft-only notice; it does not activate `program.md`
> or authorize a successor. The filename is retained for reference continuity.

## Question and interpretation

Does **one fixed adversarial refinement recipe** produce an observable, useful
1-NFE response from S3 within a small incremental cost? Compare exactly two arms:
**C: consistency-only continuation** and **A: consistency + RpGAN refinement**.
Both start from identical S3 U-Net192 EMA, train for 512 G updates at batch 32,
and receive the same consistency data and random draws. Untouched S3 is a frozen
reference, not a third training arm. Original S3 remains the few-call incumbent;
report tuned four-call quality without making it an acceptance gate.

This is a response pilot, not a convergence study, timing ablation, or search.
It tests the entire specified adversarial recipe; it cannot isolate the effect
of D preparation, ramp duration, or individual hyperparameters. One training
realization and fresh sampling confirmation cannot establish training robustness.
The 512-update horizon is a deliberately affordable proposal, not a measured
minimum for learning. A null result must not become a claim of method failure.

Evidence: [completed audit](PRE_PROGRAM_RESULTS.md),
[integration contract](INTEGRATION.md), and [inactive research map](../../program.md).
The audit verified gradients and modest D learnability, not useful guidance.
No further broad D-only audit is a prerequisite for this proposal.

## Frozen sources and initialization

- Reviewed repository revision: `60b8452` on `codex/soflow-ayf-integration`.
  A future implementation must freeze its own commit and source hashes before launch.
- S3 source: `runs/research_s3_unet192_40k_seed42/checkpoints/step_040000.pt`,
  SHA256 `9b95652fb5e503b2f877b26fd3652129abcce115bdea526db00ffe1f0daefd4a`.
  Copy its EMA into online G and G-EMA in both arms. Fresh G optimizers and G counters;
  source training step remains provenance, never the new optimizer/schedule clock.
- Partition SHA256 `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`:
  all 14,059 optimization images remain eligible for SCM; 3,020 development
  references remain evaluation only. Shared image files stay read-only.
- Arm A reuses **the first audit D seed, 20261004**, chosen by ordering rather
  than the slightly better observed score. Checkpoint:
  `runs/pre_program_audit_20261004/d_seed20261004/d_step0512.pt`, SHA256
  `3b38ca039f42d468502962b514182cabbd408ab9ead7b4771c53662f0fc33dd9`.
  It contains 512 D updates / 16,384 real and fake presentations against fixed S3.
  This is an available prepared asset, not evidence that 512 is optimal warmup.
- Restore **D weights and Adam state**, the saved D sampler state, and only
  `d_noise`/`d_flip` streams. Keep counters as inherited D=512, new D=0; never
  reset inherited exposure. G streams are fresh and common across arms.
  Do not blindly load the D snapshot as a full G training checkpoint.
- D source manifest SHA256
  `a19336e05f7f5e32b555245b32d0ccb714eb53c5ecb51e5b730dc75e0a1e7e69` is the
  [executed audit manifest](../../runs/pre_program_audit_20261004/frozen.json).
  Preserve its `fit_groups`, 604 held-out positions and diagnostic ordering.
  D fitting remains uniform-class then uniform-fit-image sampling, continuing
  the saved cursor; no additional D-only preparation or second D seed.
- The 604 images are withheld from **D optimizer updates**, not G training:
  S3 saw them previously and SCM continues to use the full optimization partition.
  The existing G update can also use them as detached real-score references.
  Thus their D statistics are diagnostics, not a pristine adaptive validation set.

Every artifact path above is relative to this repository root. Preserve source
artifacts. Verify source/partition/protected/vendor hashes before any later run.

## Proposed recipe, identical except for the adversarial intervention

| Setting | C: continuation | A: adversarial refinement |
|---|---|---|
| G updates / presentations | 512 / 16,384 | 512 / 16,384 |
| Effective batch / SCM microbatch | 32 / 2 | 32 / 2 |
| G optimizer | RAdam, lr 2e-5, betas (.9,.999), eps 1e-8, weight decay 0 | Same; fresh state |
| LR schedule / gradient clipping | Constant / none | Same |
| G-EMA | Half-life 2,048 G image presentations | Same |
| D updates | None; no D loss/backward | One D update immediately before each G update |
| New D exposure | 0 | 512 updates / 16,384 real and fake presentations |
| D optimizer | Not stepped | Restored Adam, lr 2e-5, betas (0,.99), eps 1e-8, weight decay 0 |
| RpGAN and penalties | Disabled | Existing exact RpGAN; beta=.1 for R1+R2, every D step |
| D augmentation | None required | Existing shared real/fake-pair horizontal flip p=.25 |
| G adversarial coefficient | rho=0 throughout | rho(k)=.1*min(k/64,1), completed G step k=1..512 |

No G-only delay is added. Arm A uses 64 ramp updates followed by 448 at target
strength. EMA decay is `2**(-32/2048)` = approximately .9892280132 per G update;
no warmup or separate D-driven EMA ticks. There are eight EMA half-lives overall,
seven after the ramp. Raw and EMA curves distinguish lag from no visible response.
This rules out *obvious* EMA concealment, not insufficient optimization time.

The modest G LR, rho=.1 and unchanged D recipe reuse a numerically checked
region; they are proposed first settings, not empirical optima. Fresh RAdam's
initial transient is common to both arms. D Adam is deliberately continued to
avoid a second adaptation transient after its short audited preparation.
At completion A has lifetime D=1,024, with inherited/new counts reported separately.

Retain exact clamp, conditional StyleGAN2 architecture, separate real/fake
forwards, fixed ordered D groups of four, native FP32 operators, full-vector
balancing, and mature SCM gap r=.0003125 with 1e-4 minimum gap. Preserve SCM
augmentation, time draws, residual weights and shared dropout. No range penalty,
FM term, teacher, inference redesign, lazy regularization or optimizer sweep.
The existing D object may remain inert in C to minimize code changes; report
resident and trained parameter counts separately and never export D.

## Matching and state contract to implement later

Use one declared pilot seed `20261005`. G data use a deterministic permutation
of all optimization positions for each epoch, seeded by `20261005 + epoch`;
consume a continuous sequence in batches of 32, wrapping without dropping the
last positions. Both arms receive identical position/label sequences. This is
image-uniform SCM continuation, not the audit's class-uniform D sampling.

SCM randomness must be indexed by **G update**, identically for both arms:
initialize the consistency stream for step k from the first eight SHA256 bytes
of UTF-8 `matched-pilot-v1:20261005:consistency:k`, interpreted little-endian
modulo 2**63. Fixed microbatch order then fixes augmentation, t/s, noise and
paired-dropout draws. The G-augmentation and time/noise tensors must be identical
across arms even though their gradients diverge after adversarial activation.

Likewise derive independent `g_noise` and `g_flip` step seeds with those names
in the same key. Their identity is defined even for rho=0: skipping a forward
must not shift the next step's draw. D continues its saved independent streams
and sampler; no D action advances G data or randomness. G adversarial pairs use
the current G batch's class labels and real images; D-update class priors remain
uniform and matched between real and fake. Document this difference in class
weighting between G and D; it is shared SCM sampling, not an unnoticed arm change.

Save at completed D+G boundaries (C: completed G boundaries), every 64 G updates;
retain 0/64/128/256/512, rotating other recovery saves. Persist G/D/EMA, optimizers,
origin and recipe hashes, both sampler positions, RNG streams/key convention,
counters, consumed job time and phase. If interrupted between D and G, restore
and replay the entire pair from the preceding boundary; charge discarded work.
Never apply the preceding D update twice to a restored state.

Before substantive training, minimal checks must prove: common SCM input/RNG
hashes in C/A despite extra D calls; step-indexed GAN draws through rho=0;
zero-rho equal-update control; ordered groups unchanged; restored D optimizer
and sampler identity; exact next-update recovery across steps 63→64→65; and
G-only one-/four-call independent loading. Use small disposable fixtures for
most contracts and at most two disposable B32 full-S3 D+G pairs for the new
wrapper. Do not repeat A0–A4 or benchmark additional batch sizes. Same-device
replay should match exactly under deterministic settings; cross-device gradient
comparisons use the existing declared atol/rtol tests and exact draw hashes.
Qualification states are discarded; neither arm inherits their updates.

## Measurements and fixed endpoint decision

Evaluate using the unchanged official FID implementation and the existing
CPU-per-draw-stream-v1 paired sampler, eval batch 32, category order 0..150 with
20 samples each, exactly 3,020 generated images against the 3,020 unaugmented
local development references. Official validation remains unused. Reused
training-derived development data and previous development tuning are disclosed.

Predeclare generation seeds **7301** (trajectory), **7302/7303** (sampling
confirmation). Confirm source, preprocessing, image hashes and actual network
calls; independently loading each exported G must require no D. Four-call uses
the unchanged uniform schedule. Reference S3 is reevaluated under these draws;
do not subtract its historical FID from a new noise realization.

| Evaluation | Models/checkpoints | Pass count |
|---|---|---:|
| 1-NFE trajectory, seed 7301 | Untouched S3; C/A raw and EMA at G=128 and 512 | 9 |
| 1-NFE confirmation, seeds 7302/7303 | Untouched S3; C/A **final EMA** | 6 |
| 4-NFE descriptive, seed 7301 | Untouched S3; C/A final EMA | 3 |
| **Total fixed maximum** | 15 one-call + 3 four-call passes | **18** |

Evaluate confirmation irrespective of primary outcome if both arms finish and
budget remains. Do not inspect those results to choose a checkpoint, change a
recipe or add samples. The comparison endpoint is **final EMA at G=512**, fixed
before scores. Raw/128 results explain trajectory; they cannot replace the
endpoint because they happen to score better. No best-checkpoint search.

Log each update: loss components, complete gradient norms/cosine/lambda,
parameter displacement, clipping and masked input-gradient energy, D logits,
R1/R2, actual update time, counters/exposure, allocation/reservation and host RSS.
At G=0/128/512 use the existing 604 held/fit diagnostic ordering and seed6201 for
D score/label-sensitivity diagnostics without full A3 parameter-VJP audits.
Fresh-current-G fakes and paired real/fake logical groups remain fixed in order;
measure no-augmentation scores without advancing training RNG. C needs no D audit.

Build a class-reference gallery from the evaluated images covering **all 151
classes**, with matched-noise S3/C/A panels, raw/EMA at 128/512, and fresh-noise
final-EMA panels. Reuse existing rendering/nearest-neighbor code where available.
For final EMA and S3, use only trajectory seed7301 for numerical diagnostics;
fresh seeds supply panels and sampling confirmation, not extra neighbor audits.
The neighbor pool is all optimization images **within the same class**, including
the D diagnostic holdout, excluding development and official validation.
Report per-class pairwise-distance distributions at
16x16/32x32/64x64, exact PNG duplicate counts, and nearest optimization images
under the same multiscale pixel metric (average RGB MSE over those resolutions,
minimum over reference horizontal flip). Show each class's closest match and
five least-diverse pairs. These inexpensive metrics flag cases for visual
inspection; they are not semantic diversity or memorization proofs. Inspect
flagged cases with full-resolution originals; do not introduce a pretrained
feature network or a new discriminator diagnostic campaign.

Record one brief row per class: recognizability change, pose/shape/composition
variation change, suspect copying/collapse, and image link. Broad coverage is
required; if review is incomplete or diversity evidence ambiguous, do not label
the pilot promising on FID alone. Translation/background/color variation alone
cannot satisfy within-class variation. New sampling seeds do not replicate training.

Proposed practical signal threshold: **2.0 FID points**, chosen for a small pilot
rather than asserted statistical significance. Report every per-seed paired
difference `A-C` and `A-S3`, not only an average:

- **Scientific A-versus-C judgment:** report a favorable recipe effect if the
  confirmation mean A-C is at most -2.0 and both seed differences are negative;
  adverse if the mean is at least +2.0 and both are positive; otherwise uncertain.
  This is a practical signal classification, not a significance claim.
- **Practical A-versus-S3 judgment:** independently apply that same rule to A-S3.
  A useful post-training response requires both favorable judgments plus no
  systematic new recognizability/diversity loss or unresolved copying signal.
  Improvement over S3 alone cannot be attributed to the adversarial intervention;
  improvement over C alone need not improve the available S3 asset.
- **Worse under this recipe/window:** adverse A-C or reviewed broad collapse.
  Report A-S3 separately; do not generalize to AYF or infer inevitable failure
  at longer exposure. No outcome starts another experiment.

- **Inconclusive:** small/mixed response, raw-only response, unreviewed diversity,
  or improvement versus S3 without evidence of benefit over C. This means the
  bounded pilot did not establish useful adversarial benefit, not that a longer
  run is guaranteed to help. A visible late trend is not permission to extend.
- **Invalid/incomplete:** failed pairing, numerical/resource/recovery contract,
  missing matched endpoint or timeout. Partial learning evidence remains visible
  but cannot be relabeled a completed endpoint comparison.

## Finite cost and operational proposal — not launch permission

Use the existing `fastgen_soflow` interpreter with `PYTHONNOUSERSITE=1`; no new
environment or package installation. Proposed assignment: A on physical GPU1
(`CUDA_VISIBLE_DEVICES=1`, UUID `GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888`), C on
physical GPU2 (`CUDA_VISIBLE_DEVICES=2`, UUID `GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2`),
local `cuda:0`. Verify ownership/UUID and acquire existing per-device leases
(`runs/gpu_1.lock`, `/tmp/fastgen-gpu2.lock`) before future launches. If occupied,
report pending; do not fall back to another GPU or disturb another project.
Both arms can run concurrently; evaluation runs sequentially on a free assigned
device, never concurrently with training on that same device.

The [A4 evidence](pre-program-evidence-20261005.json) measured B32 D/G(nonzero rho)
as 11.814/3.934 seconds; the single rho=0 G observation was 2.491089 seconds.
Thus update-only projections are A `512*15.748=8,062.976 s` (2.240 h), C
`512*2.491089=1,275.438 s` (.354 h). These are short-fixture extrapolations,
especially C's single measurement; not a promise of sustained speed.
[S3 evaluation evidence](../research/evidence/s3-complete-20261004.json) records
S3 pass times up to 50.629 s at one-call and 109.576 s at four-call. Eighteen
passes project about 1,089 s before startup, new diagnostics and uncertainty.

| Incremental allocation, including retries | Ceiling | Purpose |
|---|---:|---|
| Targeted wrapper qualification | 600 s | CPU/GPU contracts and at most two S3 B32 pairs |
| A training worker | 10,800 s | 8,063 projected update seconds + checkpoints, diagnostics and margin |
| C training worker | 2,100 s | 1,275 projected update seconds + checkpoints and margin |
| Evaluation/report stage | 2,700 s | 18 passes, references/gallery/statistics, loading and margin |
| **Aggregate GPU-job ceiling** | **16,200 s = 4.5 h** | No transfer of unused allowance into more updates |

The stage ceilings are cumulative, not reset on recovery; overlapping GPU jobs
are added separately, child work is counted once. Idle time while waiting for a
lease is reported separately. Bound CPU-only report construction to one elapsed
hour and report it separately when outside a GPU job. Stop on stage/global
watchdog, nonfinite state, wrong source/draw identity, or allocated memory
>=20,000,000,000 bytes. G/D each remain <100M parameters. No automatic OOM batch
reduction, recipe changes, additional seed, or early success stop. Quality
regression or apparent visual collapse does not truncate this short fixed pilot;
record it at scheduled checks. Hard numerical/resource/ownership/time failures
stop work. An unforeseen issue that invalidates continuation must be reported,
not repaired by improvising another recipe.

Allow at most **one exact resume per arm for an external interruption**, from the
last complete boundary, within its existing ceiling. No automatic retry for a
code, numerical or resource failure. Saved evidence/consumed time survives any
interruption; if completion will not fit, stop. No silent extension.

Report inherited costs without charging them again as new work: S3 complete
method cost 133,875.088 s (37.188 h); selected D audit worker 7,043.690 s (1.957 h),
of which logged D update time was 6,152.244 s (1.709 h). The D worker remainder
includes diagnostics/startup/checkpointing, not pure model fitting. The entire
completed pre-program cost was 14,323.031 s (3.979 h). Report both attributable
asset cost and total research expenditure, avoiding double-counting the selected
D worker inside the full audit. Incremental pilot costs include controls,
evaluation, failures, qualification and recovery; report A's direct training
cost separately. No inherited 12-hour campaign allowance applies.

## Minimal later implementation and deliverable

Implement only a finite two-arm runner/state manifest, step-indexed common
streams plus independent D sampler import, declared schedules/EMA, narrow
pairing/recovery contracts, and an evaluator wrapper that labels raw versus EMA
correctly. Existing `paired_evaluation.evaluation_identity` hardcodes EMA:
explicitly fix that label for raw snapshots rather than misreporting weight
source. Reuse existing update, protected evaluator, sampler, export, gallery
and accounting components. No general search controller or new training family.

Freeze config/source/asset hashes and pass the targeted contracts before launch.
Implementation and launch have now been jointly authorized; those checks are an
automatic prerequisite, not a new approval gate. Finish the fixed runs and report;
never activate the inactive research programme.

| Arm/reference | G/new D/inherited D updates | Presentations | 1-NFE raw/EMA and paired deltas | 4-NFE | Diversity/copying assessment | Train/eval/check/recovery time | Peak memory/failures | Gallery/outcome |
|---|---|---|---|---|---|---|---|---|
| S3 | 0 / 0 / 0 | inherited separately | — | — | — | — | — | — |
| C | 512 / 0 / 0 | 16,384 new G | — | — | — | — | — | — |
| A | 512 / 512 / 512 | 16,384 new G and new D | — | — | — | — | — | — |

The final report answers whether this window produced a credible response and
what remains uncertain. A successor horizon, different recipe or autoresearch
activation requires a separate decision. **The current authorization ends with this fixed pilot and report.**
