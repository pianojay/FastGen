# Direction audit: pilot versus competitive flow SiD/SiDA

> Subsequent decision, October 2: the user postponed Program 2 and chose the
> existing pilot as a bounded feasibility study. The stop/redirect recommendation
> below is historical, not an active instruction. See [current focus](FEASIBILITY_FOCUS.md).

Audit date: October 2, 2026, 13:40–13:42 KST. Source commit
`37483845ffcce0332bbd9bd74425de4c1d69cc29`.
[Read-only evidence snapshot](../resources/direction_audit_20261002.json).
This report recommends a change; it does not stop, extend or reconfigure a job.
[Program 2](../program_comparison.md) is a separately adoptable assignment.

## Finding

The current experiment is a defensible **short-exposure SiD/SiDA pilot under one
teacher**, not an adequate attempt at a competitive flow counterpart to Sangwoo's
EDM/SiDA. Its engineering qualification and pair matching are useful. Its
training horizons and teacher acceptance rule do not establish that the models
have been given a reasonable opportunity to reach competitive quality.

This came partly from the original handoff: it specified nine runs, matching and
finite watchdogs, but no teacher-quality target or source-grounded minimum
exposure. It explicitly allowed a weak but valid teacher. The implementation
agent followed that scope. The design needs correction, not simply an instruction
to be more creative or search for lower FID.

## Observed budget and quality differences

| Quantity | Current local flow experiment | Sangwoo's reported EDM/SiDA | Implication |
|---|---|---|---|
| Backbone | DDPM++ U-Net, 61,823,875 parameters | DDPM++ about 61.8M, EDM preconditioning | Family/scale reference; exact architecture identity remains unverified |
| Teacher training | 30,000 updates × 64 = **1.92M** draws | **20–25M** draws; batch 256 | Our exposure is 10.4–13.0× smaller |
| Teacher quality | **63.8874 → 45.3850 → 37.2526** at 10k/20k/30k, 32-call Euler | Holdout about **17.8**, 35 NFE | Last local interval improved 8.13 FID; no plateau was demonstrated |
| One-call student | 4,000 × 64 = **256k** G draws, plus 256k fake-score draws | About **3.5M** images, best batch 128 | About 13.7× less G exposure; fake and G draws must not be added to disguise this gap |
| Two-call student | **256k** G draws, uniform step matching | About **2M** images | About 7.8× less exposure |
| Three/four-call students | Same 256k total; expected 85.3k/64k trained outputs per ladder position | Ladders still being compared; no matched exposure available | Across-K results also reflect different exposure per position |
| Teacher objective | Raw velocity, uniform t, unit FM weight | EDM diffusion/preconditioning | Intentional scientific difference, not a missing EDM sigma setting to copy |
| Student sampling | Flow t=[1], [1,.5], [1,2/3,1/3], [1,.75,.5,.25] | EDM sigma 2.5; 2.5→.4 for two calls | Different input normalization, starting noise and ladders |
| Other choices | X-flip, dropout .25, pooled encoder D; no ADA, fake-only warmup or fake reset | X-flip/dropout .25; optional geometric ADA/head; late fake reset reported helpful | Several external details remain unavailable; these are not necessarily implementation defects |
| Hardware | TITAN RTX, teacher ~63.5 images/s | RTX PRO 6000 Blackwell, reported ~500 images/s | Similar elapsed hours do not mean similar training budgets |

Exposure/runtime figures above were read from the two image attachments in
[Sangwoo's runtime reply](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790837566479009).
The first table says teacher 20–25M / 11–14h; one-step 3.5M / ~8h; two-step 2M /
~5h. These are reported approximations, not independently verified logs; exact
counting of fake/G phases is not specified. The second table reports teacher
61.8M/96.6M peaks 13.0/16.8GB and distillation 18.4–18.9GB. Hardware is confirmed
in the same thread. See the [method report](https://kaist-aic313-f26.slack.com/archives/C0BV5UJBW3A/p1790834501987609)
and [Notion experiment](https://app.notion.com/p/3ecf35446091816e8e82e523cca6247c).

The official README checked October 2 still lists TA FIDs **39.15 / 30.90** for
one/few NFE. Our development references differ from official validation, and
Sangwoo's exact holdout identity and headline reporting split are unverified.
These numbers are planning targets, not a common leaderboard.
[Official reference](https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen#evaluation).
The 32-call teacher's 37.25 is not a one-call result beating the TA.

## What this evidence does and does not establish

Longer teacher training is the best-supported next intervention: the last
checkpoint is substantially better than the preceding one and has much less
exposure than the external reference. This is evidence for under-allocation,
not proof that another 10× training will reach FID 17.8. Overfitting, time
weighting, preprocessing, solver error and preconditioning remain possible
limits. Test 32 versus 64 teacher calls at fixed weights/noise to separate some
solver error from learning quality; more teacher inference calls do not improve
the field used for distillation.

A teacher's sampled FID is a useful readiness indicator, not a mathematical
lower bound on student FID or a complete test of its score field. Adversarial
data access can improve on the teacher. A healthy 37-FID teacher therefore does
not prove the current students will fail. Conversely, nonfinite/collapse checks
alone are too weak an acceptance standard for an expensive competitive study.

At t=1, Gaussian flow corruption contains no image information. An ideal
teacher's single denoising prediction is the class-conditional mean. A
teacher-initialized one-call generator is therefore not expected to start with
the teacher's multi-call quality or diversity. It must actually learn the
generative mapping during distillation; 256k presentations are not justified
by initialization alone.

The current trial can estimate the adversarial package's effect *at its short
fixed budget*. It cannot establish converged SiD versus SiDA, flow versus EDM,
or whether teachers prevent GAN collapse. The last claim also lacks an
adversarial-only control. One training seed per arm limits statistical claims.

## Implementation audit

- The inspected flow convention, `D=x_t-t*v`, score conversion, detach boundaries
  and intended teacher/fake input Jacobians are coherent with the stated method.
  Qualification records contain analytic gradient/accumulation tests and a
  bitwise next-update reload check. This audit did not rerun GPU tests.
- SiD coefficient remains fixed when enabling GAN; common random streams are
  separated from extra adversarial draws. This is a useful controlled design.
- Upstream multiplies BCE by the pixel dimension before its final reduction.
  Local mean reductions and 1:.01 G coefficients preserve the reference's
  nominal 100:1 SiD/GAN ratio. This is **not** an obvious missing-pixel-factor
  bug. Absolute rescaling can still affect Adam epsilon, clipping and numerical
  behavior; a future qualification should inspect effective updates and separate
  component gradients, rather than infer influence from scalar losses.
- Uniform teacher times, logit-normal student loss times, 1-t weighting and
  uniform inference ladders are recorded adaptations. Gaussian score/velocity
  equivalence does not make training recipes equivalent. The
  [flow-SiD paper](https://arxiv.org/html/2509.25127v1) discusses practical
  sensitivity to weighting, schedules and preconditioning.
- Each G/fake update trains one randomly selected rollout position. Keeping
  4,000 cycles across K matches total exposure, not per-position exposure or
  total compute. The treatment is the entire adversarial package, including
  real-data learning, not only one added scalar.
- Selected-teacher A4 qualification peaked at **15.763GB allocated**. The
  teacher peaked at **6.016GB**. No evidence here requires emergency termination.
- Current galleries cover 16 fixed classes and show diverse compositions and
  imperfect anatomy. All-class pixel flags were 0/151, but that does not certify
  semantic coverage or absence of memorization.

Source navigation: [implementation](IMPLEMENTATION.md), [engine](../flow_sid/engine.py),
[flow equations](../flow_sid/flow.py), [local recipe](../configs/campaign.json),
[upstream loss implementation](../vendor/ml-sid-dit/training/sid_dit_training_loop.py),
[qualification](../resources/qualification_summary.json).

## Completion estimate

At **13:40 KST**, T was finished and S1 was at **530/4,000**. A1, S2, A2,
S3, A3, S4, A4 had not started. A live process check confirmed the S1 worker
and a supervisor queued for S1,A1,S2,A2 on physical GPU 2. GPU 0/1 jobs were
also active and untouched.

Recent S1 updates average **3.49s**; qualified full-rollout A4 updates measured
**5.91s**. These bracket approximate throughput, not guaranteed lower/upper
bounds for every arm. At 13:42:

| Remaining scope | Approximate remaining wall time | Expected completion if uninterrupted |
|---|---:|---|
| S1 only | 3.6–4h | October 2, about 17:20–17:45 KST |
| First four students, through A2 | 17–28h | October 3, morning through late afternoon |
| All eight students | **32–57h** | **October 3 ~22:00 through October 4 ~23:00 KST** |

For all eight: 31,445 remaining updates × 3.49–5.91s = 30.5–51.6 optimization
hours, plus an estimated 2–5h for evaluations, checkpoint I/O, reports and
exports. Later arms are unmeasured at sustained throughput; update this estimate
when they run. The full 152h component allowance is a watchdog budget, not an ETA.

## Halt versus continue

| Option | Value gained | Remaining cost / drawback |
|---|---|---|
| Finish all eight pilot students | Complete short-budget NFE×adversarial matrix | ~32–57h; weak external/competitive interpretation remains |
| Finish S1/A1 only, close the remaining pilot | One paired short-budget one-call result | Roughly 8–11h; still weak teacher and student exposure |
| Stop at the next clean checkpoint, optionally after the 1k evaluation | Preserve T and implementation; redirect almost all remaining student budget | No completed S/A effect estimate; partial S1 is engineering evidence only |

**Recommendation:** stop the pilot at a clean boundary and redirect to Program 2.
The next predeclared S1 1k evaluation is a reasonable low-cost handover boundary
if it has not already passed. Finishing S1/A1 is reasonable only if the short
one-call contrast is worth ~one extra working day. Completing 3/4-NFE pilot
arms has the lowest priority for the current competitive goal.

No process was signalled during this audit. The current supervisor can launch
the next arm automatically, so a future approved handover must close the queue,
not merely stop its worker. Existing code has no cooperative signal checkpoint
hook: do not promise SIGTERM saves the current update. Verify the last complete
checkpoint and process identities first. Do not kill another project or reset a
GPU. Keep the original trial marked partial, never complete it retroactively.

## Mitigation and comparison validity

Program 2 continues the eligible scratch teacher in a new, explicitly extended
lineage; preserves the pilot; uses a finite 20M cumulative teacher budget and a
predeclared quality gate before students; and gives the primary 1/2-NFE pairs
3.5M/2M G presentations respectively. It uses one frozen stronger teacher and
matched recipes within each pair. There is no autonomous hyperparameter search.

The primary comparison is **SiD versus SiDA with a sufficiently strong shared
flow teacher**. Pilot-to-Program-2 changes are a recipe-package comparison if
batch size or other adaptations change; do not attribute them solely to duration.
A causal flow-versus-EDM comparison would additionally require an aligned local
EDM pipeline or Sangwoo's exact artifacts/configuration. Neither is silently
added here. Record external mismatches and seek existing documentation without
sending messages on the user's behalf.
