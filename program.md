# Tentative SoFlow adversarial autoresearch

> **DRAFT — NOT AUTHORIZED TO RUN AUTOMATICALLY.**
> `pre_program.md` records the completed diagnostic audit; the user clarified
> that its execution exceeded the intended documentation-only assignment.
> Its completion does not start this program or its learning pilot.

The separately authorized [fixed matched pilot](docs/adversarial/MATCHED_PILOT_DRAFT.md)
completed on October 5; [results](docs/adversarial/MATCHED_PILOT_RESULTS.md) show
A improves on C's final EMA but both regress from S3. This does not advance the
programme automatically. Explain the continuation-control regression before
broad activation/strength search. The catalogue below remains a tentative draft.

## Objective and fixed foundation

Improve 1-NFE development FID while retaining recognizable classes and credible
within-class diversity. Keep original S3 as the few-call incumbent; report
four-call changes without requiring the same tuned checkpoint to win both.
Initialize independent candidates from completed S3 EMA with fresh optimizers
and declared D seeds. Never inherit unreported extra training from a candidate.
Exact recovery restores the same recipe, state, draw position and used budget.

Keep conditional StyleGAN2, RpGAN, exact deployed clamp, fixed groups of four,
mature SoFlow consistency, architecture, partition and protected evaluator fixed.
No external pretrained training components. Source and scope details are in
[the integration contract](docs/adversarial/INTEGRATION.md). The older scratch
campaign is archived in [program.scratch-history.md](program.scratch-history.md).

## Experiments, in priority order

| Stratum | Question and experiment family | Rule for advancing |
|---|---|---|
| 0: informative trial | Untouched S3 reference; matched consistency-only continuation versus one staged adversarial pilot | Establish observable response past preparation/ramp and EMA lag at an affordable cost before recipe search |
| 1: activation and strength | Immediate versus D-prepared activation; abrupt versus gradual G adversarial activation; lower/higher relative gradient strength | One mechanism per comparison; preserve common initialization and G exposure |
| 2: optimization balance | G/D learning-rate balance and update ratio | Enter when evidence identifies weak tracking or unstable dynamics; include extra D cost |
| 3: regularization and augmentation | Nearby R1/R2 strengths and supported horizontal-flip probabilities | Enter for evidence of overfit/weak gradients; keep grouping and conditioning semantics fixed |
| 4: conditional extension | Exact-clamp baseline versus separately verified raw-output range penalty | Require clipping audit evidence, separate gradient accounting and a matched comparison |

These are ordered experimental families, not a Cartesian sweep. Each candidate
has one short mechanism-based hypothesis; coupled changes must represent one
intervention. Priority follows the outstanding bottleneck, not the cheapest
metric to improve. A loss decrease or clipping decrease does not replace FID.

## Local search versus focused investigation

Local search is appropriate when the objective and measurements are valid, a
useful trial duration exists, and nearby recipe choices address a diagnosed
tradeoff. Atomic questions requiring focused investigation take priority:
unexplained missing gradients, D exploiting boundary artifacts, conditioning
failure, numerical/recovery inconsistency, or an unaffordable observation horizon.
Stop the affected search dimension and state the unanswered question and a
bounded discriminating audit. Do not indefinitely try nearby hyperparameters.
An audit's result may require revising this program; it is not permission to
expand architecture/loss families or spend an undeclared budget.

Outcomes: **promising / worse / invalid / inconclusive / requires focused audit**.
Prefer an honest inconclusive result to extending a trial until it looks useful.

## Measurement and compact records

Use official development FID with 3,020 unaugmented reference images and exactly
3,020 generated images, 20/class. Match noise draws, checkpoint-selection rules
and G exposure within comparisons. Report real cost even when compute differs.
Use fixed and fresh-noise galleries across all classes, structural diversity,
duplicate measures and nearest optimization neighbors. Pixel diversity alone
does not prove semantic coverage or exclude memorization. Official validation
never selects recipes. Prior development-data use must remain disclosed.

One row per candidate:

| ID / outcome | Hypothesis / recipe hash / control | Source / seeds | G/D updates and exposure | 1-NFE FID / 4-NFE FID | Raw/EMA / diversity evidence | Training/eval/total GPU-job hours | Memory / failures | Gallery / decision |
|---|---|---|---|---|---|---|---|---|

Freeze evaluation draws and independent confirmation policy before scores.
New sampling draws assess sampling variation; an independent training seed is
needed to assess training variation. Report inherited S3 cost separately from
incremental fine-tuning and the complete method cost.

## Measured audit facts — October 5, 2026

[The completed audit](docs/adversarial/PRE_PROGRAM_RESULTS.md) used 3.9786
aggregate GPU-job hours; all 38 contracts and B8/B32 phase/recovery checks passed.
The two frozen-G D runs each reached 512 updates. These measurements inform the
families above; they do not activate them:

- Across 6,040 S3 samples, 14.06% of raw channel coordinates exceeded [-1,1].
  At the final D checkpoints, masked image-gradient energy was 8.25% / 10.73%,
  while actual versus diagnostic-unmasked parameter-gradient cosines were
  0.9971 / 0.9896. This is not a percentage of learning lost. Keep exact clamp
  as the reference; a range penalty is not an automatic first intervention.
- D held-out pair accuracy reached 59.52% / 59.02%, with small fit/holdout gaps.
  Correct-minus-wrong real scores were positive but small. Learnability is
  supported; useful semantic guidance, a D-readiness threshold, and absence of
  boundary-artifact reliance are not established. Do not inherit 512 as warmup.
- B8 mean D/G times were 3.07/1.39 seconds; B32 times were 11.81/3.93 seconds
  for nonzero-rho G. Peak whole-worker allocation was 13.06 GB. These establish
  cost, not a generator learning horizon. Account separately for evaluation,
  checkpointing, extra D steps and preparation.
- The range gradient is finite, with a mean norm of 0.1013 and cosine 0.4328
  to SCM in the diagnostic draws. Its effect on learning remains unknown and
  no range-training arm exists yet.

**Historical pre-pilot priority was stratum 0.** The instructions in this paragraph
and the next were fulfilled by the completed matched pilot; they do not authorize
a repeat. The original requirement was to specify a bounded, informative matched pilot
before searching timing, strength or optimizer recipes. Its protocol must state
why D preparation is informative, how EMA lag is handled, and what evidence
would distinguish an ineffective intervention from an observation window that
is too short. Conditioning failure or boundary-artifact dependence requires a
focused audit rather than repeated nearby recipe changes.

A4 verified exact replay of its disposable fixture, not matched treatment/control
stochastic draws. Before the pilot, separate G/D data samplers and explicitly
index common noise across zero/nonzero-rho phases. Extra D updates must not shift
the control's G data or common draws. Qualify that pairing, the full schedule,
and the chosen checkpoint-selection rule before substantive launches.

## Launch blockers and autonomy

These blockers concern any future search campaign. The separately frozen pilot
resolved its own recipe, qualification and finite allowance and has completed.

**UNRESOLVED:** learning-pilot recipe and horizon; numerical search ranges;
candidate count; campaign GPU-job allowance; per-run watchdogs; evaluation,
selection and confirmation allocation; EMA policy in image presentations;
acceptance thresholds/handling of ambiguous diversity evidence.

The diagnostics-only pre-program cannot resolve the generator learning horizon.
No placeholder may silently become a guessed default. Launch is refused until
the above fields are completed, the implementation is qualified, and the user
explicitly starts this assignment. Once started within those completed bounds,
in-scope candidates proceed consecutively without intermediate approval waits.
The finite campaign stops at its declared limit; it never becomes an endless
loop. Current permission for GPUs 1/2 does not itself start training.
