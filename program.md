> **Historical and inactive on `codex/soflow-ayf-integration` (October 4, 2026).**
> Current authorization is component integration and bounded smoke verification.
> Follow [AGENTS.md](AGENTS.md) and [the integration contract](docs/adversarial/INTEGRATION.md).
> The scratch-only and consecutive-run instructions below describe the previous campaign.

# Autonomous SoFlow research loop

Updated September 30, 2026. These instructions govern an actual research loop once the user starts it. Editing or reading this file alone does not launch training. A request to start or continue research authorizes consecutive in-scope experiments; do not stop for approval between them.

## Objective and scope

Find substantially better development FID at **exactly one NFE** and **two to four NFEs**, within the FastGen resource limits. Maintain separate best candidates; do not average the tracks. The scope is SoFlow and close variants, including its training objectives, curriculum, conditioning, architecture, augmentation, optimization, and compatible local-teacher methods. Direct class conditioning is the starting point. Different winning checkpoints are allowed.

**Every new training experiment starts from scratch.** The September 30 user correction applies to controls, architecture changes and longer-horizon recipes: initialize model weights, optimizer, EMA, RNG and data state at update zero, with the full horizon and all schedules declared before training. Previously trained checkpoints may be evaluation references, not initialization for a new experiment. Do not continue the completed 20k baseline or extend a completed candidate. Exact recovery of the same interrupted experiment preserves its already-declared recipe and horizon; it is not a new experiment. Earlier warm-start or continuation proposals are superseded by this rule.

Read [AGENTS.md](AGENTS.md), [current status](docs/STATUS.md), the [research log](docs/RESEARCH_LOG.md), and the [baseline results](docs/POKEMON_BASELINE_RESULTS.md). Consult the [intervention catalog](docs/AUTORESEARCH_DIRECTIONS.md) and [strategy/cost audit](docs/FASTGEN_STRATEGY.md) when relevant. Catalog entries are ingredients at different levels of a design, not equally ranked next experiments. Check existing result records and relevant branches before repeating work; a checkout's older status is not proof that another session ran no experiments. Inspect other branches without switching a checkout used by a running process.

Starting baseline: 39,667,843-parameter U-Net, 20k updates, 10.1978 GPU-job hours; mean development FID **113.9551 / 46.5978 / 40.2773 / 38.2858** at NFE 1/2/3/4. These are two sampling seeds from one trained checkpoint. TA targets checked September 29 were 39.15 / 30.90; recheck the official course page at milestone assessment. Our development references differ from the official references, so these are directional targets, not an official score comparison.

## Choose at the right level

The primary decision is **which training recipe is worth allocating meaningful compute to**, not which catalog item is easiest to execute. Use the following hierarchy to organize choices. It is a default allocation of attention, not a claim about which untested method will win.

| Level | Role | Examples and priority rule |
|---|---|---|
| Validity and targeted diagnosis | Make a comparison trustworthy or resolve a concrete design ambiguity | Fix broken conditioning, numerical errors, invalid evaluation, or resource violations first. Use an audit only when its possible outcomes change a pending design choice; do not require exhaustive diagnosis before training. |
| Recipe design | Main search for a substantial change in attainable quality at practical training cost | Objective/supervision, time and gap curriculum, conditioning, architecture/expressivity, and data/augmentation policy. Compare plausible recipes under declared compute/exposure conditions. Larger models are candidates, not an assumed cure. |
| Recipe optimization and compute scaling | Develop a promising recipe and determine how efficiently it learns | Learning rate, EMA policy, regularization, training duration, and local parameter search. Promote these earlier when specific evidence makes them a likely bottleneck. Do not extend training merely because a curve is still improving. |
| Checkpoint and sampler selection | Extract quality from an existing trained model | Online versus EMA weights, checkpoint choice, two-to-four-call schedules. These can be worthwhile cheap checks, but ease of implementation is not evidence of a large remaining gain. One-call training cannot be repaired by tuning nonexistent intermediate sampling times. |

Interventions can cross levels. A learned sampler or local teacher with its own optimization is a training-method decision with full cost, not a free inference tweak. Evidence can elevate a lower-level intervention; record why. An expensive training run can be the smallest useful experiment when shorter runs cannot answer the question.

At each major choice, compare at most three plausible options in **three short rows**: option, reason it could materially improve the target track or change the next recipe, and full cost/limitation. Then choose one in a sentence. Include a recipe-level option unless current evidence or a user constraint rules training out. Use qualitative judgment rather than invented numerical probabilities or a weighted ranking score.

- Consider plausible gain, relevance to the current gap, information that changes the next choice, and total preparation/training/evaluation cost. Cheapness is a tie-breaker, not the objective.
- State what a positive and negative result would change. A diagnostic with no actionable consequence can be deferred.
- D1 (longer training) asks about allocating more compute to a fresh run; it is not automatically the best fixed-budget recipe improvement. Compare its full from-scratch cost with another fresh recipe. A different horizon can also change the curriculum, so distinguish a complete longer-horizon recipe from a duration-only ablation. A2 (saved online versus EMA) is a checkpoint-selection check; a negative result does not resolve the training bottleneck.
- Give both tracks meaningful attention over the sequence; log the reason for focusing on one. Do not spend an indefinite series polishing the few-call sampler while the one-call learning gap remains unaddressed.
- After an unproductive local neighborhood, change the design question or recipe. Do not keep shrinking a sweep because it is easy to score. Conversely, a promising recipe merits refinement and confirmation rather than novelty for its own sake.

## Execute the loop

1. Read the latest results and active job state. Select the next option using the short comparison above.
2. Add a planned row to the log and save an exact configuration before observing candidate scores. Declare the control, intervention, training length/curriculum, seeds, evaluation rule, and job/retry limits. Reuse a campaign evaluation protocol instead of rewriting it for every experiment.
3. Implement the necessary change and run relevant contract/numerical checks. Preserve baseline artifacts and use fresh run directories. Change one factor when claiming attribution; bundled recipe changes are allowed for quality search, but do not attribute the result to one component without ablation.
4. Launch and supervise. **Hours-long training is normal.** Do not substitute a short proxy merely to finish the conversation quickly. Use durable jobs and bounded waits, check meaningful progress/checkpoint/error signals, and continue through completion. Do not poll noisily or stop at “training launched.” If the execution surface interrupts the agent, leave a recoverable record; do not promise persistence the runtime does not provide.
5. Evaluate, update the row and separate track incumbents, and record one or two sentences on the quality result and what to try next. Confirm promising candidates under the declared protocol; retain provisional status when evidence is insufficient.
6. Select and run the next experiment autonomously. A completed experiment, a negative result, a new in-scope recipe, or a long expected runtime is not a reason to wait for user approval.

Send concise nonblocking updates at launch, meaningful results, failures, and changes of direction. Ask only when a required external dependency, a genuine unresolved user constraint, or an action outside the authorized scope prevents progress. Respect a user stop/pause request and any explicit campaign limit. Do not introduce a periodic approval gate. Older briefs may contain per-experiment approval language; this policy supersedes that gate while preserving their recorded results and protocol limits. If a launcher requires an authorization reference, use the actual research-start instruction and its scope, never a fabricated approval. No extra agents unless explicitly requested.

## Compact records

Use [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for incremental decisions and results. Save full configurations and machine-readable metrics beside each run; link them instead of duplicating their contents in prose. A new essay/report per experiment is not required. Historical detailed reports remain evidence, not a required template.

Before launch, a row or linked configuration must identify:

- ID, target track, changed factor/recipe, control, source revision, fresh-initialization seed, and exact training/optimizer/EMA/curriculum/data-state policy. A checkpoint is an initialization source only for exact recovery of that same interrupted run; label reference checkpoints separately.
- Training horizon, evidence for the selected optimizer hours, supervised job-time allowance, evaluation/confirmation allowance, retry limit, selection and confirmation seeds, sample/reference identity, and the decision rule fixed before candidate scores.
- One-sentence rationale and positive/negative next decision. The three-option comparison is needed at major direction changes, not every local trial.

Append the common metrics without overwriting older outcomes:

| ID / status | Recipe delta / parent / control | Updates / image presentations | GPU-job hours: train / eval / cumulative campaign | FID 1 / 2 / 3 / 4 | Inference params | Peak allocated VRAM / host RAM | Evidence / decision |
|---|---|---|---|---|---|---|---|

Include seeds, per-seed scores, reference identity, and pairing in linked metrics. Mark unmeasured values `—` and failed evaluations `failed`; never use zero. Distinguish provisional and confirmed winners. Record sampling-only versus fresh-training replication. Log retries, interruptions and recovery costs as they occur. A small fixed visual panel can flag contradictory evidence without a full gallery essay.

## Evaluation rules

- Headline development FID retains **3,020 unaugmented training-derived references and 3,020 generated images, 20 per category**, using the protected official metric. Smaller diagnostics are labeled separately. Official validation never participates in training, candidate selection, or fitting a schedule.
- Fix selection and separate confirmation seeds and a bounded decision rule before scores. Use the same protocol for candidate and control. There is no universal FID-gain threshold; a marginal or conflicting result remains inconclusive. Do not add seeds until the desired conclusion appears. Campaign defaults can be reused; new experiments need not invent new statistical procedures.
- Preserve reference/preprocessing/batch conditions and actual matched noise where applicable. Same seed alone does not guarantee matching across NFEs. Include checkpoint, weight source, sampling schedule and RNG protocol in cache/output identities.
- Training comparisons normally evaluate both one and four calls. Report two/three calls when selected by the few-call experiment. Verify actual network-call counts, including auxiliary models.
- Intermediate checkpoints can describe a fresh run's learning curve; they are not separate completed recipes. Do not reject slow-learning recipes from early FID unless the proxy's relevance has evidence. Define the optimizer, EMA and curriculum across the full horizon from initialization. Changing total_steps changes the baseline curriculum; the 20k midpoint of a 40k schedule is not the completed 20k recipe.
- Fresh training assesses recipe variability; extra generation seeds assess sampling variability. Already-used development data are not an untouched holdout. Quality gains need not have a proven mechanism; mechanisms need not improve FID. FID does not establish class accuracy, diversity, or absence of memorization.

## Resources and boundaries

**There is no fixed training-hour cap.** On September 30 the user authorized longer training and replaced the earlier 12-hour default with evidence-justified optimizer-hour selection. The hard resource limits remain **fewer than 100M loaded inference parameters** and **peak training allocation below 20,000,000,000 bytes**. This changes the time policy; data validity, protected files, scratch initialization, GPU ownership and external-action boundaries still apply.

Choose a **finite update horizon and optimizer-hour allocation for each stage**, supported by measured learning curves, the comparison needed to answer the question, and actual update timing. State why the chosen exposure is informative and why a shorter or longer allocation is less justified now. Convert updates to optimizer hours using full augmentation, accumulation, optimizer and EMA timing; measure changed architectures before assigning their hours. Parameter-count ratios are not throughput measurements. Sparse FID points under a changing curriculum do not justify a fitted convergence forecast. Record diminishing returns and uncertainty rather than promising extrapolated FID gains.

A larger horizon requires a new from-scratch experiment and a recorded evidence-based allocation decision, including its **full** optimizer cost, control, optimizer/curriculum/EMA policy and next decision. Once research is started, this is part of autonomous execution and does not require renewed user approval. Never budget only the extra updates beyond a previous checkpoint or silently lengthen an active/completed schedule. Retain historical method costs and all failed/recovered work in campaign accounting. Recovery alone never grants more allowance; do not rename a run to hide used compute.

Budget qualification, startup/checkpoint/shutdown overhead, bounded recovery, controls, teachers, sampling and confirmation explicitly, and record total method cost alongside optimizer hours. Set a finite supervised job-time allowance large enough for the declared work using measured timings and stated uncertainty; a watchdog is not a universal scientific training limit. There is no fixed 10/2 allocation or default campaign-hour cap. Honor any later user-specified total cap or deadline. Absence of a fixed cap is not a reason to spend on repetitive low-value trials.

Reconcile used time and active allocations before launch; use existing supervised timeouts, GPU locks, and a shutdown allowance. Do not double-count nested process intervals or equate GPU utilization with elapsed job time. Per-run supervision does not automatically enforce a multi-run campaign cap. Report agent/token costs only if available.

- Use local **GPU 1**, `PYTHONNOUSERSITE=1`, and `/home/user/miniconda3/envs/fastgen_soflow/bin/python`. Leave GPU 0 and the closed CIFAR work alone.
- Preserve protected files/base Model and official evaluation. All generative lineage is scratch-trained on the optimization partition; only official FID Inception may be pretrained. Loaded inference parameters stay below 100M, peak training allocation below 20,000,000,000 bytes. Measure new paths; log host memory as well.
- Preserve effective batch when adjusting microbatch for resources. Declare whether comparisons match compute or presentations and record both. KCloud requires its own qualification; TITAN timings are not A100 predictions.
- Use dedicated branches or isolated checkouts and fresh outputs. Never reset another agent's work to reject a candidate. Retain configs, hashes, seeds, optimizer/EMA/RNG/data state, and original results. Recipe branches must not weaken exact-resume checks.
- Commit reviewable changes and small records; keep large artifacts ignored. Summarize major milestones in STATUS/Notion under existing authorization, not every trial. No Slack messages or invitations are authorized.

## Recovery and continuity

Persist the run ID, source/configuration/checkpoint identity, process start identity, artifact paths, consumed allowance, retries, and current state. A fresh session reconciles logs, checkpoint, process and lock before acting: monitor a live job instead of duplicating it; resume an interrupted one within the same allowance. Logged progress beyond saved weights is lost work. Restore required optimizer/EMA/RNG/data/curriculum state and retain strict resume validation.

A stale interruption reason is not evidence of a current failure. An unclean exit may not save accounting; reconstruct it from supervisor records or resolve the discrepancy before relaunch. An evaluation failure does not erase completed training or create a score. After a bounded unrecoverable failure, log it and choose another viable in-scope experiment; do not let one broken candidate halt the whole campaign unnecessarily.

The [CPU recovery rehearsal](docs/PROMPTWARE_RECOVERY_REHEARSAL.md) established one orderly interruption/recovery, not hard-crash, timeout-exhaustion, GPU-state or ten-hour reliability. Its older per-experiment approval policy is historical and superseded by this file. These instructions do not themselves implement a scheduler or enforce access permissions.

## Implementation readiness

Inspect the current checkout and existing research branches rather than assuming a catalog feature exists. Implement only prerequisites for the selected experiment. At baseline revision `0084d02`, configurable inference times, objective mixing and some diagnostics required work; later branches may contain reusable implementations. Fresh longer runs must set their full horizon at initialization and validate the resulting schedules. A checkpoint-continuation facility is not a prerequisite for the current scratch-only plan. Preserve strict exact-resume checks for interrupted runs without changing their declared schedules.

Use relevant repository tests for runtime changes. Documentation-only changes need consistency/link review, not another GPU experiment. This workflow adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): repeated implementation, measurement, retained evidence and autonomous iteration, with SoFlow-scale training times and two separate quality objectives.
