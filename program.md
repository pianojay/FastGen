# SoFlow autoresearch instructions

Prepared 2026-09-29, Asia/Seoul. **Documentation ready; no research session has started.** This file governs a future user-started session. Reading or editing it does not launch experiments.

## Purpose and starting evidence

Use **research agenda → question and competing explanations → bounded experiment → results → updated agenda**. FID is the quality objective; diagnostics help choose interventions and interpret their effects. A useful negative result can change the next decision without improving a checkpoint. A reproducible FID improvement can be useful even when its mechanism remains unresolved.

Read [AGENTS.md](AGENTS.md), the [completed baseline](docs/POKEMON_BASELINE_RESULTS.md), the [research catalog](docs/AUTORESEARCH_DIRECTIONS.md), and the [strategy and cost distinctions](docs/FASTGEN_STRATEGY.md). Historical running snapshots do not supersede the completed result.

Maintain two equally important objectives, with separate best candidates:

| Track | Goal | Current development baseline |
|---|---|---|
| One-call | Improve exactly-one-NFE generation | Mean FID **113.9551** |
| Few-call | Improve generation at 2, 3, or 4 NFEs | Mean FID **38.2858**, four calls |

The baselines use one trained checkpoint and two sampling seeds, not independent training repetitions. The [official TA targets](https://generative-models.kaist.ac.kr/fastgen/) checked September 29 are 39.15 and 30.90, respectively; recheck them when assessing an official milestone. Different reference sets prevent equating our development FIDs with official scores. Do not average the two tracks into one optimization target. Separate winning checkpoints are allowed.

Our scope is SoFlow and close variants. Use the four questions in the catalog to select relevant interventions; do not run every catalog entry. Keep direct conditioning as the starting recipe. Broader model-family exploration belongs to the team's other work.

## Research loop

1. **Select an uncertainty.** Review existing results for both tracks. State the decision that new evidence could change, the competing explanations, and why this probe is worth its cost now. There is no mandatory catalog order or fixed compute split between tracks.
2. **Write the experiment brief below before observing candidate scores.** Choose an audit, numerical diagnostic, controlled training comparison, or small local search. Declare exact settings only when instantiating the experiment; illustration values are not standing instructions.
3. **Prepare the comparison.** Check implementation readiness, validate any prerequisite changes, preserve the control, and establish provenance. Prefer existing artifacts when they answer the question. An early-training proxy cannot rank eventual quality unless that relationship has evidence.
4. **Run within the brief and session limits.** Change one factor for an attribution claim, or explicitly identify a bundled intervention and the later ablation needed. A new question or changed evaluation rule requires a new brief, not rewriting the original after seeing results.
5. **Interpret and confirm.** Record quality and explanatory conclusions separately. Label outcomes preliminary, confirmed under this protocol, negative within the tested scope, inconclusive, or implementation failure. Missing or failed evaluations have no numeric score.
6. **Update the agenda.** State the next question and the alternatives now more or less plausible. Preserve useful negative evidence, rejected candidates, and unexplained gains. Avoid repeatedly testing a low-information neighborhood simply because it is easy to score.

An active session may choose and implement its next probe within these boundaries without asking after every trial. Novel in-scope ideas are allowed: add their evidence and dependencies to the catalog first. A catalog change does not alter the fixed evaluation, protected files, or resource limits. This is one research agent using the assigned GPU; do not spawn additional agents without an explicit request.

## Experiment brief

Store a dated brief with the future session's records. Complete all fields before the relevant GPU work; for a CPU-only audit, mark GPU fields not applicable. The following is a record template, not an implemented CLI or configuration schema.

```text
Question ID / date / target NFE track(s):
Decision this result could change:
Evidence, sources, competing explanations, and unresolved assumptions:
Probe and why it is the smallest informative comparison:
Control / candidate(s) / exact permitted changes / fixed variables:
Starting checkpoint and code revision / prerequisites / relevant checks:
Training state policy: optimizer, EMA, curriculum, RNG, data progression:
Trial length or stopping criterion / candidate and retry limits:
Estimated GPU-job time and other cost / hard cap / confirmation reserve:
Evaluation: partition, preprocessing, sample counts, seeds, NFE, pairing:
Selection rule / separate confirmation procedure / uncertainty rule:
Diagnostics and interpretation limits:
Positive / negative / inconclusive outcomes and corresponding next questions:
```

For inference-only probes, declare training state frozen. For matched training branches, start from the same parent and preserve its state except for the named intervention. Declare whether comparisons match image presentations, elapsed compute, or both; log both quantities regardless. Controls can be reused only when their relevant conditions match.

Afterward append actual cost, artifact locations, all scores, failures, deviations, and two conclusions: **what happened to quality** and **what this says about the explanation**. Record which track's candidate changed and what evidence supports that change.

## Evaluation and confirmation

- Keep the established unaugmented **3,020 training-derived development references and 3,020 generated images, 20 per category** for headline development FID. Preserve partition identity and the official metric implementation. Any smaller diagnostic is labeled separately and never replaces this metric.
- Official validation does not participate in training, candidate selection, or schedule fitting. Fit any learned calibration on the optimization partition, and predeclare development selection and later verification. The existing development set has already informed choices: resplitting it or using new generation seeds does not create an untouched holdout.
- Fix the selection seeds and reserve separate confirmation seeds before search. Evaluate the candidate and its control under the same confirmation protocol. Repeated use of confirmation results makes them selection evidence; disclose that use in later briefs.
- Match actual initial and intermediate noise draws for sampler comparisons, where applicable. The current evaluator's identical seed values do not reliably pair images across different NFEs. Keep batch size, output processing, and reference processing fixed within the comparison.
- Training probes normally report both one-call and four-call results. Few-call selection compares the declared 2–4-call candidates; a schedule-only result at four calls says nothing about one-call learning. Verify actual network-call counts, including any auxiliary network.
- FID ranks distributional quality, not requested-class accuracy or memorization. Use predeclared category panels and diagnostics to understand changes; record contradictory findings rather than selecting attractive examples afterward. Tiny per-class sample sets do not support dependable classwise FID rankings.
- Confirmation effort and the decision rule depend on observed control variability and the claim's size. Estimate variability before candidate selection or predeclare a bounded procedure for doing so. There is **no universal FID gain threshold**. Conflicting or insufficient evidence remains inconclusive; do not add seeds until a desired conclusion appears.
- Sampling seeds assess generation variability. Fresh training repetitions assess recipe variability. Short checkpoint branches establish local adaptation evidence; a stronger recipe claim needs fresh-training confirmation. The changing consistency MSE is a diagnostic, never the quality objective.

Maintain a confirmed best candidate for each track and separately list provisional contenders. A confirmed quality improvement does not require a proven mechanism, and a supported mechanism does not imply a quality improvement. Resource and challenge validity remain prerequisites for either candidate's use.

## Budget, environment, and provenance

The **first future session has 12 cumulative GPU-job hours**, including qualification, startup, failed attempts, teacher/target work, sampling, and evaluation. Default to reserving the final **two hours for confirmation**, leaving ten for exploration. Stop starting exploratory work when it would consume that reserve. Stop at the session limit, preserve available evidence, and report unresolved questions rather than spending beyond the cap to force a conclusion. Include shutdown/checkpoint overhead in each job's allowance.

Before launching each job, sum completed job time and outstanding allocations, then cap the next job at the remaining session allowance. Reconcile actual time afterward without double-counting parent and child intervals. Existing budgets are per run; they do **not** enforce a shared 12-hour session cap. Record the aggregate accounting with the session, and enforce each job's timeout using the existing supervised-job mechanism. Do not assume copying a 24-hour baseline config implements this budget. Record agent/token expenditure if the execution surface provides it; do not invent unavailable costs.

Record incremental session cost separately from the inherited baseline's **10.1978 GPU-job hours**. Include inherited training, teacher preparation, calibration, and search when comparing total method cost; a reused checkpoint is not free pretraining. Old baseline limits remain historical and are not a replenishing allowance.

- Work in `FastGen-SoFlow`, using local **GPU 1** and `PYTHONNOUSERSITE=1` with `/home/user/miniconda3/envs/fastgen_soflow/bin/python`. Use existing GPU locks and avoid other users' processes. The closed CIFAR experiment and GPU 0 remain untouched.
- Follow [AGENTS.md](AGENTS.md): preserve official file/base-class hashes, scratch-trained lineage on the optimization partition, fewer than 100M loaded inference parameters, and peak training allocation below 20,000,000,000 bytes. Only the official FID Inception is pretrained. Measure memory for new training/calibration paths.
- Preserve effective batch when tuning microbatch for resource reasons. Record changed throughput instead of treating faster updates as equivalent learning progress. KCloud requires its own qualification and fresh run; TITAN measurements are not A100 predictions.
- Keep a dedicated experiment branch or isolated checkout and fresh run directories. Preserve the completed baseline, parent hashes, source revisions, configs, seeds, optimizer/EMA state, and data-order provenance. Commit reviewable candidate changes; do not destructively reset a shared working tree to reject a trial.
- A recipe change is an explicit branch, not exact resume. Keep the existing strict resume guard. Lengthening `total_steps` changes the consistency curriculum; define continuation behavior separately. Retain both optimizer step and curriculum position when they differ.
- Store small briefs and conclusions in versioned documentation and large artifacts in ignored run directories. Summarize measured milestones in status/Notion under the existing publication authorization. This file does not authorize Slack messages, invitations, or a new training launch merely by being read.

## Current implementation readiness

As inspected at source revision `0084d02`, the repository has a trainer, exact-resume checks, per-run budget supervision, development FID, exports, reports, and an offline gallery. It has **no custom autoresearch controller**. This document and the catalog are the operating layer for an agent using and, when needed, extending those tools.

Before relevant trials, implement and validate only the required prerequisites: explicit recipe branching, configurable inference times and matched randomness, objective mixing, or diagnostic collection. Their presence in the catalog is not evidence that they are already implemented. Existing `--run-steps` can limit an unfinished recipe but cannot continue the completed 20k checkpoint beyond its configured horizon. Do not bypass that limitation by weakening the resume check.

Use the repository's contract/numerical checks for relevant future code changes, plus targeted checks for the new behavior. Documentation-only work requires link, fact, and scenario review, not GPU training tests. The catalog's [walkthroughs](docs/AUTORESEARCH_DIRECTIONS.md#protocol-walkthroughs) exercise the intended decisions without claiming actual experimental results.

## Attribution

This workflow adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and its [program instructions](https://github.com/karpathy/autoresearch/blob/master/program.md): bounded code changes, experiments, evaluation, and retained records. We add question selection, two quality tracks, explicit interpretation, and a finite project budget. We do not copy its five-minute language-model workload or indefinite-loop instruction. The [September 28 meeting](https://app.notion.com/p/3e9f3544609181deb002cd5e50c0651a) supplies team context; [the catalog](docs/AUTORESEARCH_DIRECTIONS.md) links the scientific and local implementation evidence.
