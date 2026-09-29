# SoFlow research agenda and intervention catalog

Prepared 2026-09-29, Asia/Seoul. **Proposals and documentation walkthroughs; no follow-up experiment has been launched.** Read [program.md](../program.md) for the operating instructions. This catalog connects research questions to possible interventions; it is not a sweep specification or an ordered checklist.

## Four research directions

| Direction | Decision-relevant questions | Candidate interventions |
|---|---|---|
| **A. Understand remaining errors** | Are failures mainly conditioning, diversity, memorization, or general image quality? Are they widespread or concentrated? | Category audit, shared-noise label comparisons, nearest training neighbors, artifact diagnostics, online/EMA comparison |
| **B. Improve frozen-model sampling** | How much few-call quality depends on inference choices? Does sensitivity persist across checkpoints? | Intermediate re-noising levels, allocation of 2–4 calls, subsequently compatible learned schedules or path transformations |
| **C. Improve one-call learning** | Is the gap explained by insufficient training, weak large-jump supervision, optimization, or another limitation? | Time distributions, consistency curriculum, objective mixture, loss weighting, numerical behavior, local teacher |
| **D. Improve the baseline recipe** | Does training use the available data, compute, and capacity effectively? | Duration, augmentation, learning rate, EMA, regularization, conditioning architecture, width/depth |

Entries may serve multiple directions. Choose the next question from current evidence and the decision it could change, then instantiate an experiment brief. The initial agenda does not prescribe endpoint fractions, learning-rate grids, fixed trial lengths, or a universal FID improvement threshold.

### Evidence at the starting point

- **Measured locally:** the completed U-Net128 has 39,667,843 inference parameters. Mean development FID is 113.9551 / 46.5978 / 40.2773 / 38.2858 at NFE 1 / 2 / 3 / 4. Both one-call and four-call FID improved between 10k and 20k updates. This does not establish convergence, class accuracy, or absence of memorization. See the [completed baseline](POKEMON_BASELINE_RESULTS.md).
- **Measured cost:** 1.7201 seconds per optimizer update on average, 10.1978 cumulative GPU-job hours, and 8.613 GB peak training allocation. Final 3,020-image generation-plus-FID passes took about 40–73 seconds, excluding separate process startup. These are baseline measurements, not universal trial estimates or pure sampling latency. See [archived evidence](pokemon-baseline-evidence/metrics_summary.json) and [final evaluations](pokemon-baseline-evidence/fid_020000.json).
- **Implementation facts:** training is consistency-only, with logit-normal time draws, an exponential gap curriculum, and a warmed-up EMA. The sampler predicts an endpoint and re-noises between calls. One call has no intermediate re-noising level. See [trainer](../soflow_fastgen/training.py), [model objectives](../soflow_fastgen/models.py), and [sampler](../soflow_fastgen/sampling.py).
- **Hypotheses:** inadequate time coverage, numerical sensitivity, unsuitable augmentation, and insufficient expressivity remain possibilities. None is established as the cause of the observed FID gap. The changing MSE diagnostic does not establish a quality plateau.

## Intervention catalog

Readiness below refers to the inspected code at `0084d02`; update it after implementing a prerequisite. Costs are categories or extrapolations, to be replaced by measured estimates in an experiment brief. Published evidence motivates testing; it is not a prediction for this dataset or device.

### A1. Category fidelity, diversity, and training neighbors

- **Motivation / evidence:** current fixed panels cover eight categories; the gallery exposes all 151. Aggregate FID does not check requested labels. Reference thumbnails in the gallery are held-out examples, not nearest training neighbors. [Baseline limitations](POKEMON_BASELINE_RESULTS.md#limitations-and-next-discussion), [gallery](../scripts/gallery.html).
- **Possible probe:** audit a declared category/sample selection, summarize error types, and compare shared-noise label changes. Retrieve nearest optimization-partition images with a stated distance and show their provenance; inspect multiple neighbors and real-to-real similarities for context.
- **Readiness / dependencies:** existing gallery and images support manual audit; exporting its browser-local notes is necessary for a durable record. Neighbor retrieval and quantified diversity/conditioning diagnostics need implementation. No pretrained semantic classifier or perceptual model may be imported.
- **Cost:** existing-image audit is CPU/manual work. New matched samples or allowed diagnostics have separately budgeted generation cost.
- **Interpretation / next question:** systematic category confusion motivates conditioning probes; repeated close matches motivate duplication/memorization investigation. Pixel similarity and visual inspection cannot establish memorization or its absence. If errors are general, prioritize recipe or training questions rather than assuming label failure. Tiny per-class FIDs are not reliable rankings.

### A2. Online weights versus EMA

- **Motivation / evidence:** evaluation selects EMA weights. Actual final EMA decay was 0.9995502; the configured 0.9999 cap was never reached. [Baseline EMA analysis](POKEMON_BASELINE_RESULTS.md#training-and-resource-findings).
- **Possible probe:** compare online and EMA weights from the same immutable checkpoint using the same sampling and evaluation protocol.
- **Readiness / dependencies:** both weight sets are saved, but the evaluator currently loads EMA unconditionally. Add explicit weight-source selection and provenance before claiming a comparison; do not relabel an online checkpoint as EMA.
- **Cost:** evaluation only, plus implementation checks. Changing EMA's training policy is a separate training comparison.
- **Interpretation / next question:** sensitivity motivates a controlled smoothing study. A single checkpoint comparison does not identify the best EMA schedule or prove convergence.

### B1. Intermediate re-noising levels and call allocation

- **Motivation / evidence:** four calls use levels 0.75, 0.5, 0.25. Changing these levels tests how much quality is available from the existing model. [Sampling implementation](../soflow_fastgen/sampling.py), [schedule distinctions](FASTGEN_STRATEGY.md#what-learning-a-schedule-can-change).
- **Possible probe:** declare a small ordered time-grid family around uniform sampling, with the checkpoint and NFE fixed. Confirm promising candidates, then consider 2–3-call allocation or a narrower refinement. There is no preselected grid in this catalog.
- **Readiness / dependencies:** configurable times, explicit shared initial/re-noising draws, and schedule-aware evaluation identity are missing. Record schedule and randomness fingerprints in cache/output identity; current checkpoint/partition/NFE/seed keys cannot distinguish arbitrary schedules. Use fresh output directories and preserve old results.
- **Cost:** generation and FID only; baseline full passes were about 40–73 seconds each. Measure new timings and include startup, implementation qualification, all candidates, and confirmation.
- **Interpretation / next question:** reproducible sensitivity supports refinement. No measurable change lowers this family's priority for this checkpoint, not all possible samplers. Recheck after further training. This intervention cannot alter the current one-call sampler.

### B2. Learned schedules and alternative paths

- **Motivation / evidence:** [LD3](https://arxiv.org/html/2405.15506v4) and [BézierFlow](https://arxiv.org/html/2512.13255v3) motivate sampler adaptation around a trained model; our [cost audit](soflow-cifar10/few-nfe-training-cost-audit.md#6-learning-the-sampler-ld3-and-bézierflow) explains the missing prerequisites and pretraining cost.
- **Possible probe:** after basic sensitivity evidence, define a compatible parameterization and allowed calibration objective; compare learned coefficients with the fixed-schedule control. Distinguish training curriculum, noising path, and inference times.
- **Readiness / dependencies:** research and implementation required. The current stochastic endpoint/re-noising sampler is not a drop-in ODE solver. Generation uses `no_grad`; differentiable calibration needs a separate path and memory qualification. The cited LPIPS-based objectives cannot be imported under the current pretrained-model rule.
- **Cost:** calibration, target generation if needed, candidate scoring, and confirmation, in addition to inherited backbone training. Few trainable coefficients do not imply a cheap backward pass.
- **Interpretation / next question:** establish benefit and total cost before broadening the search. Any one-call path transformation is a new formulation, not tuning a nonexistent intermediate time. A high-NFE velocity solver needs its own quality evidence.

### C1. Training-time coverage and large jumps

- **Motivation / evidence:** [training code](../soflow_fastgen/training.py) draws start/end times from nonuniform distributions, while one-call inference queries exactly noise-to-data. This mismatch is a hypothesis about learning efficiency, not a demonstrated bug.
- **Possible probe:** inspect the joint time distribution and diagnostics by time region; then compare a declared alternative weighting/distribution against an unchanged-training branch. Explicitly state when changing sampling also changes the training objective's emphasis.
- **Readiness / dependencies:** time means/stds are config fields; endpoint mixtures and per-region diagnostics require code. Validate time ordering, gap handling, finite gradients, and effective-batch weighting. Recipe branching is required for warm-start comparisons.
- **Cost:** CPU distribution analysis first; training and matched evaluation only when justified. Existing update timing gives an estimate for unchanged architecture, not a guaranteed ranking horizon.
- **Interpretation / next question:** a selective one-call gain motivates coverage refinement and eventual fresh-training confirmation. Failure at short duration leaves optimization lag and trial length as alternatives. Do not infer poor learned behavior solely from rarely sampled endpoints.

### C2. Consistency curriculum, gap floor, and numerical behavior

- **Motivation / evidence:** MSE uses a difference quotient as the gap shrinks. The implementation imposes a 1e-4 minimum gap. Large diagnostics can reflect optimization difficulty, changed targets, or finite-precision effects. [Model code](../soflow_fastgen/models.py), [curve interpretation](STATUS.md#september-28-training-curve-readability).
- **Possible probe:** measure actual gaps, floor activation, residual magnitudes, adaptive weights, and gradients before selecting a gap-ratio or numerical intervention. Compare one declared change against the matched continuation control.
- **Readiness / dependencies:** endpoint ratios are configurable; schedule choice is hardcoded to exponential in the trainer. Additional diagnostics or schedule choices need implementation. Do not remove the numerical floor without an equivalent validated stability mechanism.
- **Cost:** distribution analysis can be CPU-only; model diagnostics and changed arithmetic cost GPU time. Training comparisons need the same state, duration, and evaluation controls as other recipe changes.
- **Interpretation / next question:** establish whether a numerical change affects generation, gradients, or merely logged values. Fresh longer curricula and terminal-gap continuations answer different questions. A lower redefined loss is not evidence of better samples.

### C3. Objective mixture and adaptive weighting

- **Motivation / evidence:** the local trainer uses only consistency learning. [SoFlow's ablations](https://arxiv.org/html/2512.15657v2#S5.SS2) explore flow-matching mixture and loss weighting on another dataset/backbone. The [local model](../soflow_fastgen/models.py) contains an unused velocity-loss path.
- **Possible probe:** compare a small declared batch allocation between flow matching and consistency against consistency-only training, with direct conditioning. Investigate the weighting exponent or smoothing constant separately if evidence points there.
- **Readiness / dependencies:** objective mixing needs trainer changes, sample-count normalization, uneven-microbatch checks, and matched branch support. Config exposes `p` and `eps`, but they are recipe changes, not runtime resume overrides. The consistency loss is a gradient surrogate; its scalar magnitude is not a comparable conventional MSE for arbitrary loss blending.
- **Cost:** measure changed forward/backward work; equal update counts need not equal compute. Count both image presentations and GPU-job time.
- **Interpretation / next question:** check whether changes help the finite jump and/or the diagonal velocity. A gain can justify a refined mix; a failed short branch does not invalidate the published method. Restore a duration control before attributing gains to a more complex objective.

### C4. Local teacher or student specialization

- **Motivation / evidence:** few-call quality is substantially better than one-call quality, suggesting a possible target source, but this does not establish a useful teacher or a good diagonal ODE field. [Strategy](FASTGEN_STRATEGY.md#cost-accounting-across-model-families), [distillation audit](soflow-cifar10/few-nfe-training-cost-audit.md#5-distillation-and-pretrained-consistency-tuning).
- **Possible probe:** first assess the proposed teacher's quality and define the conditioning/noise-to-target coupling. Only then specify a student objective, a matched additional-training control, and confirmation.
- **Readiness / dependencies:** teacher qualification, target generation, student training, and export are new work. A stochastic multi-call teacher needs an explicit target policy; naive regression across unrelated outputs can blur or remove diversity. Auxiliary networks must be absent from inference unless included in the NFE/parameter budget.
- **Cost:** teacher construction or inherited training, target generation, student fitting, and evaluation. A locally reused teacher's historical cost remains visible in total method comparisons.
- **Interpretation / next question:** a successful transfer supports specialization, not a claim of cheaper end-to-end training without accounting. Downloaded generative or perceptual weights are outside the protocol.

### D1. Training duration with an explicit continuation policy

- **Motivation / evidence:** both evaluated modes improved from 10k to 20k; longer training is an early candidate, not a last resort. [Duration clarification](POKEMON_BASELINE_RESULTS.md#follow-up-clarification-longer-training-is-an-early-candidate).
- **Possible probe:** an unchanged-recipe branch continuing at the terminal gap supplies a duration comparison and a control for other training interventions. Define optimizer, EMA global step, curriculum position, data cursor, RNG, and evaluation behavior before running.
- **Readiness / dependencies:** strict resume currently rejects semantic changes and the loop ends at the configured horizon. An explicit branch/continuation facility is required. Keep the exact-resume guard; merely increasing `total_steps` changes the gap at the branch point.
- **Cost:** +10k updates extrapolate to about 4.78 optimizer hours before overhead. This is a scale reference, not a prescribed probe length. Choose a shorter or longer comparison according to the question and budget, then report its interpretive limits.
- **Interpretation / next question:** improvement supports more training; a short flat segment does not establish a plateau. A fresh slower curriculum is a different recipe requiring its own comparison.

### D2. Augmentation and regularization

- **Motivation / evidence:** the baseline retained SoFlow's flip/scale/anisotropic-scale/translation pipeline; its suitability for these classes remains untested. [Baseline recipe](../README_LOCAL.md#baseline-recipe).
- **Possible probe:** compare the existing augmentation against a simpler pipeline or the true bypass control. Use original/augmented examples to explain geometric effects. Investigate dropout or other regularization separately when warranted.
- **Readiness / dependencies:** augmentation is currently a Boolean; false bypasses resampling entirely. Flip-only or independent transform strengths need code and consistent six-element metadata for both consistency branches. Sampling uses zero metadata. Dropout is configurable, but a changed-recipe branch remains necessary.
- **Cost:** matched training/evaluation, with throughput measured when resampling changes. Existing image inspection is inexpensive.
- **Interpretation / next question:** assess generalization and class details, not only training loss. Checkpoint adaptation and a fresh model trained with another augmentation are different claims; the small dataset alone does not establish which pipeline wins.

### D3. Optimization and EMA policy

- **Motivation / evidence:** the baseline uses FP32 RAdam at 1e-4 and warmed-up EMA. Its observed learning curve leaves optimization choices open. [Configuration](../configs/pokemon_local.json), [trainer](../soflow_fastgen/training.py).
- **Possible probe:** select a small learning-rate comparison, or an EMA-policy comparison motivated by A2; do not change both for an attribution claim. Treat effective-batch changes as learning changes, not microbatch tuning.
- **Readiness / dependencies:** learning rate, optimizer betas, weight decay, and EMA cap are config fields. LR schedules, other optimizers, and alternative EMA warmup need code. Changing only the EMA cap may do nothing if the warmup never reaches it; record the actual decay trajectory.
- **Cost:** matched branches plus confirmation; measure throughput if the computation changes.
- **Interpretation / next question:** a sensitive region warrants refinement; an apparent gain still needs comparison with ordinary continued training. Separate noisy-loss behavior from FID and sample quality.

### D4. Conditioning and model capacity

- **Motivation / evidence:** the current directly conditioned U-Net fits the resource limits, but that does not show it is too small. Label-gradient checks established connectivity, not semantic correctness. [Qualification](STATUS.md#qualification-evidence).
- **Possible probe:** use A1 evidence to decide between a conditioning change and additional capacity. Compare a declared architecture change with a matched baseline; include fresh initialization for an architecture-recipe claim.
- **Readiness / dependencies:** width, block count, and dropout are exposed. Channel multipliers and attention resolutions require code changes. Wider weights are not an exact checkpoint continuation. Before training, check loaded-model parameters, memory, throughput, and export compatibility. Baseline widths 160/192 were counted at 61,934,243/89,140,419 parameters, but their training footprints and quality remain unqualified.
- **Cost:** resource qualification and new training; report presentations and elapsed compute, since larger models alter both feasibility and learning per hour.
- **Interpretation / next question:** capacity benefit supports further allocation only under the declared resource comparison. Short runs may favor faster-learning architectures. Diagnose conditioning/optimization and ordinary training duration before claiming width is the cause.

## Illustrative translations into experiment briefs

These are examples of reasoning. They are not executable briefs: exact candidate settings, counts, seeds, costs, and stopping/confirmation rules must be supplied using [the program template](../program.md#experiment-brief) before GPU work.

| Question and decision | Probe and control | What outcomes justify |
|---|---|---|
| **A: Are class errors systematic enough to prioritize conditioning?** | Audit a declared cross-category sample; distinguish saved independent samples from truly matched-noise label comparisons. Inspect several real references and nearest optimization images. | Concentrated label failures support a conditioning probe; general artifacts redirect toward training. Ambiguous recognition motivates a better diagnostic. None establishes accuracy or memorization from a few images. |
| **B: Is uniform re-noising a significant limitation of this checkpoint?** | Freeze weights and NFE; compare a small declared ordered schedule family against uniform with explicit matched draws, then verify candidates separately. | Reproducible improvement supports refinement; no resolved effect lowers this family's priority for this checkpoint. Mixed seeds call for the predeclared uncertainty procedure, not unlimited search. |
| **C: Is weak large-jump learning contributing beyond insufficient duration?** | First characterize training-time coverage; compare a targeted distribution/objective intervention with an unchanged-training branch from the same parent. | Selective one-call improvement supports refinement; improvement in both branches points toward duration as an alternative; a short failure remains inconclusive about fresh training. |
| **D: Does geometric augmentation help held-out quality?** | Compare a declared simplification or true bypass with upstream augmentation, keeping initialization/state, training work, and evaluation explicit. | Reproducible improvement motivates a fresh-training recipe comparison. Better training diagnostics without better FID do not establish useful generalization. |

## Protocol walkthroughs

The following are **hypothetical documentation checks**, not measured results, trial records, or promises of improvement. Each exercises the rules in [program.md](../program.md).

| Scenario | Required handling | Acceptance check |
|---|---|---|
| **Informative audit**: a declared image review suggests recurring class confusion, without a new FID. | Preserve reviewed IDs and notes, state uncertainty, and propose a matched-noise conditioning diagnostic. Leave quality incumbents unchanged. | The inquiry can advance without fabricating a score or claiming that a visual audit measured class accuracy. |
| **Confirmed FID improvement**: a schedule candidate meets its predeclared confirmation rule, with mechanism still uncertain. | Preserve control and candidate scores, schedule/noise identity, resource evidence, and confirmation provenance. Update only the few-call incumbent and record mechanism as unresolved. | Quality improvement and explanatory support are separate; neither a one-call gain nor an official TA-score win is inferred. |
| **Inconclusive short run**: one sampling seed favors a training branch and another does not. | Apply only the declared confirmation procedure within remaining budget. If unresolved, retain it as inconclusive and propose a longer or fresh comparison when justified. | Do not cherry-pick a seed, redefine the threshold, or reject the whole research direction. |
| **Implementation failure**: a new branch exceeds memory or cannot resume as intended. | Save logs and actual cost, record failure with no FID, and use only the brief's allowed repair/retry allocation. Revalidate before rerunning; preserve effective batch for a microbatch repair. | A crash is not a scientific negative result; retries consume the same approved run/experiment allowance. |
| **Budget exhaustion**: exploration reaches its allowance before all questions are answered. | Start only work that fits the approved run and experiment allowances, including declared evaluation/confirmation costs. At the cap preserve state, report incomplete claims, and hand off the next question. | No unbounded loop, fabricated confirmation, or silent renewal of a training-run allowance. |

### Ongoing evidence maintenance

Keep small dated briefs and result summaries under the project docs, with links to ignored run artifacts. Every result records quality outcome, explanatory outcome, actual cost, deviations, and the next question. Update catalog evidence/readiness with links rather than rewriting rejected hypotheses as though they were never proposed. Keep provisional and confirmed candidates separate for both tracks.

For a future campaign, use the completed baseline report as the source of measured facts; keep older strategy snapshots as history. The [September 28 meeting memo](https://app.notion.com/p/3e9f3544609181deb002cd5e50c0651a) establishes the team division of work. [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) supplies the local experiment-loop inspiration; [program.md](../program.md) adds the connection to this research agenda and its finite budget.
