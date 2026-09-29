# Incremental SoFlow research log

Follow [program.md](../program.md); one research-start instruction authorizes consecutive in-scope experiments. Updated September 30 with the user's longer-training authorization and requirement that every new training experiment start from scratch. This plan revision launches no GPU work.

## Campaign context

Fill once, update when needed; link exact machine-readable settings rather than writing an essay.

| Field | Value |
|---|---|
| Current request / date | September 30: allow evidence-justified longer training within model-size/memory limits; every new experiment must start from scratch |
| Track objectives | Separate one-call and two-to-four-call FID |
| Training-time policy | No fixed hour cap; justify and charge each fresh run's full optimizer allocation, including failures and exact recovery |
| Initialization / scheduling | Fresh model, optimizer, EMA, RNG and data state at update zero; full horizon/schedules set before training; previous checkpoints are references only |
| Hard resource limits | Loaded inference parameters <100M; peak training allocation <20,000,000,000 bytes; also record host memory |
| User campaign cap / deadline | None specified; keep cumulative costs and justify each additional allocation |
| Evaluation protocol | Proposed selection seeds 3101/3102 and confirmation seeds 4101–4104, with actual matched noise and the full development metric; details below, freeze in executable configs before launch |
| Prior evidence reviewed | EXP-1; completed negative B1 (`94f2915` and local result); prepared/unrun A2 (`dfe9c77`); unlaunched C3/C1/D2 direction plan (`5d3db1b`) |
| Current active run / process identity / remaining allowance | No matching FastGen job observed at review; this revision starts none. Reconcile process/lock/accounting before execution. |

## Planned experiments

One row before launch; details belong in the linked configuration. Add status updates incrementally. No approval column or per-trial essay is required.

| ID / target track | Recipe delta / control / parent / source | Horizon and state policy | Train / eval / confirmation caps; retry limit | Config / protocol link | Why now; positive / negative next decision |
|---|---|---|---|---|---|
| S1 / proposed, both tracks | Fresh U-Net128 longer-horizon recipe, training seed 42; EXP-1 is a reference only; source reviewed `deed05b` | 0 → 40k; 5.12M new presentations; full 40k curriculum and fresh optimizer/EMA/RNG/data state | Estimated 19.112 new optimizer hours; 50 min selection/confirmation; at most one exact-recovery attempt within the declared job allocation; wall-time allowance set before launch | [Allocation evidence and initialization policy](research/evidence/scaling-20260930.json); fresh-run/evaluation configs pending | Test a full longer recipe after substantial historical 10k → 20k gains; use the complete learning curve to justify any subsequent fresh horizon |
| S2 / proposed, both tracks | Fresh U-Net160, training seed 42; S1 U-Net128 at equal exposure/schedule; other recipe settings fixed | 0 → 40k; 5.12M new presentations; the same complete 40k schedule as S1 | Optimizer hours = 40,000 × measured full-update seconds / 3,600; qualification, evaluation and finite wall-time allowance set from changed-path measurements; at most one exact-recovery attempt | [Allocation formula](research/evidence/scaling-20260930.json); wider-model qualification/config pending | Test 56% more parameters at matched exposure; a gain motivates fresh replication/capacity refinement, a negative redirects toward objective/time-coverage changes |

At a major direction change, add at most three short option rows (option, plausible gain or decision value, cost/limitation), then one sentence selecting the next experiment. Do not repeat the comparison for every local parameter trial.

### September 30 revised scaling decision

| Option | Why it can change the next decision | Full cost / limitation |
|---|---|---|
| **S1: fresh U-Net128, 40k, first** | Establish the full longer-horizon recipe and the control for capacity; historical 10k → 20k relative FID reductions were 34.4% / 33.3% at one/four calls | 19.11 new optimizer hours plus checks, overhead and evaluation. The curriculum also lengthens, so comparison with EXP-1 does not isolate duration alone. |
| **S2: U-Net160 from scratch to 40k, next** | Compare 61.93M versus 39.67M parameters at identical exposure/curriculum | Changed-path qualification and a complete fresh training run; hours remain unmeasured until profiling. Equal presentations do not mean equal compute. |
| **S3: fresh U-Net192, conditional** | A stronger capacity test if U-Net160's quality/cost supports further scaling | 89.14M parameters; memory/throughput unmeasured. No optimizer hours allocated before an actionable reason and qualification. |

**Select S1 first, then S2 to answer the user's capacity question; keep S3 conditional.** The previous C3/C1/D2 proposal remains an untested alternative, not a measured reason to exclude scaling. B1's negative sampler result changes neither quality incumbent and does not prove that capacity or duration is the bottleneck.

**Why 40k:** the baseline stopped at its prescribed horizon while both measured tracks improved. Its 10k → 20k interval consumed 4.777 optimizer hours and reduced seed-42 FID by 60.079 / 19.113. These observations motivate a fresh recipe with twice the total exposure; they do not forecast its FID or prove that 40k is optimal. Absolute gains diminished, so 80k/100k is not yet justified. Timing is stable: 1.7197 seconds/update over the last 10k versus 1.7201 overall, giving **19.108–19.112 hours for all 40k new updates**. The completed baseline's cost is historical and is not credited against this new run. [Calculation record](research/evidence/scaling-20260930.json).

**Initialization and schedules:** both S1 and S2 start from random weights at update zero with fresh optimizer, EMA, RNG and data state. Set `total_steps=40000` before training: the exponential gap ratio runs from 1 to 1/3200 across the full horizon, retaining the numerical gap floor. EMA warmup starts at zero. The inspected trainer uses RAdam at fixed LR 1e-4 with no separate LR scheduler; optimizer state, EMA and the consistency curriculum depend on progress. Any future LR schedule must also be fixed before initialization. Preserve effective batch 128, partition, augmentation, consistency-only objective, dropout and uniform sampling. S2 versus S1 isolates width under matched exposure/schedules. S1 versus historical EXP-1 compares complete horizon recipes, not a pure duration effect. A fresh replication seed/control is needed for a replicated claim.

**Assigning wider-model hours:** qualify U-Net160 on GPU 1 using real optimization images, full augmentation, accumulation, optimizer and EMA. Keep effective batch 128 while adjusting microbatch if needed. Measure a sustained update window after warmup and check the terminal-gap path, actual peak allocation, host memory and loaded/exported parameter count. Use `40,000 × measured seconds/update ÷ 3,600` for its optimizer allocation; do not multiply baseline time by the parameter ratio. Qualification uses disposable scratch weights. Set qualification, runtime-overhead, evaluation and bounded-recovery allowances before training; retain all used time after interruptions. The parameter count qualifies the candidate for profiling, not for training-memory compliance.

**Evaluation:** use all 3,020 training-derived references and 3,020 generated images per pass. Re-score EXP-1 as a historical reference and score S1's own 20k/30k/40k checkpoints at NFE 1/4 on seeds 3101/3102, with actual matched noise within each NFE. S1's 20k point is a midpoint of its 40k curriculum, not a completed 20k recipe. Primary confirmation compares S1's final 40k model against EXP-1. A track advances only if it beats the reference on both selection seeds, then all four reserved confirmation seeds; otherwise retain a negative or inconclusive result. The maximum is 32 primary passes (16 selection, 16 confirmation). At B1's slowest observed four-call pass of 75.32 seconds, the pass-time reference is 40.2 minutes; allocate 50 minutes including startup margin, charging failures. This is an estimate, not measured S1 runtime. Wider-model evaluation, fresh training replication and NFE 2/3 comparisons are budgeted separately.

Complete the declared 40k schedule rather than reject it solely from a noisy midpoint. Resource/numerical failures permit bounded exact recovery of the same experiment, with its horizon unchanged. Review the full curve and paired 30k → 40k gains before selecting any longer experiment. That experiment must start from scratch, with a newly declared complete schedule and its full cost justified. No extension of S1/S2 is planned. S2 supplies the capacity comparison; an unproductive S2 redirects toward C3/C1 rather than automatically funding S3. These decisions remain autonomous once research is started.

These are planned scientific allocations, not launch-ready configurations. Remaining prerequisites are fresh-run configs with complete schedules, paired evaluation configs, changed-architecture qualification and finite job-time allowances. No continuation implementation is needed. The user's time-policy and scratch-initialization instructions are already authoritative.

## Results

FID columns hold comparable protocol-specific means; per-seed records and uncertainty belong in the linked evidence. Use `—` for unmeasured and `failed` for failed evaluation. Cumulative campaign cost excludes inherited training here; retain inherited cost separately for method comparisons. Training versus evaluation time should stay separate when measured; never invent a split from a combined total.

| ID / status | Recipe delta / parent / control | Updates / image presentations | GPU-job hours: train / eval / cumulative campaign | FID 1 / 2 / 3 / 4 | Inference params | Peak allocated VRAM / host RAM | Evidence / decision |
|---|---|---|---|---|---|---|---|
| EXP-1 / completed baseline | SoFlow U-Net128, from scratch | 20,000 / 2,560,000 | Combined inherited job cost 10.1978; separate train/eval split not entered; new campaign — | 113.9551 / 46.5978 / 40.2773 / 38.2858 | 39,667,843 | 8.613 GB / sampled process-tree PSS 2.120 GB | [Baseline](POKEMON_BASELINE_RESULTS.md); two generation seeds, one training run |
| B1-20260930 / completed, negative within tested scope | Frozen EXP-1 EMA; four-call power 0.75 / 1.25 versus uniform | 0 training / 18,120 generated evaluation images | Train 0; selection eval 0.1275; qualification 0.0024; prior research cumulative 0.1299 | — / — / — / uniform 38.0398; candidates 38.2193 / 38.4546 | 39,667,843 | Evaluation allocation 1.398 GB / host RAM — | [Retained result](../runs/research_b1_20260930/result_summary.json); both candidates worse on both paired seeds; no confirmation |

## Track incumbents

| Track | Candidate / protocol | Evidence status |
|---|---|---|
| One-call | EXP-1 EMA; historical mean FID 113.9551 | Starting reference; not a fresh-training replication |
| Few-call | EXP-1 EMA, uniform four calls; historical mean FID 38.2858 | Starting reference; compare future controls under matched protocols |

Update provisional and confirmed candidates separately. An unchanged model with different sampling seeds is not itself an improvement. This seed table does not supersede valid results on other branches.

B1's uniform score differs from EXP-1 because sampling seeds/RNG differ; it is not a new generator improvement. A2 has zero GPU-job seconds and no scores in its [preparation record](../runs/research_a2_20260930/record.json). The C3/C1/D2 direction record at `5d3db1b` launched no model training. S1/S2/S3 above have no candidate scores.

## Short decision and event notes

Append launch, completion, failure, recovery, and major direction changes with timestamp/run ID. Normally one or two sentences per result: quality outcome, interpretation limit, next action. Link logs/configs for process identity, checkpoint and consumed allowance. Preserve negative evidence and failed scores; do not overwrite them when a candidate is rejected.

- **2026-09-30, initial longer-training plan, superseded:** user removed the fixed hour cap and required evidence for optimizer allocations. The proposed 20k → 40k continuation was never launched; the subsequent scratch-only instruction replaces it.
- **2026-09-30, scratch-only correction:** S1 and S2 now both initialize at zero with full 40k schedules. S1 costs about 19.11 new optimizer hours; historical EXP-1 is an evaluation reference only. Current code inspection confirms fixed-LR RAdam, step-dependent optimizer/EMA state and a horizon-dependent gap curriculum. Documentation/configuration-record changes only; no GPU work or new FID result.
