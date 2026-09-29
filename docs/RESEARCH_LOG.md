# Incremental SoFlow research log

Use this file when a research session starts. This template records no new launch. Follow [program.md](../program.md); one start instruction authorizes consecutive in-scope experiments, subject to resource and user limits.

## Campaign context

Fill once, update when needed; link exact machine-readable settings rather than writing an essay.

| Field | Value |
|---|---|
| Start instruction / date | Not started by this documentation edit |
| Track objectives | Separate one-call and two-to-four-call FID |
| Per-training-run cap | Default 12 GPU-job hours including failed/resumed segments |
| User campaign cap / deadline | Record if specified; no invented default |
| Evaluation protocol / selection and confirmation seeds / decision rule | Set before candidate scores; link reusable configuration |
| Prior evidence imported | Baseline below; check other research branches before selecting next work |
| Current active run / process identity / remaining allowance | None launched by this edit; reconcile actual processes at startup |

## Planned experiments

One row before launch; details belong in the linked configuration. Add status updates incrementally. No approval column or per-trial essay is required.

| ID / target track | Recipe delta / control / parent / source | Horizon and state policy | Train / eval / confirmation caps; retry limit | Config / protocol link | Why now; positive / negative next decision |
|---|---|---|---|---|---|

At a major direction change, add at most three short option rows (option, plausible gain or decision value, cost/limitation), then one sentence selecting the next experiment. Do not repeat the comparison for every local parameter trial.

## Results

FID columns hold comparable protocol-specific means; per-seed records and uncertainty belong in the linked evidence. Use `—` for unmeasured and `failed` for failed evaluation. Cumulative campaign cost excludes inherited training here; retain inherited cost separately for method comparisons. Training versus evaluation time should stay separate when measured; never invent a split from a combined total.

| ID / status | Recipe delta / parent / control | Updates / image presentations | GPU-job hours: train / eval / cumulative campaign | FID 1 / 2 / 3 / 4 | Inference params | Peak allocated VRAM / host RAM | Evidence / decision |
|---|---|---|---|---|---|---|---|
| EXP-1 / completed baseline | SoFlow U-Net128, from scratch | 20,000 / 2,560,000 | Combined inherited job cost 10.1978; separate train/eval split not entered; new campaign — | 113.9551 / 46.5978 / 40.2773 / 38.2858 | 39,667,843 | 8.613 GB / sampled process-tree PSS 2.120 GB | [Baseline](POKEMON_BASELINE_RESULTS.md); two generation seeds, one training run |

## Track incumbents

| Track | Candidate / protocol | Evidence status |
|---|---|---|
| One-call | EXP-1 EMA; historical mean FID 113.9551 | Starting reference; not a fresh-training replication |
| Few-call | EXP-1 EMA, uniform four calls; historical mean FID 38.2858 | Starting reference; compare future controls under matched protocols |

Update provisional and confirmed candidates separately. An unchanged model with different sampling seeds is not itself an improvement. This seed table does not supersede valid results on other branches.

## Short decision and event notes

Append launch, completion, failure, recovery, and major direction changes with timestamp/run ID. Normally one or two sentences per result: quality outcome, interpretation limit, next action. Link logs/configs for process identity, checkpoint and consumed allowance. Preserve negative evidence and failed scores; do not overwrite them when a candidate is rejected.
