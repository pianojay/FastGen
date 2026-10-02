# Program 2 — competitive flow SiD/SiDA comparison

**Prepared October 2, 2026. Not started or adopted.** This is a second assignment,
separate from the frozen pilot. Reading it does not stop that pilot or launch
training. An explicit instruction to start/adopt this program authorizes its
defined transition and runs without further approval waits. It then supersedes
the old nine-run mandate for the new campaign only. Preserve the original recipe,
records and incomplete outcomes.

Read [the direction audit](docs/DIRECTION_AUDIT_20261002.md), [AGENTS](AGENTS.md),
[implementation](docs/IMPLEMENTATION.md) and [frozen pilot protocol](docs/FROZEN_PROTOCOL.md).
The audit is the rationale; this file is the operational assignment.

## Objective and priority

Establish a credible class-conditioned flow SiD/SiDA comparison with a strong
teacher and realistic student exposure. The primary scientific comparison is
the effect of the adversarial package within each NFE pair. The practical aim
is quality in the range of the reported competitive results, not completing a
grid regardless of whether its teacher is adequate.

Priority: **teacher quality → one-call pair → two-call pair → report**.
Default scope is one extended teacher and four fresh students. Three/four NFE
are deferred, not prerequisites for answering the primary question. Do not
silently add them or conduct architecture, optimizer or sampling sweeps.

Targets to display in every report:

| Reference | One call | Few calls | Meaning |
|---|---:|---:|---|
| TA official reference, checked Oct 2 | 39.15 | 30.90 | Challenge targets; different reference split from local development |
| Sangwoo's reported SiDA | 28.07 | 22.79 at 2 calls | Competitive context; exact evaluation/configuration identity unavailable |
| Teacher readiness for this assignment | — | Development FID around 18–20 | Operational gate specified below, not a claim of equivalence to Sangwoo |

Do not promise those scores. Do not train until success indefinitely. A valid
negative result is failure to reach the declared target at the declared cost.
No official-validation tuning, FID-seed search or informal extra candidates.

## 1. Take ownership and preserve the pilot

After explicit adoption, check current progress, worker/supervisor identities,
saved checkpoints, GPU lease and source hashes. Do not launch another GPU 2
process alongside the pilot. Default transition is to close the pilot at the
next verified complete checkpoint; if S1 has not reached its first 1k evaluation,
that evaluation is the preferred handover boundary. If it has already passed,
use the next checkpoint instead of finishing another arm just for symmetry.
Do not wait for all eight runs.

Close the supervisor queue as well as the owned worker so A1 or later runs cannot
start inadvertently. Verify signal behavior: current `train_stage.py` does not
save on SIGTERM. Preserve the most recent complete checkpoint and report any
updates lost since it. Stop only positively identified owned processes; release
GPU 2's lease before replacement work. A host-wide reset is forbidden.

Archive immutable identities for T `latest.pt` (full raw model, optimizer, EMA,
RNG), selected T `best.pt`, completed evaluations and any pilot students. Preserve
`configs/campaign.json`, frozen source/student records and `runs/campaign/`.
Label unrun arms not started and interrupted arms partial. The 30k teacher
selected checkpoint SHA is recorded in the audit; hash the full continuation
checkpoint as well. Do not initialize continued teacher training from EMA-only
`best.pt` and call that exact continuation.

Use a new campaign ID, new config/ledger/output paths and new source revision.
Implement only the namespace/config/continuation controls needed. The existing
launcher hardcodes `runs/campaign`, T/S1/A1 identities and the pilot source hash;
passing a different JSON file alone is not sufficient. Verify that inference
uses the configured ladder rather than a hardcoded alternative. Preserve old
inference exports with their old source. No general search framework is needed.

## 2. Freeze a finite recipe with an explicit comparison ledger

Before substantive new training, write one small machine-readable recipe and a
table of **external reference / our value / reason for mismatch / evidence**.
Record architecture dimensions, parameter counts, time/preconditioning conventions,
loss reductions/weights, G/fake/D balance, warmup, resets, EMA, augmentation,
batch/exposure, solver/ladder, split/preprocessing and selection rules.

Use available Sangwoo documents; no messages to him or Notion edits are authorized.
Exact settings that remain unknown must stay labeled unknown. Parameter count
alone is not architecture matching. EDM sigma 2.5/.4 cannot be pasted into flow
time: SNR alignment would give t=sigma/(1+sigma) only with the corresponding
input/output scaling and initialization convention. This is not an instruction
to replace the flow ladder by that mapping.

Defaults deliberately preserve the current flow adaptation:

- Same 61,823,875-parameter DDPM++ U-Net, 64×64 RGB, 151 classes, no external
  pretrained components. Same flow convention and SiD Jacobians.
- Teacher continuation retains batch 64, LR 2e-4, existing Adam/EMA/RNG state,
  uniform time/unit FM weight, x-flip .5 and dropout .25. This keeps a clean
  continuation and knowingly differs from Sangwoo's reported teacher batch 256.
  Do not reset warmup, exposure or EMA ramp on continuation.
- New students use effective batch **128**, microbatch at most 16 initially,
  accumulation 8, LR 2e-5 for G and fake, Adam (0,.999), clip 10, EMA half-life
  cap 50k images and current .05-image-count ramp. Freeze any numerical execution
  changes from qualification before student training.
- Alpha=1, mean SiD weight=1, mean fake FM weight=1, D weight=1 and maximum G GAN
  weight=.01; preserve SiD scaling in both arms. One fake update per G update.
  No fake-only warmup and no fake reset by default. Record that the reported
  useful late fake reset is not reproduced because its timing/state details
  are unknown; do not invent a reset after observing FID.
- Keep 40k-image G GAN warmup and 40k-image linear ramp; define them in image
  presentations, not copied batch-64 update counts. D starts with the first
  fake update. Keep G/fake LR warmup at 6,400 images. Round to full batches and
  record exact boundaries. Shared settings apply to both members of a pair.
- Teacher/D real-data x-flip remains the only data augmentation. No new geometric
  ADA or SoFlow augmentation conditioning is introduced. Distinguish real-data
  augmentation, any discriminator-only augmentation, and denoising corruption.
- Current loss-time law and flow ladders **[1] / [1,.5]** remain fixed. Different
  ladders, loss weights, warmups or resets require a justified prelaunch protocol
  amendment from new source/correctness evidence, not trial-FID optimization.

The eightfold/tenfold exposure gap is the first problem being repaired. These
defaults are an explicit, inspectable recipe, not a claim that every external
setting has been reproduced. If a correctness check finds a genuine defect,
fix and qualify it before freezing; record why the old and new trials then
cannot be a pure duration ablation.

## 3. Strengthen and qualify one teacher

Continue the scratch teacher from the full 30k state in the new lineage to
**20,000,000 cumulative optimization draws** (312,500 updates at batch 64).
Existing 1.92M counts toward that total: the added budget is 18.08M. There is one
teacher lineage, no restart from scratch or new teacher sweep. Changing the
original horizon is a declared new experiment, not exact recovery of the pilot.
Within this new experiment, interruptions must recover exactly.

Before continuation, evaluate the preserved EMA at 32 and 64 Euler calls with
matched initial noise/classes. Use **64-call uniform Euler as the fixed primary
teacher evaluation** thereafter. Keep 32-call values as a labeled diagnostic;
do not pool solver budgets or choose a solver per checkpoint. If the difference
is large, report discretization error explicitly. No sampler search.

Evaluate at cumulative **4M, 6M, 8M, 10M, 12M, 14M, 16M, 18M, 20M** draws;
all are divisible by batch 64. Use selection seed 1729, 20 images/class and the
existing 3,020-image development reference. Keep readable learning curves,
fixed-noise galleries, all-class diagnostics and nearest optimization examples.
Continue to the full horizon even if an early checkpoint crosses a target;
exceptions are numerical failure, severe collapse, integrity/resource failure
or the predeclared watchdog. Do not silently extend past 20M.

Select the lowest primary teacher FID among declared checkpoints at **10M or
later**, earliest on ties. Confirm that checkpoint once with seed 271828,
without reselection. Teacher readiness requires all of:

1. Two-seed mean development FID **≤20**, neither seed above **22**, at the fixed
   64 calls. These are pragmatic planning thresholds around the reported 17.8,
   not statistically established equivalence margins.
2. Finite outputs, no persistent broad collapse, and class-reference review
   covering all 151 classes through paginated galleries. Pixel variance alone
   is insufficient; flag ambiguous classes and obvious copied templates.
3. No unresolved gradient/time/preprocessing problem; distinct same-class noise
   produces meaningful structural variation, with nearest-training-neighbor
   evidence and limitations reported.

If this gate fails, **report and stop before the student matrix**. That is an
underqualified teacher result, not proof flow distillation fails. Do not move
the gate, spend all student runs anyway, or start another teacher automatically.
Freeze one qualifying checkpoint and its hash for every new student.

## 4. Train the primary comparison group

| Order | ID | NFE / treatment | G presentations | Updates at batch 128 |
|---|---|---|---:|---:|
| 1 | P2-S1 | 1 / SiD | 3,500,032 | 27,344 |
| 2 | P2-A1 | 1 / SiDA | 3,500,032 | 27,344 |
| 3 | P2-S2 | 2 / SiD | 2,000,000 | 15,625 |
| 4 | P2-A2 | 2 / SiDA | 2,000,000 | 15,625 |

G, fake and EMA start freshly from the **same stronger teacher** in every arm;
fresh optimizers and saved common RNG streams within each pair. No pilot-student
warm starts and no transfer from the one-call arm to the two-call arm. Count
fake-regression and real-D presentations separately; with the default 1:1 ratio,
fake exposure equals G exposure, and A additionally consumes that many real draws.

These horizons follow the scale of Sangwoo's reported one-/two-step exposures;
they are not evidence that our models will have converged. They match exposure
within each pair, not compute across K. Train each assigned horizon even if it
passes the TA planning target early. No candidate replacement after weak FID.
Numerical/integrity failures are reported, with no silent skipped updates.

Evaluate at 0.5M, 1M, 1.5M, 2M G presentations, and also 2.5M, 3M, 3.5M for
the one-call arms; round upward to full batches. Selection uses seed 1729,
minimum development FID, earliest tie. Report final-horizon FID separately from
selected-checkpoint FID. Confirm each selected checkpoint with seed 271828
without reselection. Keep the previous 256k pilot results labeled separately;
they are not replacements for the new SiD controls.

Report the completed one-call pair concisely and continue automatically to the
two-call pair. After all four outcomes, stop and propose next directions. Do
not add 3/4-NFE students, independent training seeds, an EDM teacher or an
adversarial-only group to this default assignment.

## 5. Qualification and cost discipline

Reuse existing implementation and valid checks. Before long runs, validate the
new campaign paths and checkpoint migration, batch-128 accumulation, effective
updates under chosen reductions, intended Jacobians and frozen parameter routing,
zero-adversarial equivalence, exact next-update recovery, and actual 1/2-call
exports. Sample gradients separately for SiD/GAN and fake-FM/D on disposable
batches; record update-to-weight norms, clipping and finite FP16 gradients.
Do not select coefficients by which disposable probe has better image quality.

Verify independent checkpoint loading with only inference dependencies and the
protected official evaluation functions. Preserve starter files and licenses.
Snapshot all new source/config/teacher/partition identities. Exact recovery must
preserve RNG, optimizer, fake network, EMA, scaler, update/exposure counters and
consumed allowance. Migration from the pilot must check equivalent state and
the next update before extending its horizon, with documented identity changes.

Physical **GPU 2 only**, `CUDA_VISIBLE_DEVICES=2`, process-local `cuda:0`, expected
UUID `GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2`; verify live ownership. Hold
`/tmp/fastgen-gpu2.lock`. Preserve AFM GPU 0 and SoFlow GPU 1. Use existing
`fastgen_sid_dit`, `PYTHONNOUSERSITE=1`, and its interpreter's `python -m pip`
for any justified dependency work. Do not create another environment needlessly.
Shared images remain read-only; partition SHA stays
`825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`.

Each complete learned framework remains ≤100M parameters; full simultaneous
training/evaluation workload must remain below 20,000,000,000 allocated bytes.
Measure reserved/device/host memory too. Accumulation is a way to match effective
batch under the memory limit; copying Blackwell runtimes is not a budget method.

Measure sustained throughput with the actual final batch and alternating phases.
Before launching each stage, record its expected optimization, evaluation and
checkpoint-I/O time and a **finite full-stage watchdog**. Default calculation:
1.35 × measured projected training+I/O + declared evaluation allowance + 2h
recovery margin. Include failed/replayed work; at most two exact recoveries per
stage. Do not reset consumed time by renaming a run or relax a watchdog after
failure. No inherited 12h/16h cap. If observed throughput invalidates a budget,
report it; do not truncate exposure silently just to finish soon.

As of the audit, extending T to 20M projects **~79 optimization hours**. Four
students project **~167–283 optimization hours** at present per-image throughput.
Allowing ~20% for surrounding work gives an indicative **12–18 days** total
remaining, not a promise or a launch authorization. Reprofile batch 128 before
freezing the actual watchdogs. Full eight-student training at these scales
would instead be roughly **3–4 weeks** and is outside the default scope.

## 6. Compact reporting and interpretation

| Run | Teacher/config identity | Updates; G/fake/real exposure | Selected + final FID; confirmation | Actual NFE | Recognition/diversity/neighbor evidence | Train/eval/total hours | Peak allocated/reserved/device; host | Failures; gallery/export |
|---|---|---|---|---|---|---|---|---|
| T-extended | pending | 20M cumulative target | pending | 64 primary; 32 diagnostic | pending | include old teacher cost once | pending | pending |
| P2-S1 | pending | 3,500,032 G target | pending | 1 | pending | pending | pending | pending |
| P2-A1 | same teacher / paired recipe | same G target + real D | pending | 1 | pending | pending | pending | pending |
| P2-S2 | pending | 2M G target | pending | 2 | pending | pending | pending | pending |
| P2-A2 | same teacher / paired recipe | same G target + real D | pending | 2 | pending | pending | pending | pending |

Include separate T/G/full fake-score-D parameter counts and A−S FID/diversity
differences at each NFE. Charge original teacher training once plus continuation;
keep abandoned pilot-student and setup costs visible without charging the
teacher four times. Keep one short decision note at each stage; no repeated essays.

All-class galleries must include class references, independent noise draws and
nearest optimization examples. Saturation, duplicates and multiscale pixel
similarity are screening tools, not proof of semantic diversity or originality.
Keep sample/metric selection rules fixed and disclose reuse of development data.
Official validation is held out from all decisions in this assignment.

The controlled claim is the adversarial package's effect under this stronger flow
teacher and recipe. It is not an exact Sangwoo replication, a causal flow-vs-EDM
comparison, proof of teacher protection relative to adversarial-only training,
or evidence of statistical significance from one training seed. If the pilot
and new batch/recipe differ, their contrast is a package change, not a clean
teacher-duration ablation. Report remaining mismatches rather than hiding them.

Suggested launch instruction after reviewing this program:

> Adopt program_comparison.md. Supersede the short pilot at the documented safe
> handover boundary, preserve its artifacts, then complete the teacher-gated
> 1-/2-NFE SiD/SiDA comparison on GPU 2 without intermediate approval waits.
> Stop at the declared teacher gate failure or after the four student outcomes.
