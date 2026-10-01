# One-baseline assignment: pixel-space AFM

Status: **handoff prepared; implementation, environment and qualification pending**. Read [context](HANDOFF.md) and [boundaries](../AGENTS.md). The next explicitly started agent owns one meaningful baseline through evaluation, without intermediate approval waits. This document is the work assignment, not a frozen training configuration.

## Question and deliverable

Can a class-conditioned, teacher-free discrete AFM learn useful 64×64 Pokémon generation in exactly one generator call, at a practical measured cost?

Implement a moderate U-Net generator and class-conditioned discriminator from scratch, using official AFM losses as the reference. Begin architecture design around the familiar U-Net128; choose D and any necessary G changes with a brief rationale. Preserve the full one-step endpoint objective rather than substituting SoFlow consistency training. No teacher, VAE, guidance, few-step extension or architecture sweep belongs to this baseline.

Deliver a reproducible frozen recipe, one completed baseline (or documented infeasibility/failure), raw incremental metrics, labeled sample gallery with real class references, development FID, cost/memory measurements, and independently loadable one-NFE inference checkpoint. A poor valid FID is still an experimental result. This baseline alone is not a complete two-track course submission; leave that distinction explicit.

## Implement, qualify, then freeze one recipe

1. Inspect the pinned sources and select only required infrastructure. Implement the minimal training/evaluation/export path using course-compatible dependencies; establish the separate environment. Document source-to-adaptation mappings for G/D losses, OT decay, regularization, augmentation and gradient normalization.
2. Verify training behavior on disposable scratch weights. Check full D/G update cycles with optimizer, EMA, regularization and augmentation active. Tune execution microbatch while preserving the chosen effective batch; use modest loader/thread counts to coexist with SoFlow. Qualification is an engineering check, not a search over FID or a substitute for meaningful training.
3. Record exact architecture, class conditioning, initialization seed, G/D update ratio, separate batch sizes/data streams, precision, optimizer, EMA, loss coefficients/schedules, augmentation, full generator-update horizon and schedule units. Account for both G and D data exposure. Choose a scientifically useful horizon from measured throughput and method evidence; do not copy SoFlow's 40k or AFM's five-million-loop-step setting automatically.
4. Record training plus startup/checkpoint overhead, evaluation and bounded recovery allowances before launch. Set a finite watchdog with explicit rationale. There is no universal 12-hour cap, and elapsed hours are not a reason to abandon a healthy planned run. Fix checkpoint/sample/evaluation cadence and predeclare which checkpoint is the primary result before viewing its FID.
5. Train the single frozen recipe from fresh state, supervise it through its declared horizon, evaluate and export. Implementation fixes and disposable checks are allowed before freezing. After launch, numerical/resource failures may trigger bounded exact recovery; a changed recipe or new substantive run is a new experiment and belongs in the final proposal.

## Checks required before trusting results

- Tiny deterministic loss/gradient checks against the reference formulas: adversarial signs, OT source-noise target, finite-difference penalties, centering, detached G on D steps and frozen D parameters with gradients through its input on G steps. Include class-conditioning behavior and real/fake augmentation consistency.
- Gradient accumulation checks cover uneven batches, the regularization sample fraction, optimizer step counts and schedule units. Gradient normalization updates state per backward; do not assume equivalence across microbatch sizes. Define/test the adopted semantics and record any deviation from upstream.
- Checkpoint recovery preserves G, D, both optimizers, EMA, normalization buffers, RNG, both data-stream cursors, schedule counters, source/config/partition identity and consumed budget. Compare the next full D/G cycle after recovery. Test duplicate-job refusal and do not claim hard-crash guarantees from an orderly-stop check alone.
- Verify protected hashes/base Model, separate G/D parameter counts, peak complete-workload allocation below 20GB, host-memory reporting, exact one-call inference and no D/teacher/optimizer in the exported model. Test the official loader in an independent directory without imports from either sibling checkout.
- Before a full run, execute one end-to-end development-evaluation smoke using the official FID functions. Its untrained FID verifies plumbing only. Validate generated labels/counts, reference identity and export compatibility.

## Evaluation and concise reporting

Use the fixed 3,020-image development reference, 20 generated images per class, unchanged official preprocessing and official clean-FID functions. Never use official validation for decisions. Predeclare evaluation seeds and final/checkpoint selection policy; keep evaluation cost proportional to this one baseline, without automatically copying SoFlow's 32-pass protocol. Any subsequent sampling-seed confirmation must be reserved before selection; it is not independent training replication.

Compare against the documented SoFlow one-call reference, disclosing different seeds or noise protocols. Use actual matched draws if making a paired claim; equal integer seeds alone are insufficient. Teammates' and TA figures remain contextual until split/preprocessing/sample-count comparability is established. No claim of beating an official score based solely on local development FID.

Provide raw losses and readable trends (separate G adversarial/OT and D adversarial/regularizer series), logits/gradients, update timing and memory. Generate fixed class-labeled panels and a compact reference gallery. Inspect within-class variation and nearest optimization examples before interpreting apparent sharpness as fidelity or dismissing memorization. Do not treat low-dimensional/noisy diagnostics as definitive class accuracy or diversity measurements.

Fill these tables incrementally; link machine-readable evidence rather than writing an essay for each stage. `pending` means unmeasured, never zero.

| Stage | Source/config identity | G/D updates and image presentations | GPU hours / allowance | Outcome / evidence |
|---|---|---|---|---|
| Implementation and qualification | pending | pending | pending | pending |
| Frozen baseline plan | pending | pending | pending | pending |
| Training and any recovery | pending | pending | pending | pending |
| Evaluation and export | pending | pending | pending | pending |

| Checkpoint / seed | 1-NFE development FID | G / D parameters | Train / eval / total GPU hours | Peak allocated / reserved / device GB | Host RSS / tree PSS | Numerical failures / recovery | Gallery / metrics |
|---|---|---|---|---|---|---|---|
| pending | pending | pending | pending | pending | pending | pending | pending |

At completion, give the measured result, limitations and one or two evidence-supported next directions, then stop. Do not automatically launch another baseline, extend the completed horizon, add a second training seed or turn this into an unrestricted autoresearch loop.
