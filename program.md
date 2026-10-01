# AFM stability-search program

Prepared October 2, 2026 (Asia/Seoul). **Ready as an assignment; not started.**
This program replaces the closed one-baseline mandate for future user-started work. Reading it, preparing this handoff, or opening the repository does not start training. Once explicitly started, complete the bounded campaign without intermediate approval waits, including hours-long runs and exact recovery. No extra agents unless requested.

## Question, evidence and scope

Can the existing teacher-free, pixel-space AFM architecture learn recognizable class-conditioned samples with meaningful within-class variation under a better training recipe?

The [stopped baseline](docs/EXPERIMENT_REPORT.md) completed 10,294 G/D cycles in 2.396 training-job hours. Its complete checkpoint is at 10,000; later progress is logs only. The EMA panel strongly suggests within-class collapse. No trained FID or population-diversity measurement was obtained. Keep the frozen recipe, report, source and artifacts intact. **Do not resume the stopped baseline.** Audit its saved state read-only; do not count that audit as a new recipe.

Hold fixed: 64×64 RGB, 151 classes, G U-Net128 (39,664,771 parameters), projection D (9,918,657 parameters), fixed partition, one generator call, Gaussian image-shaped source noise, and the discrete AFM objective family. Keep the source-noise transport term; do not turn the experiment into an unregularized GAN. No teacher, VAE, perceptual training network, extra diversity loss, architectural/capacity/conditioning redesign or few-NFE extension.

Allowed recipe changes: separate G/D optimizers and learning rates, update ratios, effective/execution batches with documented normalization semantics, OT coefficients and decay, finite-difference regularization/centering, gradient-normalization settings, clipping, EMA, data augmentation and discriminator augmentation. Do not mechanically copy SoFlow's augmentation-conditioning inputs. If the fixed architecture proves limiting, report that rather than silently expanding the search.

## Audit once, then conduct experiments

1. Read [implementation mapping](docs/IMPLEMENTATION.md), the pinned AFM trainer/config and current `afm_fastgen/training.py`, `losses.py`, `models.py`, `reporting.py`, and `scripts/run_baseline.py`. Inspect the preserved checkpoint's raw G and EMA on identical independent noise draws. Check input sensitivity, outputs before/after clamping, range saturation, loss signs and separate adversarial/OT/regularizer gradient magnitudes. Engineering checks passed previously; they did not establish distribution learning.
2. Implement only missing search controls, diagnostics and bounded orchestration. Reuse the existing environment, data cache, training, atomic checkpointing and reports. The current runner always performs terminal FID and has only a repository-local lease: adapt these deliberately before the campaign; neither is already a search interface. Do not spend the campaign building a general experiment platform.
3. Calibrate the audit on the collapsed baseline, real within-class examples, repeated images, and copies altered only by noise/color/translation. Freeze the audit protocol and acceptance rules before candidate 1. If a diagnostic wrongly rewards these controls, fix it before searching. Generated controls are diagnostics, never optimization data.
4. Choose each recipe from observed evidence. Record one short causal hypothesis and the exact difference from its reference. Related parameter changes may form one intervention; do not force isolated scalar changes when a coherent recipe adjustment is needed. No automatic Cartesian sweep or fixed hierarchy disconnected from evidence.
5. Freeze configuration/source identity, initialization and audit seeds, full G-update horizon, all schedule units, audit cadence, early-stop rules, checkpoint selection, and finite watchdog before each launch. Qualify changed paths and measure sustained throughput. Count G/D steps, presentations, training, audits, failed/replayed work and overhead. There is no inherited 12-hour ceiling, no fixed exploration/evaluation split, and no silent horizon extension.
6. Train fresh. Audit at 6,000 G updates and at the preregistered cadence thereafter; retain earlier panels for trajectory context. The 6k check is not a compulsory stop. Distinguish persistent collapse, incomplete learning, a healthy candidate, and inconclusive evidence. Consider OT phase and raw-versus-EMA lag. Early termination must cite the declared rule and evidence, not a single noisy statistic or an unattractive early panel. Infrastructure/numerical failures can stop immediately.
7. Continue sequentially on GPU 0 without asking approval for in-scope trials. At most **six candidate recipes** may enter substantive training. A failed or early-stopped candidate still occupies a slot. Exact recovery is the same candidate; changing its recipe creates a new one. Disposable correctness/throughput probes must be bounded, logged and must not become uncounted quality trials.
8. The **first** candidate satisfying the frozen acceptance criteria receives **one** fresh independent training-seed confirmation with the same recipe/horizon/audit policy. Report the repeat's outcome and stop, even if it fails. If none qualifies, stop after six candidates. There is no eighth run, extra confirmation, automatic FID-tuning campaign or capacity sweep.

## What counts as success

The target is recognizable Pokémon structure plus substantial within-class variation in pose, shape or composition, sustained across checkpoints and fresh noise draws. Full recovery of every mode is not a claim these diagnostics can establish. FID is not the search objective and is not a candidate-ranking or termination gate. Do not run full FID per candidate; leave final quality benchmarking to a subsequent assignment.

Use a fixed diagnostic panel with class references, raw-G/EMA comparisons and rotating class coverage during training. Acceptance and confirmation require an audit spanning all 151 classes, with per-class results and failure counts, not only the original 12 familiar classes. Predeclare sample counts and independent confirmation draws. Retain both common draws for comparisons and fresh draws against overfitting to the panel; record actual draws or their reproducible construction.

Combine multiscale image similarity/near-duplicate statistics, low-frequency structure versus high-frequency variation, noise sensitivity, saturation, and nearest optimization examples. Inspect recognizable variation visually against multiple same-class real references. Simple pixel neighbors are a diagnostic, not proof of memorization or its absence. The official Inception is evaluation-only if used for diagnostic features; it must not become a training loss. Quantitative thresholds must come from the calibration controls and be recorded before candidate 1; label ambiguous cases inconclusive rather than moving the threshold to accept a candidate.

Random noise, color changes, shifted copies of one template, and near-copies of a small set of training images do not satisfy the question. Require persistence across at least two scheduled audits after recognizable structure emerges; do not stop at a fleeting good checkpoint. A qualifying checkpoint may advance to confirmation without exhausting a failed-run allowance, but the confirmation repeats the same declared recipe and assessment point.

## Gradient and recovery cautions

`GradientNormalization` uses a running squared gradient in a custom backward. Scaling an adversarial loss upstream can be largely canceled by its normalization, including transient state effects. Before calling a candidate “weaker adversarial pressure,” test and record the actual post-normalization gradient change and ratio to OT; preserve intended accumulation weights. Do not reinterpret a coefficient-only edit as a scientific intervention if its effective gradient is unchanged.

Future recovery must preserve G, D, both optimizers, EMA, normalization buffers, RNG, both data-stream cursors, G/D counters, schedules, source/config/data identity and consumed allowance. Bound retries before launch. A user stop is terminal unless the user later explicitly changes it. Never resume from a collapsed historical checkpoint as a purported scratch candidate.

Use a shared, physical-device-specific host lease, recommended `/tmp/fastgen-gpu0.lock`, with `flock`, GPU UUID and process identity/occupancy checks. Hold it through all GPU stages; the existing project-local lock alone does not coordinate repositories. Do not steal a lease, kill an unknown process or fall back to GPU 1/2. SiD-DiT is independently assigned GPU 2 and can run concurrently. Qualification and checks must retain the [AGENTS](AGENTS.md) resource/data boundaries.

## Incremental record and completion

Maintain one campaign record with cumulative cost, slot count, current run/process identity, calibration protocol, pending decision and evidence links. Use machine-readable metrics and this compact table; `pending` means unmeasured, never zero. Update after launches, audits, stops and recovery. Do not write an essay per experiment.

| Candidate / seed | Hypothesis / change / reference | Source + config | G/D updates; presentations | GPU hours / finite allowance | Numerical status | Diversity + recognizable classes | Decision / reason | Gallery / checkpoint |
|---|---|---|---|---|---|---|---|---|
| Baseline, historical | Original pixel AFM | `ee74486`; `configs/baseline.json` | 10,294 / 10,294; 329,408 real/stream | 2.396 train; ~2.454 incl. qualification | Finite; user stopped | Strong suspected within-class collapse; population audit pending | Closed; read-only diagnostic reference | [Report](docs/EXPERIMENT_REPORT.md) |
| Candidate 1–6 | Fill incrementally | pending | pending | pending | pending | pending | pending | pending |
| Confirmation | First qualifying recipe; independent training seed | pending | pending | pending | pending | pending | pending | pending |

At completion report whether recognizable diversity was sustained and independently confirmed, class-specific failures, uncertainty/memorization concerns, total cost and the strongest next hypothesis. A negative campaign means no success among these recipes and this fixed architecture; it does not prove AFM cannot preserve modes. Stop before further training.
