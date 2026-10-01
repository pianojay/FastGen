# One flow teacher and eight matched students

**Prepared October 2, 2026; not started.** Read [context/source map](HANDOFF.md) and [AGENTS](../AGENTS.md). This document is an experiment mandate, not an executable configuration. Source integration, environment creation, qualification and training belong to the explicitly started implementation agent.

## Question and fixed assignment

For a fixed scratch-trained, class-conditioned pixel-space flow teacher, how does adding adversarial training affect quality, diversity and cost of SiD students at exactly 1, 2, 3 and 4 neural evaluations?

Train **one** Gaussian rectified-flow teacher and **eight** separate students. Use the roughly 62M DDPM++ U-Net family as the architecture reference; use class embeddings, 64×64 RGB and no text/latent pretrained components. Sangwoo's precise configuration is not available: inspect the pinned public `SongUNet`/encoder-decoder implementations, record the exact chosen dimensions/preconditioning/parameter count before training, and describe this as family/scale matched until exact equivalence can be verified. Do not claim architecture matching from parameter count alone. No architecture sweep or substitution with a DiT belongs to this assignment.

| Order / ID | Teacher | Student objective | Inference calls | Initialization |
|---|---|---|---|---|
| T | Scratch | Conditional rectified-flow matching | Predeclared teacher solver budget | Fresh weights and optimizer |
| S1 | Frozen T* | SiD | 1 | T* student backbone; common 1-NFE initial state |
| A1 | Same T* | SiD + adversarial | 1 | Same as S1 |
| S2 | Same T* | SiD | 2 | T*; common 2-NFE initial state |
| A2 | Same T* | SiD + adversarial | 2 | Same as S2 |
| S3 | Same T* | SiD | 3 | T*; common 3-NFE initial state |
| A3 | Same T* | SiD + adversarial | 3 | Same as S3 |
| S4 | Same T* | SiD | 4 | T*; common 4-NFE initial state |
| A4 | Same T* | SiD + adversarial | 4 | Same as S4 |

Complete teacher selection once, record T* checkpoint hash and freeze it for all eight students. Every student's fake-score backbone is freshly initialized from the same teacher; discriminator-head initialization is recorded separately and matched where applicable. Fresh optimizer/EMA/RNG/data state per student; no continuing an earlier student's weights into another NFE arm. Execute S1,A1,S2,A2, report that stage, then S3,A3,S4,A4 without another approval wait. Both arms of each pair must appear in results, including failures. Never omit a weak arm from reporting or silently replace it with another trial.

## Implement, qualify and freeze

1. Build the minimal pixel/class integration outside the snapshots. Use direct class conditioning and a shared fake-score/discriminator feature backbone. Trace FM, SiD and adversarial equations and gradient paths to pinned sources. Record all deviations, especially input/time preconditioning, reductions, loss weights, pixel scaling, augmentation and discriminator head. No teacher or auxiliary pretrained network imports.
2. Create the independent environment with course-compatible dependencies. Qualify **physical GPU 2** and full simultaneous teacher/student/fake-score/D/optimizer/EMA workload. Distinguish effective batch from microbatch and training draws from discriminator real images. Disposable probes establish correctness, actual memory and sustained throughput, not uncounted quality trials. Reuse justified infrastructure; do not build a general search platform.
3. Before training T, freeze its architecture, training-time distribution/weighting, optimizer, learning schedule, augmentation/dropout/EMA, effective batch, exposure/horizon, solver/evaluation cadence, checkpoint-selection rule, bounded recovery and full finite watchdog. Use source evidence and measured throughput; neither SoFlow's horizon nor Sangwoo's GPU hours is an allocation. Include preprocessing, qualification, evaluation and failed work in total cost.
4. Train T and evaluate its predeclared checkpoints with the fixed development reference. Select T* by the declared rule. Report teacher quality and within-class coverage before distillation. A valid but weak teacher is reported as a limitation; poor FID alone does not authorize another teacher. A teacher with invalid outputs, severe persistent collapse or failed qualification is a blocker: report it rather than spending eight student runs or silently repairing the scientific scope.
5. Before student training, freeze a common distillation protocol and the four NFE-specific ladders/horizons. Each pair shares T*, initialization, alpha, distillation coefficient/time weighting, fake-score loss and update schedule, effective batch, sampling/noise policy, dropout/augmentation/EMA, training exposure, checkpoint-selection rules and evaluation draws. Declare precision/accumulation behavior and any pair-matched execution adaptation after profiling. No FID-driven retuning between S and A.
6. The treatment is the adversarial package: D real/fake learning plus the generator adversarial term, with predeclared coefficients and warmup. Preserve the common distillation coefficient when enabling GAN; do not inherit the upstream 0.5 switch as a second change. An inactive D head may be retained in S for structural equivalence, but it must not update or alter outputs. Distinguish this treatment from isolating just one scalar loss coefficient.
7. Run all assigned students with finite declared watchdogs and bounded exact recovery. Waiting hours for a healthy run is part of the assignment. Maintain a small durable campaign record and status, not approval gates or long essays. Unresolved hardware/data/correctness blockers are reported; no substitute GPU, architecture, teacher search or additional quality trials.

Recipe records must include class/data schedules, separate RNG streams, teacher/fake/G gradient policy, noise endpoints, fake-score-only warmup if used, G/fake/D update ratios, full horizon and schedule units, parameter counts, time/memory projections, all allowance components and the pair invariants. Do not move thresholds or extend budgets after seeing results. Within each NFE pair match exposure; across NFE budgets record differences and real compute rather than claiming equal compute from equal updates. Shared teacher cost is charged once; each student's stand-alone pipeline cost may be reported as teacher cost plus that student's cost, without summing that teacher eight times in the campaign total.

## Required checks before substantive training

- Analytic Gaussian/toy tests verify time direction, conditional FM target, `D=x_t-t*v`, score/velocity conversion at interior times, endpoint handling and finite values. Test label conditioning independently of noise.
- Small deterministic loss/gradient checks cover the SiD surrogate, intended input Jacobians through fixed networks, frozen teacher parameters, fake-score learning from detached generated samples, real/fake BCE signs, reductions and warmup. With adversarial coefficients zero, the matched SiDA path must reduce to the SiD path, including effective distillation scaling and common updates.
- Validate accumulation and update ratios using full training phases; test weighting, effective batch and scheduler/RNG counters. Isolate adversarial RNG/data consumption so matched class/time/noise streams remain matched within a pair. Store draw identities, not only seed integers.
- For each K=1,2,3,4, count exactly K inference network executions, verify training-step sampling/rollout boundaries and stopping gradients through prior rollout steps, and test ladder endpoint behavior and fresh-noise construction. No hidden CFG, teacher, encoder or decoder calls.
- Exact recovery preserves teacher identity, student/fake/D, optimizers, EMA, schedulers/scalers, all RNG streams, samplers/cursors, update/exposure counters, configuration/source/partition hashes and consumed allowance. Compare the next complete update after reload. Test duplicate-job refusal under the GPU 2 lease. Record the tested interruption type; no untested hard-crash guarantee.
- Verify each entire learned framework ≤100M parameters, concurrent peak allocated memory <20GB, and reserved/device/host metrics. Recheck at the most expensive 4-NFE training path. Qualification is not established by a forward pass alone.
- Smoke-test official development evaluation and independent export loading in a clean directory. Export each student with its sampler metadata and exactly the required weights; no mutable sibling imports or training-only state. An untrained smoke FID is plumbing evidence, not model quality.

## Evaluation, interpretation and compact reporting

Use unchanged official FID functions/preprocessing and the copied 3,020-image development reference, 20 generated images per class. Predeclare evaluation seeds, actual noise streams for paired S/A comparisons, checkpoint-selection policy and any sampling-seed confirmations. Official validation is not used for training, selection or protocol tuning. Additional sampling seeds do not create training replications. Use enough evaluation to compare the specified arms without turning it into a sampling search.

Maintain fixed-noise class-reference galleries, per-class diversity/duplicate diagnostics, saturation and nearest optimization examples; report pooled FID alongside coverage failures. Compare recognition and structural variation, not pixel variance alone. Teacher and students should have separately labeled quality/diversity evidence. Report raw loss curves with readable trends, G/fake/D gradients, update timing and memory without presenting unrelated consistency-MSE plots.

| Run | NFE / objective | Teacher + init identity | Config / source / checkpoint | Updates / presentations | Development FID / eval seeds | Diversity / class failures | Gallery / neighbor audit |
|---|---|---|---|---|---|---|---|
| T / T* | Teacher / FM | scratch | pending | pending | pending | pending | pending |
| S1 / A1 | 1 / SiD vs SiDA; use separate rows | pending | pending | pending | pending | pending | pending |
| S2 / A2 | 2 / SiD vs SiDA; use separate rows | pending | pending | pending | pending | pending | pending |
| S3 / A3 | 3 / SiD vs SiDA; use separate rows | pending | pending | pending | pending | pending | pending |
| S4 / A4 | 4 / SiD vs SiDA; use separate rows | pending | pending | pending | pending | pending | pending |

| Stage / run | Teacher / G / fake-score-D parameters | Qualification / training / evaluation hours | Cumulative campaign hours / allowance | Peak allocated / reserved / device GB | Host RSS / PSS | Numerical failures / recovery / outcome |
|---|---|---|---|---|---|---|
| Fill each stage and each of the nine runs | pending | pending | pending | pending | pending | pending |

Record paired FID/diversity differences at each NFE and the quality-versus-training/inference-cost curves. This is one teacher and one training realization per arm; do not claim population-level significance. Treat Sangwoo's results as reported external evidence unless partition, exact architecture, preprocessing, FID implementation, sample counts, seeds and selection policy are aligned. No claim of outperforming an official score solely from local development FID.

After S1/A1/S2/A2, send a concise interim result and continue with S3/A3/S4/A4. After all eight, report successes, failures, limitations and one or two proposed next directions, then **stop**. An adversarial-only control, training-seed replication, teacher/architecture sweep and any new student require a subsequent assignment. This pairwise design does not alone prove that teacher supervision prevents GAN collapse.
