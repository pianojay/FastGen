# Fixed SoFlow pre-program audit — October 4–5, 2026

**Completed.** A0–A4 finished within the 12 aggregate GPU-job-hour ceiling.
No sustained generator fine-tuning, FID-improvement pilot, range-penalty training,
or autoresearch run occurred. S3 remains the incumbent.

| Measurement | Result |
|---|---:|
| Contract tests | 38 passed |
| Generator / discriminator parameters | 89,140,419 / 28,930,048 |
| Frozen S3 samples | 6,040, 40/class |
| Real optimization images inspected | 14,059 |
| D diagnostic holdout | 604 images, four/class |
| Independent D runs | 2 × 512 updates, B32, 16,384 presentations/run |
| A4 disposable D/G attempts including replay | 12 / 8 |
| Total aggregate GPU-job time | 3.9786 hours |
| Failed GPU stages / automatic retries | 0 / 0 |
| Highest worker CUDA allocation | 13.059 GB |

Raw evidence and galleries: `runs/pre_program_audit_20261004/`.
[Gallery](../../runs/pre_program_audit_20261004/saturation/gallery.html) ·
[Diagnostic plots](../../runs/pre_program_audit_20261004/report.html) ·
[Compact committed evidence](pre-program-evidence-20261005.json) ·
[Frozen source/input manifest](pre-program-manifest-20261004.json).

## Saturation

Generated raw channel values outside [-1,1]: **14.06%** overall;
class means range **6.14%–27.04%**. Real exact-boundary
channel values average **15.42%** under the same equal-class weighting.
These are different quantities, not a matched measure of error. Across-class
Pearson correlation is 0.567; this is descriptive and
does not establish why the generator overshoots. Mean overshoot is
0.00770 across all coordinates, or 0.05479
conditional on clipping, in the [-1,1] scale. The exact maximum absolute raw
coordinate across these samples is 2.0129. The raw pixel quantiles in
the evidence use every 97th flattened coordinate, not exhaustive order statistics.

## Discriminator and parameter-gradient trajectories

Each cell averages the two fixed diagnostic noise sets; full per-draw values
remain in the evidence. G and EMA state hashes were unchanged at all checkpoints.
Wrong labels use a fixed 151-class permutation, not training augmentation.

| D seed | D updates | Pair accuracy: held / fit pool | Masked image energy | Parameter norm: actual / blocked | Actual vs unmasked cosine | Actual vs SCM cosine | Correct−wrong real score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20261004 | 0 | 0.469 / 0.483 | 10.25% | 3.1086 / 0.5094 | 0.9890 | -0.3334 | 0.0003 |
| 20261004 | 32 | 0.476 / 0.502 | 10.72% | 1.6407 / 0.3679 | 0.9790 | -0.2172 | -0.0001 |
| 20261004 | 128 | 0.513 / 0.531 | 10.55% | 1.6793 / 0.3142 | 0.9906 | -0.0516 | 0.0009 |
| 20261004 | 512 | 0.595 / 0.594 | 8.25% | 4.1939 / 0.4111 | 0.9971 | -0.1904 | 0.0060 |
| 20261005 | 0 | 0.465 / 0.506 | 11.01% | 4.9743 / 0.5762 | 0.9962 | 0.1999 | 0.0069 |
| 20261005 | 32 | 0.508 / 0.504 | 10.39% | 3.3588 / 0.5659 | 0.9944 | -0.1804 | 0.0098 |
| 20261005 | 128 | 0.530 / 0.527 | 10.62% | 3.5246 / 0.6498 | 0.9936 | -0.1700 | 0.0106 |
| 20261005 | 512 | 0.590 / 0.603 | 10.73% | 4.0880 / 0.9449 | 0.9896 | -0.0447 | 0.0127 |

The actual vector includes the clamp derivative. The blocked contribution is
a diagnostic vector–Jacobian product only; its sum with the actual vector is
not the gradient of the deployed clamped objective. Parameter contributions can
reinforce or cancel, so these norm ratios are not a percentage of learning lost.
All A3 vectors average 604 examples per noise set; they are not measurements of
the stochastic B32 training-update distribution.

SCM and squared-overshoot gradients were computed without updates and cached
because G was frozen. Their norms and cosines are in the evidence. The optional
range penalty passed analytic and finite-difference-related contracts, but its
effect on learning, image quality, and diversity remains untested.

## Readiness and cost

Phase transitions 0→.05→.1 and exact next-update replay passed at B8 and B32,
including sampler position, optimizers, EMA, RNG and counters. The original
35 contracts plus three audit-specific contracts passed. GPU-stage startup,
checkpointing and all GPU child stages are included once in the shared ledger; parallel jobs
are summed separately. Host RSS measures workers, not an aggregate host peak.

| Effective batch | D mean seconds/update | G mean seconds/update, rho>0 | D+G sum |
|---|---:|---:|---:|
| 8 | 3.069 | 1.393 | 4.462 |
| 32 | 11.814 | 3.934 | 15.748 |

These are short fixture measurements with data loading, exact penalties,
accumulation, optimizer, EMA and diagnostic update logging live. They exclude
checkpoint serialization and A3 analysis time. They establish execution cost,
not the number of updates required for useful learning. The native FP32 exact-
regularization path remains the qualified reference; no kernel optimization
or lazy-regularization substitution was introduced.

## Interpretation boundaries

- The D holdout was excluded from D updates, but S3 previously trained on those
  optimization images. It is not a generator memorization holdout.
- Fit-pool diagnostic images are sampled from the D-fit pool, not guaranteed
  to have appeared in its updates. The evidence reconstructs actual exposure
  and checks it against saved sampler states at every checkpoint. Pair accuracy
  is not semantic image quality.
- There is no range-penalty learning arm, generator response curve, FID claim,
  or evidence establishing the ideal warmup, ramp, optimizer or training horizon.
- A4's shared fixture sampler verifies exact recovery, not treatment/control
  draw matching. A future paired pilot must use separate G/D data samplers and
  declare GAN noise indexing across zero/nonzero-rho phases so extra D updates
  cannot shift the control's G data or common stochastic draws.
- Results from two D seeds describe this fixture; they are not independent
  generator-training replications. The tentative program remains inactive.

## Assessment for the next assignment

**The component path is ready; the learning experiment is not yet specified.**
All fixed A0–A4 procedures completed. No failed or timed-out stage was relabeled
as success. Both D runs reached exactly 512 updates, and the A4 fixtures used
12 D / 8 G attempts including replay. GPUs 1/2 were idle after worker exit.

D learned modest separation with this fixture: final held-out pair accuracies
were **59.52% / 59.02%**, compared with **46.94% / 46.52%** at initialization.
The final fit-pool accuracies were **59.44% / 60.35%**, giving little evidence of
a large fit/holdout gap at this exposure. Reconstructed sampling recorded
9,317 / 9,293 unique fit images; 423 / 427 of the 604 fit diagnostic images had
actually appeared in training. These facts support learnability, not a claim
that D is ready to guide G or that 512 updates is an appropriate warmup.
Correct-label real scores exceeded wrong-label scores by only **0.0060 / 0.0127**
at the endpoint. This positive but small sensitivity does not establish semantic
conditioning or rule out reliance on boundaries/backgrounds. The latter remains
unmeasured and must trigger focused investigation if suspected in a later pilot.

At the final D checkpoints, clamp masking removed **8.25% / 10.73%** of visible
image-gradient squared norm. Actual parameter-gradient norms were **4.194 / 4.088**;
blocked-coordinate contributions had norms **0.411 / 0.945**. Their sums remained
closely aligned with the actual gradients: mean cosines **0.9971 / 0.9896**.
There is no evidence here of wholesale parameter-gradient disappearance. The
blocked contribution is material in magnitude for one seed, so clamping is not
proved harmless either. Existing detached balancing compensates scalar norm
changes when uncapped; it does not repair a changed direction. The comparison
holds the visible-image gradient fixed and is not an unclamped-objective trial.
It must not be described as a percentage of learning lost.

Mean SCM/range gradient norms were **0.1432 / 0.1013**, with cosine **0.4328**;
the range term supplies a finite, partly aligned local corrective signal.
At fixed visible outputs, that says nothing about its eventual effect on FID
or diversity. Its training coefficient stayed zero, and it has not been wired
into a training arm. The current evidence does not warrant prioritizing this
objective extension over establishing an informative adversarial trial.

| Dimension | Readiness after the audit | Remaining question |
|---|---|---|
| Execution, gradients, phase state and recovery | Passed at the declared fixtures | Production pilot must still freeze a full recipe and matched control streams |
| Activation timing and relative strength | Components support controlled comparisons | Define useful G response horizon, EMA lag and a justified D-readiness criterion first |
| G/D optimization balance | Costs and weak-to-modest D learning measured | No LR/update-ratio optimum established; enter only for a diagnosed tracking problem |
| R1/R2 and flip probability | Exact reference path numerically qualified | No demonstrated large overfit gap or calibrated reason to change these now |
| Clamp/range extension | Diagnostic gradient verified; extension remains conditional | Show an actual learning bottleneck and qualify a matched range-training arm before search |
| Boundary-artifact reliance and semantic conditioning | Not settled | Focused controls are needed if the pilot depends on either claim |

B32 costs about **15.75 seconds for one D plus one nonzero-rho G update** in
short fixtures: roughly 4.37 GPU-job hours per 1,000 such pairs before evaluation,
checkpoint overhead and any additional D preparation. This arithmetic is not
a proposed horizon. Establishing an affordable observation horizon is the
central unresolved experiment-design question; diagnostics cannot answer it.
The tentative program remains **inactive**, with no numeric search ranges,
candidate count or campaign duration silently filled in.

## Final repository verification

After completion, all frozen source-file hashes, S3 checkpoint and partition
hashes, protected files/base Model, and the 23-file StyleGAN2 vendor identity
were rechecked. The historical program matches its bytes at `858663d`; the
committed frozen manifest matches the executed manifest exactly. New document
links and report assets were checked; generated plots were visually inspected.
Both device leases were available and both workers had exited. The CPU-only
report renderer reconstructs actual sampling exposure and checks it against
each saved D sampler state. No GPU checks were needlessly repeated.
