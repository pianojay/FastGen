# Matched SoFlow adversarial pilot — completed October 5, 2026

**This recipe did not improve the S3 asset.** At the predeclared final-EMA endpoint,
adversarial A beat consistency-only C by **2.788 FID** on the two confirmation
sampling seeds, but remained **12.867 FID worse than untouched S3**. Both confirmation
seeds agree in direction. This is a response to the tested recipe, not evidence
that adversarial refinement generally fails, that longer training would fix it,
or that the difference is statistically significant.

The explicitly authorized fixed pilot is complete. Exactly two arms finished;
no successor, timing sweep or autoresearch was launched. Preserve S3 as incumbent.
The [frozen protocol](MATCHED_PILOT_DRAFT.md) and [manifest](matched-pilot-manifest-20261005.json)
remain unchanged. Execution source commit: `7065116c8e95741a8ce7f3a8fc8fcd3ba337b12a`.

## Fixed endpoint and measurement

All values use the official FID implementation with the existing **3,020-image
local development references**, 20 generated images/class, 151 classes, matching
noise within each seed, and actual one-/four-call accounting. They are local
measurements, not official challenge scores. Prior development-data use remains
disclosed; no official validation tuning occurred. Seeds 7302/7303 are sampling
confirmation, **not independent training replications**. Final EMA 512 was selected
before scores; raw and update 128 observations are diagnostics, not alternatives
selected after seeing FID.

| Final EMA | 1 NFE, 7301 trajectory | 1 NFE, 7302 confirmation | 1 NFE, 7303 confirmation | Confirmation mean | 4 NFE, 7301 only |
|---|---:|---:|---:|---:|---:|
| Untouched S3 | 68.6920 | 67.4560 | 66.5407 | 66.9983 | 25.0951 |
| C: consistency continuation | 85.1867 | 83.1208 | 82.1867 | 82.6537 | 31.5657 |
| A: consistency + RpGAN | 82.6231 | 80.2509 | 79.4803 | 79.8656 | 30.0128 |
| A − C | −2.5637 | −2.8699 | −2.7064 | −2.7882 | −1.5529 |
| A − S3 | +13.9310 | +12.7949 | +12.9396 | +12.8672 | +4.9177 |

Scientific judgment: **favorable final-EMA A-versus-C response** under the declared
2-FID practical threshold, with limited generalization from one training seed.
Practical judgment: **adverse A-versus-S3 result**; not a useful replacement.
Four-call regression is reported rather than used as the one-call acceptance gate.

| Seed 7301 diagnostic | C | A |
|---|---:|---:|
| Update 128 raw | 84.8072 | 92.8045 |
| Update 128 EMA | 75.7644 | 79.0768 |
| Update 512 raw | 88.9791 | 89.3232 |

A was worse at update 128 and did not beat C in final raw weights. The favorable
final-EMA contrast therefore does not establish a universal improvement across
checkpoints or weight choices. Both raw and EMA regress from S3; this is not an
otherwise successful raw refinement hidden only by EMA lag.

## Executed recipe and controls

Both G arms started from the exact same S3 U-Net192 EMA as online G and G-EMA,
with fresh RAdam state: learning rate 2e-5, betas (0.9,0.999), eps 1e-8, no clipping
or weight decay, 512 updates at batch 32 with SCM microbatch 2. Each consumed
16,384 G presentations (about 1.165 optimization-partition passes). Fixed mature
SCM gap 0.0003125/minimum 1e-4, original augmentation, shared dropout and
residual weighting remained. EMA half-life was 2,048 presentations/64 updates,
without warmup. Actual GAN/inference endpoint retained exact clamp.

A used audited D20261004 after 512 frozen-G updates, selected by first seed order,
not the best audit score. It restored D weights, Adam state, D noise/flip streams
and sampler. D had one new update per G update, Adam lr 2e-5/betas(0,0.99), exact
0.1(R1+R2) each update, paired hflip 0.25 and fixed StyleGAN2 groups of four.
Its lifetime D counter reached 1,024:512 inherited plus512 new. Rho ramped linearly
from 0.1/64 to 0.1 over 64 G updates, then stayed fixed for 448. This is one tested
activation policy; the audit did not establish an optimal warmup or readiness.
C used rho 0 and no D updates; its inert D allocation remained resident.

G used all 14,059 optimization images in continuous image-uniform epoch
permutations. D retained its class-uniform sampler and 604-image diagnostic
holdout. That holdout is D-fit-excluded, not a pristine G-validation set: G SCM
may see it. G SCM/noise/flip streams were step-indexed and independent of D.
All 512 production steps matched positions, labels and SCM RNG hashes exactly.
Additional D calls could not consume the control arm's SCM draws.

## Diversity and visual limits

[Gallery](../../runs/matched_pilot_20261005/report/gallery.html) ·
[training curves](../../runs/matched_pilot_20261005/report/training.png) ·
[per-class review](matched-pilot-class-review-20261005.csv) ·
[fresh-draw panels](../../runs/matched_pilot_20261005/report/fresh_review.jpg) ·
[nearest-neighbor review](../../runs/matched_pilot_20261005/report/nearest_review.jpg).

Visual inspection covered final-EMA draws 0–5 for every class and all three models,
plus six classes at both fresh seeds. Matched compositions and pose variation
largely persist. No obvious new within-class template collapse was found in these
reviewed draws. Rendering, contrast and detail change; C is sometimes washed out,
and A is not uniformly better or more recognizable. Some inherited malformed or
ambiguous shapes remain. This is a coarse relative review, not an expert class
accuracy score or an exhaustive review of every generated image.

The fixed seed 7301 numerical diagnostic covers all 20 draws/class, all 151 classes.
Its distance is mean RGB MSE at 16/32/64 resolutions on pixels 0..1.

| Final EMA | Mean within-class pair distance | Exact PNG duplicates within classes | Median class-median nearest distance |
|---|---:|---:|---:|
| S3 | 0.18220 | 0 | 0.05242 |
| C | 0.17016 | 0 | 0.04919 |
| A | 0.17830 | 0 | 0.04971 |

A retains more pixel variation than C but slightly less than S3. Pixel variation
can reflect color/background/noise; it does not establish semantic coverage.
Neighbors were searched against the **full same-class optimization partition**,
including the D diagnostic holdout, with reference horizontal flips allowed;
development/official-validation images were excluded. The 12 closest classes per
arm were visually reviewed. Shared poses/shapes occur already in S3, with visible
differences in retrieved examples; low distances often favor pale backgrounds.
No memorization threshold was predeclared, and this metric/review cannot rule out
memorization. The evidence supports “no obvious new collapse,” not a diversity
or memorization clearance.

D held-out/fit no-augmentation pair accuracy fell to 42.38%/44.54% at 512
(43.87%/46.36% at 128), while its target G changed. This is not a fixed-target D
learning curve and does not isolate weak semantic guidance. A's last 64-update
median clipped-channel fraction was 17.25%, gradient cosine −0.0687 and detached
balancing coefficient 0.01578. Adversarial gradients were nonzero; clamping's causal
role in the regression was not tested. C's logged zero clipping is an unused-GAN
sentinel, not a measured absence of clipping.

## Verification, resource use and accounting

Seven targeted tests passed, covering paired streams, control/ramp recovery,
actual zero-rho SCM, RpGAN signs and R2/flip routing. Two disposable full-S3 B32
D+G attempts verified bitwise completed-update replay and restored D optimizer.
No production recovery or failed GPU stage occurred. The already completed
integration/pre-program tests were not needlessly rerun.

Final generator-only exports from both arms independently loaded through the
official Model interface, with one/four actual calls and no critic imports.
G 89,140,419 / D 28,930,048 parameters each satisfy the separate 100M limit.
Maximum measured whole-worker allocation was 12.109 GB; maximum reservation 13.067 GB,
under the 20 GB allocation ceiling. Maximum individual-worker peak host RSS was 3.235 GB; this is not concurrent aggregate host memory.
All 18 FID passes completed, with 95 batches and 95/380 network calls for 1/4 NFE.
Protected files, partition, vendor, frozen execution sources and inherited
S3/D checkpoint hashes were checked again after completion. Both device leases
were released and no pilot process remained. GPU 1/2 only; no other project changed.

| New stage, including its startup/checkpoint/diagnostic overhead | GPU-job seconds | Hours |
|---|---:|---:|
| Targeted qualification and replay | 93.193 | 0.02589 |
| A training | 8,228.718 | 2.28575 |
| C training | 1,393.305 | 0.38703 |
| All 18 evaluations | 1,372.332 | 0.38120 |
| **Total new spending** | **11,087.548** | **3.07987** |

This is aggregate GPU-job time, including overlapping A/C jobs, below the 4.5 h
ceiling. A's logged update bodies sum 8,113.904 s; C's 1,348.701 s. The remaining
training worker time includes initialization, diagnostics, saves and teardown;
it is already charged above, not free. Recovery qualification is charged in the
qualification stage. CPU export/diagnostic/gallery generation took 133.756 s;
additional CPU-only visual review/closeout stayed within its separate one-hour
allowance. No failed attempt or retry was omitted.

Inherited costs remain visible: S3 complete method cost 37.1875 h (including
qualification/evaluation); selected D audit worker 1.9566 h
(of which 1.7090 h were logged D updates). The entire earlier pre-program cost 3.9786 h
and **includes** that selected D worker. S3 + selected D + this full controlled
pilot totals 42.2240 h; counting the entire pre-program instead totals 44.2460 h.
These are transparent inherited-plus-study totals, not a claim that all control
and diagnostic overhead is required to train a deployable model.

## Disposition

Stop here. The bounded window produced an observable but practically adverse
response, so it is not merely “no measurable movement.” It does not establish
eventual potential or justify an automatic extension. Before broad GAN-strength
search, the highest-value question is why the consistency-only continuation
recipe degrades the starting asset and how to retain a useful continuation
control. Fresh optimizer, learning rate, batch/exposure and EMA policy all belong
to the tested continuation package; this pilot does not isolate their causes.
No additional audit, run or programme is authorized by this recommendation.

[Compact evidence](matched-pilot-evidence-20261005.json) retains exact values,
resource records, exports and judgments. Large checkpoints/images remain ignored
under `runs/matched_pilot_20261005/`. The repository was committed locally only;
no remote push, Slack message or Notion edit was performed.
