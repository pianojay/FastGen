# First-wave results: color and Fourier transfers

Completed October 7, 2026. Both dispatched experiments passed their frozen
contracts. **Same-class donors did not by themselves protect distinguishing
details, and clipping changed more than display range.** These are working
appearance interventions; they do not yet establish an identity-independent
style factor or useful augmentation for generative training.

The master compiled this report from worker summaries and structured handoffs.
Workers owned implementation, numerical verification, actual visual inspection
and replay. The master checked contract/cohort identity, declared completion
counts, resource limits, proof references and artifact-manifest digests;
[acceptance record](/home/user/jw_soflow/exploration/transfer_aug/shared/handoff_acceptance.json).

## Completed scope

Both methods used the same **604 A1 content images**, four distinct 64-pixel
images per category, with matched intra-class donors. FDA additionally used
one uniformly sampled other-class donor per content image. Seed: 20261007.
Inputs were the verified official 64 × 64 RGB export. No A2/B images, learned
models, training, classifier/discriminator evaluation or FID entered this round.

| Experiment | Fixed grid | Transformed images + controls | Visual evidence |
| --- | --- | ---: | --- |
| TA-001: Color Transfer | Intra-class; Lab interpolation 0.25, 0.5, 1.0 | 1,812 + 604 | 48 representative judgments + 8 tails; 24 inspected sheets |
| TA-002: FDA | Intra/inter separately; radius 0, 1, 2, 4 × strength 0.25, 0.5, 1.0 | 14,496 + 604 | 384 representative judgments + 8 tails; 34 inspected sheets |

Representative review covered one fixed content image in each of 16 seeded
categories across all conditions. Tails were selected by the frozen clipping
rule. These are **unblinded AI visual judgments on a small, repeated-condition
subset**, not category-accuracy estimates or population failure rates.

## Color Transfer: same-class pairing is a restriction, not a guarantee

The NumPy CIELAB implementation aligned/interpolated the requested moments to
numerical precision. One near-achromatic channel followed the specified
mean-only fallback, so its donor variance was deliberately not matched.
RGB clipping then prevented exact realization of those Lab statistics.

| Strength | Median RGB RMS change | Median clipped-channel fraction |
| --- | ---: | ---: |
| 0.25 | 0.0619 | 0.59% |
| 0.50 | 0.1156 | 2.71% |
| 1.00 | 0.2180 | 9.36% |

RMS is measured on clipped floating RGB in [0,1]; clipping fraction counts
raw scalar channels outside that interval. At full strength, the p95 clipping
fraction was 33.96%. Worker review found background tinting, reduced Primeape
eye contrast, and flat saturated Tentacool regions. At that strength, anatomy/
markings were judged retained in 7, uncertain in 7, and disrupted in 2 of the
16 representative outputs. Source blur and card clutter limited confidence.

**Implication:** intra-class color transfer can still erase information used to
recognize the subject. Whole-image moments combine the subject with its
background; matching them is not evidence of transferring rendering style.

[Worker summary](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-001/SUMMARY.md)
· [Visual gallery](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-001/gallery.html)
· [Condition measurements](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-001/condition_metrics.json)
· [Structured evidence](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-001/result.json)

## FDA: the intended spectral change is exact until clipping

The implementation matched the pinned NumPy FDA convention. Before clipping,
the maximum complex FFT change outside the selected window was only
**4.55 × 10⁻¹³**. Radius zero was verified to equal per-channel RGB mean
interpolation. It is a global color/brightness intervention, not an unchanged
control.

At full strength:

| Measurement | Intra-class donors | Inter-class donors |
| --- | ---: | ---: |
| Median RMS change, radius 0 | 0.190 | 0.231 |
| Median RMS change, radius 4 | 0.238 | 0.274 |
| Median outside-window relative FFT change after clipping, radius 4 | 0.203 | 0.244 |
| p95 outside-window relative FFT change after clipping, radius 4 | 0.542 | 0.538 |

The relative FFT measure is the L2 norm of the change outside the window divided
by the original image's outside-window L2 norm. It is **not an artifact score**.
Clipping spreads changes into frequencies that the raw operator preserved.

Worker inspection found largely recognizable layouts alongside stronger color
patches, halos and altered surface markings at larger radii/strengths. Across
the 192 representative condition-output judgments per donor arm, markings were
judged disrupted in 15 intra-class and 33 inter-class cases. These repeated
judgments are correlated. The fixed inter-class donors caused larger median
RGB changes in every matched condition; this does not isolate a causal class
effect because donor appearance/background also differed.

**Implication:** retaining raw Fourier phase is insufficient to establish
identity-safe RGB augmentation. Any later discriminator study must account for
the actual clipped images. This round did not measure discriminator responses.

[Worker summary](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-002/SUMMARY.md)
· [Visual gallery](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-002/gallery.html)
· [Condition measurements](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-002/condition_metrics.csv)
· [Structured evidence](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-002/result.json)

## A shared diagnostic limitation: clipping count is not visible damage

The frozen tail rule exposed a useful limitation. Color Transfer's highest-
clipping Mewtwo case had **61.95%** out-of-range channels but RMS change only
**0.00649** and looked almost unchanged. A Dewgong case clipped fewer channels
(45.77%) but had RMS **0.384** and visibly lost white-body detail. FDA's four
intra-class tails likewise appeared mild despite 61.8–66.0% clipping. Their
excursions were small real overshoots, not numerical roundoff.

Large nearly white areas can dominate a strict count. Keep clipping frequency,
excursion magnitude, appearance displacement and inspected subject details
separate. The original ranking was retained; no favorable replacement examples
were selected after seeing results.

## What this supports next

No winning method, strength, radius or donor policy is selected. No new
experiment is launched by this report.

- Investigate whether the transferred statistics predominantly describe the
  subject or its background. A white-pixel threshold alone is not a verified
  foreground mask.
- For FDA, distinguish raw operator behavior from gamut handling before testing
  a revised frequency window or downstream training objective.
- For color moments, investigate nearly achromatic channels and excessive
  variance scaling as explicit follow-up hypotheses, preserving this fixed run.
- If a discriminator experiment is later proposed, distinguish adding modified
  images to the real target distribution from applying a symmetric training
  augmentation. The present evidence cannot establish which helps SiDA.

These directions concern what the interventions do and what they destroy.
Producing more images remains separate from demonstrating a better training
distribution.

## Verification and resources

Workers verified unchanged controls, self-donor reconstruction, constant-input
behavior, declared moment/spectral properties, finite outputs, input integrity,
and exact replay of **all 17,516 raw/display outputs**, measurement tables,
tail selections and contact sheets. Every selected original A1 hash and all
20,099 source-file size/mtime records remained unchanged. Integrity-only
metadata checks did not turn excluded images into analysis inputs.

| Resource | Color Transfer | FDA |
| --- | ---: | ---: |
| Learned parameters / GPU seconds | 0 / 0 | 0 / 0 |
| Measured maximum process host memory | 0.418 GB | 1.706 GB |
| Conservative assignment wall time | 14.3 min | 17.9 min |

Wall time includes implementation, debugging, visual review and reporting;
it is not inference throughput. Host memory is the largest measured process
peak, not aggregate concurrent memory. Both workers finished within their
two-hour CPU assignment caps. Color Transfer preserved one failed numerical
fixture and its algebraically equivalent stable-reduction fix.

The persistent [master instructions](/home/user/jw_soflow/exploration/transfer_aug/MASTER.md),
[contract template](/home/user/jw_soflow/exploration/transfer_aug/EXPERIMENT_TEMPLATE.md)
and [ledger](/home/user/jw_soflow/exploration/transfer_aug/EXPERIMENT_LEDGER.json)
are ready for subsequent bounded dispatches. Frozen contracts and worker
reproduction commands retain the evidence for this completed round.
