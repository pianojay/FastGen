# Round 2 appearance-domain preparation — revision 1

Authorized by user: derive and visually inspect A1-only appearance groups for
the next CycleGAN/CUT pair. Worker writes only this directory; no models,
training, GPU, pretrained features, environment changes or A2/B analysis.
Use the existing CPU launcher and pinned official-export pixel cache, with A1
indexing and integrity checks as in the first-wave cohort preparation.

Question: can a reproducible, class-controlled contrast in low-level appearance
provide source/target domains for a bounded unpaired-translation pilot? These
are statistical domains, not established semantic styles.

Procedure (seed 20261008):
1. Measure every A1 image using 64px RGB pixels: luminance mean/std, saturation
   mean/std, near-white fraction (all channels>.95), near-black fraction
   (all channels<.05), mean absolute x/y neighbor luminance gradient averaged,
   mean squared luminance Laplacian (interior 5-point stencil), normalized
   32-bin luminance histogram entropy, non-DC Fourier energy fraction at
   max(|fx|,|fy|)>=16, and outer-4px-border versus interior luminance mean
   absolute difference. State exact conventions in implementation/results.
2. Deduplicate within each class using official-export pixel hashes; retain the
   lexicographically first path per identical pixel group for domain selection.
   This does not alter official manifests. Report duplicate exclusions.
3. Subtract each class mean from the descriptor vector; scale each descriptor
   by pooled class-balanced residual RMS, dropping zero-variance descriptors.
   Fit unwhitened PCA with equal total weight per class (SVD of residual rows
   multiplied by inverse sqrt(class size)). Use only PC1. Orient its sign to
   correlate positively with normalized luminance entropy (fall back to edge
   energy if correlation is numerically zero). Record loadings/variance.
4. Within each class, sort by PC1 score then pixel hash/path and take the
   bottom floor(n/3) as D0 and top floor(n/3) as D1. Middle samples are unused
   in training domains. This preserves equal per-class D0/D1 support; it
   does not prove equivalent styles across classes. No classes are discarded.
5. Within each class/domain, reserve two pixel-distinct examples by independent
   seeded hash order as a fixed A1 diagnostic holdout; all remaining domain
   members form the optimization pools. Require >=4 optimization images per
   class/domain, or report blocked. These are auxiliary subpartitions of A1,
   not modifications to official A1/A2/B. Holdout pictures never train models.
6. Visually inspect a fixed 16-category subset using seed20261008; show two
   optimization and two diagnostic examples per domain for each selected
   category. Inspect all sheets with image tools, record uncertainty and
   whether the within-class contrast recurs across categories. Do not rename
   groups as semantic styles unsupported by this review.
7. Verify disjoint pixel hashes per class across optimize/holdout/domain pools,
   membership in A1, class counts, exact regeneration of selection, source
   integrity, and finite descriptors. Report cross-class duplicate hashes
   separately; do not hide them or permit train/holdout pixel overlap globally.
   If cross-class collisions would leak across the holdout boundary, report
   details to master before finalizing rather than silently changing the rule.

Deliver domain_manifest.json (per-path class/domain/role/hash/score), scores.csv,
descriptor/PCA definitions, global and per-class counts, review sheets/record,
verification JSON, exact commands and SHA manifest. SUMMARY.md <=500 words and
result.json carry findings, limitations, proof paths and a clear ready/blocked
recommendation. Master will freeze accepted manifests into the learned-model
contracts. Spend at most one CPU-hour; no additional agents or method search.
