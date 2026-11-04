# Round 2 appearance-domain candidate — ready

Measured all 14,059 A1 images and excluded 1,072 within-class exact-pixel duplicates, retaining 12,987 images across all 151 categories. D0 and D1 each contain 3,975 optimization images and 302 fixed diagnostic holdout images; 4,433 middle images are unused. Every category has equal D0/D1 support and at least 6 optimization images per domain.

Class-balanced, residual-normalized PC1 explains 27.04% of weighted variance. Its strongest positive loadings are luminance standard deviation, neighbor gradient, Laplacian energy, border contrast and near-black fraction; mean luminance loads negatively. Entropy sign correlation is 0.239.

All four review sheets were inspected with image tools: 128 images, 16 seeded categories, two optimization plus both holdout examples per domain/category. D1 repeatedly shows stronger light-dark contrast/edges and frequent dark backgrounds; D0 often appears paler/softer. Recurrence is weaker or heterogeneous for Dugtrio, Shellder and Venusaur. Mixed media, poses, scale, object detail and backgrounds remain confounded; these are statistical domains, not established semantic styles.

All descriptors are finite. Full extraction, PCA, membership and image-sheet regeneration were byte-identical. Synthetic constant and checkerboard checks passed. Every A1 source file was rehashed; protected-input hashes match the first-wave frozen manifest. Six cross-class duplicate pairs exist: five entirely optimize within the same domain, one entirely unused. There is zero global optimization/holdout pixel overlap; the prescribed selection rule required no changes.

Candidate: `domain_manifest.json`; exact definitions: `DEFINITIONS.md`; numerical/integrity proofs: `verification.json`, `regeneration.json`; visual evidence: `review_1.png`–`review_4.png`, `visual_review.json`; counts, PCA, measurements and exclusions have dedicated JSON/CSV files. `artifact_manifest.json` pins outputs and implementation.

Recommendation: freeze this candidate for a bounded unpaired-translation pilot, retaining diagnostic holdouts. No training, model loading, GPU use, external holdout evaluation, class-correctness assessment or usefulness/FID claim occurred. Near duplicates may remain. Resource details are in `resource_usage.json`; CPU use is well below one hour.
