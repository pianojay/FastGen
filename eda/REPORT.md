# Pokémon dataset EDA results

Completed on October 5, 2026, using the FastGen-SoFlow-aligned environment and physical GPU 0. This report compares real-image datasets only. No project model was created, trained, tuned, sampled, or evaluated, and all fixed partitions remain unchanged.

## Main findings

A and B differ in their visual composition. The difference persists after matching sample counts and balancing Pokémon categories. B emphasizes isolated character illustrations and white pixels; A contains more scenic artwork and photographed objects. These descriptions come from inspected examples and image measurements, not an automatically inferred style taxonomy.

The most important split audit finding is exact: **in all 151 categories, B consists of the first 20 filenames in lexicographic order**. This is a verified manifest property, not a reconstruction of the original split-generation code. Filename ordering is a plausible mechanism for the composition shift; these data alone do not establish causal attribution.

| Population | Images | Role |
| --- | ---: | --- |
| A | 17,079 | Official training pool |
| A1 | 14,059 | Existing optimization subset of A |
| A2 | 3,020 | Existing development subset of A |
| B | 3,020 | Official evaluation reference |

There are 343 B images (11.36%) with exact image matches in A. The full populations are retained; duplicate-clean comparisons are explicitly supplementary.

## FID and size controlled baselines

**Full FID(A, B) = 15.710899**, using all 17,079 A images and 3,020 B images. This is a real-data distribution distance, not a generator score. It agrees with clean-fid's reference implementation to the tolerance recorded in [the numerical check](reports/analysis/fid_full.json).

The table compares equally sized sets. U and V are disjoint subsets drawn from A; B remains fixed. Each scheme has 20 predefined seeds. Values are median [25th percentile, 75th percentile]; the last column is the median paired difference FID(U,B) − FID(U,V).

| Sampling scheme | Images per set | A subset versus B | Within A | Paired difference |
| --- | ---: | ---: | ---: | ---: |
| Observed category mix | 3,020 | 21.139 [20.795, 21.307] | 14.924 [14.824, 14.949] | +6.259 |
| 20 per category | 3,020 | 20.828 [20.619, 21.024] | 14.800 [14.749, 14.874] | +6.001 |
| Unique pixels; B excludes A matches | 2,677 | 24.665 [24.357, 24.845] | 16.813 [16.748, 16.848] | +7.750 |

The matched comparisons, rather than the unequal-size full score versus a baseline, support the conclusion that the observed A/B difference exceeds the measured within-A subsampling spread. Balancing categories does not remove it. The repeated-sampling ranges are not population confidence intervals or formal significance tests; equal counts do not remove every source of FID bias.

The unique-pixel sensitivity keeps one representative per 64 × 64 RGB group and removes B groups that match A. Its B population and sample count differ, so compare its A/B result only to its own within-A baseline. It also removes exact content sharing between U and V. Near-duplicates can remain.

Supplementary FID(A1, A2) = 9.868. This uses unequal counts and the existing special development selection, so it is not the primary random within-A baseline.

![FID baselines](reports/analysis/fid_baselines.png)

Data: [all regular repeats](reports/analysis/fid_repeats.csv), [duplicate sensitivity repeats](reports/analysis/fid_duplicate_sensitivity.csv), [summary and ranges](reports/analysis/fid_summary.json).

## Major directions of variation

A single centered, unwhitened PCA basis was fitted to A and used to project both sets. The first 64 components explain **58.66%** of A's feature variance. PCs 1 and 2 together explain only 9.26%, so overlap in the scatter plot cannot establish overall distribution equivalence.

| PC | A variance explained | B mean minus A mean, in A standard deviations | KS distribution distance |
| --- | ---: | ---: | ---: |
| 1 | 5.10% | +0.130 | 0.081 |
| 2 | 4.16% | +0.417 | 0.209 |
| 3 | 3.59% | -0.200 | 0.101 |
| 4 | 2.87% | +0.264 | 0.130 |
| 5 | 2.52% | -0.240 | 0.105 |
| 6 | 2.21% | +0.173 | 0.089 |
| 7 | 1.91% | -0.230 | 0.114 |
| 8 | 1.81% | +0.190 | 0.092 |

PC2 has the largest absolute mean shift among the first ten components. Its low-end examples include scenic/card artwork, while high-end examples include isolated character images on bright backgrounds. This is an interpretation of examples; it is not a pure style axis. PC2 correlates with brightness within A (Spearman 0.529) and within B (0.553). Other components mix color, texture, pose, background, and Pokémon identity.

![PCA scatter](reports/analysis/pca_scatter.png)

![PCA distributions](reports/analysis/pca_distributions.png)

Inspect [PC endpoint images](reports/analysis/pca_endpoints.png), [variance explained](reports/analysis/pca_variance.png), [per-PC empirical CDFs](reports/analysis/pca_ecdfs.png), [PC effects](reports/analysis/pca_effects.csv), [within-category PC shifts](reports/analysis/pca_within_category.csv), and [property correlations](reports/analysis/pca_property_correlations.csv). PCA signs are arbitrary; all reported signs refer to the saved basis.

## Individual B samples and coverage

| B to A nearest-neighbor description | Distance |
| --- | ---: |
| Median | 8.820 |
| 90th percentile | 12.869 |
| 95th percentile | 13.801 |
| 99th percentile | 15.230 |
| Maximum | 17.234 |
| Median after excluding exact-matched B queries | 9.604 |
| Median with A restricted to the same category | 8.883 |

The nearest A image has the same category as its B query in 82.42% of cases. Feature neighborhoods capture more than Pokémon identity: color, background, and rendering can dominate a match.

**B is not globally more isolated than A.** Against the same randomly chosen 14,059-image A gallery, the median B distance is lower than the median held-out A distance in all 20 repeats; the median paired difference is -1.177. That observation can coexist with a distribution shift: proportions can change while many individual B images are well represented. Exact matches also affect these distributions.

The raw A→B median is 12.917, but its gallery has only 3,020 images, so it must not be directly compared with B→all-A to infer directional difficulty. The existing A2→A1 median is 11.623; B→A1 is 9.766. A2's duplicate-aware construction makes it a different selection from a random holdout.

Visual inspection of the tail found sprites, cards, game-interface images, unusual crops, and renderings. Some are poorly matched by category; others have recognizable artwork in A with altered framing. In particular, the Marowak tail example on page 3 shares recognizable artwork with its first A neighbor despite a large feature distance. Therefore, these are **coverage-review candidates**, not proven absent content or labels to change.

![Nearest-neighbor distances](reports/analysis/neighbor_distributions.png)

[Tail examples 1–8](reports/analysis/neighbor_tail_1.png) · [9–16](reports/analysis/neighbor_tail_2.png) · [17–24](reports/analysis/neighbor_tail_3.png) · [central-distance examples](reports/analysis/neighbor_central.png).

Each sheet shows the B query and three closest A images, with original-aspect thumbnails alongside the actual 64 × 64 input. Full filenames, five neighbors, distances, same-category neighbors, cosine sensitivity, and duplicate flags are in [the B neighbor table](reports/analysis/B_nearest_neighbors.csv). Also see [gallery-size controls](reports/analysis/neighbor_gallery_comparisons.csv), [A leave-one-out distances](reports/analysis/A_leave_one_out.csv), [A to B neighbors](reports/analysis/A_reverse_neighbors.csv), and [summary](reports/analysis/neighbor_summary.json).

## Modes overrepresented and underrepresented in B

K-means was fitted on A's first 64 unwhitened PCs, with k=32, n_init=10, and seed 20261005. B was assigned to the same centers. Cluster labels below are tentative descriptions based on the displayed examples, not exhaustive semantic annotations.

| Cluster | Share of A | Share of B | B minus A | Visual interpretation |
| --- | ---: | ---: | ---: | --- |
| 19 | 4.16% | 11.39% | +7.23 pp | Isolated illustrated characters, often orange or tan |
| 2 | 3.52% | 8.44% | +4.92 pp | Isolated illustrations, often blue or pale |
| 29 | 3.83% | 8.54% | +4.71 pp | Rounded character illustrations and cutouts |
| 30 | 3.33% | 6.99% | +3.66 pp | Outlined character illustrations, often blue |
| 9 | 3.31% | 1.03% | -2.28 pp | Colorful scenic and action artwork |
| 3 | 4.75% | 2.68% | -2.07 pp | Photographed toys and plush figures, plus some rendered figures |
| 15 | 3.17% | 1.13% | -2.04 pp | Scenic illustrations and shaded renderings |
| 4 | 3.12% | 5.07% | +1.95 pp | Flat outlined illustrations and icons |

Cluster-composition total variation is **0.261**; after equal category weighting it is **0.260**. For comparison, the median TV between two category-balanced A subsets of 3,020 is 0.055. These are descriptive controls on the fixed clusters, not a population test.

The k=16 and k=64 sensitivity runs give TV 0.235 and 0.270, respectively. B's median distance to the k=32 center is 8.229, versus 8.379 for A. Assignment to A's centers can hide a B-only mode, and 64 PCs omit 41.34% of A's feature variance; use the full-feature neighbor review alongside these clusters.

![Cluster composition](reports/analysis/cluster_composition.png)

Representative sheets show three central and two distant images from each available split, avoiding exact RGB duplicates within each row:

- [Cluster 19](reports/analysis/cluster_01_k32_19.png)
- [Cluster 2](reports/analysis/cluster_02_k32_2.png)
- [Cluster 29](reports/analysis/cluster_03_k32_29.png)
- [Cluster 30](reports/analysis/cluster_04_k32_30.png)
- [Cluster 9](reports/analysis/cluster_05_k32_9.png)
- [Cluster 3](reports/analysis/cluster_06_k32_3.png)
- [Cluster 15](reports/analysis/cluster_07_k32_15.png)
- [Cluster 4](reports/analysis/cluster_08_k32_4.png)

Data: [k=32 composition](reports/analysis/clusters_k32.csv), [all assignments](reports/analysis/cluster_assignments.csv), [sampling baselines](reports/analysis/cluster_sampling_baselines.csv), and [k sensitivity summary](reports/analysis/cluster_summary.json).

## Metadata and image properties explain part of the shift

| Property | A | B |
| --- | ---: | ---: |
| Median fraction of near-white pixels | 0.29% | 37.52% |
| Median brightness, scale 0 to 1 | 0.615 | 0.762 |
| PNG files | 147 (0.86%) | 1,336 (44.24%) |
| Median native width | 436 px | 200 px |
| Median native height | 435 px | 200 px |

B has a higher median near-white fraction within **all 151 categories**. This is not explained solely by different Pokémon frequencies. Near-white means every RGB channel exceeds 0.95 after the official 64 × 64 export; this is a pixel statistic, not a foreground/background segmentation. File format and naming patterns are observable metadata, not verified source provenance.

![Image properties](reports/analysis/image_properties.png)

See [the exact split-order audit](reports/analysis/split_order_audit.csv), [metadata summary](reports/analysis/metadata_shift.json), [white fraction per category](reports/analysis/white_fraction_by_category.csv), and [all image properties](reports/analysis/image_properties.csv).

## Reproducibility and limits

The frozen clean-fid 0.1.35 Inception extractor produces 2,048-dimensional float32 features. Inputs use the supplied dataset's white-composited RGB conversion, bilinear 64 × 64 resize, normalization and PNG export round-trip, followed by clean-fid's float bicubic 299 × 299 resize. Thirty samples, including all six non-RGB source images, matched the supplied export pixel-for-pixel and the clean-fid loader tensor-for-tensor. Every image file was checked against the preparation hash during extraction.

FID uses full unnormalized features and float64 means, covariances, and matrix operations; the symmetric eigendecomposition result was checked against clean-fid on the full A/B sets, and B/B self-FID was checked near zero. Neighbor search exhaustively scans the gallery in chunks, selects 64 float32 candidates, and reranks those candidates using float64 distances. The first five Euclidean distances for 32 randomly selected B queries matched independent full-gallery SciPy calculations to 1e-8 tolerance. Gallery-size comparisons check that the stored candidate lists contain eligible neighbors. Self-matches are excluded from A leave-one-out distances.

Exact-match flags use pixel hashes rather than feature-distance thresholds. Of the 343 exact-matched B queries, 340 have zero nearest feature distance and three have small nonzero distances, at most 0.002, despite identical input pixels. These small residuals are consistent with float32 batch/JIT numerical variation; no distances or features were artificially forced to zero. See [validation details](reports/analysis/validation.json).

Feature extraction took 113.5 seconds; features and input pixels are cached under `eda/features/`. GPU 0 was used under the shared device lock. Numerical analyses use bounded CPU threads. GPU jobs have finished. Original starter files and all split manifests passed their hash checks.

This run completed the four primary analyses plus category controls, exact-duplicate sensitivity, reverse neighbors, and basic image/filename metadata. It did not perform a systematic near-duplicate-family audit, native-resolution feature extraction, MMD, or pooled A/B PCA/clustering. Inception is one measurement representation and can overweight texture, color, framing, or other cues. The results describe these fixed datasets; they do not establish model performance or authorize changes to other experiments.

Reproduction commands, after the completed preparation and Conda setup:

```bash
bash eda/run_python.sh eda/extract_features.py
bash eda/run_python.sh eda/analyze_features.py
bash eda/run_python.sh eda/analyze_neighbors.py
bash eda/run_python.sh eda/enrich_results.py
bash eda/run_python.sh eda/build_report.py
```

Completed feature and analysis caches are reused. If inputs or analysis settings change, use a new output directory or explicitly invalidate the relevant caches; do not silently reuse results across protocols. See [the preparation and methods](README.md), [feature provenance](features/metadata.json), and [qualified environment](resources/qualified_environment.json).
