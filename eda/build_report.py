"""Assemble completed dataset EDA results into local repository documentation."""
import csv
import html
import json
from pathlib import Path

import numpy as np
from common import ROOT,OUT,RES,CACHE,verify,save_json,sha


def read(name):return json.loads((OUT/name).read_text())
def csvrows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def fmt(value):return f'{value:.3f}'


def main():
    verify()
    fid=read('fid_summary.json');pca=read('pca_summary.json');nn=read('neighbor_summary.json')
    clusters=read('cluster_summary.json');meta=read('metadata_shift.json');properties=read('image_properties_summary.json')
    effects=csvrows('pca_effects.csv');cr=csvrows('clusters_k32.csv');baselines=csvrows('cluster_sampling_baselines.csv')
    assert len(csvrows('fid_repeats.csv'))==40 and len(csvrows('fid_duplicate_sensitivity.csv'))==20
    assert len(csvrows('B_nearest_neighbors.csv'))==3020 and len(csvrows('cluster_assignments.csv'))==20099
    for k in [16,32,64]:
        rr=csvrows(f'clusters_k{k}.csv');assert sum(int(r['n_A']) for r in rr)==17079;assert sum(int(r['n_B']) for r in rr)==3020
    fid_table=[]
    for scheme,label in [('uniform','Observed category mix'),('balanced','20 per category'),('unique_pixels','Unique pixels; B excludes A matches')]:
        s=fid['schemes'][scheme]
        fid_table.append(f'| {label} | {s["n"]:,} | {s["FID_U_B"]["median"]:.3f} [{s["FID_U_B"]["p25"]:.3f}, {s["FID_U_B"]["p75"]:.3f}] | {s["FID_U_V"]["median"]:.3f} [{s["FID_U_V"]["p25"]:.3f}, {s["FID_U_V"]["p75"]:.3f}] | {s["paired_difference"]["median"]:+.3f} |')
    interpretations={19:'Isolated illustrated characters, often orange or tan',2:'Isolated illustrations, often blue or pale',29:'Rounded character illustrations and cutouts',30:'Outlined character illustrations, often blue',9:'Colorful scenic and action artwork',3:'Photographed toys and plush figures, plus some rendered figures',15:'Scenic illustrations and shaded renderings',4:'Flat outlined illustrations and icons'}
    cluster_table=[]
    for r in sorted(cr,key=lambda r:abs(float(r['p_B_minus_p_A'])),reverse=True)[:8]:
        c=int(r['cluster']);cluster_table.append(f'| {c} | {100*float(r["p_A"]):.2f}% | {100*float(r["p_B"]):.2f}% | {100*float(r["p_B_minus_p_A"]):+.2f} pp | {interpretations[c]} |')
    btv=[float(r['TV_U_V']) for r in baselines if r['k']=='32' and r['scheme']=='balanced']
    pc_table='\n'.join(f'| {r["PC"]} | {100*float(r["A_variance_fraction"]):.2f}% | {float(r["B_minus_A_mean_in_A_SD"]):+.3f} | {float(r["KS_distance"]):.3f} |' for r in effects[:8])
    text=f'''# Pokémon dataset EDA results

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

**Full FID(A, B) = {fid['full']:.6f}**, using all 17,079 A images and 3,020 B images. This is a real-data distribution distance, not a generator score. It agrees with clean-fid's reference implementation to the tolerance recorded in [the numerical check](reports/analysis/fid_full.json).

The table compares equally sized sets. U and V are disjoint subsets drawn from A; B remains fixed. Each scheme has 20 predefined seeds. Values are median [25th percentile, 75th percentile]; the last column is the median paired difference FID(U,B) − FID(U,V).

| Sampling scheme | Images per set | A subset versus B | Within A | Paired difference |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(fid_table)}

The matched comparisons, rather than the unequal-size full score versus a baseline, support the conclusion that the observed A/B difference exceeds the measured within-A subsampling spread. Balancing categories does not remove it. The repeated-sampling ranges are not population confidence intervals or formal significance tests; equal counts do not remove every source of FID bias.

The unique-pixel sensitivity keeps one representative per 64 × 64 RGB group and removes B groups that match A. Its B population and sample count differ, so compare its A/B result only to its own within-A baseline. It also removes exact content sharing between U and V. Near-duplicates can remain.

Supplementary FID(A1, A2) = {fid['FID_A1_A2']:.3f}. This uses unequal counts and the existing special development selection, so it is not the primary random within-A baseline.

![FID baselines](reports/analysis/fid_baselines.png)

Data: [all regular repeats](reports/analysis/fid_repeats.csv), [duplicate sensitivity repeats](reports/analysis/fid_duplicate_sensitivity.csv), [summary and ranges](reports/analysis/fid_summary.json).

## Major directions of variation

A single centered, unwhitened PCA basis was fitted to A and used to project both sets. The first 64 components explain **{100*pca['retained_variance_64']:.2f}%** of A's feature variance. PCs 1 and 2 together explain only {100*sum(pca['explained_variance_ratio'][:2]):.2f}%, so overlap in the scatter plot cannot establish overall distribution equivalence.

| PC | A variance explained | B mean minus A mean, in A standard deviations | KS distribution distance |
| --- | ---: | ---: | ---: |
{pc_table}

PC2 has the largest absolute mean shift among the first ten components. Its low-end examples include scenic/card artwork, while high-end examples include isolated character images on bright backgrounds. This is an interpretation of examples; it is not a pure style axis. PC2 correlates with brightness within A (Spearman 0.529) and within B (0.553). Other components mix color, texture, pose, background, and Pokémon identity.

![PCA scatter](reports/analysis/pca_scatter.png)

![PCA distributions](reports/analysis/pca_distributions.png)

Inspect [PC endpoint images](reports/analysis/pca_endpoints.png), [variance explained](reports/analysis/pca_variance.png), [per-PC empirical CDFs](reports/analysis/pca_ecdfs.png), [PC effects](reports/analysis/pca_effects.csv), [within-category PC shifts](reports/analysis/pca_within_category.csv), and [property correlations](reports/analysis/pca_property_correlations.csv). PCA signs are arbitrary; all reported signs refer to the saved basis.

## Individual B samples and coverage

| B to A nearest-neighbor description | Distance |
| --- | ---: |
| Median | {nn['B_to_A']['median']:.3f} |
| 90th percentile | {nn['B_to_A']['p90']:.3f} |
| 95th percentile | {nn['B_to_A']['p95']:.3f} |
| 99th percentile | {nn['B_to_A']['p99']:.3f} |
| Maximum | {nn['B_to_A']['max']:.3f} |
| Median after excluding exact-matched B queries | {nn['B_to_A_excluding_exact_matched_queries']['median']:.3f} |
| Median with A restricted to the same category | {nn['B_to_A_same_category']['median']:.3f} |

The nearest A image has the same category as its B query in {100*nn['same_category_first_neighbor_rate']:.2f}% of cases. Feature neighborhoods capture more than Pokémon identity: color, background, and rendering can dominate a match.

**B is not globally more isolated than A.** Against the same randomly chosen 14,059-image A gallery, the median B distance is lower than the median held-out A distance in all 20 repeats; the median paired difference is {nn['gallery_control_median_difference']['median']:.3f}. That observation can coexist with a distribution shift: proportions can change while many individual B images are well represented. Exact matches also affect these distributions.

The raw A→B median is {nn['A_to_B']['median']:.3f}, but its gallery has only 3,020 images, so it must not be directly compared with B→all-A to infer directional difficulty. The existing A2→A1 median is {nn['fixed_A2_to_A1']['median']:.3f}; B→A1 is {nn['fixed_B_to_A1']['median']:.3f}. A2's duplicate-aware construction makes it a different selection from a random holdout.

Visual inspection of the tail found sprites, cards, game-interface images, unusual crops, and renderings. Some are poorly matched by category; others have recognizable artwork in A with altered framing. In particular, the Marowak tail example on page 3 shares recognizable artwork with its first A neighbor despite a large feature distance. Therefore, these are **coverage-review candidates**, not proven absent content or labels to change.

![Nearest-neighbor distances](reports/analysis/neighbor_distributions.png)

[Tail examples 1–8](reports/analysis/neighbor_tail_1.png) · [9–16](reports/analysis/neighbor_tail_2.png) · [17–24](reports/analysis/neighbor_tail_3.png) · [central-distance examples](reports/analysis/neighbor_central.png).

Each sheet shows the B query and three closest A images, with original-aspect thumbnails alongside the actual 64 × 64 input. Full filenames, five neighbors, distances, same-category neighbors, cosine sensitivity, and duplicate flags are in [the B neighbor table](reports/analysis/B_nearest_neighbors.csv). Also see [gallery-size controls](reports/analysis/neighbor_gallery_comparisons.csv), [A leave-one-out distances](reports/analysis/A_leave_one_out.csv), [A to B neighbors](reports/analysis/A_reverse_neighbors.csv), and [summary](reports/analysis/neighbor_summary.json).

## Modes overrepresented and underrepresented in B

K-means was fitted on A's first 64 unwhitened PCs, with k=32, n_init=10, and seed 20261005. B was assigned to the same centers. Cluster labels below are tentative descriptions based on the displayed examples, not exhaustive semantic annotations.

| Cluster | Share of A | Share of B | B minus A | Visual interpretation |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(cluster_table)}

Cluster-composition total variation is **{clusters['32']['total_variation']:.3f}**; after equal category weighting it is **{clusters['32']['category_balanced_total_variation']:.3f}**. For comparison, the median TV between two category-balanced A subsets of 3,020 is {np.median(btv):.3f}. These are descriptive controls on the fixed clusters, not a population test.

The k=16 and k=64 sensitivity runs give TV {clusters['16']['total_variation']:.3f} and {clusters['64']['total_variation']:.3f}, respectively. B's median distance to the k=32 center is {clusters['32']['B_distance_quantiles']['median']:.3f}, versus {clusters['32']['A_distance_quantiles']['median']:.3f} for A. Assignment to A's centers can hide a B-only mode, and 64 PCs omit 41.34% of A's feature variance; use the full-feature neighbor review alongside these clusters.

![Cluster composition](reports/analysis/cluster_composition.png)

Representative sheets show three central and two distant images from each available split, avoiding exact RGB duplicates within each row:

{chr(10).join(f'- [Cluster {int(r["cluster"])}](reports/analysis/cluster_{rank+1:02d}_k32_{int(r["cluster"])}.png)' for rank,r in enumerate(sorted(cr,key=lambda r:abs(float(r['p_B_minus_p_A'])),reverse=True)[:8]))}

Data: [k=32 composition](reports/analysis/clusters_k32.csv), [all assignments](reports/analysis/cluster_assignments.csv), [sampling baselines](reports/analysis/cluster_sampling_baselines.csv), and [k sensitivity summary](reports/analysis/cluster_summary.json).

## Metadata and image properties explain part of the shift

| Property | A | B |
| --- | ---: | ---: |
| Median fraction of near-white pixels | {100*properties['white_fraction']['A']['median']:.2f}% | {100*properties['white_fraction']['B']['median']:.2f}% |
| Median brightness, scale 0 to 1 | {properties['brightness']['A']['median']:.3f} | {properties['brightness']['B']['median']:.3f} |
| PNG files | {meta['A']['formats']['PNG']:,} ({100*meta['A']['formats']['PNG']/17079:.2f}%) | {meta['B']['formats']['PNG']:,} ({100*meta['B']['formats']['PNG']/3020:.2f}%) |
| Median native width | {meta['A']['median_native_width']:.0f} px | {meta['B']['median_native_width']:.0f} px |
| Median native height | {meta['A']['median_native_height']:.0f} px | {meta['B']['median_native_height']:.0f} px |

B has a higher median near-white fraction within **all {meta['categories_with_higher_B_median_white_fraction']} categories**. This is not explained solely by different Pokémon frequencies. Near-white means every RGB channel exceeds 0.95 after the official 64 × 64 export; this is a pixel statistic, not a foreground/background segmentation. File format and naming patterns are observable metadata, not verified source provenance.

![Image properties](reports/analysis/image_properties.png)

See [the exact split-order audit](reports/analysis/split_order_audit.csv), [metadata summary](reports/analysis/metadata_shift.json), [white fraction per category](reports/analysis/white_fraction_by_category.csv), and [all image properties](reports/analysis/image_properties.csv).

## Reproducibility and limits

The frozen clean-fid 0.1.35 Inception extractor produces 2,048-dimensional float32 features. Inputs use the supplied dataset's white-composited RGB conversion, bilinear 64 × 64 resize, normalization and PNG export round-trip, followed by clean-fid's float bicubic 299 × 299 resize. Thirty samples, including all six non-RGB source images, matched the supplied export pixel-for-pixel and the clean-fid loader tensor-for-tensor. Every image file was checked against the preparation hash during extraction.

FID uses full unnormalized features and float64 means, covariances, and matrix operations; the symmetric eigendecomposition result was checked against clean-fid on the full A/B sets, and B/B self-FID was checked near zero. Neighbor search exhaustively scans the gallery in chunks, selects 64 float32 candidates, and reranks those candidates using float64 distances. The first five Euclidean distances for 32 randomly selected B queries matched independent full-gallery SciPy calculations to 1e-8 tolerance. Gallery-size comparisons check that the stored candidate lists contain eligible neighbors. Self-matches are excluded from A leave-one-out distances.

Exact-match flags use pixel hashes rather than feature-distance thresholds. Of the 343 exact-matched B queries, 340 have zero nearest feature distance and three have small nonzero distances, at most 0.002, despite identical input pixels. These small residuals are consistent with float32 batch/JIT numerical variation; no distances or features were artificially forced to zero. See [validation details](reports/analysis/validation.json).

Feature extraction took {json.loads((CACHE/'metadata.json').read_text())['seconds']:.1f} seconds; features and input pixels are cached under `eda/features/`. GPU 0 was used under the shared device lock. Numerical analyses use bounded CPU threads. GPU jobs have finished. Original starter files and all split manifests passed their hash checks.

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
'''
    (ROOT/'eda/REPORT.md').write_text(text)
    # Standalone local gallery with no external assets or services.
    images=sorted(OUT.glob('*.png'))
    cards='\n'.join(f'<figure><a href="{html.escape(p.name)}"><img loading="lazy" src="{html.escape(p.name)}" alt="{html.escape(p.stem)}"></a><figcaption>{html.escape(p.stem.replace("_"," "))}</figcaption></figure>' for p in images)
    gallery='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Pokémon dataset EDA gallery</title><style>body{font:16px system-ui;margin:32px auto;padding:0 24px;max-width:1400px;color:#172033;background:#f5f7fa}figure{margin:24px 0;padding:16px;background:white;border:1px solid #dde3eb;border-radius:8px}img{max-width:100%;height:auto}figcaption{margin-top:10px}a{color:#2458aa}</style><h1>Pokémon dataset EDA gallery</h1><p>Real-image dataset comparisons. Click any figure for full resolution. <a href="../../REPORT.md">Read the report</a>.</p>'+cards+'</html>'
    (OUT/'gallery.html').write_text(gallery)
    save_json(OUT/'completion.json',{'complete':True,'scope':'four primary dataset EDA analyses plus documented controls','source_sha256':{p.name:sha(p) for p in (ROOT/'eda').glob('*.py')},'output_files':len(list(OUT.iterdir())),'protected_files_unchanged':True})
    print(ROOT/'eda/REPORT.md')


if __name__=='__main__':main()
