"""Compile the completed EDA into concise, linked Notion publication payloads.

This only reads existing EDA artifacts. It does not run dataset analysis or models.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports'


def table(rows):
    return '<table header-row="true">\n' + '\n'.join(
        '<tr>' + ''.join('<td>' + str(c) + '</td>' for c in row) + '</tr>'
        for row in rows
    ) + '\n</table>'


def toggle(title, body):
    return '<details>\n<summary>' + title + '</summary>\n' + '\n'.join(
        '\t' + line if line else '' for line in body.splitlines()
    ) + '\n</details>'


def fig(key, caption):
    return '![%s](__FIG_%s__)' % (caption, key)


def ref(n, label=None):
    return '[%s](__EDA_%d__)' % (label or 'EDA-%d' % n, n)


def base(extra=''):
    return 'Shared setup: fixed A/A1/A2/B partitions and official 64 × 64 image preparation' + extra + '; see ' + ref(1, 'EDA-1 — detailed base setup') + '.\n\n'


reports = []


def add(n, title, content, sources):
    reports.append({'number': n, 'properties': {'Title': f'EDA-{n} — {title}', 'Challenge': 'FastGen 2026', 'Status': 'Completed'}, 'content': content.strip(), 'local_sources': sources})


add(1, 'FID and sample-size controls', '''**A and B differ beyond the measured within-A sampling spread. The difference survives class balancing.** This is evidence about the reference distribution, not a generator score or an achievable FID floor.

''' + table([
    ['Comparison', 'Images per set', 'A subset vs B', 'Within A'],
    ['Observed category mix', '3,020', '**21.14**', '**14.92**'],
    ['20 images per category', '3,020', '**20.83**', '**14.80**'],
]) + '''

Values are medians over 20 seeds. Each seed draws disjoint U and V from A and compares FID(U,B) with FID(U,V); B stays fixed. Full, unequal-size **FID(A,B) = 15.71** is supplementary. Repeated subsampling variability is descriptive, not a confidence interval for model performance.

''' + fig('fid_baselines', 'Real-image FID: A/B versus within-A controls. The unique-pixel panel changes population and sample count; interpret it against its own baseline.') + '''

**Implication for SiDA / DMD2.** Both methods combine teacher-derived signals with a real-image adversarial signal. Better matching the training distribution can coexist with a mismatch to B; increasing adversarial weight is therefore not an established cure for the reference shift. This is our inference from the EDA, not a result of a model ablation. Method context: [SiDA paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ce953d71deeb33d9ffa2c879b518d273-Abstract-Conference.html), [DMD2 paper](https://papers.nips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html).

**How to use this result.** Keep the fixed A1/A2 development protocol, sample count, category schedule and export pipeline constant when comparing future recipes. Do not subtract 15.71 or the roughly six-point controlled gap from a generator FID; FID has no such additive decomposition. These baselines do not show that a model scoring near 20 has exhausted its improvement potential. Finite-sample FID bias is also model-dependent: [Chong & Forsyth](https://arxiv.org/abs/1911.07023).

The strongest next direction is a small, controlled study of augmentation and coverage within A1/A2, with image quality and diversity checked alongside FID. EDA does not select a learning rate, loss weight, number of critic updates or augmentation probability.

''' + toggle('Detailed base setup — shared by the EDA series', '''**Scope and dates.** Dataset EDA completed on <mention-date start="2026-10-05"/>; interpretation compiled on <mention-date start="2026-10-07"/>. No project generator or teacher was created, trained, tuned, sampled or evaluated. PCA and clustering are descriptive fits only.

**Dataset.** [Kaggle Pokémon Generation One — 22k](https://www.kaggle.com/datasets/bhawks/pokemon-generation-one-22k): 20,099 readable images across 151 categories. All listed images were present and decoded successfully.

''' + table([
    ['Partition', 'Images', 'Role'],
    ['A', '17,079', 'Official training pool'],
    ['A1', '14,059', 'Existing optimization subset'],
    ['A2', '3,020', 'Existing development subset; 20/category'],
    ['B', '3,020', 'Official evaluation reference; 20/category'],
]) + '''

A1 and A2 are the existing immutable partition of A, copied from FastGen-SoFlow and verified byte-identical to the saved AFM and SiD-DiT partitions. A2 uses the existing seed-42, duplicate-aware selection; it is not an arbitrary fresh random holdout. No split was regenerated. These checks do not establish that every teammate's separate run used that same partition.

**Image path.** Supplied white compositing for transparency → RGB → supplied bilinear 64 × 64 resize → tensor normalization and official uint8 PNG export round-trip → clean-fid clean-mode float bicubic 299 × 299 resize. EDA therefore describes what the official 64 × 64 pipeline retains, not all native-resolution detail. Thirty verification images, including all six non-RGB sources, matched the supplied export pixel-for-pixel and the clean-fid loader tensor-for-tensor.

**Representation.** Frozen standard clean-fid Inception-v3 features, 2,048 dimensions; no learned project checkpoint. FID uses full unnormalized features and float64 statistics. The numerical implementation agreed with clean-fid on full A/B to about 1.5e-12; B/B self-FID was zero. PCA is fitted on A only; cluster centers are fitted on A only; B uses those fixed transforms. Full A includes development A2 for dataset description, not model training.

**Sampling controls.** Twenty predefined seeds, 20261005–20261024; each regular set contains 3,020 images. Uniform sampling preserves A's observed category mixture in expectation; the balanced control takes 20 per category in each subset. Exact duplicates are retained in the main analysis and addressed separately in ''' + ref(7) + '''.

**Environment and integrity.** Independent Conda copy aligned with FastGen-SoFlow: Python 3.10.21, torch 2.7.1+cu126, torchvision 0.22.1+cu126, clean-fid 0.1.35, NumPy 1.26.4, SciPy 1.15.3; scikit-learn 1.7.2 added for EDA. Feature extraction used GPU 0. Original files and split manifests passed hash checks; sibling projects were unchanged.

**Boundary for later tuning.** B has been inspected for this descriptive audit. Do not use B images, B-derived cluster weights/background targets, or repeated B scores to choose training settings. Select on A2 and disclose this EDA exposure when describing B as held out. The supplied README reserves B for evaluation and permits hyperparameter selection only within A.

**Current team context.** The SiDA log describes a development holdout, while the DMD2 log describes train-split FID selection. Before comparing methods, reconcile the reference split/hash, sample count and generation protocol. The EDA does not independently verify the reported frontier ranking.
<mention-page url="https://app.notion.com/p/3ecf35446091816e8e82e523cca6247c"/>
<mention-page url="https://app.notion.com/p/3eef3544609180aa98ded4e4182036b0"/>

**Local provenance.** `/home/user/jw_soflow/exploration/eda/REPORT.md`, `/home/user/jw_soflow/exploration/eda/features/metadata.json`, `/home/user/jw_soflow/exploration/eda/resources/provenance.json`, and `/home/user/jw_soflow/exploration/eda/reports/analysis/validation.json`. These are local artifact identifiers, not web links.''') + '''

''' + toggle('Other methodology reports', '\n'.join('- ' + ref(i, s) for i, s in [
    (2, 'EDA-2 — PCA: directions of visual variation'),
    (3, 'EDA-3 — Nearest neighbors: individual coverage'),
    (4, 'EDA-4 — Clusters: mode proportions'),
    (5, 'EDA-5 — Split ordering and metadata'),
    (6, 'EDA-6 — Image properties and augmentation hypotheses'),
    (7, 'EDA-7 — Duplicate overlap and sensitivity'),
])), ['analysis/fid_summary.json', 'analysis/fid_repeats.csv', 'analysis/validation.json'])

add(2, 'PCA directions of visual variation', base(', using frozen Inception features') + '''**B shifts along several major feature directions; brightness and illustration style are prominent, but not the whole explanation.**

**Method.** Principal component analysis (PCA) finds directions of greatest variation in A's image features. We project both datasets onto the same centered, unwhitened basis and compare their distributions. This shows *where* they differ; components are not predefined semantic labels.

- PC2 has the largest mean shift among the first ten PCs: **+0.417 A standard deviations** for B.
- PC2 correlates with brightness within A (**0.529**) and B (**0.553**, Spearman). Its low-end examples include scenic/card artwork; high-end examples include isolated characters on bright backgrounds.
- The first two PCs explain only **9.26%** of A's variance. Overlapping points in this scatter do not imply equivalent distributions.

''' + fig('pca_scatter', 'A and B projected onto A-fitted PCs. Equal-size display samples; the plotted axes capture only a small part of total variation.') + '''

**Generative-model implication.** A single global FID can conceal whether a change affects style, background, color or subject appearance. These directions provide a compact diagnostic vocabulary for a later, separately authorized comparison of generated samples with A2. They are not evidence that the current SiDA or DMD2 output has any particular defect.

**Guidance.** Treat brightness/color augmentation as a distribution-changing intervention that needs inspection at 64 × 64. Do not optimize a model toward B's PC2 mean or declare a PC to be a pure “quality” axis. Inspect whole characters and within-class diversity as well as aggregate feature shifts.

''' + toggle('Per-PC distributions and visual anchors', fig('pca_distributions', 'Distributions of the first ten A-fitted components. Signs are arbitrary and refer to the saved basis.') + '\n\n' + fig('pca_endpoints', 'Low and high extremes of the first four PCs; original-aspect thumbnail beside actual 64 × 64 input. Endpoint examples illustrate an axis, not its exclusive meaning.') + '''

The first 64 PCs explain 58.66% of A's variance. Other components mix identity, pose, texture and color. PCA captures linear variation in this one representation; it is neither a complete semantic map nor a causal explanation.'''), ['analysis/pca_effects.csv', 'analysis/pca_property_correlations.csv'])

add(3, 'Nearest neighbors and individual coverage', base(', using full 2,048-dimensional Inception features') + '''**B is not globally more isolated from A. Distribution shift here does not primarily mean every B image lacks a training analogue.**

**Method.** For each B image, find the closest A image by Euclidean feature distance. Large distances identify candidates for visual coverage review. Compare held-out A and B against the *same* A gallery so gallery size does not determine the conclusion.

''' + table([
    ['Measure', 'Result'],
    ['B → all A: median / 95th percentile', '8.82 / 13.80'],
    ['B queries with exact matches in A', '343 / 3,020 (11.36%)'],
    ['Median excluding those exact-matched B queries', '9.60'],
    ['Nearest A image has the same Pokémon category', '82.42%'],
    ['Same-gallery comparison, 20 repeats', 'B median lower than held-out A in all 20'],
]) + '''

''' + fig('neighbor_distributions', 'Nearest-neighbor distance distributions. Use the common-gallery controls for comparisons; raw reverse A→B has a smaller gallery.') + '''

**What the tail shows.** Unusual sprites, cards and crops appear among the distant examples. Some contain recognizable artwork already in A under another framing. In the Marowak example below, a large distance coexists with visibly shared artwork. Feature distance is not proof of novel content or a missing semantic mode.

''' + toggle('Selected high-distance examples', fig('neighbor_tail_1', 'Tail ranks 1–8. Each B query is followed by its three closest A images; each cell includes the native-aspect and 64 × 64 views.') + '\n\n' + fig('neighbor_tail_3', 'Tail ranks 17–24. The first Marowak row illustrates a framing-sensitive large distance despite recognizable shared artwork.')) + '''

**Generative-model implication.** Improving sharpness alone and discovering missing modes are different problems. The current evidence makes mode proportions and variation within familiar modes important hypotheses; it does not exclude difficult rare cases. For future augmentation design, modest framing robustness is worth testing inside A1/A2, but aggressive crops can remove the Pokémon or create an unintended target distribution.

**Limit.** This is real-to-real coverage, not generated-sample precision/recall, diversity or memorization measurement. Exact duplicates affect the result. A-fitted feature distance may emphasize color or composition over class identity; review the images before interpreting any tail sample.''', ['analysis/neighbor_summary.json', 'analysis/neighbor_gallery_comparisons.csv', 'analysis/B_nearest_neighbors.csv'])

add(4, 'Cluster composition and mode proportions', base(', using the first 64 A-fitted PCs') + '''**The strongest discrepancy is how often visual modes occur. Balancing Pokémon categories barely changes it.**

**Method.** K-means groups A images into 32 feature-space clusters; B is assigned to the same centers. Comparing each cluster's share, p_A(k) versus p_B(k), reveals over- and underrepresented visual modes. Names below are tentative descriptions from representative images, not ground-truth style labels.

''' + table([
    ['Illustrative cluster', 'A share', 'B share', 'B − A'],
    ['19: isolated orange/tan illustrations', '4.16%', '11.39%', '+7.23 pp'],
    ['2: isolated blue/pale illustrations', '3.52%', '8.44%', '+4.92 pp'],
    ['9: scenic/action artwork', '3.31%', '1.03%', '−2.28 pp'],
    ['3: toys/plush photographs and some renders', '4.75%', '2.68%', '−2.07 pp'],
]) + '''

''' + fig('cluster_composition', 'Shares of A and B in the same 32 A-fitted feature clusters.') + '''

Cluster total variation, half the sum of absolute share differences, is **0.261**; with equal category weighting it is **0.260**. The median between two balanced A subsets is **0.055**. TV summarizes a shift in cluster proportions; 0.261 does not mean 26.1% of B contains missing content. Sensitivity runs at 16 and 64 clusters retain substantial shifts (0.235 and 0.270).

**Generative-model implication.** Class-balanced generation can still miss the *within-class style mixture*. A model may depict all 151 categories and still overproduce one pose, rendering type or background. Stronger adversarial matching to A1 does not automatically correct the mismatch to B.

**Guidance.** In later authorized model work, inspect generated mode occupancy and within-class variety against A2 alongside FID. A lower FID accompanied by fewer visual modes should prompt review. Do not set training weights to p_B(k), remove scenic images to mimic B, or treat these unsupervised groups as class labels.

''' + toggle('Representative images and limits', fig('cluster_01_k32_19', 'Cluster 19, enriched in B. Three central and two more distant distinct-image representatives per split row.') + '\n\n' + fig('cluster_05_k32_9', 'Cluster 9, depleted in B. Examples emphasize scenic/action artwork.') + '\n\n' + fig('cluster_06_k32_3', 'Cluster 3, depleted in B. Photographs of physical objects are visible alongside some rendered images.') + '''

The retained PCs explain 58.66% of A variance. Assigning B to A's centers can hide a B-only mode. Cluster descriptors are visual interpretations, not exhaustive annotations or a formal mode-collapse test.'''), ['analysis/clusters_k32.csv', 'analysis/cluster_summary.json', 'analysis/cluster_sampling_baselines.csv'])

add(5, 'Split ordering and metadata audit', base() + '''**B has an exact filename-order signature. It should not be interpreted as an ordinary random holdout from A's mixture.**

**Method.** Audit the fixed manifests, category counts, lexicographic filename ranks, file formats and native image sizes. This detects observable selection structure without fitting a model or inferring undocumented image provenance.

- In **all 151 categories**, B equals exactly the **first 20 lexicographically sorted filenames** in the combined A∪B list for that category.
- A contains **46–183 images per category**; B contains **20 each**. A1 contains **26–163 per category**.
- PNG share is **0.86% in A versus 44.24% in B**. Median native dimensions are **436 × 435 in A versus 200 × 200 in B** (separate width/height medians).

''' + table([
    ['Observable fact', 'Interpretation'],
    ['Lexicographic first-20 pattern in every category', 'Strong evidence of structured selection; historical split code was not reviewed.'],
    ['Format and size differ substantially', 'Possible collection/style associations; metadata alone does not establish source provenance.'],
    ['Category balancing leaves the FID gap', 'Category frequency alone is insufficient to explain the visual shift.'],
]) + '''

**Generative-model implication.** A student and teacher trained only within A can learn a different visual mixture from the fixed FID reference. A2 remains the required development set, but success on A2 is not a guarantee of the same gain on B. This is a dataset property shared by methods; it cannot explain the SiDA–DMD2 ranking on its own.

**Guidance.** Keep all partitions fixed. Use the official 20-per-category generation schedule for comparable reporting, and consider class-balanced A1 sampling as an A2-tested design choice—not a remedy for the entire shift. Avoid selecting “early” filenames or PNG-only training samples to reproduce B. There is no evidence here for converting files, lowering resolution or changing the official loader.

**Limit.** The ordering pattern is exact; its causal responsibility for style differences is a plausible interpretation, not proven historical attribution. The EDA does not establish any challenge-rule change or permission to tune against B.''', ['analysis/metadata_shift.json', 'analysis/split_order_audit.csv', 'category_composition.csv'])

add(6, 'Image properties and augmentation hypotheses', base() + '''**Brightness and near-white pixels differ strongly, even within every Pokémon category. Augmentation should preserve useful variation rather than force images toward B's appearance.**

**Method.** Measure simple properties on the actual 64 × 64 RGB inputs: near-white pixel fraction, brightness, saturation and edge/texture summaries. These interpretable measurements complement Inception features and help identify what an augmentation would change.

''' + table([
    ['Per-image statistic (median)', 'A', 'B'],
    ['Near-white pixel fraction', '0.29%', '37.52%'],
    ['Brightness, 0–1 scale', '0.615', '0.762'],
    ['Saturation, 0–1 scale', '0.279', '0.207'],
]) + '''

B's median near-white fraction is higher in **all 151 categories**. Near-white means all three RGB channels exceed 0.95; this is not a background segmentation or an estimate of subject size.

''' + fig('image_properties', 'Distributions of measured 64 × 64 input properties. Near-white pixels may include both subject and background.') + '''

**Implication for adversarial distillation.** A discriminator could exploit background, framing or color cues, but this EDA does not show that the current SiDA/DMD2 discriminators do so. Distinguish two interventions: modifying the real-image distribution used by a teacher changes its learning target; augmenting discriminator inputs aims to regularize the real/fake comparison.

''' + table([
    ['Candidate for a future A1/A2 ablation', 'What to check before expecting a benefit'],
    ['Small translations or mild scale changes', 'Character remains complete at 64 × 64; padding introduces no artificial border cue.'],
    ['Horizontal flips where semantically acceptable', 'Inspect text/cards and asymmetry; establish whether the current recipe already uses flips.'],
    ['Mild brightness/saturation changes', 'Preserve species cues and style diversity; color invariance is not automatically desirable.'],
    ['Aggressive crop, cutout, background removal/replacement', 'Lower-priority, higher target-distortion risk; requires separate evidence from A.'],
]) + '''

For a DiffAugment-style adversarial branch, use the same augmentation policy/distribution on real and generated inputs and preserve gradients through the generated-image transform. This does not require pairing identical random transforms between unrelated images. Its effectiveness in standalone GANs is motivation, not evidence of a gain in these distillation implementations. [DiffAugment paper](https://arxiv.org/abs/2006.10738).

**Guidance.** Keep an unaugmented control and vary one augmentation family/strength at a time, selecting on A2. Do not automatically apply image transforms to teacher or fake-score regression targets: coordinate transforms, noise/time conditioning and targets must remain mathematically consistent. Do not whiten backgrounds, remove scenic images, or choose augmentation strength to hit B's 37.52% statistic. These are hypotheses for later work; no augmentation experiment was run.''', ['analysis/image_properties.csv', 'analysis/white_fraction_by_category.csv', 'analysis/metadata_shift.json'])

add(7, 'Exact duplicate overlap and FID sensitivity', base() + '''**The evaluation reference contains exact training-pool matches, yet the A/B shift persists when those matches are excluded. Coverage and generalization need separate interpretations.**

**Method.** Hash file bytes and decoded RGB pixels, including the actual 64 × 64 inputs. Identical hashes flag exact content sharing; a second, explicitly changed-population FID analysis tests whether the shift persists without it.

- **343 of 3,020 B images (11.36%)** have exact matches in A, confirmed at both native RGB and 64 × 64 RGB levels.
- Across the full dataset, **1,416 RGB duplicate groups** contain **1,421 redundant files**.
- There are **no exact RGB duplicate groups crossing A1/A2**. This does not rule out near-duplicate artwork with different crops, compression or borders.

''' + table([
    ['Unique-pixel sensitivity: 2,677 images/set, 20 seeds', 'Median FID'],
    ['Unique A subset vs B with A-matching groups excluded', '**24.66**'],
    ['Within A, disjoint unique-pixel groups', '**16.81**'],
]) + '''

The sensitivity keeps one representative per 64 × 64 RGB group and prevents content sharing between its two A subsets. It changes B and sample count, so compare **24.66 with 16.81 within this scheme**; the difference from the main 3,020-image score is not a quantified “duplicate penalty.”

**Generative-model implication.** Memorizing some A images could reproduce part of B's reference content; this audit establishes that possibility, not that any current model memorizes. Low reference FID alone therefore cannot establish novel-image generation. Duplicates also increase the effective sampling frequency of certain artworks, which matters in small-data adversarial training.

**Guidance.** Preserve the official data and score. In later authorized work, inspect generated-to-A1 neighbors and within-class diversity, and build a perceptual near-duplicate-family audit within A for a supplementary generalization check. Any duplicate-aware sampler is a separate A2-tested intervention, not an automatic cleanup. Do not remove examples because they happen to match B.

**Connection to development results.** A2's duplicate-aware construction makes it structurally different from B. That helps explain why A2 and B can give different coverage impressions, but it does not predict a direction or numerical offset for model FID. See ''' + ref(3, 'EDA-3 — nearest-neighbor controls') + ' and ' + ref(1, 'EDA-1 — main FID controls') + '.', ['audit.json', 'analysis/fid_duplicate_sensitivity.csv', 'analysis/validation.json'])

# Verify every cited local result exists before allowing publication.
for report in reports:
    for src in report['local_sources']:
        assert (OUT / src).is_file(), src
    assert report['properties']['Title'].startswith('EDA-' + str(report['number']) + ' — ')
    assert 'Experiment ID' not in report['properties']

(OUT / 'notion_report_drafts.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2))
print(json.dumps({'reports': len(reports), 'words': [len(r['content'].split()) for r in reports]}))
