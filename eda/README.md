# Pokémon dataset EDA preparation

This workspace studies the real-image dataset and its fixed partitions. It does not create, train, tune, or evaluate generative or predictive models. Preparation downloaded an independent Kaggle copy, preserved the existing split, and audited the data. The four primary feature-based analyses are complete; see [the EDA results](REPORT.md) and [figure gallery](reports/analysis/gallery.html). This document retains the preparation record and analysis protocol.

## Dataset and fixed partitions

| Name | Meaning | Images | Local manifest |
| --- | --- | ---: | --- |
| A | Official training pool | 17,079 | `resources/A.txt` |
| A1 | Existing optimization subset of A | 14,059 | `resources/A1.txt` |
| A2 | Existing development subset of A | 3,020 | `resources/A2.txt` |
| B | Official validation file, used as evaluation reference | 3,020 | `resources/B.txt` |

A1 and A2 are disjoint and their union is A. A and B have disjoint paths. A2 is not B, despite both having 3,020 images. The primary requested comparison is all of A versus B; A1 versus A2 is supplementary context only.

The saved development partition was read from `../FastGen-SoFlow/artifacts/development_split.json`. Its bytes agree with `../FastGen-AFM/resources/development_split.json` and `../FastGen-SiD-DiT/resources/development_split.json`. All three have SHA256 `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`. The recorded training-manifest and category-map hashes match this checkout. `resources/development_split.json` is an exact copy; no partition was regenerated. Derived manifests retain original ordering within each saved index list.

The original partition selected 20 globally byte-unique training files per category using seed 42 and SHA256 ordering. This makes A1/A2 a useful fixed descriptive comparison, but not an interchangeable random within-A baseline.

## Initial audit findings

All 20,099 downloaded images decode successfully and match the official manifest union, with no missing or unlisted files. The supplied compositor converted 1,476 transparent images; none retain transparency. Five images remain grayscale and one is opaque palette-mode, so the RGB conversion in the data pipeline remains necessary.

B and A2 each have exactly 20 images per category. A ranges from 46 to 183, and A1 from 26 to 163. The total variation distance between A and B's category proportions is 0.0701. This establishes a category-composition difference but says nothing yet about within-category appearance differences. The largest absolute differences are Pikachu, Nidoran-f, and Gengar. See [category composition](reports/category_composition.png).

There are 343 B images (11.36%) with exact file-byte matches in A after white compositing. Decoded native RGB and 64 × 64 RGB checks also find 343 B images matching A. There are no duplicate groups crossing A1/A2 under any of these three checks. Across the complete dataset, there are 1,415 repeated file-byte groups and 1,416 repeated RGB groups. These are measured duplicates, not a near-duplicate audit. Preserve the splits and annotate these matches in FID and neighbor interpretation; deduplicated results must remain supplementary.

The machine-readable preparation evidence is [audit.json](reports/audit.json), with paths and hashes in [inventory.csv](reports/inventory.csv) and duplicate group lists alongside it. Completed feature comparisons and the finding that B equals the first 20 lexicographic filenames per category are documented in [the results](REPORT.md).

## Preparation and reproducibility

The dataset source is [Kaggle Pokemon Generation One](https://www.kaggle.com/datasets/bhawks/pokemon-generation-one-22k). The public archive was downloaded directly from Kaggle because `kagglehub` was absent in the initial system-Python environment. That download required no package changes. The subsequent isolated Conda setup is described below.

`resources/download.json` records the downloaded archive hash, size, file count, and transparency conversion count. The original archive remains in `.cache/downloads/`. Extraction validates paths and ZIP member CRCs. Only this checkout's new image files are composited onto white using the supplied `download_dataset.py` function. No shared data symlinks or hardlinks are used.

To repeat the audit after extraction:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python3 eda/prepare.py
```

`python3 eda/prepare.py --extract` is for initial materialization only and refuses to overwrite an existing image tree. It expects `.cache/downloads/pokemon-generation-one-22k.zip`. The audit needs only Pillow and the Python standard library, does not import the model or evaluation entry point, and uses no GPU.

Audit outputs in `reports/` include image dimensions, modes, formats, decode failures, missing and unlisted files, category composition, and duplicate groups. Duplicate checks separately cover file bytes after compositing, decoded RGB pixels at original dimensions, and decoded RGB pixels after 64 × 64 bilinear resizing. None of these hashes detect all near-duplicates. Preserve the original populations; any deduplicated sensitivity analysis must be separately labeled and must not rewrite manifests.

After the audit, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 eda/plot_audit.py` creates `reports/category_composition.png` using the available matplotlib installation and a workspace-local cache.

`resources/provenance.json` pins the starter files, manifests, and copied partition. The audit checks these hashes before and after execution. All outputs, caches, and environments belong inside `exploration`. The image audit used bounded CPU resources; subsequent GPU 0 environment qualification is recorded below.

## Shared image representation

For the main analyses, use the frozen Inception feature extractor supplied by clean-fid solely as an image measurement instrument. No learned component will be updated and no project checkpoint will be loaded. PCA and clustering are descriptive transforms of these fixed features, not predictive-model experiments.

The primary input view should match the provided image pipeline: white-composited RGB, bilinear resize to 64 × 64, then the same normalization and uint8 export behavior as `dataset.py` and `evaluate.py`. Verify the export round-trip against those functions before extracting features. Cache one feature vector per manifest path in stable order, with image hashes, package versions, extractor weight hash, resize settings, feature dimension, and dtype. Pin `mode="clean"` and `model_name="inception_v3"` explicitly. FID uses full, unnormalized 2,048-dimensional features; do not substitute PCA or cosine-normalized features.

[clean-fid documents the sensitivity of FID to resizing and quantization](https://github.com/GaParmar/clean-fid). Native-resolution images can be a separately labeled sensitivity view, never mixed into the primary feature cache. Original images should accompany 64 × 64 examples so that compression or resizing artifacts can be recognized.

The isolated Conda environment is now qualified; use the FastGen-SoFlow reference described below. The earlier system-Python inventory in `resources/environment.json` is historical and is not the EDA runtime.

## Qualified environment

At the user's direction, FastGen-SoFlow is the reference project for the EDA runtime. `.conda/eda` is an independent `conda create --clone ... --copy` copy of `/home/user/miniconda3/envs/fastgen_soflow`, including its Python interpreter. No sibling environment was changed. All nine package versions in SoFlow's `requirements-fastgen-lock.txt` were checked against the local runtime.

| Component | Version |
| --- | --- |
| Python | 3.10.21 |
| PyTorch and CUDA runtime | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| clean-fid | 0.1.35 |
| Pillow | 12.3.0 |
| matplotlib | 3.10.3 |
| scikit-learn, added for EDA | 1.7.2 |

Scikit-learn's supporting additions are joblib 1.6.0, threadpoolctl 3.7.0, and cloudpickle 3.1.2. `pip check` passed. The complete installed snapshot is `resources/eda-environment.lock.txt`; SoFlow's reference requirements and package snapshot are saved alongside it. The snapshot includes a Conda-origin local packaging URL, so it is an inventory rather than a portable pip-only installer.

Run Python through the wrapper to disable user-site packages, constrain CPU threads, direct caches into this workspace, and select physical GPU 0:

```bash
bash eda/run_python.sh eda/prepare.py
bash eda/run_python.sh eda/plot_audit.py
bash eda/run_python.sh eda/check_environment.py
```

GPU 0 is a 24 GiB NVIDIA TITAN RTX, UUID `GPU-e9d30df5-3b84-1a7e-c505-c6fff0a0ce7f`, compute capability 7.5. The user authorized GPU 0 for EDA. Device access requires execution outside the restricted sandbox; the prior `nvidia-smi`/CUDA failure was sandbox visibility, not a host driver failure. No driver changes are needed. Future GPU scripts must hold the existing `/tmp/fastgen-gpu0.lock`, verify UUID and occupancy, and never terminate other processes. The wrapper selects the device; it does not acquire the lock itself.

`check_environment.py` holds that lock without modifying its contents, verifies imports originate inside the local Conda prefix, checks all SoFlow reference pins, runs CPU PCA, and extracts finite 2 × 2,048 features from synthetic inputs on GPU 0. The successful check peaked at 125,640,192 allocated GPU bytes. This is an environment smoke check, not dataset EDA or model evaluation. See `resources/qualified_environment.json` for the measured evidence.

The standard clean-fid Inception weight file is independently copied under `.cache/clean-fid/`, SHA256 `f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4`. Its source was the AFM cache; the clean-fid extractor implementation is byte-identical in AFM and SoFlow. It is the standard frozen measurement network, not a project-trained checkpoint. Preprocessing equivalence and all-image feature extraction passed; see [feature provenance](features/metadata.json).

## FID and within A baselines

Report full FID(A, B) with counts 17,079 and 3,020. Treat this as a descriptive distance between real-image feature distributions, not a model score or a significance test.

The primary size-controlled experiment uses m = 3,020 and 20 fixed repeats with seeds 20261005 through 20261024. In each repeat, sample 2m distinct A paths without replacement, divide them into disjoint U and V, and compute both FID(U, B) and FID(U, V). Each pair then has the same sample counts. Save the sampled indices before extracting or comparing results. Report the median, interquartile range, and full range of both distributions and of their paired differences. B is fixed, so this spread describes subsampling variation, not a population confidence interval. Path disjointness does not ensure content disjointness: annotate duplicate-group overlap and add a sensitivity analysis that keeps duplicate groups from crossing U/V.

`python3 eda/prepare_sampling.py` has saved all uniform and balanced sampling indices in `resources/fid_sampling.json.gz`, with manifest hashes and seeds. All 40 pairs were checked for sizes, index bounds, and within-pair disjointness; the balanced pairs were checked against B's category counts. These are analysis subsets, not replacements for A1/A2. A later duplicate-group sensitivity design must be stored separately.

Run a separate category-balanced version: sample 40 distinct A paths per category and split them into 20 for U and 20 for V. Every category has at least 46 A paths, so this is feasible. B already has 20 per category. Compare the unstratified and balanced results to distinguish category proportions from within-category appearance shifts. Equal sample sizes improve comparability but do not eliminate all [finite-sample FID bias](https://arxiv.org/abs/1911.07023). A1/A2 has a different selection mechanism and is not the only null baseline. Avoid full-dimensional per-category FID with only 20 B examples.

Deliverables: full-set result, per-repeat table, sampling manifests, baseline distribution plot, and duplicate sensitivity notes.

## PCA directions and distributions

Fit one centered, unwhitened PCA basis on A features and project both A and B through the same centering and basis. This describes how B lies relative to A's major directions; never fit separate coordinate systems for the two sets. Report explained variance and cumulative variance. A pooled, equally weighted A/B PCA can be a separately labeled sensitivity view, because an A-only basis may miss directions unique to B.

Show PC1/PC2 and PC3/PC4 with equal-size display samples and transparency or density contours. Compute distributions from all observations. Use common-bin normalized histograms and empirical CDFs for PCs 1–10. Report mean shifts in A-standard-deviation units and distribution differences; show image examples near both extremes of each prominent PC. PC axes are visual variation directions, not automatically semantic concepts. Category-conditioned views help separate Pokémon identity from background, rendering style, pose, or cropping.

Deliverables: explained-variance plot, common-axis scatter plots, per-PC distributions, effect-size table, and PC endpoint contact sheets.

## B to A nearest neighbors

For each B image, search all A features, preserving B's identity and ordering. Use Euclidean distance in full Inception space as the primary measure and cosine distance after L2 normalization as a labeled sensitivity check. Compute distances in bounded chunks, keeping the nearest five A paths and distances rather than materializing the full matrix. Repeat with A restricted to the same Pokémon category.

Plot the full empirical CDF and histogram, report median, 90th, 95th, and 99th percentiles, and list the largest distances. Show the top 24 B queries with their three closest A neighbors, plus representative central-distance cases and metadata. Include both original and 64 × 64 views, exact-duplicate flags, category, and filenames. High distance means poor coverage in this representation; visual inspection is required before calling a sample novel, mislabeled, or absent from A.

For a reference distribution, use A leave-one-out distances with self matches excluded; report a version excluding same duplicate-group matches. For fairer gallery-size control, compare held-out A queries and B queries to the same fixed A gallery of size 14,059, with query sets of 3,020. Repeat that random-gallery comparison using saved seeds, optionally category-balanced. Keep it distinct from the primary B→all-A result and the nonrandom A2→A1 comparison.

Deliverables: per-B neighbor table, distance plots, category-conditioned summaries, and tail contact sheets.

## Cluster composition and representative images

As a descriptive summary, fit a shared clustering on A and assign B to the same centers. Proposed fixed choices are A-centered, unwhitened PCA to 64 dimensions followed by k-means with k = 32, random_state = 20261005, and n_init = 10. Report retained variance. Use k = 16 and 64 as predefined sensitivity checks, without selecting a k based on the strongest A/B separation. These are dataset summaries, not trained generators or predictors.

For every cluster, report n_A(k), n_B(k), p_A(k) = n_A(k)/|A|, p_B(k) = n_B(k)/|B|, and signed difference p_B(k) − p_A(k). Rank by absolute difference, displaying the sign and sample counts. Report total variation, one-half the sum of absolute differences. Avoid ranking only by raw counts or unstable ratios for tiny clusters. Repeat proportions after giving categories equal weight to separate category imbalance from style imbalance.

For the most shifted eight clusters, show the real images closest to each center, representatives from each split when present, and some farther examples to reveal within-cluster variety. Annotate Pokémon composition and B distance-to-center tails: assigning all B images to A centers can hide a genuinely new mode. A balanced pooled descriptive clustering is an optional complementary view for such cases. Do not assign semantic names without inspecting images. If intervals are added, account for duplicate groups and distinguish fixed-dataset descriptions from population claims.

Deliverables: cluster-composition table, signed proportion-difference plot, representative-image sheets, and sensitivity comparisons.

## Interpretation boundary

B is available here only for the requested dataset audit. Findings must not become training examples, checkpoint selection, sampling-schedule selection, or hyperparameter tuning in the sibling experiments. Keep measured data properties distinct from hypotheses about generation quality. No model-performance claims follow from this EDA.
