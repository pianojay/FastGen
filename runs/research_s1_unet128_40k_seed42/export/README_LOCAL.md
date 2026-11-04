# Pokémon SoFlow on the local TITAN server

This adds a directly class-conditioned pixel-space SoFlow baseline to the official FastGen starter. The official README and protected files remain intact. See docs/STATUS.md for measured progress and docs/MIGRATION.md for KCloud.

## Environment

The initial local environment was cloned from the repaired soflow environment into fastgen_soflow; the original was not modified. Kagglehub and clean-fid were installed into the clone. User-site packages are disabled.

For a fresh machine:

```bash
conda create -n fastgen_soflow python=3.10 pip -y
conda activate fastgen_soflow
hash -r
export PYTHONNOUSERSITE=1
python -m pip install -r requirements-fastgen-lock.txt
python -m pip check
```

Run all following commands from this repository's root with that environment active. Locally, the absolute interpreter is /home/user/miniconda3/envs/fastgen_soflow/bin/python.

## Data and validation

```bash
python download_dataset.py
python -m scripts.prepare_data
CUDA_VISIBLE_DEVICES=1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.profile_batch
```

The fixed development partition holds out 20 globally byte-unique official TRAIN images per category. This avoids exact duplicate files across the optimization/development boundary while retaining 14,059/3,020 images. Near-duplicates are not excluded; the development distribution deliberately excludes byte-duplicate holdout candidates. Official validation images never drive development choices.

## Training and monitoring

```bash
python -m scripts.launch --config configs/pokemon_local.json --gpu 1
tail -f runs/pokemon_unet128_seed42/console.log
python -m scripts.report --run-dir runs/pokemon_unet128_seed42
```

The detached workflow trains up to 20,000 updates, evaluates terminal EMA weights at 1/2/3/4 NFE with two seeds, chooses the few-call mode by development FID, and exports a self-contained package. A 24-hour cumulative job budget includes profiling and evaluation. Four hours are reserved for terminal evaluation.

For a stopped run, use the same launch command with --resume. Do not relaunch a live workflow. To stop gracefully, signal the training PID (not the CIFAR process); the trainer checkpoints at the next update boundary. SIGKILL/power loss can lose work since the last checkpoint. Full resume state includes optimizer, EMA, RNG and deterministic data-order cursor. A hardware change does not promise bitwise-identical arithmetic.

Reports: metrics.jsonl, status.json, budget.json, REPORT.md, report.html, curves.png, samples/, evaluations/, export/. Reports refresh with sample grids every 500 updates and at workflow exit. Use `python -m scripts.report` to refresh sooner. Parent RSS and sampled process-tree PSS include a distinction between trainer memory and total memory including loader workers. `active_job.json` identifies the current training/evaluation child.

Training curves occupy one full-width row per metric. The default trend is a centered 201-record rolling median, with a local 10th–90th percentile band and all raw values retained. MSE also shows a rolling mean and uses a log y-axis when all values are positive; `curves_linear.png` provides the full-range linear view. The band describes local observed variability, not uncertainty in the trend. Windows shorten at the boundaries. Set `--smooth-window 501` (or another positive odd length) when regenerating the report to adjust smoothing. The changing consistency curriculum means MSE is not a fixed validation objective.

## Offline visual gallery

```bash
python -m scripts.gallery --run-dir runs/pokemon_unet128_seed42
```

Open `runs/pokemon_unet128_seed42/gallery.html` directly in a browser, beside the existing `report.html`. The gallery needs no server, CDN, model loading or GPU. It references existing PNGs, so keep it with the run's `evaluations/`, `reference_dev/` and `samples/` directories. It is not a self-contained image archive; the original SFTP bundle excludes bulk evaluation PNGs.

Browse all 151 categories, checkpoint/seed combinations and call counts, enlarge images, inspect 20 held-out real references per category, or switch to the 41 saved training panels. Evaluation images are not guaranteed to share initial noise across NFE; the gallery explains that distinction. The progression panels do reset initial noise across NFE/checkpoints. References are held-out development examples, not nearest optimization images.

Review notes are stored per category/checkpoint/seed in the current browser when supported. Download the audit JSON for a portable copy; notes are not automatically written to the run directory or Notion. Moving the HTML or using a different browser may not preserve browser-local notes. Rebuild the gallery after additional evaluations to refresh its embedded index.

## Inference-only package

```bash
python -m scripts.export --checkpoint runs/pokemon_unet128_seed42/checkpoints/latest.pt --output dist/local_snapshot --few-nfe 4
python -m scripts.verify_export --package dist/local_snapshot --device cpu
```

The export includes only EMA inference weights in checkpoints/one_nfe.ckpt and checkpoints/few_nfe.ckpt. Each loaded model has one backbone, no EMA duplicate or optimizer. Architecture settings accompany the package in inference_config.json; the protected loader needs no extra arguments. Root constructors default to the 128-width baseline.

Development FID uses clean-fid through the official utility functions, on unaugmented training-derived references. It is neither the SoFlow TensorFlow/CIFAR evaluator nor an official TA score. Do not run official validation FID as a tuning loop.

## Baseline recipe

39,667,843 parameters; 151 category IDs; six augmentation inputs; widths [128,256,256,256]; two blocks/level; attention at 16; dropout 0.1. Direct conditioning consumes one backbone evaluation per sampling call. Training uses SoFlow consistency-only learning, FP32 RAdam, effective batch 128, learning rate 1e-4, EMA warmup capped at 0.9999, and the original time distributions/weighting with a 20,000-update curriculum.

Augmentation is upstream SoFlow: horizontal flip with approximately 50% probability; scale, anisotropic scale, and translation each enabled with probability 12%. These transform existing training images to discourage memorizing their exact position and size. The six augmentation inputs tell the network which transformation was applied; both consistency branches receive the same transformed image context and metadata, while sampling requests the original distribution with zero metadata. This is useful as an initial baseline but does not establish that mirroring and stretching are optimal for Pokémon. Monitor the saved original/augmented image sheet and compare a matched augmentation-off run later if distortions hurt quality. The off control bypasses the entire pipeline because setting its probability to zero still resamples images.

Architecture, batch, dropout, EMA warmup and shortened curriculum are FastGen adaptation choices, not reproduction of the CIFAR paper recipe. Widths 160 and 192 give 61,934,243 and 89,140,419 parameters; later comparisons require matching training settings and separate resource qualification. They are not launched automatically.
