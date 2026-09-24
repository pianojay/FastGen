# KCloud migration

Destination: private [pianojay/fastgen-soflow](https://github.com/pianojay/fastgen-soflow) for source; SFTP for large artifacts. The KCloud hostname, username and destination directory must be supplied when service returns. Never commit credentials or copy Conda environments between machines.

1. Clone the private code repository on KCloud; check out the documented experiment commit.
2. Create Python 3.10 plus pip and install requirements-fastgen-lock.txt with PYTHONNOUSERSITE=1. Record pip check, package locations, GPU model, driver/CUDA and dependency versions. CUDA 12.6 wheels require a compatible driver; do not change shared/server drivers to accommodate this project.
3. Use the official downloader and fixed manifests. Recreate the development partition with python -m scripts.prepare_data. Compare manifest hashes with the local experiment. Dataset transfer is optional: SFTP the already composited data directory and marker if re-downloading is impractical.
4. Repeat contract tests and actual augmented training-memory qualification on the A100 vGPU. Use CUDA_VISIBLE_DEVICES=0 if that is its assigned GPU. TITAN timings and BF16 support assumptions do not transfer.
5. Create a fresh configuration with run_dir runs/pokemon_kcloud_fresh and the selected microbatch. Keep effective batch 128 and all objective settings for the first comparison. Start without --resume. Local checkpoints are evidence, not initialization for this fresh baseline.
6. Transfer the source bundle, exported model package, checkpoints, logs, figures, partition and environment manifest with SFTP when wanted. Run SHA-256 verification after transfer. Exact resume is supported within an experiment, but KCloud's first substantive run starts from scratch as agreed.

Bundle locally:

```bash
python -m scripts.bundle --config configs/pokemon_local.json --output dist/migration
```

The bundle contains source.tar.gz, artifacts.tar.gz, bundle_metadata.json and SHA256SUMS. Source contains Git-tracked files, including the official directory layout and added tools, but no dataset images or weights. Commit new source/config files before bundling; unrelated untracked work from other agents is excluded. Artifact archive includes development manifests, qualification reports, the latest full training checkpoint, export checkpoints, sample grids, FID reports and environment provenance; it excludes the raw image dataset and bulk generated FID PNGs. A bundle taken during training is a progress snapshot; the workflow creates an updated terminal bundle in dist/RUN_NAME after evaluation and export.

Interactive transfer (fill in connection details locally):

```text
sftp USER@HOST
mkdir REMOTE_DIRECTORY
cd REMOTE_DIRECTORY
put dist/migration/source.tar.gz
put dist/migration/artifacts.tar.gz
put dist/migration/bundle_metadata.json
put dist/migration/SHA256SUMS
bye
```

On KCloud, use `sha256sum -c SHA256SUMS` before extracting. The exported package also has a per-file SHA256SUMS.json. Run `python -m scripts.verify_export --package PATH_TO_EXPORT --device cuda` before scoring. Official validation scoring is a final frozen-recipe check, never a development selector.
