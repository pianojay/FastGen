# Restore on another machine

This restores selected archival state, not a live job or a complete training environment. Authenticate GitHub CLI (`gh auth login`) using your normal account; the two repositories and release are private. Python 3.10+ and Git are sufficient for byte restoration. No package installation or GPU execution is performed by the restore tool.

## Minimal verified restoration

Run from a parent directory where you want the archive:

```bash
git clone --branch FastGen https://github.com/pianojay/master_root.git FastGen-master
git clone https://github.com/pianojay/FastGen.git FastGen-code
python3 FastGen-code/tools/restore.py --output "$PWD/FastGen-restored"
```

Keep the output separate from either Git checkout. The tool downloads only the named release assets, checks the committed manifest and checksum list, rejects unexpected/unsafe archive members, reconstructs `FastGen-restored/legacy/`, and verifies every retained file and checkpoint. It refuses to overwrite differing existing files. Repeating it rechecks already downloaded/restored files without silently clobbering changes.

The complete file-level mapping is `migration/inventory.json`: original path, archived asset, restored path, byte length, SHA-256, provenance and any Git branch/path. Historical absolute paths beginning `/home/user/jw_soflow/` map to `FastGen-restored/legacy/`. Open the generated `INDEX.html` for portable links; inspect `reference-availability.json` for retained versus omitted Markdown/HTML references. Historical files remain byte-identical, so their embedded absolute links are not rewritten.

Exact inference locations after restoration:

```text
FastGen-restored/legacy/FastGen-SoFlow/runs/research_s3_unet192_40k_seed42/export/
FastGen-restored/legacy/fastgen_zip/fastgen_sida_1007/
```

Both contain their original model source and `checkpoints/one_nfe.ckpt` / `few_nfe.ckpt`. S3's configuration is width 192; do not load its weights using the historical root default width 128. The two S3 files share model tensors but retain distinct original sampling-count buffers. Peer uses different architectures for its one- and two-call models.

## Optional Master/worker worktree layout

The index clone above is enough to inspect/restore everything. To recreate the template's bare-anchor/worktree layout instead, from the Master clone:

```bash
cd FastGen-master
mkdir -p workers/FastGen
git clone --bare https://github.com/pianojay/FastGen.git workers/FastGen/.FastGen.git
git --git-dir=workers/FastGen/.FastGen.git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git --git-dir=workers/FastGen/.FastGen.git fetch origin
for name in soflow afm sid-dit peer-sida exploration cifar-soflow; do
  git --git-dir=workers/FastGen/.FastGen.git worktree add "workers/FastGen/$name" "archive/$name"
done
```

The bare clone already imports remote branches as local heads; omit `-b` when allocating these existing branches. Keep the shared bare metadata while any linked worktree exists. Worktrees do not isolate GPUs, processes or environments.

## Environment and dataset reconstruction

Original environments and dataset images are deliberately excluded. Historical reconstruction inputs are:

| Codebase | Preserved environment record |
|---|---|
| SoFlow reference | `legacy/FastGen-SoFlow/requirements-fastgen-lock.txt` and `artifacts/environment-freeze.txt` |
| AFM | `legacy/FastGen-AFM/resources/fastgen_afm_requirements.lock` |
| Flow SiD | `legacy/FastGen-SiD-DiT/resources/environment.lock.txt` |
| EDA/transfer | `legacy/exploration/eda/resources/eda-environment.lock.txt` and `qualified_environment.json` |
| Peer | Its README and requirements, plus the independently tested inference environment reported in `cpu-inference-verification.json` |
| Original CIFAR | `legacy/SoFlow/requirements_soflow.txt`, `requirements_soflow_eval.txt`, and AGENTS environment notes |

For the tested shared reference stack, the historical installation recipe is:

```bash
conda create -n fastgen_reference python=3.10 pip -y
conda activate fastgen_reference
PYTHONNOUSERSITE=1 python -m pip install -r "$FASTGEN_RESTORED/legacy/FastGen-SoFlow/requirements-fastgen-lock.txt"
```

Set `FASTGEN_RESTORED` to the absolute restoration output first. Use `python -m pip` and `PYTHONNOUSERSITE=1`. The CUDA build in the archived lock is historical, not a claim about the new machine's GPU compatibility. Any changed software stack or fresh GPU work requires its own qualification; do not blindly reuse old device numbers, UUIDs or host paths.

Dataset reconstruction uses the preserved official `download_dataset.py`, fixed `train_split.txt`, `val_split.txt` and `category_to_id.json`. Run a downloader only in a newly chosen data location, never against preserved evidence. The original official split has 17,079 training / 3,020 validation images; all three local worker experiments used the identical 14,059 / 3,020 training-derived partition. Compare split/manifests and content hashes to the retained provenance before interpreting a reconstructed experiment. Source size/mtime statistics are historical observations and will not reproduce on another filesystem.

EDA features, shared pixel arrays, teacher weights and other outputs must be recomputed if later authorized; the archive does not contain them. Downloading/restoring does not authorize training, evaluation or automatic continuation. The private release does not expire like an Actions artifact, but continued access still depends on retaining the repository/account.

## Optional CPU inference check

With an already prepared compatible environment:

```bash
PYTHONNOUSERSITE=1 python FastGen-code/tools/verify_inference.py --legacy-root "$PWD/FastGen-restored/legacy" --report "$PWD/FastGen-restored/cpu-inference-verification.json"
```

This loads all four checkpoints on CPU, verifies finite parameters/outputs and actual network calls 1/4 for S3 and 1/2 for peer. It is not a FID test, GPU qualification, training reproduction or proof of peer training provenance.
