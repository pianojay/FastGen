# FastGen migration report — verified November 4, 2026

**The selected backup passed remote-only restoration and CPU inference verification.** This is deletion readiness for the selected scope; the original directory and old remotes have not been deleted.

## Destinations and checks

- Private worker archive: https://github.com/pianojay/FastGen
- Coordination project: https://github.com/pianojay/master_root/tree/FastGen
- Private release: https://github.com/pianojay/FastGen/releases/tag/migration-2026-11-04
- Template base remains `739fd3fcafaebca25e69950876c82de83abc8c3a`; shared template main was not changed.
- Fresh HTTPS clone passed `git fsck --full` and verified all **6 archive refs and 10 ordinary historical refs** against their frozen full SHAs.
- Remote-only release restoration verified **4,152 source/evidence files** (754,267,277 original bytes) and **4 checkpoints**. No source-directory copy was used during restoration.
- GitHub's uploaded-asset digests/sizes matched the staged files; downloaded/restored bytes independently matched the committed manifest and inventory.
- Pre-publication and post-restore checks found zero selected-source/Git mismatches. Host /proc checks found no matching Python process tied to the original workspace. No GPU, training or FID workload was launched.

The committed restore/inference tools were exercised from worker main `326652dd766e87858780839d38f18dfdb3e2d7da`. Subsequent closeout edits add records/navigation only; historical source, frozen assets and tested tools remain unchanged. Ref and per-file hashes are in migration/expected-refs.json and migration/inventory.json.

## CPU loader and call-count observations

| Restored model | Registered parameters | Actual backbone calls per batch | Finite output / shape / bound checks |
|---|---:|---:|---|
| soflow-one | 89,140,419 | 1 | pass |
| soflow-few | 89,140,419 | 4 | pass |
| peer-one | 61,823,875 | 1 | pass |
| peer-few | 96,561,763 | 2 | pass |

Each test sampled shape (2,3,64,64) with category IDs 0 and 150, using CPU and two threads in the existing fastgen_soflow environment, PyTorch 2.7.1+cu126. All parameters and outputs checked finite, outputs lay in [-1,1], and registered counts stayed below 100M. These are inference plumbing/compatibility observations, not image-quality or scientific-reproduction results. Detailed timings and paths are in migration/verification/cpu-inference-verification.json.

## Size and reproducibility

The 13 core release assets total **1,758,378,283 bytes**. Exactly **1,347,327,916 bytes** are the four original parameter files; the remaining **411,050,367 bytes** are compressed source/evidence, inventory and checksum manifests. Small final verification attachments are reported separately in release verification-summary.json.

Final Git pack measurements and combined payload totals are recorded after closeout publication in the release's verification-summary.json. Figures describe retained Git/release payload, excluding HTTP/TLS overhead and downloads for testing. Verification itself downloaded the full core payload again. Token/service accounting was not instrumented; GPU usage was zero.

Read [RESTORE.md](RESTORE.md) for authenticated clone/download, complete checksum validation, safe extraction, optional bare-anchor/worktree reconstruction, archived dependency records and CPU-check commands. The restored INDEX.html gives portable retained-file links; reference-availability.json identifies omitted or historical external Markdown/HTML references. Embedded historical links remain byte-identical.

The archive distinguishes original starting/executed identities from newly created archival commits. Full historical scientific reports remain source evidence rather than freshly verified findings. Peer README training claims are still attributed to the peer; source/checkpoint bytes and inference behavior were verified, not original training provenance.

## Retention limits

Only S3 one-/four-call and peer one-/two-call inference exports are retained. S1/S2, teachers, all other weights and optimizer/resume states are intentionally excluded. This archive cannot exactly resume those training jobs. Parent training-checkpoint hashes may remain as provenance even where those files were deliberately omitted.

Datasets, feature/output arrays, bulk per-image collections, most intermediate/unreviewed panels, environments/caches, duplicate ZIP/distribution bundles, Git garbage/hidden agent refs, audit/session exports and the unrelated Python cleanup plan are excluded. The NVIDIA incident record was already moved by the user and was not touched.

The shallow original CIFAR checkout is preserved as a source snapshot with its upstream SHA and local patch, not full upstream ancestry. Exploration remains a related independent study; peer remains an externally supplied copy. Historical instructions do not authorize fresh execution. New-machine GPU/software/data qualification and future research are separate assignments.

## User cleanup checklist

1. Inspect the Master, worker archive, private release and verification records using the account that will be used on the new machine.
2. Confirm the explicitly excluded material is not additionally needed; especially note the absence of training-resume/teacher states and dataset arrays.
3. Keep this report and [RESTORE.md](RESTORE.md) accessible independently of the old machine.
4. You may then remove `/home/user/jw_soflow` yourself. This migration has left it intact and has not stopped or changed another project's processes.
5. Optional old-remote deletion is user-only after the consolidated backup is accepted:
   - https://github.com/pianojay/fastgen-soflow/settings
   - https://github.com/pianojay/fastgen-afm/settings
   - https://github.com/pianojay/fastgen-sid-dit/settings

No deletion or credential changes are performed by this task. Staging/verification copies are disposable; all selected recovery inputs are remote.
