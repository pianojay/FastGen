# FastGen migration archive

Private, selected archival backup of the former `~/jw_soflow` workspace, created November 4, 2026. Source histories and compact records live on independent branches; main is an index and restoration interface.

The November 5–7 A100 campaign completed fresh S3 60k and FM25 40k runs. Code and configurations are on [deadline/soflow-a100-20261105](https://github.com/pianojay/FastGen/tree/deadline/soflow-a100-20261105). Research decisions, setup measurements, and campaign records are on the [Master project branch](https://github.com/pianojay/master_root/tree/FastGen). Historical archive instructions retain their original scope.

| A100 recipe | One-call development FID | Four-call development FID |
|---|---:|---:|
| Fresh S3 60k | 57.0394 | 22.7310 |
| Fresh FM25 40k | 71.8435 | 24.0835 |

Scores average four paired sampling seeds on the original training-derived development reference. Each recipe has one training seed. The [campaign results](https://github.com/pianojay/master_root/blob/FastGen/programs/soflow-a100-campaign-20261105/results.json) include the per-seed metrics and local checkpoint identities. The new model weights are retained locally under the user's November 7 publication instruction. The existing migration release continues to hold the original archival assets.

- [Owning Master project](https://github.com/pianojay/master_root/tree/FastGen): charter, research map, historical relationships, dispatch and diary.
- [Restore guide](RESTORE.md): clone, authenticated download, checksum verification, original-path reconstruction and optional worktrees.
- [Migration report](MIGRATION_REPORT.md): verification state, limitations and cleanup boundary.
- [Private release](https://github.com/pianojay/FastGen/releases/tag/migration-2026-11-04): selected evidence and exactly four inference checkpoints.
- [Complete retained-file inventory](migration/inventory.json), [release manifest](migration/release-manifest.json), and [preserved ref map](migration/expected-refs.json).

| Archival branch | Frozen archival commit | Historical result / interpretation |
|---|---|---|
| [archive/cifar-soflow](https://github.com/pianojay/FastGen/tree/archive/cifar-soflow) | `561396e885e27f76dd412e377aef5031ee0ae847` | Stopped; finite early optimization, no image-quality evaluation. Last logged step 130; recoverable main checkpoint 5. No CIFAR weights retained. |
| [archive/soflow](https://github.com/pianojay/FastGen/tree/archive/soflow) | `f3b413505df78d4ce7f35b09764daf900052a116` | S3 remains the retained local incumbent: confirmation development FID 66.9684 / 25.0072 at one/four calls. The later matched adversarial pilot did not improve S3. No official challenge score or training-seed replication is claimed. |
| [archive/afm](https://github.com/pianojay/FastGen/tree/archive/afm) | `0951be5ea7b9f7dd828c97553715dbe7f0909695` | Baseline stopped; six-recipe stability campaign completed with no qualifying recipe, no confirmation and no trained-model FID benchmark. |
| [archive/sid-dit](https://github.com/pianojay/FastGen/tree/archive/sid-dit) | `65d70e496eeffc0b427cc32b59a2e7bc88c230b5` | Teacher and eight students completed. SiDA improved the tested one-call arm but degraded the two-/three-/four-call comparisons. One training realization per arm. |
| [archive/exploration](https://github.com/pianojay/FastGen/tree/archive/exploration) | `67cf7fb0f27571beea41b3b17a2a6ca55b2681b7` | Related independent master-worker study, not retrospectively subordinate to the FastGen coordinator. EDA and six transfer studies completed; no useful SiDA augmentation or generative FID improvement was established. |
| [archive/peer-sida](https://github.com/pianojay/FastGen/tree/archive/peer-sida) | `e7309c619f883c857b3d7d2bd69efa6eaa12c0db` | Externally supplied source/checkpoint copy. README method and training claims remain attributed to the peer; original training logs and write-up were not supplied. Migration validates bytes and CPU inference only. |

The ordinary historical branches are retained under `history/soflow/*`, `history/afm/main`, `history/sid-dit/main` and `history/exploration/main`, preserving original commit IDs. Hidden agent snapshots, reflog-only objects and temporary Git packs are excluded. The shallow original CIFAR checkout is a source snapshot with its base SHA and patch, not a full upstream history.

The peer branch is a supplied copy over the challenge starter ancestry; inherited starter assets are not evidence supplied by the peer. Exploration was an independent EDA/master-worker session. Neither is retrospectively represented as a subordinate FastGen worker experiment.

## Selected parameters

| Release asset | Bytes | SHA-256 |
|---|---:|---|
| `soflow-s3-one_nfe.ckpt` | 356,696,109 | `d03a959d0117a7d34ce3b82929e7b816becd3cd97199e7423cdc92ba299819c5` |
| `soflow-s3-few_nfe.ckpt` | 356,696,109 | `fc6337a046f6a1272ffa2f0afe92ca1c7b514f642032ecdb83aec99baf5d400b` |
| `peer-sida-one_nfe.ckpt` | 247,492,073 | `7134046f1654da72ad3d7905ef1e2762d87b5eaa0840947aece32eac6be958ee` |
| `peer-sida-few_nfe.ckpt` | 386,443,625 | `2246bfc0054026a95f7ecc4b1f54c99f896c75b9d514f9f4a24c5653345a5690` |

Total parameter bytes: **1,347,327,916**. These are inference references. No teacher or optimizer/resume checkpoint is retained. S3's exact inference source/configuration is in its evidence snapshot and archived branch under `runs/research_s3_unet192_40k_seed42/export`.

## Deliberate exclusions

- All parameter/optimizer/resume files except the four named inference exports.
- Datasets, data caches, features, numpy arrays, and per-image generated collections.
- Intermediate/unreviewed visual panels except the explicitly selected early/midpoint/final samples.
- Environments, package/download caches, cached paper PDFs, bytecode, and locks.
- ZIP/distribution duplicates, Git temporary packs, hidden agent/checkpoint refs and reflog-only objects.
- claude-audit, codex-session exports, machine-python-user-site-plan.json, and moved NVIDIA incident record.

The original directory and three earlier private remotes remain untouched. Their eventual deletion is a separate user action after inspecting the verification report.
