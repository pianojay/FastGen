# Migration report — November 4, 2026

State: prepared; publication and fresh remote-only restoration are pending. Do not delete the source based on this prepared state.

Frozen selection: 4,152 source/evidence files (754,267,277 original bytes), six compressed snapshots, four unchanged inference checkpoints (1,347,327,916 bytes). Exact asset identities are committed in the release manifest. Final network payload measurements and verification observations will be added after remote restoration.

The archived source preserves ordinary reachable histories and current files, including untracked EDA/transfer work and local CIFAR modifications. It excludes Git garbage and hidden agent snapshots. Source histories prove preservation of revisions, not reproducibility of the historical scientific conclusions.

The S3 checkpoint parent-training identity is retained as provenance even though that full parent checkpoint is deliberately not backed up. No optimizer or complete training state is retained. No new scientific result or official score was measured during migration.

## Cleanup boundary

Keep `/home/user/jw_soflow` until the final report confirms remote refs, every selected file/weight checksum and all four CPU inference checks. Then the user may remove it after deciding no additional excluded material is needed. The migration itself does not delete it.

After verified consolidation, user-only old repository deletion links:

- https://github.com/pianojay/fastgen-soflow/settings
- https://github.com/pianojay/fastgen-afm/settings
- https://github.com/pianojay/fastgen-sid-dit/settings

Deleting those repositories is not necessary for restoration. This task does not change credentials or request repository-deletion scope.
