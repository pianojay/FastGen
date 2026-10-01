# Flow SiD/SiDA campaign results

Local development evaluation only: 3,020 train-derived reference images, 20 generated images per class. One teacher and one training realization per arm. Sampling confirmations are not training replications.

| Run | State / updates | NFE | Selection FID | Confirmation FID | Flagged classes | Charged stage hours |
|---|---|---:|---:|---:|---:|---:|
| T | training / 10070 | 32 | 63.8874 | — | 0 | 3.084 (active) |
| S1 | not started / 0 | 1 | — | — | — | 0.000 |
| A1 | not started / 0 | 1 | — | — | — | 0.000 |
| S2 | not started / 0 | 2 | — | — | — | 0.000 |
| A2 | not started / 0 | 2 | — | — | — | 0.000 |
| S3 | not started / 0 | 3 | — | — | — | 0.000 |
| A3 | not started / 0 | 3 | — | — | — | 0.000 |
| S4 | not started / 0 | 4 | — | — | — | 0.000 |
| A4 | not started / 0 | 4 | — | — | — | 0.000 |

| NFE | Paired selection ΔFID (A−S) | Paired confirmation ΔFID (A−S) |
|---:|---:|---:|
| 1 | — | — |
| 2 | — | — |
| 3 | — | — |
| 4 | — | — |

Training/evaluation stage hours charged so far: 0.000; teacher charged once. Setup, preprocessing and qualification are recorded separately in `runs/cost.jsonl`. Each stage includes checkpoint I/O, evaluation, failed work and replay, where applicable.

Checkpoint/source/teacher identities, memory snapshots, raw per-update curves, galleries, all-class diversity and optimization-neighbor audits are retained in each `runs/campaign/<run>/` directory. Potential-collapse flags are pixel diagnostics and require visual interpretation.

Teacher 10k progress: 640,000 optimization presentations; FID **63.8874** at 32 calls / seed 1729. Teacher stage cost through this evaluation: **3.084 hours**, plus **0.445 hours** of recorded pre-teacher setup/qualification wall time. The teacher remains active; the completed-stage ledger total above excludes its in-flight attempt. Selection is provisional until the 30k horizon.

No classes triggered the predefined duplication/diversity-collapse flag (151 assessed); all generated images were unique within class. Median relative area-pooled pixel diversity was 0.653 of the development reference. The fixed 16-class gallery contains broad class cues and varied compositions, with distorted anatomy and sometimes ambiguous identities. These pixel flags do not establish semantic class coverage.
