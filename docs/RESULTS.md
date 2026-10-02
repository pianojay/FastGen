# Flow SiD/SiDA campaign results

Local development evaluation: 3,020 fixed train-derived reference images and 20 generated images per class. Selection seed 1729; confirmation seed 271828 without reselection. One training realization per arm.

| Run | State / completed updates | Selected update | NFE | Selection FID | Confirmation FID | Pixel flags / 151 | Stage hours |
|---|---|---:|---:|---:|---:|---:|---:|
| T | complete / 30000 | 30000 | 32 | 37.2526 | — | 0 | 9.250 |
| S1 | training / 1018 | 1000 | 1 | 85.6491 | — | 0 | 1.029 |
| A1 | not started / 0 | — | 1 | — | — | — | 0.000 |
| S2 | not started / 0 | — | 2 | — | — | — | 0.000 |
| A2 | not started / 0 | — | 2 | — | — | — | 0.000 |
| S3 | not started / 0 | — | 3 | — | — | — | 0.000 |
| A3 | not started / 0 | — | 3 | — | — | — | 0.000 |
| S4 | not started / 0 | — | 4 | — | — | — | 0.000 |
| A4 | not started / 0 | — | 4 | — | — | — | 0.000 |

Active-stage hours include elapsed work; completed-stage hours use the external allowance ledger. Stage cost includes optimization, evaluation, diagnostics, checkpoint I/O, startup and any failed/replayed attempts.

| NFE | Selection ΔFID (A−S) | Confirmation ΔFID (A−S) | Δ median relative pixel diversity (A−S) |
|---:|---:|---:|---:|
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |
| 4 | — | — | — |

| Run | Optimization hours | Generation + FID hours | Allocated / reserved GB | Sampled device GB | Host RSS / PSS / HWM GB | Attempts |
|---|---:|---:|---:|---:|---:|---:|
| T | 8.399 | 0.467 | 6.016 / 6.392 | 6.470 | 4.881 / 4.776 / 4.953 | 1 |
| S1 | 0.994 | 0.014 | 15.761 / 15.903 | 16.186 | 4.152 / 4.048 / 4.165 | 1 |
| A1 | — | — | — | — | — | 0 |
| S2 | — | — | — | — | — | 0 |
| A2 | — | — | — | — | — | 0 |
| S3 | — | — | — | — | — | 0 |
| A3 | — | — | — | — | — | 0 |
| S4 | — | — | — | — | — | 0 |
| A4 | — | — | — | — | — | 0 |

Recorded pipeline cost: **10.732 hours** = 10.279 run-stage hours (teacher once) + 0.445 pre-teacher setup wall hours + 0.008 subsequent qualification/export-check hours. Setup includes overlapping installation, preprocessing and disposable probes; those components are not added twice. Pre-environment read-only review was not instrumented. The finite component allowances sum to 152 hours.

Every teacher, student and full fake-score/discriminator framework has **61,823,875 learned parameters**; the shared encoder’s pooled discriminator output adds zero parameters. EMA storage and optimizer states are included in measured simultaneous memory. The strict allocated-memory limit is 20,000,000,000 bytes. Device usage is sampled; allocated/reserved and host HWM are high-water measurements.

Pixel flags use the predeclared within-class duplication/diversity rule. They do not certify semantic recognition. Nearest optimization examples are same-class neighbors in 16×16 pixel space, not a perceptual test.

- **S1 at 1,000 (provisional):** The 16 fixed classes retain varied compositions and recognizable broad class cues, but noise, edge artifacts and less coherent anatomy/detail are apparent relative to the selected teacher. Pixel-diversity flags are 0/151; this is not a semantic-coverage score. This checkpoint is provisional until the fixed student horizon finishes. [Gallery](../runs/campaign/S1/eval_001000/gallery_fixed_0.png)
- **Selected T at 30,000:** The fixed 16-class gallery shows varied compositions and recognizable class cues, with clearer outlines and details than earlier checkpoints. Anatomy and some class-specific shapes remain imperfect, notably in Gyarados, Metapod and Scyther. No severe common-output collapse is apparent in this subset. All-class pixel diagnostics flag 0/151 classes at both 20k and 30k and at the selected checkpoint. Pixel diversity does not establish all-class semantic coverage. [Gallery](../runs/campaign/T/eval_030000/gallery_fixed_0.png)
- **T at 10,000:** Six fixed-noise generations in each of 16 fixed classes show varied color and composition with broad class cues, but distorted anatomy and sometimes ambiguous class features. No severe common-output collapse is apparent in this displayed subset; all-class pixel flags are 0/151. [Gallery](../runs/campaign/T/eval_010000/gallery_fixed_0.png)
- **T at 20,000:** Class features are clearer in the fixed gallery, particularly Eevee, Gengar, Kabuto and Starmie, and compositions remain varied. Limb/body distortions and ambiguous shapes remain visible in classes such as Gyarados, Metapod and Scyther. This is a qualitative review of 16 fixed classes, not an all-class recognition score; all-class pixel flags are 0/151. [Gallery](../runs/campaign/T/eval_020000/gallery_fixed_0.png)

Raw loss/gradient/timing records and SVG trends, all-class diversity, saturation, same-class neighbor audits and fixed-noise galleries are retained under `runs/campaign/<run>/`. Minimal inference sources and selected student checkpoints are under `artifacts/exports/`.

Interpretation is limited to this single teacher and training realization. The treatment adds the full adversarial package (real/fake D learning and G adversarial loss) while preserving the SiD coefficient. This experiment cannot establish that teacher supervision prevents GAN collapse without an adversarial-only control. External reported scores use incompletely aligned settings and are contextual, not controlled comparisons.

Protocol: [frozen recipe](FROZEN_PROTOCOL.md), [implementation/deviations](IMPLEMENTATION.md), [source identity](../resources/campaign_source.json).
