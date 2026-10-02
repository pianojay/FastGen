# Flow SiD/SiDA campaign results

Local development evaluation: 3,020 fixed train-derived reference images and 20 generated images per class. Selection seed 1729; confirmation seed 271828 without reselection. One training realization per arm.

| Run | State / completed updates | Selected update | NFE | Selection FID | Final FID | Confirmation FID | Pixel flags / 151 | Stage hours |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T | complete / 30000 | 30000 | 32 | 37.2526 | 37.2526 | — | 0 | 9.250 |
| S1 | complete / 4000 | 4000 | 1 | 66.4564 | 66.4564 | 67.0671 | 0 | 4.059 |
| A1 | training / 1022 | 1000 | 1 | 61.6596 | — | — | 0 | 1.139 |
| S2 | not started / 0 | — | 2 | — | — | — | — | 0.000 |
| A2 | not started / 0 | — | 2 | — | — | — | — | 0.000 |
| S3 | not started / 0 | — | 3 | — | — | — | — | 0.000 |
| A3 | not started / 0 | — | 3 | — | — | — | — | 0.000 |
| S4 | not started / 0 | — | 4 | — | — | — | — | 0.000 |
| A4 | not started / 0 | — | 4 | — | — | — | — | 0.000 |

Active-stage hours include elapsed work; completed-stage hours use the external allowance ledger. Stage cost includes optimization, evaluation, diagnostics, checkpoint I/O, startup and any failed/replayed attempts.

| NFE | Selection ΔFID (A−S) | Final ΔFID (A−S) | Confirmation ΔFID (A−S) | Δ median relative pixel diversity (A−S) |
|---:|---:|---:|---:|---:|
| 1 | -4.7968 | — | — | -0.1005 |
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| 4 | — | — | — | — |

| NFE | Matched updates | G and fake presentations each | SiD FID | SiDA FID | ΔFID (A−S) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 64,000 | 85.6491 | 61.6596 | -23.9895 |
| 1 | 2000 | 128,000 | 77.4692 | — | — |
| 1 | 4000 | 256,000 | 66.4564 | — | — |

| Run | Teacher presentations | G / fake / real D presentations | G presentations by ladder position |
|---|---:|---:|---|
| T | 1,920,000 | 0 / 0 / 0 | — |
| S1 | 0 | 256,000 / 256,000 / 0 | 256,000 |
| A1 | 0 | 65,408 / 65,408 / 65,408 | 65,408 |
| S2 | — | — | — |
| A2 | — | — | — |
| S3 | — | — | — |
| A3 | — | — | — |
| S4 | — | — | — |
| A4 | — | — | — |

Same-NFE initial-baseline FID: **unavailable**. The teacher’s 32-call evaluation is context, and the unrelated untrained plumbing FID is not a scientific control. Equal student cycles match total G/fake exposure; per-position exposure and compute differ across NFE. Interpretation follows [the feasibility focus](FEASIBILITY_FOCUS.md), written after the first S1 result; it is not a retrospective preregistered success threshold.

| Run | Optimization hours | Generation + FID hours | Allocated / reserved GB | Sampled device GB | Host RSS / PSS / HWM GB | Attempts |
|---|---:|---:|---:|---:|---:|---:|
| T | 8.399 | 0.467 | 6.016 / 6.392 | 6.470 | 4.881 / 4.776 / 4.953 | 1 |
| S1 | 3.917 | 0.058 | 15.765 / 16.108 | 16.186 | 4.859 / 4.755 / 5.834 | 1 |
| A1 | 1.102 | 0.015 | 15.761 / 15.926 | 16.209 | 4.525 / 4.421 / 4.525 | 1 |
| S2 | — | — | — | — | — | 0 |
| A2 | — | — | — | — | — | 0 |
| S3 | — | — | — | — | — | 0 |
| A3 | — | — | — | — | — | 0 |
| S4 | — | — | — | — | — | 0 |
| A4 | — | — | — | — | — | 0 |

Recorded pipeline cost: **14.902 hours** = 14.449 run-stage hours (teacher once) + 0.445 pre-teacher setup wall hours + 0.008 subsequent qualification/export/analysis hours. Setup includes overlapping installation, preprocessing and disposable probes; those components are not added twice. Pre-environment read-only review was not instrumented. The finite component allowances sum to 152 hours.

Every teacher, student and full fake-score/discriminator framework has **61,823,875 learned parameters**; the shared encoder’s pooled discriminator output adds zero parameters. EMA storage and optimizer states are included in measured simultaneous memory. The strict allocated-memory limit is 20,000,000,000 bytes. Device usage is sampled; allocated/reserved and host HWM are high-water measurements.

Pixel flags use the predeclared within-class duplication/diversity rule. They do not certify semantic recognition. Nearest optimization examples are same-class neighbors in 16×16 pixel space, not a perceptual test.

- **A1 at 1,000 (provisional):** Some class contours are clearer than the matched S1 checkpoint, but quality is mixed. The Gyarados row contains compact blue figures with little expected serpentine structure; Metapod repeatedly uses upright rounded forms rather than clear angular cocoons. Noisy texture and wrong anatomy remain elsewhere. Zero pixel-collapse flags do not rule out restricted structural variation or class confusion. The lower pooled FID is provisional evidence, not a complete quality/diversity conclusion. [Gallery](../runs/campaign/A1/eval_001000/gallery_fixed_0.png)
- **S1 at 1,000 (provisional):** The 16 fixed classes retain varied compositions and recognizable broad class cues, but noise, edge artifacts and less coherent anatomy/detail are apparent relative to the selected teacher. Pixel-diversity flags are 0/151; this is not a semantic-coverage score. This checkpoint is provisional until the fixed student horizon finishes. [Gallery](../runs/campaign/S1/eval_001000/gallery_fixed_0.png)
- **S1 at 2,000 (provisional):** Fixed-noise compositions remain varied, with recognizable class cues in many displayed samples. Noisy textures, blurred or fragmented anatomy and inconsistent outlines remain. Higher pixel diversity does not establish improved semantic diversity. All-class pixel-collapse flags remain 0/151; selection is still provisional. [Gallery](../runs/campaign/S1/eval_002000/gallery_fixed_0.png)
- **S1 confirmation seed:** Different noise produces visibly different compositions in the 16 fixed classes. Gengar, Eevee and Nidorina retain class cues; Gyarados and Metapod still show substantial structural errors, and image texture remains noisy. Confirmation FID is 67.0671 with 0/151 pixel-collapse flags. [Gallery](../runs/campaign/S1/confirmation/gallery_fixed_0.png)
- **Selected S1 at 4,000:** The fixed-class gallery has varied poses/compositions and recognizable class cues in many samples; outlines and faces are clearer than at 1k. Grain/noise, merged parts and incorrect anatomy remain, especially among more complex shapes. This review covers 16 fixed classes, not an all-class recognition score; pixel-collapse flags are 0/151. [Gallery](../runs/campaign/S1/eval_004000/gallery_fixed_0.png)
- **Selected T at 30,000:** The fixed 16-class gallery shows varied compositions and recognizable class cues, with clearer outlines and details than earlier checkpoints. Anatomy and some class-specific shapes remain imperfect, notably in Gyarados, Metapod and Scyther. No severe common-output collapse is apparent in this subset. All-class pixel diagnostics flag 0/151 classes at both 20k and 30k and at the selected checkpoint. Pixel diversity does not establish all-class semantic coverage. [Gallery](../runs/campaign/T/eval_030000/gallery_fixed_0.png)
- **T at 10,000:** Six fixed-noise generations in each of 16 fixed classes show varied color and composition with broad class cues, but distorted anatomy and sometimes ambiguous class features. No severe common-output collapse is apparent in this displayed subset; all-class pixel flags are 0/151. [Gallery](../runs/campaign/T/eval_010000/gallery_fixed_0.png)
- **T at 20,000:** Class features are clearer in the fixed gallery, particularly Eevee, Gengar, Kabuto and Starmie, and compositions remain varied. Limb/body distortions and ambiguous shapes remain visible in classes such as Gyarados, Metapod and Scyther. This is a qualitative review of 16 fixed classes, not an all-class recognition score; all-class pixel flags are 0/151. [Gallery](../runs/campaign/T/eval_020000/gallery_fixed_0.png)

[Quality versus NFE](../artifacts/report/quality_vs_nfe.svg) · [Quality versus stand-alone pipeline cost](../artifacts/report/quality_vs_pipeline_cost.svg). Plots use confirmation FID; these are sampling-seed checks, not independent training repetitions.

Raw loss/gradient/timing records and SVG trends, all-class diversity, saturation, same-class neighbor audits and fixed-noise galleries are retained under `runs/campaign/<run>/`. Completed student exports are written under `artifacts/exports/`; minimal inference sources are packaged at the milestones.

Interpretation is limited to this single teacher and training realization. The treatment adds the full adversarial package (real/fake D learning and G adversarial loss) while preserving the SiD coefficient. This experiment cannot establish that teacher supervision prevents GAN collapse without an adversarial-only control. External reported scores use incompletely aligned settings and are contextual, not controlled comparisons.

Protocol: [frozen recipe](FROZEN_PROTOCOL.md), [implementation/deviations](IMPLEMENTATION.md), [source identity](../resources/campaign_source.json).
