# Flow SiD/SiDA campaign results

Local development evaluation: 3,020 fixed train-derived reference images and 20 generated images per class. Selection seed 1729; confirmation seed 271828 without reselection. One training realization per arm.

| Run | State / completed updates | Selected update | NFE | Selection FID | Final FID | Confirmation FID | Pixel flags / 151 | Stage hours |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T | complete / 30000 | 30000 | 32 | 37.2526 | 37.2526 | — | 0 | 9.250 |
| S1 | complete / 4000 | 4000 | 1 | 66.4564 | 66.4564 | 67.0671 | 0 | 4.059 |
| A1 | complete / 4000 | 4000 | 1 | 52.4695 | 52.4695 | 52.3536 | 0 | 4.471 |
| S2 | complete / 4000 | 4000 | 2 | 36.6304 | 36.6304 | 36.5497 | 0 | 4.504 |
| A2 | complete / 4000 | 4000 | 2 | 44.9048 | 44.9048 | 45.6651 | 0 | 4.885 |
| S3 | complete / 4000 | 4000 | 3 | 35.8246 | 35.8246 | 36.0900 | 0 | 4.932 |
| A3 | training / 2019 | 2000 | 3 | 59.4129 | — | — | 0 | 2.653 |
| S4 | not started / 0 | — | 4 | — | — | — | — | 0.000 |
| A4 | not started / 0 | — | 4 | — | — | — | — | 0.000 |

Active-stage hours include elapsed work; completed-stage hours use the external allowance ledger. Stage cost includes optimization, evaluation, diagnostics, checkpoint I/O, startup and any failed/replayed attempts.

| NFE | Selection ΔFID (A−S) | Final ΔFID (A−S) | Confirmation ΔFID (A−S) | Δ median relative pixel diversity (A−S) |
|---:|---:|---:|---:|---:|
| 1 | -13.9869 | -13.9869 | -14.7135 | +0.0168 |
| 2 | +8.2744 | +8.2744 | +9.1154 | -0.0292 |
| 3 | +23.5884 | — | — | -0.0340 |
| 4 | — | — | — | — |

| NFE | Matched updates | G and fake presentations each | SiD FID | SiDA FID | ΔFID (A−S) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 64,000 | 85.6491 | 61.6596 | -23.9895 |
| 1 | 2000 | 128,000 | 77.4692 | 55.6283 | -21.8409 |
| 1 | 4000 | 256,000 | 66.4564 | 52.4695 | -13.9869 |
| 2 | 1000 | 64,000 | 40.0471 | 56.5236 | +16.4766 |
| 2 | 2000 | 128,000 | 37.6426 | 60.9626 | +23.3201 |
| 2 | 4000 | 256,000 | 36.6304 | 44.9048 | +8.2744 |
| 3 | 1000 | 64,000 | 41.7539 | 60.1346 | +18.3807 |
| 3 | 2000 | 128,000 | 37.3777 | 59.4129 | +22.0352 |
| 3 | 4000 | 256,000 | 35.8246 | — | — |

| Run | Teacher presentations | G / fake / real D presentations | G presentations by ladder position |
|---|---:|---:|---|
| T | 1,920,000 | 0 / 0 / 0 | — |
| S1 | 0 | 256,000 / 256,000 / 0 | 256,000 |
| A1 | 0 | 256,000 / 256,000 / 256,000 | 256,000 |
| S2 | 0 | 256,000 / 256,000 / 0 | 127,040, 128,960 |
| A2 | 0 | 256,000 / 256,000 / 256,000 | 127,040, 128,960 |
| S3 | 0 | 256,000 / 256,000 / 0 | 83,328, 85,696, 86,976 |
| A3 | 0 | 129,216 / 129,216 / 129,216 | 40,960, 44,544, 43,712 |
| S4 | — | — | — |
| A4 | — | — | — |

Same-NFE initial-baseline FID: **unavailable**. The teacher’s 32-call evaluation is context, and the unrelated untrained plumbing FID is not a scientific control. Equal student cycles match total G/fake exposure; per-position exposure and compute differ across NFE. Interpretation follows [the feasibility focus](FEASIBILITY_FOCUS.md), written after the first S1 result; it is not a retrospective preregistered success threshold.

| Run | Optimization hours | Generation + FID hours | Allocated / reserved GB | Sampled device GB | Host RSS / PSS / HWM GB | Attempts |
|---|---:|---:|---:|---:|---:|---:|
| T | 8.399 | 0.467 | 6.016 / 6.392 | 6.470 | 4.881 / 4.776 / 4.953 | 1 |
| S1 | 3.917 | 0.058 | 15.765 / 16.108 | 16.186 | 4.859 / 4.755 / 5.834 | 1 |
| A1 | 4.318 | 0.058 | 15.761 / 16.131 | 16.209 | 5.908 / 5.804 / 6.882 | 1 |
| S2 | 4.329 | 0.076 | 15.765 / 16.169 | 16.247 | 5.598 / 5.494 / 6.571 | 1 |
| A2 | 4.719 | 0.076 | 15.762 / 16.167 | 16.245 | 5.455 / 5.351 / 6.428 | 1 |
| S3 | 4.746 | 0.095 | 15.767 / 16.110 | 16.188 | 7.361 / 7.257 / 8.299 | 1 |
| A3 | 2.565 | 0.046 | 15.762 / 16.133 | 16.211 | 5.434 / 5.331 / 6.039 | 1 |
| S4 | — | — | — | — | — | 0 |
| A4 | — | — | — | — | — | 0 |

Recorded pipeline cost: **35.217 hours** = 34.755 run-stage hours (teacher once) + 0.445 pre-teacher setup wall hours + 0.017 subsequent qualification/export/analysis hours. Setup includes overlapping installation, preprocessing and disposable probes; those components are not added twice. Pre-environment read-only review was not instrumented. This accounting sums run-stage and ancillary task durations; CPU analysis can overlap training, so it is neither end-to-end elapsed wall time nor measured GPU-active time. The finite component allowances sum to 152 hours.

Every teacher, student and full fake-score/discriminator framework has **61,823,875 learned parameters**; the shared encoder’s pooled discriminator output adds zero parameters. EMA storage and optimizer states are included in measured simultaneous memory. The strict allocated-memory limit is 20,000,000,000 bytes. Device usage is sampled; allocated/reserved and host HWM are high-water measurements.

Pixel flags use the predeclared within-class duplication/diversity rule. They do not certify semantic recognition. Nearest optimization examples are same-class neighbors in 16×16 pixel space, not a perceptual test.

- **A1 at 1,000 (provisional):** Some class contours are clearer than the matched S1 checkpoint, but quality is mixed. The Gyarados row contains compact blue figures with little expected serpentine structure; Metapod repeatedly uses upright rounded forms rather than clear angular cocoons. Noisy texture and wrong anatomy remain elsewhere. Zero pixel-collapse flags do not rule out restricted structural variation or class confusion. The lower pooled FID is provisional evidence, not a complete quality/diversity conclusion. [Gallery](../runs/campaign/A1/eval_001000/gallery_fixed_0.png)
- **A1 at 2,000 (provisional):** Pooled FID improves further, with recognizable contours in classes such as Eevee and Gengar. Strong semantic/shape errors persist: Gyarados samples remain compact blue figures rather than serpentine creatures, and Metapod uses repeated upright rounded forms. Other classes still have wrong anatomy and texture artifacts. Median pixel diversity is close to matched S1, but that does not resolve structural coverage. Selection remains provisional. [Gallery](../runs/campaign/A1/eval_002000/gallery_fixed_0.png)
- **A2 at 1,000 (provisional):** FID is 56.5236 versus matched S2 40.0471, a 16.4766-point disadvantage at 64k presentations per phase. The fixed gallery has varied compositions and class cues, but grainy textures and substantial structural errors: simplified blue Machamp bodies, incomplete Gyarados anatomy, rounded or worm-like Metapod shapes and altered Victreebel structures. Pixel flags remain 0/151; larger pixel diversity does not establish better semantic variation. This reverses the one-call FID direction at the first checkpoint; it is provisional, with no recipe change. [Gallery](../runs/campaign/A2/eval_001000/gallery_fixed_0.png)
- **A2 at 2,000 (provisional):** FID worsens from 56.5236 to 60.9626 while matched S2 improves to 37.6426; the A−S disadvantage widens to 23.3201. The fixed gallery still has varied compositions, but altered class anatomy persists: Gyarados frequently becomes short blue creatures, Metapod uses rounded or worm-like shapes, and Machamp/Victreebel structures are inconsistent. Cleaner surfaces in some examples do not establish better class fidelity. Pixel flags remain 0/151 and median relative pixel diversity is 1.0495, nearly unchanged from A2 at 1k; these diagnostics miss semantic errors. The predeclared horizon and recipe remain unchanged. [Gallery](../runs/campaign/A2/eval_002000/gallery_fixed_0.png)
- **A3 at 1,000 (provisional):** FID is 60.1346 versus matched S3 41.7539, an 18.3807-point disadvantage at 64k presentations per phase. Fixed-noise compositions vary, but class structure is often simplified or altered: Scyther acquires large rounded faces, Metapod appears as rounded or worm-like forms, and Gyarados remains anatomically unclear. Dewgong includes incomplete bodies. Pixel flags are 0/151 and median relative pixel diversity is 1.0516; these do not measure semantic fidelity. This is provisional evidence under the unchanged recipe. [Gallery](../runs/campaign/A3/eval_001000/gallery_fixed_0.png)
- **A3 at 2,000 (provisional):** FID changes from 60.1346 to 59.4129, while matched S3 reaches 37.3777: A−S is +22.0352. The fixed gallery preserves distinct compositions but simplified or incorrect structures persist in Scyther, Metapod, Gyarados, Machamp and Dewgong examples. Median relative pixel diversity is 0.9814 and flags are 0/151; neither establishes semantic fidelity. The 1k–2k FID change is small, with final quality still pending. [Gallery](../runs/campaign/A3/eval_002000/gallery_fixed_0.png)
- **Completed 1-NFE pair, broad review:** Broad unblinded review covered fixed sample indices 00/01 for every one of 151 classes, plus six samples in the preselected Eevee/Gyarados/Metapod panels and the confirmation fixed-class gallery. Many classes show distinct compositions and recognizable cues; A1 often has clearer outlines and less noisy surfaces. The improvement is not uniform: both arms have serious shape errors, and A1 sometimes simplifies or changes class structure. Examples include Charizard, Gyarados, Lapras, Metapod and Weedle. No all-class semantic-accuracy score was measured. Pixel diversity is similar between the arms; zero pixel flags do not certify semantic coverage or rule out near-memorization. [Gallery](../artifacts/report/comparisons/broad_pair_1_0.png)
- **Completed 2-NFE pair, broad review:** Broad unblinded review covered fixed sample indices 00/01 for all 151 classes, six samples in each preselected Eevee/Gyarados/Metapod panel, and the A2 confirmation gallery. Both arms retain recognizable cues and varied compositions in many samples, but anatomy errors remain widespread. A2 often shifts toward flatter or simplified forms; some individual examples look cleaner, while Gyarados, Metapod, Lapras and Machamp show clear structural failures. S2 also fails complex shapes. The poorer A2 FID persists at every matched checkpoint and the confirmation seed, despite substantial late recovery. Similar pixel-diversity summaries and zero flags do not establish semantic coverage or rule out near-memorization. [Gallery](../artifacts/report/comparisons/broad_pair_2_0.png)
- **S1 at 1,000 (provisional):** The 16 fixed classes retain varied compositions and recognizable broad class cues, but noise, edge artifacts and less coherent anatomy/detail are apparent relative to the selected teacher. Pixel-diversity flags are 0/151; this is not a semantic-coverage score. This checkpoint is provisional until the fixed student horizon finishes. [Gallery](../runs/campaign/S1/eval_001000/gallery_fixed_0.png)
- **S1 at 2,000 (provisional):** Fixed-noise compositions remain varied, with recognizable class cues in many displayed samples. Noisy textures, blurred or fragmented anatomy and inconsistent outlines remain. Higher pixel diversity does not establish improved semantic diversity. All-class pixel-collapse flags remain 0/151; selection is still provisional. [Gallery](../runs/campaign/S1/eval_002000/gallery_fixed_0.png)
- **S1 confirmation seed:** Different noise produces visibly different compositions in the 16 fixed classes. Gengar, Eevee and Nidorina retain class cues; Gyarados and Metapod still show substantial structural errors, and image texture remains noisy. Confirmation FID is 67.0671 with 0/151 pixel-collapse flags. [Gallery](../runs/campaign/S1/confirmation/gallery_fixed_0.png)
- **S2 at 1,000 (provisional):** The fixed 16-class gallery has varied compositions, class cues and relatively clear contours in examples including Eevee, Gengar and Scyther. Distorted limbs, merged anatomy and unclear class-specific shapes remain, including Gyarados and Metapod examples. Zero pixel-collapse flags are not a semantic-coverage measure. Compared across NFE budgets, inference calls, training seed/draws and per-position exposure differ; selection remains provisional. [Gallery](../runs/campaign/S2/eval_001000/gallery_fixed_0.png)
- **S2 at 2,000 (provisional):** FID improves from 40.0471 to 37.6426 on identical evaluation draws. The fixed gallery retains recognizable cues and distinct compositions, including Eevee and Gengar, but mixed or incorrect anatomy remains in Gyarados, Metapod, Machamp and other examples. The modest FID gain does not imply uniform visual improvement. Pixel flags remain 0/151; median relative pixel diversity rises from 0.9434 to 1.0433, which is not a semantic-coverage measure. Selection remains provisional. [Gallery](../runs/campaign/S2/eval_002000/gallery_fixed_0.png)
- **S2 confirmation seed:** Second-seed FID is 36.5497. Different noise gives visibly different poses, backgrounds and compositions in the fixed gallery; class-shape errors remain, including Gyarados, Machamp and Metapod. All-class pixel flags are 0/151 in both final draws. This does not certify semantic coverage or absence of near-memorization. [Gallery](../runs/campaign/S2/confirmation/gallery_fixed_0.png)
- **S3 at 1,000 (provisional):** FID is 41.7539 at three calls. The fixed gallery shows varied class cues and compositions, including Eevee, Gengar and Scyther, but incomplete bodies and distorted anatomy remain. Dewgong includes very thin or fragmentary shapes; Gyarados, Metapod and Pidgeot include substantial structural errors. Pixel flags are 0/151 and median relative pixel diversity is 1.0354. No same-NFE initialization baseline was measured; comparisons across NFE also change training draws and per-position exposure. Selection remains provisional. [Gallery](../runs/campaign/S3/eval_001000/gallery_fixed_0.png)
- **S3 at 2,000 (provisional):** FID improves from 41.7539 to 37.3777 on identical evaluation draws. Fixed-noise compositions remain varied, with recognizable Eevee/Gengar/Scyther cues and clearer contours in some samples. Dewgong still includes incomplete thin bodies; Gyarados, Metapod and Pidgeot retain shape and anatomy errors. Median relative pixel diversity is 0.9591, with 0/151 flags; its change is not a semantic-diversity score. Selection remains provisional. [Gallery](../runs/campaign/S3/eval_002000/gallery_fixed_0.png)
- **S3 confirmation seed:** Confirmation FID is 36.0900. The second noise draw produces varied poses and compositions, with recognizable cues in many samples, but malformed bodies and inconsistent class-specific shapes remain. Both final draws have 0/151 pixel flags and about 1.01 median relative pixel diversity; this is not a semantic-coverage score. [Gallery](../runs/campaign/S3/confirmation/gallery_fixed_0.png)
- **Selected A1 at 4,000:** Final/selected FID is 52.4695 and confirmation FID is 52.3536, compared with S1 66.4564 and 67.0671. The FID advantage persists at all three matched checkpoints but narrows from about 24 points at 1k to 14 points at 4k. Pixel-collapse flags are 0/151 in both final evaluation draws. Visual interpretation remains mixed; see the all-class pair review. [Gallery](../runs/campaign/A1/eval_004000/gallery_fixed_0.png)
- **Selected A2 at 4,000:** FID trajectory is 56.5236 → 60.9626 → 44.9048, with final checkpoint selected and confirmation FID 45.6651. The final A−S gap is +8.2744; confirmation gap is +9.1154. All-class pixel flags remain 0/151 in both final draws. Late recovery narrows the gap but does not reverse the two-call disadvantage. [Gallery](../runs/campaign/A2/eval_004000/gallery_fixed_0.png)
- **Selected S1 at 4,000:** The fixed-class gallery has varied poses/compositions and recognizable class cues in many samples; outlines and faces are clearer than at 1k. Grain/noise, merged parts and incorrect anatomy remain, especially among more complex shapes. This review covers 16 fixed classes, not an all-class recognition score; pixel-collapse flags are 0/151. [Gallery](../runs/campaign/S1/eval_004000/gallery_fixed_0.png)
- **Selected S2 at 4,000:** FID improves 40.0471 → 37.6426 → 36.6304 across 1k/2k/4k; the final checkpoint is selected. The 16 fixed classes retain distinct compositions and recognizable cues, with cleaner shapes in several samples but continuing merged or incorrect anatomy. Metapod varies in orientation and shape but often departs from its expected cocoon structure. Gyarados and Machamp remain difficult. This is a within-run trend, not a measured improvement over an unavailable same-NFE initialization baseline. [Gallery](../runs/campaign/S2/eval_004000/gallery_fixed_0.png)
- **Selected S3 at 4,000:** FID improves 41.7539 → 37.3777 → 35.8246 across the fixed 1k/2k/4k checkpoints, selecting the final checkpoint. The fixed 16-class gallery retains distinct compositions and recognizable cues; structural errors remain in Dewgong, Gyarados, Machamp, Metapod and Pidgeot examples. The small FID difference versus two-call SiD does not isolate an inference-call effect because seeds, per-position exposure and compute also differ. [Gallery](../runs/campaign/S3/eval_004000/gallery_fixed_0.png)
- **Selected T at 30,000:** The fixed 16-class gallery shows varied compositions and recognizable class cues, with clearer outlines and details than earlier checkpoints. Anatomy and some class-specific shapes remain imperfect, notably in Gyarados, Metapod and Scyther. No severe common-output collapse is apparent in this subset. All-class pixel diagnostics flag 0/151 classes at both 20k and 30k and at the selected checkpoint. Pixel diversity does not establish all-class semantic coverage. [Gallery](../runs/campaign/T/eval_030000/gallery_fixed_0.png)
- **T at 10,000:** Six fixed-noise generations in each of 16 fixed classes show varied color and composition with broad class cues, but distorted anatomy and sometimes ambiguous class features. No severe common-output collapse is apparent in this displayed subset; all-class pixel flags are 0/151. [Gallery](../runs/campaign/T/eval_010000/gallery_fixed_0.png)
- **T at 20,000:** Class features are clearer in the fixed gallery, particularly Eevee, Gengar, Kabuto and Starmie, and compositions remain varied. Limb/body distortions and ambiguous shapes remain visible in classes such as Gyarados, Metapod and Scyther. This is a qualitative review of 16 fixed classes, not an all-class recognition score; all-class pixel flags are 0/151. [Gallery](../runs/campaign/T/eval_020000/gallery_fixed_0.png)

[Quality versus NFE](../artifacts/report/quality_vs_nfe.svg) · [Quality versus stand-alone pipeline cost](../artifacts/report/quality_vs_pipeline_cost.svg). Plots use confirmation FID; these are sampling-seed checks, not independent training repetitions.

Raw loss/gradient/timing records and SVG trends, all-class diversity, saturation, same-class neighbor audits and fixed-noise galleries are retained under `runs/campaign/<run>/`. Completed student exports are written under `artifacts/exports/`; minimal inference sources are packaged at the milestones.

Interpretation is limited to this single teacher and training realization. The treatment adds the full adversarial package (real/fake D learning and G adversarial loss) while preserving the SiD coefficient. This experiment cannot establish that teacher supervision prevents GAN collapse without an adversarial-only control. External reported scores use incompletely aligned settings and are contextual, not controlled comparisons.

Protocol: [frozen recipe](FROZEN_PROTOCOL.md), [implementation/deviations](IMPLEMENTATION.md), [source identity](../resources/campaign_source.json).
