# Flow-SiD feasibility handoff: completed 1–4 NFE

Does adding adversarial training improve this bounded pixel-flow distillation recipe? **It lowers FID at 1 NFE and raises FID at 2, 3 and 4 NFE**, at every matched checkpoint and on the declared confirmation draw. Many samples retain recognizable class cues and variation, but substantial anatomy errors remain. One scratch teacher and all eight students are complete; the campaign stopped on October 4, 2026 (Asia/Seoul). Program 2 remains postponed.

Local development FID uses 3,020 fixed train-derived reference images, 20 generated images per class, selection seed 1729 and confirmation seed 271828 without reselection. Teacher training used 30k updates; every student completed 4k. S4 selects its 1k checkpoint; all other selected checkpoints are final. One training realization per arm; sampling checks are not training replication.

| Run | NFE | Teacher / G / fake / real-D presentations | Selected update | Selected / final FID | Confirmation FID | Selected median relative pixel diversity / flagged classes | Stage hours |
|---|---:|---|---:|---:|---:|---:|---:|
| T | 32 | 1,920,000 / 0 / 0 / 0 | 30,000 | 37.2526 / 37.2526 | — | 0.661 / 0 | 9.250 |
| S1 | 1 | 0 / 256,000 / 256,000 / 0 | 4,000 | 66.4564 / 66.4564 | 67.0671 | 0.928 / 0 | 4.059 |
| A1 | 1 | 0 / 256,000 / 256,000 / 256,000 | 4,000 | 52.4695 / 52.4695 | 52.3536 | 0.945 / 0 | 4.471 |
| S2 | 2 | 0 / 256,000 / 256,000 / 0 | 4,000 | 36.6304 / 36.6304 | 36.5497 | 1.054 / 0 | 4.504 |
| A2 | 2 | 0 / 256,000 / 256,000 / 256,000 | 4,000 | 44.9048 / 44.9048 | 45.6651 | 1.024 / 0 | 4.885 |
| S3 | 3 | 0 / 256,000 / 256,000 / 0 | 4,000 | 35.8246 / 35.8246 | 36.0900 | 1.015 / 0 | 4.932 |
| A3 | 3 | 0 / 256,000 / 256,000 / 256,000 | 4,000 | 43.8224 / 43.8224 | 43.6696 | 1.131 / 0 | 5.266 |
| S4 | 4 | 0 / 256,000 / 256,000 / 0 | 1,000 | 35.0944 / 35.6887 | 34.8934 | 1.082 / 0 | 5.272 |
| A4 | 4 | 0 / 256,000 / 256,000 / 256,000 | 4,000 | 45.8363 / 45.8363 | 44.6908 | 1.121 / 0 | 5.714 |

Matched 1k → 2k → 4k FIDs:

- S1: 85.6491 → 77.4692 → 66.4564; A1: 61.6596 → 55.6283 → 52.4695.
- S2: 40.0471 → 37.6426 → 36.6304; A2: 56.5236 → 60.9626 → 44.9048.
- S3: 41.7539 → 37.3777 → 35.8246; A3: 60.1346 → 59.4129 → 43.8224.
- S4: 35.0944 → 36.8754 → 35.6887; A4: 54.5247 → 55.8006 → 45.8363.

A1's advantage narrows with training. A2/A3/A4 recover substantially after 2k but still trail matched final SiD by 8.2744 / 7.9979 / 10.1476 FID points. Selected-checkpoint confirmation gaps are +9.1154 / +7.5795 / +9.7975. S4 does not improve beyond its first measured checkpoint; this does not prove convergence or overtraining. Its selected model received 64,000 G and fake presentations each, while all 256,000 presentations and the full 4k cost are charged. Same-NFE initialization FID was **not measured**; the teacher's 32-call result is not that control.

Actual presentations by ladder position over the full horizon are identical within each S/A pair:

| NFE | G presentations by position | Fake presentations by position |
|---:|---|---|
| 1 | 256,000 | 256,000 |
| 2 | 127,040 / 128,960 | 128,640 / 127,360 |
| 3 | 83,328 / 85,696 / 86,976 | 88,832 / 83,648 / 83,520 |
| 4 | 62,784 / 63,040 / 65,152 / 65,024 | 64,960 / 64,320 / 60,864 / 65,856 |

SiD's FID decreases markedly from one to two calls and only modestly thereafter. SiDA's confirmation FID is lowest at three calls. Cross-NFE seeds, per-position exposure and compute differ, so these are observed budget outcomes rather than an isolated inference-call effect.

The teacher is charged once: **48.354 total run-stage hours**, or **48.837 hours including recorded setup and ancillary checks**. These are summed task durations, not end-to-end elapsed wall time or measured GPU-active time. All nine training stages completed in one attempt with no recovery. The 67 disposable qualification updates, including failed/replayed probes, were charged to setup or subsequent qualification and discarded. Every teacher, student and full fake-score/discriminator framework has 61,823,875 parameters; the shared discriminator adds no parameters. Peak allocated/reserved GPU memory was **15.768 / 16.169 GB**; maximum sampled device usage was **16.247 GB**. Worker host high-water memory was **8.299 GB**; an aggregate host high-water mark including data-loader and archiver processes was not instrumented. [Full measurements](RESULTS.md) and [machine-readable results](../resources/campaign_results.json).

Broad unblinded review covered fixed samples 00/01 from all 151 classes for each selected pair, plus the preselected six-sample panels and confirmation galleries. Both methods retain many class cues and distinct compositions. A1 often improves outlines over noisy S1 outputs. A2/A3/A4 sometimes sharpen contours but also simplify or change class structure. Gyarados, Metapod, Lapras and Scyther expose persistent failures; neither the FID benefit at one call nor higher pixel variance at three/four calls establishes uniform semantic improvement. All 35 campaign evaluations had zero pixel-collapse flags and no exact within-class duplicate images. These diagnostics and same-class 16×16 pixel neighbors cannot establish semantic coverage or rule out near-memorization.

The following classes were chosen before the first adversarial result. Panels show six fixed-index samples with matched noise within each S/A pair; noise differs across NFE. S4 is labeled at its selected 1k checkpoint. Images are unchanged evaluation outputs.

![Eevee: useful class cues and variation](../artifacts/report/comparisons/class_Eevee_through_nfe4.png)

![Gyarados: persistent anatomy failures](../artifacts/report/comparisons/class_Gyarados_through_nfe4.png)

![Metapod: incorrect class-specific shape despite pixel variation](../artifacts/report/comparisons/class_Metapod_through_nfe4.png)

Exact training source identity: `49ae213d200a0f1253dafea2d382b114534e5e31263fd5943fd4e4aa9ad028f1` (original commit `3b1502c467c7a6c82fd0da21398dd21b66beeb54`). Canonical configuration: `28c371ef0253453e7f651da593865158765529cf9a9d53e4fda0b80c84d5f9ea`. Partition: `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`. Selected teacher checkpoint: `8bafcc5cb0e5d1e0f7a644fe6789640ccda9b513813633bc27a40eb81d17b3cf`.

The [passed campaign audit](../resources/audit_through_nfe4.json) contains every selected checkpoint and export hash and verifies all 4,000 common training draws plus all evaluation draws within each pair. [Actual-export checks](../resources/export_verification_through_nfe4.json) confirm independent loading, bitwise equality to selected weights, only student/sampler state, and exactly 1/2/3/4 calls. Inference sources, category map, runtime manifest and all eight checkpoints are under `artifacts/exports/` outside Git. All [27 declared candidate archives](../resources/evaluated_weights_inventory.json) were hash-verified; that inventory also records the auxiliary CPU watcher's terminal status 143, with no lost archive or training replay. The [1–2 NFE milestone](../resources/milestone_nfe2.json) remains preserved.

Material differences from Sangwoo's reported EDM setup include our weak scratch flow teacher, direct velocity prediction with mean-reduced flow losses, effective batch 64 rather than reported 128, fresh-noise flow ladders, independent phase-position draws, no geometric ADA or fake-score reset, and a 4k student horizon. We match the reported U-Net family and scale, not an unavailable exact configuration. Hardware, partition and evaluation alignment are incomplete; his reported scores are context rather than a controlled benchmark. [Full recipe/deviations](IMPLEMENTATION.md).

One future test worth considering is a **matched two-call continuation from 4k to 8k**, preserving this teacher and recipe, with predeclared 6k/8k evaluations and the same confirmation rule. A shrinking A−S gap with maintained class structure would support delayed adversarial adaptation; a persistent gap or worsening structure would weaken that explanation. The cost extrapolated from this pair is roughly another 9.4 stage hours on this device. This is **a proposal only, outside the completed assignment**. This pilot has no adversarial-only control and cannot prove that teacher supervision prevents GAN collapse. No external message or page edit has been sent.
