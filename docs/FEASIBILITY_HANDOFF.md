# Flow-SiD feasibility handoff: completed 1–2 NFE

Does adding adversarial training improve this bounded pixel-flow distillation recipe? **At 1 NFE it lowers FID; at 2 NFE it raises FID**, with both directions surviving the declared second sampling seed. The adaptation produces useful class cues and variation, but substantial anatomy errors remain. S3/A3/S4/A4 are continuing under the same frozen protocol; Program 2 remains postponed.

Local development FID uses 3,020 fixed reference images, 20 generated images per class, selection seed 1729 and confirmation seed 271828. All five selected checkpoints are their final checkpoints. One training realization per arm; these sampling checks are not training replication.

| Run | NFE | Teacher / G / fake / real-D presentations | Selected / final FID | Confirmation FID | Median relative pixel diversity / flagged classes | Stage hours |
|---|---:|---|---:|---:|---:|---:|
| T, 30k | 32 | 1,920,000 / 0 / 0 / 0 | 37.2526 / 37.2526 | — | 0.661 / 0 | 9.250 |
| S1, 4k | 1 | 0 / 256,000 / 256,000 / 0 | 66.4564 / 66.4564 | 67.0671 | 0.928 / 0 | 4.059 |
| A1, 4k | 1 | 0 / 256,000 / 256,000 / 256,000 | 52.4695 / 52.4695 | 52.3536 | 0.945 / 0 | 4.471 |
| S2, 4k | 2 | 0 / 256,000 / 256,000 / 0 | 36.6304 / 36.6304 | 36.5497 | 1.054 / 0 | 4.504 |
| A2, 4k | 2 | 0 / 256,000 / 256,000 / 256,000 | 44.9048 / 44.9048 | 45.6651 | 1.024 / 0 | 4.885 |

Matched 1k → 2k → 4k FIDs:

- S1: 85.6491 → 77.4692 → 66.4564; A1: 61.6596 → 55.6283 → 52.4695.
- S2: 40.0471 → 37.6426 → 36.6304; A2: 56.5236 → 60.9626 → 44.9048.

A1's advantage narrows with training. A2 recovers substantially after 2k but still trails S2 by 8.2744 final-FID points and 9.1154 confirmation points. Same-NFE initialization FID was **not measured**; the teacher's 32-call result is not that control. Two-call G presentations by position are 127,040 / 128,960, and fake presentations are 128,640 / 127,360, identical within the pair. Cross-NFE seeds, per-position exposure and compute differ.

The teacher is charged once: **27.170 total run-stage hours**, or **27.632 hours including recorded setup and ancillary checks** through this milestone. These are summed task durations, not end-to-end elapsed wall time or measured GPU-active time. Each complete learned framework has 61,823,875 parameters; the shared discriminator adds no parameters. Across these runs, peak allocated/reserved GPU memory is 15.765/16.169 GB, maximum sampled device usage is 16.247 GB, and host high-water memory is 6.882 GB. No training recovery was needed. [Full measurements and identities](../resources/milestone_nfe2.json).

The broad unblinded review covered fixed samples 00/01 from all 151 classes. Both methods retain many recognizable class cues. A1 often improves outlines over noisy S1 outputs; A2 often shifts toward flatter or simplified forms. Neither result is uniform semantic improvement. Gyarados, Metapod, Lapras and Machamp expose structural failures. Zero pixel flags and same-class 16×16 pixel neighbors cannot establish semantic coverage or rule out near-memorization.

These three classes were chosen before the first adversarial result. Panels show six fixed-index samples with matched noise within each S/A pair; noise differs across NFE. Images are unchanged evaluation outputs.

![Eevee: useful class cues and variation](../artifacts/report/comparisons/class_Eevee_through_nfe2.png)

![Gyarados: persistent anatomy failures](../artifacts/report/comparisons/class_Gyarados_through_nfe2.png)

![Metapod: incorrect class-specific shape despite pixel variation](../artifacts/report/comparisons/class_Metapod_through_nfe2.png)

Exact training source identity: `49ae213d200a0f1253dafea2d382b114534e5e31263fd5943fd4e4aa9ad028f1` (original commit `3b1502c467c7a6c82fd0da21398dd21b66beeb54`). Canonical configuration hash: `28c371ef0253453e7f651da593865158765529cf9a9d53e4fda0b80c84d5f9ea`. Partition hash: `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`. The [passed campaign audit](../resources/audit_through_nfe2.json) contains every selected checkpoint and export hash, and verifies all 4,000 common training draws plus all evaluation draws within each pair. [Actual-export checks](../resources/export_verification_through_nfe2.json) confirm independent loading, bitwise equality to selected weights, only student/sampler state, and exactly 1 or 2 calls. Inference sources and checkpoints are under `artifacts/exports/` outside Git.

Material differences from Sangwoo's reported EDM setup include our weak scratch flow teacher, direct velocity prediction with mean-reduced flow losses, effective batch 64 rather than reported 128, fresh-noise flow ladders, independent phase-position draws, no geometric ADA or fake-score reset, and a 4k student horizon. We match the reported U-Net family and scale, not an unavailable exact configuration. Hardware, partition and evaluation alignment are incomplete; his reported scores are context rather than a controlled benchmark. [Full recipe/deviations](IMPLEMENTATION.md).

One future test worth considering is a **matched two-call continuation from 4k to 8k**, preserving this teacher and recipe, with predeclared 6k/8k evaluations and the same confirmation rule. A shrinking A−S gap with maintained class structure would support delayed adversarial adaptation; a persistent gap or worsening structure would weaken that explanation. This would cost roughly another pair-stage budget (~9.4 hours on this device), and is **a proposal only, outside the current assignment**. This pilot has no adversarial-only control and cannot prove that teacher supervision prevents GAN collapse. No external message or page edit has been sent.
