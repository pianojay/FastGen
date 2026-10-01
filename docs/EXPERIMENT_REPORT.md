# Experiment report: U-Net-first, teacher-free AFM

**Date:** October 1, 2026. **Disposition:** stopped by the user at 21:17 KST owing to strongly suspected conditional GAN mode collapse. GPU 0 was released; the GPU 1 SoFlow job was left running.

The experiment tested class-conditioned, 64×64 pixel-space generation in one generator call. A fresh U-Net128 generator and projection discriminator used the discrete AFM objective: relativistic adversarial losses, source-noise OT, finite-difference R1/R2 penalties, centering and gradient normalization. There was no teacher or pretrained training network. The fixed optimization partition contained 14,059 images; official validation was unused.

The [frozen recipe](../configs/baseline.json) used seed 20261001, FP32, batch 32, one D update per G update, AdamW at 1e-4 and EMA decay .999. OT decreased from .2 to .005 over the first 10,000 G updates. The planned horizon was 50,000 G/D cycles.

| Measure | Result at termination |
|---|---|
| Completed updates | 10,294 G + 10,294 D; 20.59% of the planned horizon |
| Completed-cycle exposure | 329,408 real presentations per stream; 658,816 total real and 658,816 generated |
| Parameters | G: 39,664,771; D: 9,918,657 |
| Measured TITAN RTX cost | 2.396 training-job hours; approximately 2.454 hours including qualification and its evaluation smoke |
| Recent throughput | 0.827 seconds per complete D/G cycle, last 1,000 cycles |
| Main-run memory | 7.522 GB peak allocated; 8.154 GB peak reserved; 8.382 GB maximum sampled device usage |
| Host memory | Maximum sampled RSS 2.016 GB; process-tree PSS 2.003 GB |
| Retained checkpoint | Complete training state at 10,000 updates; later 294 completed cycles retained in logs only |
| Trained development FID | Unmeasured; planned final evaluation and inference export were not reached |

Memory uses decimal GB. Completed-cycle counters exclude the interrupted next D phase; job time includes that partial work. [Final accounting and checkpoint hash](../runs/afm_unet128_50k_seed20261001/stop_summary.json) preserve the exact measurements.

**Findings.** The [10,000-update EMA panel](../runs/afm_unet128_50k_seed20261001/panels/step010000.png) shows nearly repeated templates across eight distinct Gaussian inputs within each of twelve displayed classes. Class-dependent colors and silhouettes emerged, but pose, layout and shape varied little, while substantial texture and outline defects remained. This is strong qualitative evidence of within-class mode collapse. Its scope is limited to the displayed samples; population diversity, memorization and the causal mechanism remain unmeasured.

All 10,294 logged cycles had finite metrics, with no numerical failure, resource failure or recovery. [Qualification](../artifacts/qualification_summary.json) passed six objective tests, exact next-cycle recovery, memory/parameter limits, protected-source checks and an independent one-call loader check. Those checks establish engineering behavior; they do not establish successful distribution learning. The earlier untrained FID smoke is solely a pipeline check. No trained FID or quantitative comparison with SoFlow is available.

**Conclusion and proposed next work.** This interrupted recipe did not demonstrate useful, diverse one-call generation. Logs, [loss curves](../runs/afm_unet128_50k_seed20261001/training_curves_final.svg), panels and the checkpoint are preserved; the stop is recorded as user-directed termination, with the original interrupt traceback retained.

1. Audit noise sensitivity, raw-generator versus EMA outputs, output saturation, and adversarial/OT/regularizer gradient balance against the pinned AFM reference. OT decay and G/D balance are hypotheses, not established causes.
2. After that audit, design one separately scoped change with predeclared within-class diversity diagnostics alongside development FID. No further training or evaluation was launched.
