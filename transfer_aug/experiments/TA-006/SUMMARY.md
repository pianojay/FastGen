# TA-006 StarGAN v2: appearance variation, preservation failure

**Completed:** 20,000 scratch iterations, batch8, final EMA; 5,436 outputs and all required EDA/reviews. Total **5.548 GPU-hours** including qualification/retry/generation. Peak allocated **12.704GB**, reserved19.541GB; GPU2 lease released. Framework **131,065,096 parameters including EMA**, outside the standard100M cap under the user's explicit exception.

The model learned both statistical appearance directions. It also produced substantial visible variation, but **did not establish identity-preserving augmentation**. References visibly control palette/background while faces, wing/flower patterns, mouths, limbs and body shapes often change. Latent variants frequently introduce unrelated colors and clutter. Self-style controls also distort details, exposing a reconstruction/preservation problem beyond cross-domain transfer.

Means below first average four variants per source, then302 sources per direction. RMS uses nominal[0,1] RGB; raw values may exceed that range. Clipped metrics use floating RGB before PNG quantization. Diversity is the mean six pairwise RMS distances, not semantic diversity.

| Direction/arm | PC1 shift | Source RMS raw/clipped | Diversity raw/clipped | Out-of-range channels |
|---|---:|---:|---:|---:|
| D0→D1 reference | +3.320 | .414/.413 | .384/.382 | 8.92% |
| D0→D1 latent | +3.488 | .403/.400 | .405/.400 | 7.97% |
| D1→D0 reference | −4.157 | .424/.422 | .277/.274 | 12.47% |
| D1→D0 latent | −3.213 | .384/.382 | .318/.314 | 5.78% |

Average clipping magnitudes were only .0014–.0029 for translations; clipping frequency alone is not a damage measure. Self-control clipped source RMS was .121(D0)/.163(D1). No exact duplicate excess appeared within the reported arms, but this cannot establish useful diversity.

Direct inspection covered all16 representative pages, two separate tail pages and four milestone pages. High-RMS tails frequently include background inversion; low-diversity tails still show color variation. Initial noise evolves into structured images, with persistent anatomical errors; no checkpoint was selected visually.

Checks passed: pinned mechanism parity, exact GPU checkpoint recovery,512 quantized replay outputs, finite outputs, PNG/raw agreement and unchanged data/split hashes. One pre-update qualification layout error was fixed with contiguous NCHW input; its3.81seconds remain charged.

**Guidance:** do not yet treat these outputs as acceptable real-data augmentation. The next question is an A1-only preservation constraint that protects anatomy/markings, not merely greater pixel variation. No A2/B, FID or SiDA-benefit test was performed; diagnostics are not independent validation.

Evidence: `gallery.html`, `visual_review.json`, `aggregate_metrics.json`, `final_verification.json`, `gpu_resource_ledger.json`, `result.json`.
