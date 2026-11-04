# TA-003 CycleGAN — completed

Completed **20,000 scratch-training updates** on the frozen A1 domains, then generated **604 diagnostic translations** (302 per direction), plus 604 cycle and 604 identity controls. All five required review milestones were retained. This is completion of the fixed schedule, not evidence of convergence.

| Direction | Median appearance-score shift | Median RGB RMS change |
|---|---:|---:|
| D0 → D1 | +2.9343 | 0.15257 |
| D1 → D0 | −3.0837 | 0.17683 |

RMS uses raw generated RGB mapped to [0,1]. Appearance descriptors use quantized PNGs and the frozen class-centered PCA. The score is a statistical appearance axis, not a style or quality score. Outputs moved closer to their class’s target optimization-score mean in 300/302 forward and 302/302 reverse cases. Per-class target descriptor distributions and nearest-target pixel distances are also recorded.

**Small unblinded AI review:** all eight representative sheets (64 sources across 16 categories, all five milestones) and the separately selected eight largest-RMS tails were directly inspected. Representative D0→D1 outputs often preserve coarse pose/layout while adding dark contours, grain and stronger shading; some replace pale backgrounds with black or introduce bars/halos. D1→D0 often softens contours and shading, but can wash out markings and replace black backgrounds with bright white. Fine anatomy/identity judgments remain uncertain for small or obscured details.

The selected tails—five forward, three reverse; one overlaps the representatives—show severe background inversion, halos and lost/corrupted detail (RGB RMS 0.732–0.787). These are selected failures, not an incidence estimate. Median cycle RMS was only 0.06257/0.06570, illustrating that close cycle reconstruction does not ensure a faithful translated image. No exact duplicate final images or out-of-range raw output values were found; that does not establish semantic diversity or correctness.

GPU qualification and checkpoint recovery passed exactly. Production reset after qualification; all 64 quantized review regenerations matched. Inputs remained unchanged and diagnostics were excluded from optimization. Framework: **17,001,352 parameters**; peak allocated/reserved: **2.197/2.466 GB**; total GPU time: **3,706.565 seconds (1.030 hours)**. GPU lease released; no retries, external weights or package changes.

Evidence: `result.json`, `eda/aggregate.json`, `eda/per_image.csv`, `visual_review.json`, `gallery.html`, `generation_verification.json`, `gpu_ledger.json`, and `output_hashes.json`.

These A1 diagnostics influenced domain discovery and are not independent validation. No class-accuracy, FID or downstream augmentation-usefulness claim is supported. Primary cross-method comparison is D0→D1; CycleGAN’s bidirectional cycle/identity objectives differ from CUT. The next research question is whether appearance shifts can be separated from background replacement and fine-content damage; no successor run was launched.
