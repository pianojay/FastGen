# CycleGAN and CUT: appearance shifts learned, preservation unresolved

Both scratch-trained models completed their fixed schedules and learned the intended statistical appearance shift. Both also changed backgrounds and damaged fine details in reviewed examples. These pilots demonstrate feasible learned transfer at 64 × 64, but do not establish augmentation suitable for SiDA's real-data targets.

## Shared experiment

A1-only discovery used eleven image statistics, class centering and class-balanced PCA. The low/high within-class thirds formed **D0** (usually softer/paler) and **D1** (usually stronger contrast/edges, sometimes darker backgrounds). PC1 explains 27.04% of standardized residual variance. These groups mix media, pose, scale and background; they are not established art-style domains.

Each domain supplied **3,975 optimization images and 302 diagnostics**, across all 151 classes. Models received uniformly sampled classes and unpaired same-class source/target images, with horizontal flips. Both trained from scratch for **20,000 batch-16 updates**, with a fixed learning-rate schedule and final-checkpoint evaluation. A2 and B were unused. Diagnostics were excluded from network optimization but participated in domain discovery, so this is not independent validation.

[Domain evidence](/home/user/jw_soflow/exploration/transfer_aug/domain_discovery/round2/SUMMARY.md) · [Frozen procedure](/home/user/jw_soflow/exploration/transfer_aug/shared/round2/TRAINING_PROTOCOL_v1.md)

## Common-direction comparison

The appearance score projects an image's statistics onto the frozen class-centered PC1. RGB RMS measures change from the source on a [0,1] scale. Neither measures semantic fidelity or image quality; larger shifts are not inherently better.

| Final diagnostic translations | Median appearance-score shift | Median RGB RMS change |
|---|---:|---:|
| CycleGAN D0 → D1, 302 images | +2.9343 | 0.15257 |
| CUT D0 → D1, 302 images | +3.1752 | 0.18055 |
| CycleGAN D1 → D0, 302 images | −3.0837 | 0.17683 |

CUT changes pixels more in the shared direction, but this does not rank its augmentation quality. Appearance descriptors use quantized viewing images; distances use raw generated floats mapped to [0,1]. Objectives, bidirectional coverage and antialiasing differ between implementations.

**CycleGAN:** coarse pose/layout often remains, while forward translations add dark contours, grain and shading. Some replace pale backgrounds with black or add halos/bars. Reverse translations can wash out outlines and markings and replace dark backgrounds with white. Median cycle-reconstruction RMS is only 0.06257/0.06570 despite visible translation failures: recovering the input does not certify a faithful intermediate image.

**CUT:** stronger contours and altered colors also appear, with fine-detail uncertainty. Four of 32 representative translations visibly replace pale backgrounds with black. All eight deliberately selected largest-change tails do so; reviewed Geodude/Arcanine examples lose important structure or markings. Target-domain identity controls also change images (median RMS 0.08990), so identity PatchNCE is not an identity guarantee.

These are **small, unblinded AI reviews**: CycleGAN inspected 64 representative sources and eight selected tails; CUT inspected 32 translations, 32 identity controls and eight tails. Each worker directly inspected nine sheets across fixed milestones. Tails may overlap representatives and cannot estimate failure prevalence. No exact duplicate final outputs were found; that does not establish semantic diversity.

[CycleGAN gallery](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-003/gallery.html) · [CUT gallery](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-004/gallery.html)

## Implication for augmentation

The present domain definition permits background/contrast changes that can satisfy distribution matching while compromising desired content. Cycle consistency and patch correspondence do not resolve that choice automatically. Before mixing outputs into generative training, define which background, palette and marking changes are acceptable and test whether they can be controlled separately. Background-aware grouping or preservation constraints are hypotheses for later contracts, not validated fixes.

No winner, convergence claim, class-accuracy estimate, generative FID improvement or SiDA benefit follows from this round. MUNIT and StarGAN v2 remain the queued subsequent pair. Their additional output diversity would still need the same preservation scrutiny. StarGAN v2 retains the user's exploratory parameter-cap exception; other limits remain.

## Execution and evidence

| Framework | Parameters | Peak allocated GPU memory | Cumulative GPU time |
|---|---:|---:|---:|
| CycleGAN | 17,001,352 | 2.197 GB | 1.030 hours |
| CUT | 9,061,060 | 2.569 GB | 1.028 hours |

Both passed checkpoint-recovery and deterministic-generation checks, stayed below the 20 GB/eight-hour limits, and released their GPU leases. Input hashes remained unchanged. Workers own implementation verification and artifact inspection; the master accepted their summaries, contract/input identity, counts, resource accounting and 59 resolving proof references.

[CycleGAN handoff](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-003/SUMMARY.md) · [CUT handoff](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-004/SUMMARY.md) · [Acceptance record](/home/user/jw_soflow/exploration/transfer_aug/shared/round2/handoff_acceptance.json) · [Ledger](/home/user/jw_soflow/exploration/transfer_aug/ROUND2_LEDGER.json)
