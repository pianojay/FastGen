# MUNIT and StarGAN v2: variation without reliable preservation

Both pilots completed. Both generate visibly different outputs from the same input, but reviewed variants also alter Pokémon anatomy and markings. Reference images often control palette and background; this does not establish that an independently transferable art style has been isolated. Neither pilot validates mixing its outputs into SiDA's real-data targets.

## Shared setup

We reused the [round2 A1 appearance groups](/home/user/jw_soflow/exploration/transfer_aug/ROUND2_REPORT.md): D0 tends toward softer/paler images, D1 toward stronger contrast/edges. Each domain has 3,975 optimization images and 302 diagnostics across all 151 classes. A2/B were unused. Diagnostics contributed to domain discovery, so they are not independent validation.

Each model trained from scratch for **20,000 batch-8 iterations**, with its pinned method-specific losses and schedule. Each diagnostic source received four fixed same-class target-domain references and four fixed sampled codes. Each method produced **4,832 translations plus 604 self-controls**. The same 64 representative sources, fixed milestone traces, and eight separately selected large-change/low-diversity tails were reviewed.

MUNIT encodes separate content and style representations; reference styles or an eight-dimensional Gaussian code drive its decoder. StarGAN v2 uses domain-specific style encoders and a learned latent-to-style mapping, evaluated through EMA networks. Its two domain labels describe appearance groups, not Pokémon species. Neither method used pretrained features or facial alignment/masks.

[Frozen protocol](/home/user/jw_soflow/exploration/transfer_aug/shared/round3/TRAINING_PROTOCOL_v1.md) · [Design and limits](/home/user/jw_soflow/exploration/transfer_aug/ROUND3_DESIGN.md)

## Measured variation and observed failures

The common D0→D1 comparison below averages four variants within each source, then averages 302 sources. Source change is clipped-floating RGB RMS on a nominal 0–1 scale. Diversity averages the six pairwise distances among a source's four outputs. **Neither is a quality score.** PC1 measures movement along the frozen statistical appearance axis.

| Method / arm | Mean PC1 shift | Mean source change | Mean within-source diversity |
|---|---:|---:|---:|
| MUNIT / reference | +1.17 | 0.410 | 0.307 |
| MUNIT / latent | +3.13 | 0.441 | 0.450 |
| StarGAN v2 / reference | +3.32 | 0.413 | 0.382 |
| StarGAN v2 / latent | +3.49 | 0.400 | 0.399 |

**MUNIT:** reference variants often retain coarse pose while following donor backgrounds/palettes, but smear or erase facial, wing, shell and body markings. Sampled styles can produce hollow outlines, streaks or distorted blobs. Mean self-reconstruction RMS is only 0.054/0.062 across the two source domains, versus roughly 0.410–0.449 for translations: close reconstruction does not certify safe transfer.

**StarGAN v2:** reference variants show strong palette/background control alongside altered faces, mouths, limbs, flowers, wings and body shapes. Sampled styles can recolor the Pokémon and introduce clutter. Even self-style controls visibly alter details, with clipped RMS 0.121/0.163. Its statistical movement toward D1 is stronger than MUNIT's reference arm here, but that does not make its outputs better augmentation.

Both reverse mappings also move PC1 toward D0, with preservation concerns recorded in the galleries. All judgments are **small, unblinded AI reviews**, not class accuracy or population failure rates. Selected tails are intentionally unrepresentative.

[MUNIT gallery](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-005/gallery.html) · [StarGAN v2 gallery](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-006/gallery.html)

MUNIT had no out-of-range outputs. StarGAN's unbounded RGB generator placed about 5.78–12.47% of translated channel values outside the viewing range, depending on arm/direction, but average clipping magnitude was only 0.0014–0.0029. Clipping frequency alone cannot substitute for assessing visible damage. Raw arrays and clipped views are retained.

## Implication for generative augmentation

The new evidence is that explicit style codes and multiple outputs do not resolve the ambiguity of what should change. Pixel diversity can coexist with erased identity cues; statistical target matching can coexist with visibly damaged content. Background changes need an explicit policy, while anatomy/marking preservation needs separate evidence.

The next useful question is whether a narrowly permitted factor can be varied while protecting foreground structure and markings. That is a proposed future contract, not an implemented fix. No augmentation winner, convergence claim, FID improvement or SiDA benefit is established. These are single-seed 64-pixel pilots with shortened schedules; they do not reject either method under other settings. Compared with round2, batch size and optimizer phases differ, so this is not an equal-compute benchmark.

## Resources and verification

| Framework | Registered parameters | Peak allocated memory | Cumulative GPU time |
|---|---:|---:|---:|
| MUNIT | 46,594,716 | 1.970 GB | 1.147 hours |
| StarGAN v2 | 131,065,096, including EMA | 12.704 GB | 5.548 hours |

StarGAN uses the explicitly authorized exception to the normal 100M cap. Its peak includes recovery verification; peak reserved memory was 19.541 GB. Both stayed within the 20-billion-byte allocation/eight-hour limits and released their leases.

All **10,872 final outputs**, **1,024 replay checks**, and **44 reviewed sheets** are accounted for. Checkpoint recovery and protected input hashes passed. Master acceptance checked compact handoffs and 46 proof references; workers owned implementation checks and direct artifact inspection.

A targeted handoff check found that both reports initially used PNG-rounded distances for metrics labeled “clipped.” Workers corrected them to floating-point clipping. Descriptor/PCA values, model outputs, checkpoints, reviewed sheets and tail selection were unchanged. The comparison above uses corrected metrics.

[MUNIT summary](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-005/SUMMARY.md) · [StarGAN summary](/home/user/jw_soflow/exploration/transfer_aug/experiments/TA-006/SUMMARY.md) · [Acceptance record](/home/user/jw_soflow/exploration/transfer_aug/shared/round3/handoff_acceptance.json) · [Ledger](/home/user/jw_soflow/exploration/transfer_aug/ROUND3_LEDGER.json)
