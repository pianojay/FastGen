# CycleGAN and CUT: what this pilot can establish

The shared question is whether a model trained from scratch can move images between two A1 appearance distributions while retaining recognizable Pokémon identity. This round does not measure augmentation benefit during SiDA training.

The domains were derived from eleven simple image descriptors after subtracting each class's mean and balancing classes in PCA. The first component explains 27.04% of standardized residual variance. Within each class, the low and high thirds define D0 and D1; the middle third is unused. D0 tends toward softer/paler images and D1 toward stronger contrast/edges, sometimes with darker backgrounds. The inspected groups contain mixed rendering media, poses and scales. They should not be called discovered art styles.

Each domain has 3,975 optimization images and 302 diagnostic images, with all 151 classes represented. Uniform class sampling and same-class unpaired draws reduce class-frequency confounding; they do not impose an identity-preserving loss. Diagnostic images are excluded from network optimization, but contributed to domain discovery. A2 and B remain unused in this round.

| Comparison | Interpretation |
|---|---|
| D0→D1 appearance-score shift | Whether outputs move along the predeclared statistical axis; moving farther is not necessarily better. |
| Target-relative descriptor changes | Whether a shift resembles target statistics or overshoots them. |
| Pixel changes, edges and range | Size and character of the intervention; these are not semantic quality measures. |
| Within-class target nearest neighbors | Pixel resemblance and possible copying warnings; distance alone cannot establish memorization or novelty. |
| Fixed representative images | Typical visible behavior on the preselected subset, with uncertain judgments allowed. |
| Eight largest-change tails | Failure discovery; deliberately unrepresentative of prevalence. |
| Cycle/identity reconstructions | Objective-specific behavior, not proof that translations preserve category. |

CycleGAN has two generators and two discriminators (17,001,352 parameters); CUT has one generator, one discriminator and five patch-projection heads (9,061,060 parameters). Both use six-block generators at 64 pixels. Their objectives and antialiasing differ, so an equal update count does not isolate an architectural cause. The primary common direction is D0→D1; CycleGAN's D1→D0 is supplementary.

Both receive a fixed maximum of 20,000 updates and eight cumulative GPU-hours, including qualification and inference. The final completed checkpoint is reported without selecting a preferred checkpoint using diagnostic images. Exhausting this finite pilot cannot establish convergence or a method's ultimate capability.

A useful next-stage candidate would need plausible appearance changes with limited visible identity damage, followed by separate confirmation and an objective-specific SiDA experiment. Strong target-statistic matching alone is insufficient: a translated target distribution could still contain undesirable artifacts or lose useful original modes.

Evidence: [domain handoff](/home/user/jw_soflow/exploration/transfer_aug/domain_discovery/round2/SUMMARY.md), [frozen protocol](/home/user/jw_soflow/exploration/transfer_aug/shared/round2/TRAINING_PROTOCOL_v1.md), [round ledger](/home/user/jw_soflow/exploration/transfer_aug/ROUND2_LEDGER.json). Completed results: [round 2 report](/home/user/jw_soflow/exploration/transfer_aug/ROUND2_REPORT.md).
