# MUNIT and StarGAN v2: diversity with preservation

This round continues the same frozen A1 appearance-domain experiment. MUNIT and StarGAN v2 offer reference-guided and randomly sampled variations of the same source image. The question is whether those variations preserve recognizable identity, anatomy and markings, rather than merely producing different pixels.

D0/D1, optimization images and diagnostic sources are unchanged from round2. For each of604 diagnostic sources, the master froze four distinct same-class opposite-domain optimization references and four method-specific Gaussian draws. MUNIT's prior is eight-dimensional; StarGAN's mapping input is sixteen-dimensional. Their coordinates are not matched style meanings. Both methods produce4,832 translations and604 self-controls at the final checkpoint, with the same64 representative sources and independently identified failure/diversity tails.

The primary comparisons separate direction and reference/latent arm. Four outputs are averaged within each source before summarizing the302 sources per direction. Diversity is the mean of six pairwise pixel distances among a source's four variants, measured before and after viewing-time clipping. This quantifies variation, not semantic diversity or quality. High values may reflect destructive changes; low values may reveal ignored style inputs. Fixed appearance-PC1 statistics complement direct review; no pretrained LPIPS/FID/classifier is introduced.

MUNIT retains its audited scratch-capable configuration with VGG loss disabled:46.6M parameters. StarGAN v2 uses two appearance domains, disables FAN/high-pass facial masks, and includes EMA copies:131.1M total, explicitly outside the normal100M rule under the user's exception. Its linear RGB output is unbounded, making clipping diagnostics important. Both start from scratch and retain the20GB/eight-GPU-hour ceilings.

Both receive batch8 and at most20,000 full iterations, with their method-specific losses and optimizers. This differs from round2's batch16 and from original published training budgets. An iteration also contains different optimizer phases between methods. Comparisons are bounded pilot observations, not equal-compute benchmarks or claims of convergence. StarGAN's diversity-loss decay is shortened to20,000 iterations; MUNIT's original long StepLR remains constant during this pilot. Full choices are frozen in contracts.

Review targets background replacement and fine-detail damage observed previously, plus whether multiple outputs differ plausibly, follow donors, collapse or vary mainly through noise. The chosen groups remain statistical appearance domains entangled with background/content. Adding a style code does not establish that style has been isolated. A1 diagnostics participated in domain discovery; A2 remains for later independent confirmation and B is excluded.

[Protocol](/home/user/jw_soflow/exploration/transfer_aug/shared/round3/TRAINING_PROTOCOL_v1.md) · [Evaluation freeze](/home/user/jw_soflow/exploration/transfer_aug/shared/round3/freeze.json) · [Ledger](/home/user/jw_soflow/exploration/transfer_aug/ROUND3_LEDGER.json)

Completed findings and galleries: [round3 report](/home/user/jw_soflow/exploration/transfer_aug/ROUND3_REPORT.md). Both schedules completed; preservation remains unresolved.
