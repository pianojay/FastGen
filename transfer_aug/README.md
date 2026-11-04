# Transfer augmentation research

Investigate whether transferable factors can be identified within the Pokémon training dataset and used to construct useful augmentation. The transferred “something” is deliberately unspecified: it might involve texture, rendering, color statistics, geometry, or another factor, and it may not be separable from identity reliably.

This is continued EDA oriented toward applied augmentation. Establish what can be transferred and whether the resulting images are reasonable before considering integration with SiDA or another generative model.

## Current status

The dispatch workflow and first-wave experiments completed on October 7, 2026. Independent workers executed TA-001 (intra-class Color Transfer) and TA-002 (intra/inter-class FDA) on the shared frozen A1 cohort, generating 16,308 transformed images and 1,208 controls. The [comparative report](/home/user/jw_soflow/exploration/transfer_aug/FIRST_WAVE_REPORT.md) records the mechanism/failure findings and their limits. Both contracts passed; no winning augmentation was selected. No learned model training or GPU use occurred.

CycleGAN (TA-003) and CUT (TA-004) completed on October 8, 2026: 20,000 scratch-training updates each, approximately 1.03 GPU-hours per method. Both shifted the intended A1 appearance statistics, but reviewed outputs show background replacement and fine-detail damage. See the [round 2 comparison and galleries](/home/user/jw_soflow/exploration/transfer_aug/ROUND2_REPORT.md). No augmentation winner or SiDA benefit was established. MUNIT (TA-005) and StarGAN v2 (TA-006) also completed on October 8, 2026: 20,000 scratch iterations and 5,436 final outputs each. Both showed substantial variation alongside content-preservation failures. See the [round3 comparison and galleries](/home/user/jw_soflow/exploration/transfer_aug/ROUND3_REPORT.md). GPU use was 1.147/5.548 hours; both leases are released. StarGAN's 131.1M registered parameters used the user's exploratory exception; all other resource and data limits remained in force.

## Research documents

- [First-wave results](/home/user/jw_soflow/exploration/transfer_aug/FIRST_WAVE_REPORT.md): completed color/Fourier comparisons, visual evidence, verification and implications.
- [Master instructions](/home/user/jw_soflow/exploration/transfer_aug/MASTER.md): read first on resumed sessions; defines dispatch, delegation, evidence review and resource boundaries.
- [Experiment contract template](/home/user/jw_soflow/exploration/transfer_aug/EXPERIMENT_TEMPLATE.md): reusable deterministic worker procedure and handoff requirements.
- [Experiment ledger](/home/user/jw_soflow/exploration/transfer_aug/EXPERIMENT_LEDGER.json): frozen revisions, worker assignments and current status.
- [Study interpretations](/home/user/jw_soflow/exploration/transfer_aug/STUDY_INTERPRETATIONS.md): corrections to the initial method impressions and implications for the research sequence.
- [Methodology audit](/home/user/jw_soflow/exploration/transfer_aug/METHODOLOGY_AUDIT.md): comparison, applicability findings, and proposed research sequence.
- [Source registry](/home/user/jw_soflow/exploration/transfer_aug/SOURCE_REGISTRY.md): primary papers, pinned codebases, configuration anchors, and evidence limitations.
- [Parameter results](/home/user/jw_soflow/exploration/transfer_aug/parameter_audit.json): static component counts, including auxiliary networks and applicable EMA copies.
- [Research plan](/home/user/jw_soflow/exploration/transfer_aug/RESEARCH_PLAN.md): questions, evidence sequence, candidate methods, and criteria for proceeding.
- [Research log](/home/user/jw_soflow/exploration/transfer_aug/RESEARCH_LOG.md): dated decisions, findings, and unresolved questions.
- [Completed dataset EDA](/home/user/jw_soflow/exploration/eda/REPORT.md): historical context. Its B-specific measurements must not become augmentation targets or selection criteria.

## Working boundaries

Use A1 for discovery, fitting, content images, and transfer donors. Retain A2 as the existing development set for later validation; keep B out of this research's fitting and selection. Preserve all existing split manifests. A contains 17,079 images, partitioned into A1 with 14,059 and A2 with 3,020; B contains 3,020 evaluation images.

The [project rules](/home/user/jw_soflow/exploration/README.md:188) limit each learned model to 100M parameters, including its auxiliary components; separate models are not summed into one parameter allowance. Learned models used in training or sampling must be trained from scratch on the allowed data, and peak training memory must remain within 20 GB. Check the full dependency chain of a replication, including frozen encoders, decoders, and perceptual networks. Consider a smaller backbone when necessary, documenting that change as an adaptation.

Keep research documentation, code and artifacts under this directory. Reuse the existing isolated EDA environment without modification; do not alter sibling projects or running processes. The first round authorizes the two statistical-transform experiments. Later bounded learned augmentation experiments require filled contracts and retain the scratch-only, parameter and memory rules, with a default cap of 8 cumulative GPU-hours per experiment on GPU 1 or 2. SiDA integration remains a subsequent stage.

## Documentation practice

Keep the plan current and append dated findings to the log. Distinguish observations, hypotheses, and decisions. For future measurements, record the data partition, representation, controls, artifact locations, and implementation revision. Record unsuccessful methods as well as successful ones; generated image count alone is not evidence of improvement.
