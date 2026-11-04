# Research log

## October 7 2026

Initialized persistent research documentation under `transfer_aug/` following the completed dataset EDA and discussion of transfer-based augmentation.

### Agreed direction

- Treat this as continued EDA with an applied augmentation objective.
- Keep the transferred factor unspecified until evidence clarifies what can change and what must remain invariant. Reliable separation may not be possible.
- Use intra-class and inter-class comparisons to investigate identity and appearance. Do not equate within-class variation with style without further evidence.
- More synthetic samples or more apparent styles do not imply a lower FID.
- Screen candidate replications against the full parameter and resource limits; consider smaller backbones when required.
- Establish reasonable augmentation before considering SiDA integration.

### Current evidence and hypotheses

The existing EDA provides cached image measurements and Inception features. Those are possible starting points for A1-only comparisons. The earlier A/B analyses describe the dataset but do not supply augmentation targets for this work.

AdaIN and MixStyle are conceptual references. No implementation, backbone, transferable factor, transformation strength, donor policy, or mixing ratio has been selected. A dedicated transfer generator may be unnecessary; its value must be compared with explicit transforms and possible reuse of existing components.

### Open questions

1. What recurring factors can be distinguished from category identity, pose, and framing using A1 alone?
2. Can those factors be changed in image space while preserving distinguishing anatomy and markings at 64 × 64?
3. When does an inter-class donor add useful variation, and when does it introduce identity contamination?
4. What measurements distinguish useful transfer from no change, corruption, or duplicated content?
5. Does a candidate require learned synthesis, or can a simpler statistical operation test the same hypothesis?
6. What compute budget would a viable implementation require beyond parameter compliance?

### Status

Documentation only. No new dataset analysis, implementation replication, augmentation generation, model training, or SiDA modification was performed in this setup step. Existing data, splits, and sibling projects remain unchanged.

### Format for future entries

Record the question, evidence or artifact, interpretation, decision, and next unresolved question. Include the exact partition and relevant settings for measurements. Keep hypotheses distinct from findings, and retain the rationale when a direction is rejected or superseded.

## October 7 2026 — methodology and implementation audit

**Question:** Which existing appearance-transfer or related augmentation methods have a plausible path under our data, resolution, scratch-training, and parameter constraints?

**Evidence:** Read primary papers and selected source/configuration files from 14 pinned codebases covering 12 method families. Recorded original dataset scales/resolutions, transfer mechanisms, dependencies, and configuration-specific parameter arithmetic in the [audit](/home/user/jw_soflow/exploration/transfer_aug/METHODOLOGY_AUDIT.md), [registry](/home/user/jw_soflow/exploration/transfer_aug/SOURCE_REGISTRY.md), and [parameter record](/home/user/jw_soflow/exploration/transfer_aug/parameter_audit.json).

**Findings:** Color moments and Fourier amplitude transfer need no learned network. MixStyle has no additional parameters but produces features, not RGB augmentation. AdaIN's active encoder/decoder is about 7.01M and the inspected WCT port about 34.24M, but their standard pretrained visual features conflict with the scratch-only rule. CUT is about 14.70M including its discriminator and patch MLPs; a suitable appearance-domain partition is still missing. Swapping Autoencoder is relevant to learning factors without domain labels, but its released Church framework totals 109.03M and cannot directly accept 64-pixel input through its default global encoder. StarGAN v2's three-domain Solver totals 137.06M with registered EMA copies.

**Interpretation:** “Appearance-factor transfer” remains a provisional description. Neither a named style code nor statistical moment matching establishes separation from identity. Parameter fit alone does not establish compute affordability or useful augmentation. Original face/landscape/high-resolution demonstrations are not direct evidence for heterogeneous 64-pixel Pokémon.

**Research direction:** Discover candidate factors in A1; then compare narrow donor-driven color/Fourier interventions with unchanged inputs and donor-free controls. Reserve learned translation/factorization for a demonstrated need. This is a proposed sequence, not an adopted transform or authorization to start model training.

**Outstanding:** Actual transferable factors, identity failure rates, useful donor policies, transformation strength, runtime/VRAM, and eventual generative-model benefit are unmeasured. WCT's exact selected decoder-training corpus size remains unverified.

**Scope completed:** Documentation, cached public source/paper reading, and pure-Python parameter arithmetic only. No weights, model execution, augmentation generation, dataset/split changes, GPU use, or sibling-project modifications.

## October 7 2026 — authorized dispatch round started

The user explicitly requested implementing the agreed dispatch plan, including worker execution and a comparative result. Created MASTER.md, the reusable experiment template, method-interpretation corrections, a CPU-only environment wrapper, the experiment ledger, and two frozen contracts. This extends the earlier documentation-only scope to the bounded first-wave statistical-transform experiments.

The common cohort contains 604 A1 content images (four distinct official-export pixel hashes per category), matched intra-class donors and separately sampled inter-class FDA donors. Its 16-category review set is seeded and frozen. An independent regeneration reproduced the cohort identically; cache path alignment, A1 membership and frozen hashes passed master preflight. A2 and B are excluded from the analysis; B manifest hashes and source-file metadata are used only for integrity checks.

Dispatched TA-001 and TA-002 to separate workers with exclusive writable experiment directories. Workers own numerical verification, execution, low-level artifacts and actual visual inspection. The master will inspect their <=500-word summaries, structured results and evidence references. Both experiments are CPU-only, with no model training. Future learned contracts retain scratch-only and resource limits, with an eight-GPU-hour default cumulative cap. The round ends with a mechanism/failure comparison, not a recipe selection or SiDA integration.

## October 7 2026 — first dispatch round completed

Both workers completed their frozen contracts and the master accepted the compact handoffs. Delivered 16,308 transformed images plus 1,208 controls, 448 recorded visual judgments, and 58 actually inspected contact sheets. Exact output/metric/sheet replay and protected-input checks passed. Work used CPU only; GPU usage and learned parameters were zero. The master checked contract/cohort identity, counts, budgets, proof paths and artifact-manifest hashes, without redoing the delegated low-level inspection.

**Findings:** intra-class Color Transfer sometimes weakened markings and changed backgrounds, despite accurate pre-gamut moment transfer. FDA's raw spectral changes stayed in the declared window, while RGB clipping introduced outside-window changes; larger windows/strengths showed more spatial color patches/halos in the reviewed examples. Inter-class FDA had larger median appearance displacement in this frozen cohort, without isolating a causal class effect. Both methods demonstrated that high clipping frequency can accompany visually mild changes on large white backgrounds.

**Limitations/decision:** this is a mechanism/failure map on A1, with a small unblinded AI review subset. No preferred recipe, category-accuracy estimate, discriminator response, FID effect, or SiDA integration was established. No successor was launched. Follow-up hypotheses and proof links are in [FIRST_WAVE_REPORT.md](/home/user/jw_soflow/exploration/transfer_aug/FIRST_WAVE_REPORT.md). The master instructions, reusable template and immutable contract revisions are ready for subsequent bounded dispatches.

## October 8 2026 — CycleGAN/CUT round authorized

The user requested CycleGAN and CUT as the next pair because their core recipes can train without pretrained models, then MUNIT and StarGAN v2 as the subsequent pair. The user explicitly permits ignoring StarGAN v2's parameter-count problem for now; that exploratory exception does not relax other data/memory/budget constraints.

The user selected deriving and inspecting shared appearance groups from A1. Dispatched a CPU domain-preparation worker and separate CycleGAN/CUT implementation-preparation workers. Training remains gated on frozen inspected domains and final training contracts. CUT setup revision 2 corrects the two-layer discriminator receptive field from 34px to 40px when its original antialias filters are retained; revision 1 is preserved. Round state is in ROUND2_LEDGER.json.

## October 8 2026 — shared domains accepted; training dispatched

Accepted the domain worker’s summary and structured verification: 12,987 within-class distinct A1 images; each domain has 3,975 optimization and 302 diagnostic examples across all 151 classes. PC1 explains 27.04% of class-balanced standardized residual variance. D0/D1 mainly distinguish softer/paler from stronger contrast/edges; they are not established semantic styles. Visual inspection covered 128 images and exact regeneration passed.

Froze an A1-only 8,554-image official RGB64 bundle and final TA-003/TA-004 training contracts. CPU setups passed 29/29 CycleGAN and 8/8 CUT checks, with 17,001,352 and 9,061,060 parameters respectively. Dispatched workers for GPU qualification, up to 20,000 updates within 8 cumulative GPU-hours each, deterministic diagnostic generation and EDA. No A2/B fitting or SiDA integration is authorized.

## October 8 2026 — CycleGAN/CUT round completed

Both workers completed 20,000 scratch-training updates and released their GPU leases. CycleGAN used 17,001,352 parameters, 2.197 GB peak allocated memory and 1.030 GPU-hours; CUT used 9,061,060 parameters, 2.569 GB and 1.028 GPU-hours. Final outputs comprise 604 CycleGAN translations plus cycle/identity controls, and 302 CUT translations plus 302 identity controls. Each worker directly inspected nine fixed representative/tail sheets. Checkpoint recovery, deterministic generation, protected-input integrity and resource checks passed. Master acceptance checked compact handoffs and 59 proof references, without duplicating low-level inspection.

Both learned the chosen contrast/edge appearance direction. D0→D1 median appearance shifts/RGB RMS were +2.9343/0.15257 (CycleGAN) and +3.1752/0.18055 (CUT). Reviewed outputs sometimes replace pale backgrounds with black and damage fine markings; CycleGAN's reverse mapping can wash out details. Close cycle reconstruction did not certify preservation in the translated image. These are descriptive small unblinded AI reviews, not class accuracy or population failure rates.

The shared domains remain entangled with background/content. This round establishes computational feasibility and mechanism/failure evidence, not readiness for SiDA real-data augmentation, convergence or improved FID. MUNIT/StarGAN v2 remain queued with the recorded StarGAN parameter exception. Full comparison and galleries: ROUND2_REPORT.md; accepted metadata: shared/round2/handoff_acceptance.json.

## October 8 2026 — MUNIT/StarGAN v2 round authorized

The user requested continuing with MUNIT and StarGAN v2. Dispatched CPU implementation/verification workers under frozen setup contracts TA-005/TA-006, retaining the round2 A1 domains and diagnostic sources. Plan reference-guided and sampled-style outputs to examine diversity alongside content preservation. StarGAN v2 uses two appearance domains and disables FAN/high-pass facial masks; its full parameter count including EMA will be reported under the user exception. Both retain scratch-only, 20 GB training allocation, assigned GPU1/2 and eight cumulative GPU-hour ceilings. No A2/B or SiDA integration.

CPU preparation accepted: MUNIT46,594,716 parameters, StarGAN v2 131,065,096 including EMA; exact synthetic checkpoint recovery and source/mechanism checks passed. Frozen final training contracts and shared reference/latent evaluation manifest. Dispatched GPU1/GPU2 qualification and complete bounded runs. Each method has604 diagnostic sources with4 reference/4 sampled-style translations plus604 self-controls. The fixed review includes all64 representative sources and separate destructive-change/low-diversity tails. Reporting separates raw and clipped differences because StarGAN has unbounded RGB output.

TA-005 MUNIT completed its full20,000-iteration pilot,5,436 final outputs and22 reviewed sheets, using1.147 GPU-hours and1.970 GB peak allocation. Exact replay and protected hashes passed; GPU1 released. Worker review finds donor background/palette transfer with damaged markings, and Gaussian variations often distorted; high pixel diversity did not establish identity preservation. Master accepted27 proof references and frozen contract/data/resource identity. TA-006 remains training under its original contract; no settings changed using MUNIT results.

## October 8 2026 — multimodal round completed and accepted

TA-005 MUNIT and TA-006 StarGAN v2 both completed 20,000 scratch iterations. Generated 10,872 final outputs total, verified 1,024 exact replay translations, and workers directly reviewed 44 sheets. MUNIT used 1.147 GPU-hours / 1.970 GB peak allocation; StarGAN 5.548 GPU-hours / 12.704 GB recovery-inclusive peak. StarGAN's 131,065,096 registered parameters include EMA and use the explicit exception. Both GPU leases released; protected inputs unchanged.

Both reference/latent arms produce substantial variation and move appearance statistics, but reviewed outputs alter anatomy and markings as well as backgrounds/palettes. Close MUNIT self-reconstruction did not ensure safe transfer; StarGAN self-controls also distorted details. No reliable identity-preserving augmentation, FID improvement, convergence or SiDA benefit was established. Results remain bounded single-seed 64-pixel pilots with different objectives/exposure from round2.

Master caught a reporting-definition discrepancy: clipped distances used quantized PNGs. Targeted CPU-only corrections replaced those distances with floating-point clipping, checked all outputs/source groups, and verified that models, generated images, descriptors/PCA, reviewed sheets and tail choices were unchanged. Updated result hashes and 46 proof references accepted in shared/round3/handoff_acceptance.json. Full comparison: ROUND3_REPORT.md. No successor launched.
