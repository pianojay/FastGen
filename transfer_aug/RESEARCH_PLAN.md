# Research plan

The research question is whether statistics and comparisons within A1 reveal factors that can vary while Pokémon identity remains intact. The user authorized a finite first round of explicit color/Fourier interventions to test this question, following the dispatch contracts recorded in the [ledger](/home/user/jw_soflow/exploration/transfer_aug/EXPERIMENT_LEDGER.json). These operators are probes; naming their statistics does not establish semantic separation from identity.

Current execution has progressed beyond the first statistical probes: CycleGAN/CUT completed under [round2 contracts](/home/user/jw_soflow/exploration/transfer_aug/ROUND2_REPORT.md); MUNIT and StarGAN v2 completed under [round3 contracts](/home/user/jw_soflow/exploration/transfer_aug/ROUND3_LEDGER.json). These pilots examine preservation and diversity; SiDA integration remains a separate later stage. MASTER.md records current authorization and limits.

## Problem formulation

Conceptually, an image may contain identity, instance content, and appearance factors. Their separation is a hypothesis, not an established property of this dataset.

A candidate transfer operator takes a content image, a donor image or donor statistic, and a strength parameter. Its desired output retains the content image's category while changing a specified factor. Once a candidate factor is identified, state explicitly what is preserved, what changes, how donors are chosen, and what would count as failure.

An augmented training distribution mixes original and transformed images. Both the transformation and its sampling frequency matter. More images, greater feature variance, or broader apparent style coverage do not by themselves imply better FID or better generalization.

## Evidence sequence

### Discover candidate factors within A1

Compare variation within the same category with variation across categories. Use native images for interpretation and the actual 64 × 64 inputs for task-relevant measurements. Begin with interpretable properties and existing frozen EDA features where appropriate.

Useful questions include:

- Which appearance patterns recur across different Pokémon categories?
- Which changes occur within a category while its distinguishing anatomy and markings remain recognizable?
- Are apparent differences driven by pose, cropping, background, rendering, or duplicate artwork rather than a transferable factor?
- Do candidate patterns remain stable across category-balanced samples and duplicate-aware controls?

Within-class variation is not automatically style: it also contains pose, geometry, framing, and noise. Across-class differences are not automatically identity: category and appearance may be correlated. Class-centered feature residuals or between/within-class covariance decompositions can be exploratory tools, but neither proves disentanglement or guarantees that a feature direction can be realized as a valid image transformation.

Any new representation, component basis, cluster assignment rule, or donor-selection rule intended for augmentation should be derived from A1. Earlier EDA PCA and clustering were fitted on full A, including A2, so they are descriptive context rather than ready-made development transforms. The frozen per-image features may be reused with the correct A1 indexing.

### Define a restricted transfer hypothesis

For each promising factor, specify a narrow intervention and the expected invariants. Separate background and framing changes from rendering changes unless evidence supports treating them jointly.

Compare intra-class and inter-class donors. Same-class transfer is a useful control, not a guarantee of preserved identity; cross-class transfer tests whether a factor generalizes beyond its original category. Donor selection and strength must not use B-derived statistics.

Keep an unchanged-image control. A future comparison should distinguish donor-driven transfer from a generic perturbation of comparable magnitude, so an apparent benefit is not incorrectly attributed to transferable structure.

### Assess augmentation quality before integration

Inspect identity preservation, anatomical integrity, artifacts, actual factor change, and variation among outputs. Check examples across categories and transfer strengths at 64 × 64. Similarity to the input alone rewards doing nothing, while large feature displacement alone rewards corruption.

Start with a qualitative rubric and measured failure rates. Define acceptance thresholds before selecting a method, rather than choosing them after seeing favorable examples. Automated category agreement, if introduced later, is supporting evidence rather than a semantic guarantee.

The next stage is justified only if a candidate produces repeatable, interpretable changes with acceptable identity preservation and resource cost. Failure to identify such a candidate is a valid research outcome.

### Consider SiDA integration later

Only after establishing reasonable augmentation should we choose its location in SiDA: teacher training data, student adversarial targets, or real/fake discriminator inputs. These locations change different objectives and must not be conflated.

Any later model study should use fixed A1/A2 partitions, matched evaluation settings, and a compute-matched unaugmented control. Separate augmentation effects from extra optimization time or extra teacher sampling. No integration or model experiment is scheduled by this plan.

## Candidate references

| Reference | What it may contribute | What needs scrutiny |
| --- | --- | --- |
| [AdaIN](https://arxiv.org/abs/1703.06868) | A reference-based feature-statistic transfer formulation | Content preservation, decoder and encoder dependencies, pretraining assumptions, full parameter count, and scratch-training feasibility |
| [MixStyle](https://arxiv.org/abs/2104.02008) | Mixing feature statistics without a standalone image generator | It operates in feature space; recognition/domain-generalization results do not establish valid image transfer or improved generative FID |
| Explicit statistical transforms | Cheap tests of a narrowly defined factor | Whether they produce plausible changes rather than arbitrary appearance distortion |
| Existing teacher reuse | Potential variation without training another generator | A class-conditional teacher is not automatically a controllable transfer operator; its artifacts and coverage limitations can be inherited |

The [October 7 methodology audit](/home/user/jw_soflow/exploration/transfer_aug/METHODOLOGY_AUDIT.md) expands these references to 12 method families. Its historical audit involved no runtime replication. The subsequent authorized first round implements Color Transfer and FDA as explicit probes. CUT remains conditional on defining appearance domains, and a reduced Swapping Autoencoder remains a later factorization reference. The released Swapping framework and StarGAN v2 Solver exceed 100M in the audited configurations; standard AdaIN/WCT recipes face a pretrained-feature dependency. Scratch-only remains mandatory.

## Replication screening

For each implementation investigated, record the paper and repository, pinned revision, license, expected inputs, external data, pretrained components, and training objectives. Inventory all learned components and verify parameter counts rather than relying on the headline backbone size.

Check the 100M per-model limit, 20 GB training-memory limit, and scratch-training requirement before treating an implementation as a feasible candidate. Frozen weights still count where included in the model framework. A pretrained feature extractor allowed for EDA measurement is not automatically allowed as a training loss network.

If a smaller backbone is needed, document the substituted components and which claims from the original method no longer constitute a direct replication. Estimate memory, training time, and augmentation throughput separately; fitting the parameter cap does not establish affordability. Do not choose a smaller architecture until the relevant bottleneck is known.

## Immediate next research question

At 64 × 64, what do intra-class Lab moment transfer and intra/inter-class Fourier amplitude transfer actually change, and when do clipping, identity ambiguity, background contamination or visible artifacts arise? The first round reports a fixed strength/radius grid and uncertainty, without selecting a winning recipe or using A2/B. Future learned pilots have an 8 cumulative GPU-hour default cap and require a frozen contract; they are not automatically launched by this round.
