# FastGen strategy after the CIFAR experiment and training-cost audit

Update 2026-09-28: the Pokémon baseline has completed. See [final results](POKEMON_BASELINE_RESULTS.md) for measured one/few-call quality and the resulting next-experiment discussion. The running observations below are historical snapshots from September 24.

Consolidated 2026-09-24, Asia/Seoul. This note synthesizes the parallel agent's [formal CIFAR report](soflow-cifar10/EXPERIMENT_REPORT.md), [training-cost analysis](soflow-cifar10/cifar10-training-cost.md), [literature audit](soflow-cifar10/few-nfe-training-cost-audit.md), and our [Pokémon run record](STATUS.md). It adds interpretation and future experiment proposals. The running Pokémon configuration remains unchanged; no calibration or new training experiment was launched for this review.

## What the experiments establish

The closed CIFAR run established initial trainability and measured throughput. Its 800,000-update schedule projected to 125.95 days on the assigned TITAN RTX. There were 130 logged updates, a recoverable main checkpoint at update 5, and no generation-quality evaluation. The feasibility decision concerns reproducing that full recipe within the local budget. Useful quality on a shorter recipe remains an empirical question.

Our Pokémon run is a separate, deliberately smaller experiment:

| Quantity | Original CIFAR recipe | Current Pokémon adaptation |
|---|---:|---:|
| Resolution / conditioning | 32×32 / unconditional | 64×64 / 151 categories |
| Model parameters | About 55.93M | 39,667,843 |
| Optimizer updates | 800,000 | 20,000 |
| Effective batch | 1,024 | 128 |
| Image presentations | 819.2M | 2.56M |
| Equivalent passes over optimization data | 16,384 | About 182 |
| Measured update time | 13.603 s, sustained window | About 1.72 s, early live run |
| Update-only projection | 125.95 days | About 9.6 hours |

The presentation budget is **320 times smaller**. Both configurations currently process roughly 75 images/second, despite different architectures and resolutions. Most of the runtime reduction comes from choosing less training work. The eightfold smaller batch also makes update-to-update timing an inadequate measure of algorithmic acceleration. Microbatch controls memory and throughput; effective batch and update count determine how many images the optimizer processes.

Pokémon also changes the curriculum horizon, EMA treatment, dropout and architecture. Its consistency-gap curriculum reaches the same terminal ratio over 20,000 updates. This is a changed learning recipe whose quality must be evaluated, rather than a partial reproduction of the 800,000-update schedule.

At 22:06 KST, the live run had reached update 1,650 with finite statistics, 8.610 GB peak tensor allocation and about 1.82 GB sampled process-tree PSS. Visual inspection of the saved update-1,500 grids found coarse foreground shapes and lighter backgrounds at four calls; one-call outputs remained noisy. Several categories retained very similar silhouettes under paired noise. These are qualitative observations from a small fixed panel; semantic fidelity and diversity remain unestablished. No development FID for this substantive run had yet been produced; the scheduled first pass is at update 5,000. The earlier four-update FID belongs to pipeline qualification.

## Cost accounting across model families

Use total cost to reach a specified quality at the required NFE:

`backbone training + reference/teacher generation + tuning or distillation + evaluation and search`

The [SoFlow](https://arxiv.org/html/2512.15657v2#A1) and [MeanFlow](https://arxiv.org/html/2505.13447v1#A1) CIFAR recipes both prescribe 800K updates at batch 1,024. The literature audit therefore provides no basis for expecting a method-name change to solve our budget problem. Its iCT and Shortcut comparisons use different budgets and, in some cases, different datasets, latent representations and architectures. They warrant controlled experiments rather than a portable runtime claim.

The audit separates three routes:

| Route | What gets learned | Cost that remains |
|---|---|---|
| Direct few-call training | A generator able to make large jumps | Learning the dataset distribution |
| Distillation or consistency tuning | A student, or adapted generator weights | Training a suitable starting model and, where required, producing targets |
| Sampler calibration | Times or path coefficients around a fixed generator | Training that generator, calibration/reference generation, and evaluation |

Pretrained ECT/SFD results are evidence about inexpensive adaptation after suitable initialization. FastGen's scratch requirement keeps the cost of obtaining suitable generative weights inside our project. A locally trained teacher could support a later stage; the stopped CIFAR checkpoint is not evidence of such a teacher for Pokémon. A100/A6000/H100/TPU timings and published CIFAR FIDs are not TITAN/Pokémon predictions.

## What learning a schedule can change

Three distinct quantities need separate names:

| Schedule | Role in our implementation | Consequence of changing it |
|---|---|---|
| Training curriculum | `get_middle_time` shrinks the consistency gap as optimizer updates progress | Changes optimization and requires training evidence |
| Noise/interpolant path | `Scheduler` defines clean-image/noise mixing and the solution parameterization | Requires a mathematically consistent model/sampler transformation |
| Inference times | The sampler chooses the noise level for each denoising call | Can be compared while keeping generator weights frozen |

[LD3](https://arxiv.org/html/2405.15506v4) optimizes discretization and query times for a fixed denoiser. Its roughly six-minute, one-A100 CIFAR result is at 10 NFE and starts from a trained model. [BézierFlow](https://arxiv.org/html/2512.13255v3) learns a transformed interpolant path; it reports eight-minute CIFAR calibration at four NFE on an A6000, including an EDM/iPNDM result of FID 6.93. These demonstrate useful sampler adaptation, with backbone training still required. They do not demonstrate faster SoFlow backbone convergence. BézierFlow's reported CIFAR one-NFE FID of 125.03 also leaves a substantial one-call quality challenge.

Our [sampler](../soflow_fastgen/sampling.py) makes an endpoint prediction, then adds noise before the next endpoint prediction. It is a stochastic denoise-and-renoise procedure. An ODE discretization method needs adaptation to this procedure. The diagonal `predict_velocity(x,t)` offers another possible interface, but its accuracy and any higher-NFE reference solver must be measured; more calls do not automatically make a better teacher for this finite-interval model.

For N calls, the present sampler has N−1 intermediate re-noising levels. Four calls use levels 0.75, 0.5 and 0.25. Three fixed scalar levels are an attractive low-dimensional target for a later comparison. At one call, the current endpoints are fixed at 1 and 0, leaving no intermediate level to adjust. Broader path transformations can retain parameters at one NFE; they are separate experiments and are not ruled out by this narrower observation.

Our implementation currently uses `torch.no_grad()` for generation. Gradient-based calibration would need a separate differentiable path through inputs and times while weights remain frozen. Its activation memory must be measured under the same 20 GB constraint: a small number of trainable scalars does not imply a small backward-pass memory footprint.

## Proposed next decision, after baseline evidence

Continue the current baseline's scheduled grids and development FID, separately tracking one-call and few-call behavior. Aggregate FID measures an image distribution and cannot by itself establish that each requested category is correct. Retain the paired-noise category panels and inspect recognizability, label changes, diversity and possible memorization alongside FID.

| Observed outcome | Most informative next investigation |
|---|---|
| Useful few-call samples; small changes in sampling improve quality | Freeze a checkpoint and compare a small family of global re-noising schedules |
| Useful few-call samples; one-call output remains poor | Treat one-call learning or local-teacher tuning as its own budgeted problem |
| Weak structure or class fidelity across all NFEs | Investigate optimization, conditioning, data/augmentation and training duration before assuming a larger model is the remedy |
| Better training behavior but worsening held-out quality/diversity | Investigate overfitting, duplication and regularization before adding capacity |

A first sampler experiment could compare a few predeclared power-spaced time grids with the existing uniform grid at the same NFE and frozen checkpoint. This requires no new learned network, teacher or perceptual-loss model. Fix random draws for fair comparisons, use only training-derived data for selection, and predefine separate calibration/verification data and new seeds before searching. Repeated selection on the same development FID is not an unbiased final assessment. Preserve the existing 3,020-image baseline metric when reporting any additional smaller-partition metrics.

Gradient-based LD3/BézierFlow-inspired calibration can follow if that inexpensive sensitivity check supports it. Their published image experiments use LPIPS. Our current local protocol permits only the official FID Inception as a pretrained component; importing LPIPS would change that protocol and add a dependency. The official README explicitly bans pretrained image-generation models, while the stricter local restriction is recorded in AGENTS.md. Keep that distinction clear. Use an allowed objective or resolve the protocol before such an experiment; avoid optimizing against the official validation set.

Keep any learned schedule as coefficients and arithmetic with verified total backbone-call counts; an auxiliary trained scheduler network could consume the challenge's NFE budget. Recheck export compatibility, actual parameter count, training/calibration peak memory, inference time, and total search/evaluation cost. All proposals here remain unlaunched and require a defined allocation within a subsequent experiment budget.

The decision criterion is useful, diverse, category-appropriate samples at one and at most four NFEs within the available compute. The current baseline will supply the missing evidence about time to that quality. Its results will guide whether to spend the next budget on sampling, the training recipe, or model capacity.
