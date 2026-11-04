# Literature audit: training cost of direct few-NFE models, distillation, and sampler optimization

**Date:** 2026-09-24, Asia/Seoul  
**Scope:** Selected primary papers and official code; literature review following the stopped SoFlow/CIFAR-10 experiment  
**Local evidence:** [Formal experiment report](EXPERIMENT_REPORT.md)  
**Execution:** Documentation only; no training, profiling, or image-quality evaluation performed for this audit

## 1. Conclusions

The available evidence does **not** identify SoFlow as uniquely expensive to train. Its closest comparison, MeanFlow, prescribes the same CIFAR-10 update and image budget. Some alternatives prescribe fewer image presentations, and Shortcut Models have a relatively inexpensive objective, but none of those facts establishes a one-day reproduction on our TITAN RTX. The detailed source comparisons are below.

Distillation can greatly reduce the **additional cost after a suitable model has been trained**. This advantage depends on the method, target quality, and permitted inference NFE. Older distillation recipes remain substantial training jobs. Several newer results make two-NFE generation a plausible small-budget research direction when pretrained initialization is permitted.

**Follow-up:** LD3 and BézierFlow add a third route: optimize a sampler around a fixed trained generator. Section 6 covers their scope, costs, and applicability to this checkout. They broaden the options for obtaining few-step samples, while leaving the initial generator-training expense in the project budget.

For a new dataset with no suitable pretrained model, the accounting must include learning the original distribution. Our step-5 checkpoint does not provide evidence of a usable teacher. The experiment's approximately 126-day estimate remains specific to its full prescribed schedule; it does not establish a minimum time to useful image quality.

## 2. How costs are compared

- **NFE** counts network evaluations during generation. It does not count optimizer updates during training. Two guided evaluations can also make one sampling step cost two NFEs.
- **Image presentations = optimizer updates × effective batch**, when the training loop uses one such batch per update. This counts repeated presentations, not unique images or all network passes. Synthetic trajectories require separate accounting.
- **Time/update** depends on architecture, resolution, precision, target construction, accumulation, compiler, and hardware. Parameter count alone is insufficient.
- **GPU-hours = GPU count × wall-clock hours** for a fixed accelerator type. TPU chip-hours, A100-hours, H100-hours, and TITAN RTX-hours cannot be converted one-for-one.
- **Total project cost** includes teacher pretraining or adaptation, target generation, student optimization, evaluation, and any search. Count online teacher calls inside the distillation stage only once. Shared pretrained weights can eliminate that training expense for the current user while retaining its historical cost.

The tables are recipe comparisons, not a controlled quality-versus-compute ranking. Published FIDs provide context for what each budget bought; architecture and evaluation differences limit comparisons. A published final schedule is not a measured minimum convergence time.

## 3. Local reference point

Our unconditional, pixel-space CIFAR-10 experiment used a roughly 55.93M-parameter UNet, FP32, effective batch 1,024, and microbatch 64 on one TITAN RTX. The selected microbatch improved short-run throughput by approximately 15.8% over 16. Sustained production timing was 13.603 seconds/update, or 75.28 image presentations/second.

The requested 800,000-update schedule therefore meant 819.2M presentations, 16,384 equivalent passes over 50,000 images, and approximately 125.95 days excluding periodic overhead. One day at that throughput permits about 6,352 updates, or 0.794% of the schedule. These are local measurements and arithmetic, not paper-reported runtimes. Only 130 updates were logged, generation quality was unevaluated, and the latest recoverable main checkpoint is step 5. See the [report and archived evidence](EXPERIMENT_REPORT.md).

## 4. Direct training for few-NFE generation

### Prescribed budgets

| Method / experiment | Updates × effective batch | Presentations, calculated | Hardware/runtime evidence and source |
|---|---:|---:|---|
| SoFlow, CIFAR-10 | 800,000 × 1,024 | 819.2M | H100 GPUs; CIFAR wall time and GPU count unspecified. [SoFlow v2, Appendix A](https://arxiv.org/html/2512.15657v2#A1) |
| MeanFlow, CIFAR-10 | 800,000 × 1,024 | 819.2M | About 55M parameters in pixel space. [MeanFlow v1, Appendix A](https://arxiv.org/html/2505.13447v1#A1) |
| Improved Consistency Training, CIFAR-10 main iCT/iCT-deep results | 400,000 × 1,024 | 409.6M | A100 cluster; main recipe differs from batch-512 ablations. [iCT v1, Appendix B](https://arxiv.org/html/2310.14189v1#A2) |
| Original Shortcut Models, CelebA-HQ 256 | 400,000 × 64 | 25.6M | DiT-B in pretrained VAE latents. [Shortcut v3, Appendix B](https://arxiv.org/html/2410.12557v3#A2) |
| Original Shortcut Models, ImageNet 256 | 800,000 × 256 | 204.8M | DiT-B main comparison; latent representation. Typical runs take 1–2 days on TPUv3 nodes; node size is unspecified there. [Shortcut v3, Appendix B](https://arxiv.org/html/2410.12557v3#A2) |

### Does switching to Shortcut solve our cost problem?

Shortcut's objective uses bootstrap targets for one-quarter of its batch. The authors approximate its overhead as 16.7% over ordinary flow/diffusion training: forward cost 1, backward cost 2, and two extra target forwards on one-quarter of the batch, totaling 3.5 versus 3 units. Its image benchmarks do not provide a matching CIFAR-10 run. [Shortcut, Section 3.1](https://arxiv.org/html/2410.12557v3#S3.SS1)

For our local unconditional SoFlow path, `get_consistency_loss` in [models.py](../../../SoFlow/models.py) executes one differentiable network forward and one target forward without gradients; the trainer performs one backward. Applying the same rough backward-cost assumption gives 4 units. **Our inference:** 3.5/4 = 0.875, or roughly 12.5% less network work for Shortcut at otherwise identical settings. This is an illustrative operation count, not a benchmark or convergence prediction. It excludes implementation overhead and any change in required training length.

The smaller published Shortcut presentation budgets are interesting, but changing from CIFAR pixels/UNet to other datasets/VAE latents/DiT changes the comparison. Adapting Shortcut to our CIFAR setup would be a new experiment. The published evidence does not establish a dramatic runtime saving at matched CIFAR quality.

### Is SoFlow itself an inefficient choice?

SoFlow's own controlled PyTorch comparison reports 2.94 versus 2.39 updates/second and 35.44 versus 51.45 GB for SoFlow with efficient attention versus MeanFlow with math attention. That is approximately 23% higher throughput and 31% lower peak memory in that benchmark. Its configuration is insufficiently specified to convert this into our UNet runtime. [SoFlow, Appendix D](https://arxiv.org/html/2512.15657v2#A4)

**Assessment:** MeanFlow is not supported as a training-budget shortcut. iCT halves the prescribed CIFAR presentation count, which is a meaningful reduction but far short of the roughly 126-fold reduction needed to fit our full workload into one day. Halving our workload at unchanged throughput would still mean about 63 days; this is only a scale illustration, not an iCT timing estimate. The main obstacle is attempting a large benchmark budget in the measured single-device configuration. This audit does not establish that our implementation is optimally efficient.

## 5. Distillation and pretrained consistency tuning

| Method | Training-cost evidence | Interpretation |
|---|---|---|
| Original Consistency Distillation (CD) | CIFAR: 800,000 × 512 = 409.6M student presentations; 8 GPUs; pretrained EDM teacher. [Consistency Models, Table 3](https://proceedings.mlr.press/v202/song23a/song23a.pdf#page=25) | Distillation itself remains long, and teacher work is additional. The label “distillation” does not imply a cheap run. |
| Progressive Distillation (PD) | CIFAR teacher: 800,000 updates; batch 128. Each halving stage uses 50,000 updates, with 100,000 for each of the last two stages. Training plus distillation takes about a day on **8 TPUv4 chips**. [PD, Appendix E](https://arxiv.org/pdf/2202.00512#page=15) | An example reporting end-to-end time, but on substantially different hardware and architecture. Stage count depends on target NFE. |
| Simple and Fast Distillation (SFD) | CIFAR: **2 NFE, FID 4.53, 0.64 A100-hours**; about 200,000 generated trajectories at batch 128. A separate second-stage one-NFE result is FID 5.83 with a listed 4.88 A100-hours. [SFD v1, Table 2 and Section 4](https://arxiv.org/html/2409.19681v1#S4) | Strong evidence for cheap few-step adaptation from pretrained EDM weights. The two-NFE budget is not a one-NFE result. |
| Easy Consistency Tuning (ECT) | CIFAR: **8,000 × 128 = 1.024M presentations**, 2-NFE FID **2.73**, within one hour on one A100 40GB. [ECT v2, Appendix B](https://arxiv.org/html/2406.14548v2#A2) | Initializes from a pretrained diffusion model; no frozen diffusion teacher is queried for targets. This is pretrained consistency tuning, distinct from teacher-based distillation and from scratch training. |

SFD's times are normalized A100 cost estimates: Appendix B describes timing iterations on two A100s and multiplying by device count; main experiments use up to four GPUs. Teacher pretraining is excluded. That appendix estimates CIFAR diffusion pretraining at about 200 A100-hours. It also estimates original CD at about 1,156 A100-hours, using its own reimplementation. These are authors' estimates, not our measurements or universal hardware timings. [SFD, Appendix B](https://arxiv.org/html/2409.19681v1#A2)

ECT also investigates paying for pretraining: one controlled ablation uses pretraining plus tuning equal to 56.25% of the from-scratch iCT budget and improves its reported DINOv2 distribution metric. Thus, benefits need not disappear when pretraining is counted, but that result does not establish a one-day TITAN experiment or a matched SoFlow comparison. [ECT, Appendix B, pretraining ablation](https://arxiv.org/html/2406.14548v2#A2)

### An official-script budget discrepancy

ECT's current [run_ecm_1hour.sh](https://github.com/locuslab/ect/blob/main/run_ecm_1hour.sh) supplies `--duration=25.6 --batch=128` and transfers a pretrained unconditional CIFAR EDM checkpoint. In [ct_train.py](https://github.com/locuslab/ect/blob/main/ct_train.py), duration is measured in millions of images. The [training loop](https://github.com/locuslab/ect/blob/main/training/ct_training_loop.py) stops at that image count, not after an hour. Therefore the script specifies **200,000 updates**, 25 times the paper's 8,000-update result. A future reproduction must explicitly set the intended image/update limit and account for evaluation time; the filename is insufficient. These repository observations were checked on the audit date and may change.

## 6. Learning the sampler: LD3 and BézierFlow

**Added 2026-09-24 following the user's scheduler-optimization question.** This extension is a literature and source-code inspection, with no additional model runs.

### Published evidence

| Method | Learned component | Illustrative CIFAR-10 result |
|---|---|---|
| LD3 | Discretization and network-query times; fixed denoiser. Soft teacher forcing also optimizes training-sample initial states. | iPNDM: FID 2.38 at 10 NFE, approximately 6 minutes on one A100. [LD3 v4, Sections 4 and Appendix C/D](https://arxiv.org/html/2405.15506v4) |
| BézierFlow | Bézier coefficients for an interpolant scheduler, transforming the sampling path around a fixed model. | EDM/UniPC: FID 2.09 at 10 NFE; CIFAR scheduler training takes 15 minutes on one A6000. At 4 NFE, EDM/iPNDM reaches FID 6.93; the reported CIFAR training cost is 8 minutes. [BézierFlow v3, Tables 1 and 15](https://arxiv.org/html/2512.13255v3) |

BézierFlow's low-NFE appendix reports CIFAR EDM FID 125.03 at one NFE. Thus, its reported improvements do not establish competitive one-call generation. Its continuous scheduler remains parameterized at one NFE, even when there are no intermediate integration nodes to move. [BézierFlow, Appendix E.1](https://arxiv.org/html/2512.13255v3#A5.SS1)

The official workflows separately require pretrained checkpoints, teacher-output generation, sampler optimization, and FID evaluation. BézierFlow's CIFAR example generates 400 pairs; LD3's example generates 100 and splits them into 50 training and 50 validation pairs. These are synthetic calibration pairs, not evidence that a generator can learn CIFAR from 50 or 200 real examples. [BézierFlow implementation](https://github.com/KAIST-Visual-AI-Group/BezierFlow), [LD3 implementation](https://github.com/vinhsuhi/LD3)

**Cost interpretation:** Account for backbone training, reference generation, calibration, and evaluation separately. The minute-scale numbers describe sampler learning; they do not establish end-to-end costs on our TITAN. Fewer trainable parameters also does not eliminate network computation: gradients must pass through the sampler to its learned inputs. Compare methods at the same inference NFE and quality criterion. A 10-NFE success cannot establish a 1–4-NFE success.

### Three different meanings of schedule in our checkout

| Quantity | Current implementation | Relevance of these papers |
|---|---|---|
| Training curriculum | `get_middle_time` uses optimizer progress and `l_schedule` to choose the consistency gap. | Neither method, as published, learns this curriculum or reduces our 800,000-update prescription. |
| Noise/interpolant coefficients | `Scheduler.alpha/beta` define how clean images and noise are mixed. | BézierFlow changes an inference path with a corresponding model-coordinate transformation. Replacing these functions alone would not implement it. |
| Inference times | `eval_time` selects the noise levels used during repeated generation. | Optimizing these is a plausible local adaptation once usable weights exist. |

Sources for these local observations: [models.py](../../../SoFlow/models.py), [train.py](../../../SoFlow/train.py), and [inference.py](../../../SoFlow/inference.py).

### Compatibility with SoFlow: our assessment

SoFlow exposes a finite-interval `solution_operator(x,t,s)`. Its current inference loop predicts all the way to time zero, adds fresh noise at the next `eval_time`, then predicts again. This is a stochastic denoise-and-renoise procedure. Directly inserting an ODE discretization learner would change the sampling algorithm and requires a defined adapter and objective.

For two to four calls, a modest extension could learn the intermediate re-noising levels while freezing the network. Use controlled random draws during calibration, a training-derived development criterion, and an independent evaluation. This is an untested proposal inspired by schedule learning, not a claim of reproducing either paper. The current `torch.inference_mode()` wrapper would also need a differentiable calibration path; freezing weights is different from disabling all gradients.

Alternatively, `predict_velocity` queries the diagonal `network(x,t,t)`, offering a possible ODE interface. Its accuracy and the quality of a reference solver would need to be established before using it as a teacher. The existing smoke test provides no such evidence. More integration steps are not guaranteed to produce a better teacher from a model trained for finite jumps.

At one NFE, our existing sampler makes a fixed call from time 1 to 0. There is no intermediate re-noising level to optimize. A broader transformation of the sampler would be a different experiment. Changing the backbone's training noise schedule or jointly learning a curriculum is also a separate research proposal; the two cited papers do not demonstrate faster SoFlow backbone convergence.

**Recommendation:** Include conventional diffusion/flow training followed by sampler calibration among future alternatives, especially when 4–10 NFE is acceptable. This could be economical if useful multi-step backbone quality is reached sooner, but that must be measured. For the current checkpoint, first establish generator quality; calibration cannot be assumed to recover a data distribution that has not been learned. Under FastGen's existing rules, a backbone trained locally from scratch could retain compliant initialization, but downloaded generative weights and the papers' pretrained LPIPS loss require separate consideration: the current instructions permit only the official pretrained FID Inception. A permitted calibration loss would change the reproduction recipe. No FastGen policy or experiment was changed here.

## 7. What follows for our next experiment

| Constraint | Assessment / recommended direction |
|---|---|
| A suitable pretrained CIFAR model is permitted; 2–4 NFE is acceptable | Prioritize ECT and SFD for an explicitly capped reproduction. Their reported budgets justify investigation. Measure local throughput before assigning a TITAN runtime. |
| A trained diffusion/flow backbone is available; 4–10 NFE is acceptable | Include LD3/BézierFlow sampler calibration as inexpensive candidates. Count target generation and evaluation, and compare at matched NFE. |
| Every generative weight must start from scratch | Account for teacher training inside the same budget. Compare direct training with a diffusion-pretraining-plus-tuning pipeline; neither route has a demonstrated one-day result on our hardware. |
| Exactly one NFE is mandatory | Use a separate quality/cost target. Successful two-NFE results do not establish equivalent one-NFE quality. |
| Different dataset or conditioning | A CIFAR checkpoint is not evidence of a suitable teacher. Include adaptation/training cost and assess whether its distribution and conditioning match the task. |
| Faithful paper reproduction | Preserve the selected paper's model, batch, schedule, and evaluation protocol. A shorter schedule is an adapted experiment with its own outcome. |

The current FastGen [AGENTS.md](../../AGENTS.md) requires all generative weights to start from scratch on its training-derived optimization partition. Downloaded pretrained generative teachers or ECT initialization are outside that protocol. A locally trained teacher could satisfy the initialization rule but would consume the experiment budget. The separate CIFAR audit does not change these constraints or the Pokémon experiment.

For resource-limited development, the decision metric should be **time to a predefined acceptable sample-quality result at the required NFE**, including evaluation. A finite loss smoke test answers whether training executes; a full leaderboard recipe answers a different research question. Our earlier record establishes the first and estimates the cost of the second. It leaves intermediate useful quality unresolved.

## 8. Audit limits and provenance

- This is a focused audit of nine comparison methods plus SoFlow, including closely related diffusion/consistency methods and learned samplers. It is not an exhaustive review of every flow objective or the latest distillation variant.
- Paper versions are fixed in the links: SoFlow v2, MeanFlow v1, Shortcut v3, iCT v1, ECT v2, SFD v1, LD3 v4, BézierFlow v3, plus the original CD and PD publications.
- Table presentation counts and cost-ratio examples are our arithmetic. Hardware timings and quality scores are attributed to their authors.
- No matched TITAN benchmarks, new FIDs, or model changes were made. Published final scores do not supply our missing early-training quality measurements.
- The closed CIFAR experiment remains closed. This document supports choosing a future experiment; it does not launch one.
