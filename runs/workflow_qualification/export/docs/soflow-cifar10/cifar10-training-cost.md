# CIFAR-10 training scale and runtime

Recorded: 2026-09-24. Canonical document location: `FastGen-SoFlow/docs/soflow-cifar10/`. These measurements describe the local SoFlow CIFAR-10 run on GPU 0, an NVIDIA TITAN RTX. They are separate from the Pokémon/FastGen adaptation in the sibling `FastGen-SoFlow` checkout.

## Why the full run takes months

The main cost is the prescribed **800,000 optimizer updates at an effective batch size of 1,024**, executed on one GPU. CIFAR-10 contains only 50,000 training images, but this schedule processes them repeatedly:

| Quantity | Value |
|---|---:|
| Training images | 50,000 |
| Effective batch per optimizer update | 1,024 |
| Scheduled optimizer updates | 800,000 |
| Total image presentations | 819,200,000 |
| Equivalent complete passes over the dataset | 16,384 |

Calculations:

```text
800,000 updates × 1,024 images/update = 819,200,000 image presentations
819,200,000 / 50,000 = 16,384 equivalent dataset passes
```

The loader uses `drop_last=True`, so a loader epoch contains `floor(50,000 / 1,024) = 48` updates, processing 49,152 images. The full schedule therefore spans approximately **16,667 loader epochs**. The difference from 16,384 equivalent passes comes from dropping the final incomplete batch each epoch.

At 13.25 seconds/update, one loader epoch takes about **10.6 minutes**. The long total runtime comes from the number of epochs. Stopping is controlled by `total_steps`, not by a requested epoch count. Reducing dataset size alone while keeping batch size and total updates fixed would largely preserve the amount of computation and increase data reuse.

The local UNet has approximately **55.93 million parameters**. Its consistency objective performs two network forward passes, one with gradients and one without, followed by a backward pass. With microbatch 64, each optimizer update requires 16 sequential chunks to accumulate the effective batch of 1,024. These chunks fit the work into GPU memory; they do not reduce the number of images processed.

## What the paper reports

The original paper's CIFAR-10 implementation details specify **batch size 1,024 and 800K iterations**, consistent with MeanFlow. Appendix A also states that the experiments used **NVIDIA H100 GPUs**. These settings appear in both the original submission and the revised paper:

- [Original SoFlow paper, Appendix A (v1)](https://arxiv.org/html/2512.15657v1#A)
- [Revised SoFlow paper, Appendix A (v2)](https://arxiv.org/html/2512.15657v2#A)

The paper supports this training scale. It does **not provide an explicit total CIFAR-10 elapsed training time or the GPU count for that experiment**. Our approximately four-month estimate is a local hardware extrapolation, not a runtime reported by the authors.

[Appendix D, Table 5](https://arxiv.org/html/2512.15657v2#D) reports 2.94 iterations/second for SoFlow with efficient attention. Its benchmark configuration is not specified sufficiently to treat that figure as the throughput of our CIFAR-10 UNet. Do not use it to calculate a directly comparable CIFAR-10 completion time.

“One-step” describes generation after training; it does not describe the number of training updates.

## Local throughput measurements

All candidates used the actual FP32 training loop, effective batch 1,024, and the original 800,000-update schedule. Each successful candidate ran five updates; the reported time is the median of updates 3–5 after two warmup updates. Device memory was sampled using `nvidia-smi` and includes more than PyTorch's live tensor allocation.

| GPU microbatch | Accumulation passes/update | Peak device memory | Median seconds/update |
|---|---:|---:|---:|
| 16 | 64 | 4.09 GiB | 14.73 |
| **64 — selected** | **16** | **10.33 GiB** | **12.72** |
| 128 | 8 | 18.72 GiB | 13.22 |
| 256 | 4 | Out of memory | — |

Microbatch 64 improved measured throughput by approximately **15.8%**, reducing time per update by about **13.6%** versus microbatch 16. Microbatch 128 used more memory and was slightly slower in this short comparison. The resumed run showed approximately 99% GPU utilization. Higher memory occupancy alone did not imply higher throughput.

The 12.72-second comparison result projects to **117.8 days** for 800,000 updates. Subsequently, the production log at **20:50:13 KST on 2026-09-24** recorded **132.51 seconds for 10 updates**, or **13.251 seconds/update**:

```text
800,000 × 13.251 / 86,400 = 122.69 days
```

Allow approximately **118–123 days as an early estimate**, plus periodic sampling, checkpoint overhead, and interruptions. This is not a guaranteed range; sustained throughput and hardware conditions may change it. Sampling generates 32 demo images every 1,000 updates and 50,000 images every 25,000 updates.

The initial 122-day estimate came from the very first update. The later 118-day estimate came from the short microbatch comparison. The subsequent 123-day estimate came from a production logging interval; these are different measurements of the same run configuration.

## Implications and evidence

Preserving the full paper schedule on one TITAN RTX entails over 16,000 equivalent dataset passes. Substantially shorter elapsed training requires a smaller training budget or substantially more compute. Changing the effective batch or shortening the schedule changes the experimental setup; microbatch tuning preserves the effective batch and offers a modest speed improvement. A successful training loop does not establish convergence or reproduce the paper's FID.

Source files and local evidence:

- [Run configuration](../../../SoFlow/cifar_gpu0.yaml): effective batch 1,024, microbatch 64, 800,000 updates.
- [Training implementation](../../../SoFlow/train.py): accumulation, stopping condition, and loader behavior.
- [Training run record](TRAINING_RUN.md): commands, modifications, environment repair, and restart history.
- [Raw throughput measurements](../../../SoFlow/runs/cifar10_throughput/results.json): timings, memory, and commands.
- [Production training log](../../../SoFlow/runs/cifar10_gpu0/log.txt): subsequent timings and progress.

The raw measurements and logs are local artifacts and may not accompany a repository clone. This document preserves the key numbers for that case. Later runtime reports should identify their measurement window and distinguish measured throughput from projected completion time.
