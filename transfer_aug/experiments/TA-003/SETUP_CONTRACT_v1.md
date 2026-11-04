# TA-003 CycleGAN implementation preparation — revision 1

The user authorized CycleGAN as the next learned augmentation experiment,
scratch-trained within A1. This assignment prepares its isolated implementation
while a separate worker derives shared appearance domains. Write only this
experiment directory. Read MASTER.md and EXPERIMENT_TEMPLATE.md; do not edit
shared docs/ledger, sibling projects, existing contracts or environment.

Read pinned upstream `audit_sources/cyclegan` (commit
2a7afba2895d52556dd5dfe07e8555ef657ced6f). Implement a standalone CPU-testable
adaptation using torch/numpy/Pillow already installed. No package installation,
external weights, GPU allocation or dataset/model training during preparation.
Attribute upstream code and preserve applicable license notices locally.

Proposed pilot architecture: RGB64, ResNet generator with 6 residual blocks,
ngf64, two stride-2 downsamples, reflection padding, instance normalization
(no running statistics or affine parameters), no dropout. Two generators,
two 2-layer width64 PatchGAN discriminators (34px receptive field). Preserve
standard CycleGAN LSGAN, L1 cycle weight10 each direction, identity factor0.5,
50-image replay pools. Explain 64px/six-block/34px-discriminator adaptations.

Prepare training/evaluation code for a frozen manifest with D0/D1 and
optimization/diagnostic roles. Sample class IDs uniformly, then one unpaired
image from each domain of the same class; do not wire class labels into losses.
Batch16; Adam lr0.0002, betas(.5,.999); 20,000 maximum updates; constant LR for
10,000, linear decay thereafter; seed20261008. Optional horizontal flips are
independent per sampled image (p=.5), no random crops/resizing beyond RGB64.
No pretrained perceptual losses, classifiers, FID, external features or EMA.

Prepare CPU synthetic tests: shape/range/finite checks, actual registered
parameter count per network and whole framework, gradients, replay behavior,
complete optimizer/RNG/sampler checkpoint recovery. Worker owns details and
verification. Return <=500-word SETUP_SUMMARY.md and setup_result.json with
proof paths and blockers. Master freezes the final training contract and data
before GPU profiling/training starts. GPU 1 and an eight cumulative GPU-hour
ceiling are planned, not permission to start before the final contract.
