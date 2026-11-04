# TA-004 CUT implementation preparation — revision 2

The user authorized CUT as the next learned augmentation experiment,
scratch-trained within A1. Prepare its isolated implementation while shared
appearance domains are derived separately. Write only this experiment
directory. Read MASTER.md and EXPERIMENT_TEMPLATE.md; all shared docs/ledger,
sibling projects, contracts and environment are read-only to this worker.

Read pinned `audit_sources/cut` (commit
b3ac297708dfb6f7589d04662277e53c0d579c27). Implement a standalone CPU-testable
adaptation using installed torch/numpy/Pillow. Attribute upstream code and
preserve license notices. No package installation, external weights, GPU
allocation, dataset training or extra agents during preparation.

Proposed architecture: RGB64 ResNet G, six residual blocks, ngf64, instance
normalization without affine/running statistics, no dropout; width64 2-layer
PatchGAN (40px receptive field including the upstream blur filters). Preserve CUT's anti-aliasing and encode-only
feature interfaces, learned 256-dimensional patch MLPs, per-image negatives
and stopped key gradients. Use nce_layers=[0,4,8,12,16], up to 256 patches/layer,
temperature .07, lambda_GAN=1, lambda_NCE=1, nce_idt=True, LSGAN, no image pool,
and no FastCUT flip-equivariance branch. Confirm actual layer shapes/indices
and count all MLP parameters after initialization; report any incompatibility
before changing the architecture or feature mapping.

The planned pilot is one-direction D0->D1 using the same optimization/diagnostic
manifest as CycleGAN. Uniform class sampling; one unpaired D0 and D1 image
from the same class per draw, with no class-conditioned loss. Batch16;
Adam lr.0002, betas(.5,.999); 20,000 maximum updates; constant LR10,000 then
linear decay; seed20261008. Independent p=.5 horizontal flips only. No EMA,
pretrained feature/loss network, classifier or FID.

Prepare CPU synthetic verification: tensor shapes/finite outputs, feature
index mapping, per-image negatives, stopped keys and nonzero G/MLP gradients,
actual whole-framework parameter count and optimizer/RNG/sampler checkpoint
recovery. Worker owns detailed verification. Deliver SETUP_SUMMARY.md <=500
words and setup_result.json with proof paths/blockers. Master must freeze the
data and final training contract before profiling or training. GPU 2 and an
eight cumulative GPU-hour ceiling are planned, not permission to launch yet.

Revision 2 corrects the receptive-field accounting: 34 pixels applies to a
non-antialiased strided PatchGAN. Retained CUT antialias filters expand it to
40 pixels. Preserve those filters; do not silently match CycleGAN by deleting
upstream antialiasing. Revision 1 is retained as superseded history.
