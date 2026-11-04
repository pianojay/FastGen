# TA-003 CycleGAN preparation

Preparation only: no cohort training, GPU use, or final domain selection has run.

Reproduce CPU verification from workspace root:

```bash
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-003/test_setup.py
```

`networks.py` contains exact selected definitions from the pinned upstream
`junyanz/pytorch-CycleGAN-and-pix2pix` commit
`2a7afba2895d52556dd5dfe07e8555ef657ced6f`. See `upstream_provenance.json`
for source hashes, URLs and `UPSTREAM_LICENSE` for all original notices.
The architecture is instantiated in `engine.py`; the trainer and sampler are
local adaptations of the upstream CycleGAN optimization equations. Replay
behavior is implemented locally from its standard algorithm.

RGB64 has a 16px bottleneck after two stride-2 convolutions. Six residual blocks
retain the upstream smaller ResNet option. Width is 64 throughout the base
configuration; InstanceNorm has neither affine parameters nor running stats.
There is no dropout. Four discriminator convolutions use strides 2,2,1,1,
producing 14x14 logits and a nominal convolutional receptive field of 34px.
This reduces the usual three-downsample 70px PatchGAN to suit 64px inputs.
InstanceNorm uses whole-feature-map statistics, so 34px is not a claim that
its full mathematical dependence is confined to a 34px input region.

Generators minimize two LSGAN terms, two L1 cycle terms weighted 10, and two
L1 identity terms weighted 5 (=10*0.5). Each discriminator minimizes half the
sum of its real-target-one and replay-fake-target-zero squared errors. Replay
holds 50 detached samples per domain, returns the current sample with half
probability after filling, otherwise returns/replaces a uniformly chosen old
sample. There are no class labels in networks or losses, external features,
pretrained weights, EMA, perceptual objectives, or auxiliary networks.

`ManifestSampler` currently accepts JSON `records` entries with absolute `path`,
`sha256`, `partition` (A1 only), `class_id`, `domain` (D0/D1), and `role`
(optimization/diagnostic). Native dimensions must be 64x64; decoding converts
to RGB, then maps to [-1,1]. Every optimization class must contain both domains.
Classes are sampled uniformly with replacement, then one image per domain
uniformly and independently within that class. Optional flips are independent.
Diagnostic records never enter optimization. Image hashes are checked before
sampling. This provisional schema awaits the final master domain manifest.

The engine supports batch16; Adam uses lr 0.0002 and betas (0.5,0.999).
One-based updates 1..10000 use the constant rate; updates 10001..20000 use
`0.0002*(20000-update)/10000`. Thus the last allowed update has zero LR.
Final contract must accept or supersede this explicit endpoint convention.
`Trainer.step`, `evaluate`, `save`, and `load` provide optimization, both
translation/cycle outputs and atomic checkpoint recovery. Saved state includes
all networks, both Adam optimizers, exact update/schedule position, two replay
pools and RNGs, global Python/NumPy/torch RNGs, optional CUDA RNGs, sampler RNG,
manifest/contract hashes and configuration. Only trusted local checkpoints may
be loaded. GPU recovery remains untested; deterministic CPU recovery is exact.

No executable dataset training entrypoint is provided before final contract,
GPU lease/preflight, budget accounting, checkpoint cadence and EDA freeze.
Those orchestration controls are prerequisites to a subsequent authorized run.
CPU tests use native full-width networks and batch1 optimizer calls, and test
the nominal batch16 sampler. They do not establish GPU runtime, VRAM, actual
training stability, image quality, preservation, augmentation value, or FID.
Synthetic checkpoint payloads are deleted after verification; reproducible
source, checkpoint hash and exact before/after state hashes are retained.
