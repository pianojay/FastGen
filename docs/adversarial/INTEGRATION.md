# SoFlow endpoint adversarial integration

Current contract, October 4, 2026. This branch implements components and bounded
smoke verification. It does not define or authorize a post-training experiment,
FID study, warmup/ramp schedule, or autonomous research loop.

## Decisions and preserved context

The branch starts at stopped-C3 commit `53882e7ef8fe901b6a3a89bef23a7df946bc68b4`.
S1/S2/S3 completed 40k updates, with paired-development 1/4-call FIDs respectively
82.3062/29.4079, 74.8758/26.5204, and 66.9684/25.0072. Those results and their
artifacts remain untouched. C3 was stopped after 65 logged updates; only its
step-zero checkpoint was recoverable. Its objective mixture is not used here.

[The preserved theory report](soflow_adversarial_theory.md) is byte-identical to
`/tmp/soflow_adversarial_theory.md`, SHA256
`6d6a2d462df536557cf4172f2768b382f7fc825f4bcff198af37850cec0d78f6`.
Its original U160/projection-critic and training-recipe recommendations are
historical. **This document supersedes those choices:** S3 U192 EMA is the smoke
asset, D is a class-conditioned StyleGAN2 discriminator, and its minibatch
statistics require the group-based regularizer below. Local `/tmp` references
inside the preserved report are historical; its public discussion link remains
the durable source. No numerical training schedule in that report is active.

## Source and architecture

G uses the existing `Soflow` architecture unchanged: 64x64 RGB, 151 classes,
six augmentation coordinates, no CFG. Load the source architecture and EMA
weights into online G and new G-EMA, with fresh optimizers. U128/U160/U192 are
supported; their counts are 39,667,843 / 61,934,243 / 89,140,419.

Default source: `runs/research_s3_unet192_40k_seed42/checkpoints/step_040000.pt`,
SHA256 `9b95652fb5e503b2f877b26fd3652129abcce115bdea526db00ffe1f0daefd4a`.

D comes from [NVIDIA's official PyTorch StyleGAN2 implementation](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/d72cc7d041b42ec8e806021a205ed9349f87c6a4).
Selected source files are unmodified under `vendor/stylegan2_ada`; its provenance
manifest records the immutable revision, archive/file hashes, and retained
NVIDIA license. `soflow_fastgen/stylegan2.py` guards upstream import origins and
selects reference operators at runtime. It performs no extension compilation,
loads no pretrained features, and imports no mutable sibling checkout.

The configuration is ResNet D, RGB 64x64, c_dim=151, channel_base=32768,
channel_max=512, cmap_dim=512, eight conditioning-mapping layers, FP32,
no convolution clamp, default [1,3,3,1] resampling, and one minibatch-standard-
deviation channel with group size four. There are **28,930,048 D parameters**.
Small channel overrides exist only for numerical fixtures. Real and fake class
priors match: each fake uses its paired optimization image's class. This is
unpaired distributional supervision, not noise-to-image reconstruction.

[AYF Appendix F.2/F.3](https://arxiv.org/html/2506.14603v1) specifies a StyleGAN2
discriminator with RpGAN. Its exact D settings and adaptive-weight function are
not available in released code. Our chosen official PyTorch variant and class
conditioning are explicit adaptations; retaining SCM instead of external-teacher
EMD also makes this AYF-inspired post-training, not an AYF reproduction.

## Probability and gradient contract

The raw map is M(x,t,s,c,a)=x+(s-t)N(x,t,s,c,a). The adversarial fake is exactly
clamp(M(z,1,0,c,0),-1,1). Never move clipping into M. G uses eval mode for this
endpoint, with gradients enabled for its update. SCM keeps its training-mode
paired dropout and original augmentation metadata.

L_D_adv = mean softplus(d_fake-d_real), L_G_adv = mean softplus(d_real-d_fake).
L_D = L_D_adv + beta*(R1+R2). Larger D scores mean more real. D receives separate
real and fake forwards with matched class vectors. D updates detach G output
before creating fresh visible-image gradient leaves. G updates freeze D weights
but differentiate its fake-input path; the detached real score remains in the
relativistic difference. R2 never regularizes G or differentiates its clamp.

For a fixed ordered group X of four, R(X) is

    mean_j sum_pixels (d/dX_j sum_i D_i(X,C))^2.

This follows the upstream summed-logit input-gradient convention and includes
cross-image derivatives from minibatch statistics. It is not the Jacobian
diagonal of four independent critics. Logical groups are always four images,
including on accumulated updates; the wrapper rejects incomplete or repacked
groups. Each group contributes 4/B of its loss. Changing accumulation boundaries
must not change group membership. Real and fake never share one D forward.

SCM retains source t/s distributions, p=.75, eps=.001, cfg_scale=1, and the
exact surrogate gradient. Its maturity is fixed at r=.0003125:
l=min(t-r*(t-s),t-1e-4). This is independent of new optimizer counters. The
source config remains immutable; the new components never enter the C3 FM path.

Collect full effective-batch parameter gradients g_C and g_A before balancing:

    lambda = stopgrad(rho * min(10000, norm(g_C)/(norm(g_A)+1e-12)))
    g = g_C + lambda*g_A.

An exactly zero adversarial gradient gives lambda=0 and a diagnostic. Nonfinite
gradients stop the update. This full-vector rule is our explicit adaptation,
not a reconstruction of AYF's unspecified adaptive function. Scalar SCM loss
ratios are invalid for its signed surrogate. Log cosine, component/combined
norms, coefficient, and actual optimizer displacement. No FM, composition loss,
external teacher, fake-score network, or additional feature loss is present.

SCM augmentation is unchanged. D augmentation is separately specified identity
or shared per-pair horizontal flips. No G conditioning coordinates describe
those flips. Exact R penalties differentiate through the flip; geometric
`grid_sample` augmentation is excluded from that second-order path. Raw pixels
outside [-1,1] lose direct GAN gradient: record clipping and the fraction of
visible-image gradient energy masked by the clamp (reported as the mean of per-group fractions), without substituting an STE
or tanh. A successful backward pass does not establish useful learning.

## Interfaces, state, and disposable fixtures

`AdversarialComponents` loads an EMA source and offers separate single D/G
updates, save/load, and named RNG streams. The caller supplies the optimization
batch. G/D update and image-exposure counters are distinct; the maturity phase
does not advance. Saves occur at completed-update boundaries, carry weights,
both optimizers, EMA, component configuration, source identity, named RNG states,
and caller metadata. Changed recipe/source state is rejected on recovery.
The fixed-batch smoke stores its batch identity. A future data-loading training
loop must also save/restore its own sampler position; that loop is not built here.

Smoke-only fixture: B=8, SCM microbatch=2, D groups=4, rho=.1, beta=.1 (equivalent
to gamma=.2 under gamma/2), shared flip p=.25, G RAdam lr=2e-5 betas=(.9,.999),
D Adam lr=2e-5 betas=(0,.99), no weight decay, fixed EMA decay=.99, seed=20261004.
These constants exercise code; none is a recommended post-training recipe.
No automatic warmup, adversarial ramp, LR schedule, horizon, or search exists.

Inference continues using the protected Model interface and unchanged sampler.
G-only state dictionaries retain its existing architecture and sampling-NFE
buffer. Loading inference requires no D imports, optimizer, or EMA duplicate.

## Verification and operating boundary

Use the existing `fastgen_soflow` interpreter with `PYTHONNOUSERSITE=1`.
The sole GPU command is:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 /home/user/miniconda3/envs/fastgen_soflow/bin/python -m scripts.smoke_adversarial --run-dir runs/adversarial_integration_smoke_NEW_ID
```

The supervisor checks physical GPU 1 UUID
`GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888`, checks occupancy, and acquires the
existing GPU lease. It runs the contract suite and S3 worker under a 900-second
wall watchdog. S3 may attempt at most four D and three G updates, counting replay.
It refuses an existing output directory and never launches a successor experiment.
Use the recorded run rather than repeating it without a relevant code change.

The original 14,059/3,020 manifest is unchanged (SHA256
`825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`). Smoke reads
eight optimization images only. No development or official-validation images
participate, and no FID is computed. Both model counts must be <100M separately;
the simultaneous GPU workload must allocate <20,000,000,000 bytes.

Verification covers source hashes; upstream forward/first-/second-derivative
parity; RpGAN signs; exact group penalties and R2 routing; augmentation second
derivatives; fixed-group accumulation; zero-adversarial control; source endpoint
identity; D-only invariance; exact next-update recovery; independent G-only
one-/four-call loading; and parameter/memory checks. Test weights, checkpoints
and output logs live under ignored `runs/`. [The compact results report](SMOKE_RESULTS.md) records what passed and what remains unmeasured.
