# Pixel-space flow SiD/SiDA implementation

Implementation began on October 2, 2026 after explicit authorization. The dated
handoff and its provenance file remain historical source records. Current run
status belongs to `runs/current_status.json`, qualification to
`resources/qualification_summary.json`, and the frozen recipe to
`configs/campaign.json`. Large artifacts and operational logs are excluded from Git.

## Sources and architecture

`flow_sid/unet.py` adapts the DDPM++ primitives and SongUNet from the pinned
NVIDIA-derived [`training/networks.py`](../vendor/sid-sida/training/networks.py).
It retains the NVIDIA CC BY-NC-SA notice. Persistence decorators and unrelated
architectures were removed; optional activation checkpointing and the pinned
SiDA encoder channel-pooling interface were added outside the vendor tree.
There are no runtime imports from vendor or sibling repositories.

The architecture is 64×64 RGB, 151 class labels, base width 128, channel
multipliers [1,2,2,2], four residual blocks per resolution, embedding multiplier
4, positional time embedding, standard encoder/decoder, [1,1] resampling,
one-head attention at 16×16 and the 8×8 bottleneck, and dropout 0.25. Each
teacher/student/full fake-score-discriminator framework has **61,823,875
parameters**, including frozen parameters. The discriminator is the shared
encoder with channel-mean 8×8 logits; its additional parameter count is zero.
The same parameter-free pooling computation is present in both student arms.
This is matched within this experiment and matched to the reported external
DDPM++ family/scale; exact Sangwoo architecture equivalence is not established.

EMA is an alternative FP32 weight state, maintained as tensors for later
selection, not an attached inference module. Its storage is included in measured
concurrent memory. No export contains both raw and EMA models: only selected
EMA student weights and sampler buffers are exported.

## Flow convention and gradient routing

Inputs use the official RGB/bilinear 64×64 preprocessing and [-1,1] scaling.
Clean is t=0 and standard Gaussian noise is t=1. The network directly predicts
velocity from `(x_t, 1000*t, class)` without EDM input/output preconditioning.
The target is noise minus clean image. Teacher time sampling is uniform on
[0,1), with unit velocity MSE weight; loss reduction averages pixels and examples.
The teacher's solver is uniform Euler from 1 to 0 with 32 network calls.

The clean prediction is `D = x_t - t*v`. At positive t the Gaussian score is
`-(x_t + (1-t)*v)/t`; score division is never used at zero. Toy conditional
Gaussian tests check this identity and the integration sign independently.

The generator objective adapts the pinned
[`sid_dit_training_loop.py`](../vendor/ml-sid-dit/training/sid_dit_training_loop.py)
lines 1510–1656, as described in
[Score Distillation of Flow Matching Models](https://arxiv.org/html/2509.25127v1).
For corrupted generated image X, R and F are frozen real-teacher and fake-model
clean predictions. The loss is the pixel/example mean of
`(R-F)*((R-X)-alpha*(R-F))*(1-t)`, alpha=1. **Both R and F retain input
Jacobians through X**, while their parameters are frozen during G updates.
The teacher never receives parameter gradients. Fake-score velocity regression
uses detached generated images and target `noise - detached_image`.

Unlike the upstream summed-pixel objective, all losses use mean-pixel reduction.
The common SiD coefficient is 1; the GAN coefficient is 0.01, preserving the
upstream default 100:1 relative scale while dividing their shared overall scale.
Enabling GAN never halves the SiD loss. D uses half the sum of real-positive and
fake-negative BCE, averaged spatially and across examples, coefficient 1 in
addition to unit fake velocity MSE. G uses positive-label BCE weighted by 1-t.
Logits are clamped to [-10,10] as in the flow reference. This treatment includes
both real-data D learning and the G adversarial term. It is not an isolated
scalar-coefficient experiment. See also the
[SiDA construction](https://arxiv.org/html/2410.14919v4).

Loss times for both student phases are sigmoid(N(log(2),1.6²)), clamped to
[0.01,0.99]. The teacher and fake networks are in evaluation mode during G
updates. The fake model uses dropout 0.25 during its own update. Generator
calls always disable dropout. Adam uses FP32 master parameters/states; G and
fake optimizers have betas (0,0.999), epsilon 1e-8 and learning rate 2e-5.
Teacher Adam uses betas (0.9,0.999), epsilon 1e-8 and learning rate 2e-4.
No weight decay is applied. Gradients are clipped at global norm 10.

## Sampling, matching and execution

K-call ladders are [1], [1,1/2], [1,2/3,1/3] and [1,3/4,1/2,1/4]. Each
call denoises `(1-t)*previous_clean + t*fresh_noise`; there is no final extra
network. For training, each phase independently draws a uniform step position,
executes preceding calls without gradients, detaches that rollout, and trains
only the chosen call. Independent phase positions are a documented local
variation from the reference's shared position per cycle. The fake phase uses
the same distribution of truncated rollouts as G. The update ratio is one fake
update followed by one G update. There is no fake-only warmup or fake reset.

Teacher and D data draw a uniform class then a uniform optimization image, with
independent probability-0.5 horizontal flips. Student class draws are uniform.
No geometric ADA or other image augmentation is used. Fake/G noise, class/time
draws, teacher data/dropout, and D data/noise/dropout have separate saved RNG
streams. Both arms of a pair have identical initial states and common draws;
draw hashes and counters are logged for every complete update. Extra D draws
cannot advance fake or G streams. The fake-score/D framework is initialized
freshly from the selected teacher for every student; there is no learnable
head needing separate random initialization.

Physical GPU 2 is UUID `GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2`, TITAN RTX,
capability 7.5. `scripts/run_gpu.sh` sets CUDA_VISIBLE_DEVICES=2; PyTorch verifies
the mapped UUID at local cuda:0. A host-wide `/tmp/fastgen-gpu2.lock` flock,
process start ticks/boot identity, and occupancy checks protect every GPU stage.
The lease refusal test ran a second process. No sibling process or environment
was modified. CPU and preprocessing pools use two threads.

FP16 autocast uses FP32 reductions/master weights and loss scaling 65,536.
Autocast weight caching is disabled for training because a rollout mixes
no-gradient and gradient-enabled calls to the same network. The initial probe
identified missing convolution gradients with that cache enabled; it was fixed
before substantive training. A scalar norm alone would not have caught it.
Parameter-gradient presence is asserted for the output convolution. Nonfinite
losses or gradients are surfaced as failures; no sample filtering or skipped
optimizer updates is permitted. Full teacher/student/fake/D/optimizers/EMA and
activation memory is measured, with an allocator ceiling at 19.5 billion bytes
and an independent strict allocated-peak check below 20 billion bytes.

The qualified effective batch is 64 with microbatch 16 and four accumulation
rounds. A loss is multiplied by microbatch/effective batch before backward.
There is no activation checkpointing in the qualified recipe. Student dropout
and all execution settings match within each pair. D uses 64 additional real
images per cycle, reported separately from the 64 fake-regression and 64 G
presentations. More calls across K cost more compute despite equal exposure.

## Evaluation, selection and recovery

`evaluate.py` remains unchanged. Local evaluation calls its exact PNG sampling
and clean-FID functions on the copied 3,020-image development reference. The
reference renderer reproduces its normalization and PNG quantization. Official
validation is never used. The official FID feature extractor is evaluation-only.
Evaluation stores the actual class/noise draw hash; paired seed integers alone
are not the matching check.

Selection uses the predeclared checkpoints and seed in `configs/campaign.json`,
minimum pooled development FID, with the earliest checkpoint winning ties.
Each selected student receives one additional sampling seed confirmation, with
no reselection. These are sampling checks, not training replications. Fixed
class galleries show two development examples, six generated images and the
nearest optimization example to the first sample. All classes receive exact
duplicate, 16×16 area-pooled pixel diversity, and saturation diagnostics. Pixel
neighbors and variance do not establish recognition or perceptual diversity;
gallery inspection is reported separately.

Recovery saves complete FP32 G/fake/EMA and optimizer tensors, scalers, all RNG
streams/global RNG states, data presentation counters, update counters,
teacher/config/source/partition identity. The LR scheduler is a deterministic
function of the saved complete-update count. Qualification compared the next
complete teacher and 4-NFE adversarial update bit-for-bit after serialization,
object destruction and reloading, including optimizer/EMA and RNG state. This
tests a clean checkpoint round trip; it is not a hard-crash guarantee.

The external allowance ledger is never restored from an older checkpoint.
It charges failed and replayed attempts; at most two exact recoveries from
external interruptions are allowed per run. Numerical/data/ownership/source
failures stop and surface their evidence. The supervisor has a finite whole-run
watchdog and a no-progress watchdog. Renaming a run cannot reset its declared
scientific horizon or allowance. Source, dependency, partition and recipe hashes
accompany every checkpoint and export. Inference loading has been checked from
a clean directory with no training files, vendor or sibling imports at exactly
1, 2, 3 and 4 backbone calls.
