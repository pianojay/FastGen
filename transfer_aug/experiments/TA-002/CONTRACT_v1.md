# TA-002: FDA at 64×64 — contract revision 1

Follow `../../MASTER.md` and frozen `../../shared/PROTOCOL_V1.md`. Only this
experiment directory is writable by the worker. The master ledger pins this
contract, common protocol and bundle before dispatch.

## Question and falsifiable hypothesis

How do Fourier window radius, strength and donor class affect appearance and
artifacts at 64×64? Hypothesis: amplitude transfer is localized before clipping,
but out-of-gamut correction can introduce changes outside the selected spectrum.
Inter-class donors may change class-linked appearance despite retained phase.
Discriminator response, augmentation utility and FID are not measured here.

## Implementation and inputs

No learned parameters; training is not applicable. Reference: official
`YanchaoYang/FDA` commit `b9a0cdf0bcc9b787c00e39df73eda5673706f219`, NumPy branch
in `utils/__init__.py`. Preserve its fftshift-centered symmetric square
convention; do not port the differing torch branch's corner slicing.

Compute per-channel 2D FFT on float64 sRGB [0,1]. Define binary centered mask M
for |fx|<=r and |fy|<=r, using integer radii r={0,1,2,4}. Replace amplitude by
A'=(1-alpha*M)*A_content+alpha*M*A_donor for alpha={0.25,0.5,1.0}; retain content
phase, inverse transform, and preserve raw real RGB before clipping. Record
maximum imaginary residual. No tapering, padding, reflection, phase mixing,
segmentation or frequency-window tuning in this experiment.

Use all 604 content images, with separate intra and inter donor arms:
**14,496 transformed outputs** (604×2×4×3), plus 604 explicit unchanged controls.
Radius 0 changes the DC coefficient. Alpha=0 is an exact bypass. Store raw
float arrays and every display PNG. Self-donor checks cover all 604 sources and
all 12 conditions. Use no A2/B images or new learned measurement network.

## Mechanism checks and EDA

In addition to common checks: centered mask sizes must be 1,9,25,81 and conjugate
symmetric; imaginary residual <=1e-10; raw complex FFT change outside mask
max absolute <=1e-8; declared inside-mask amplitude interpolation max error
<=1e-8. Radius 0 must equal per-channel RGB mean interpolation within 1e-10
before clipping. Check linear interpolation at each strength and constant inputs.

Measure outside-mask complex spectral change before and after clipping as
both absolute L2 norm and relative L2 norm (denominator content outside-mask
norm; use an explicit zero-denominator flag rather than silently invent a ratio).
Report artifact-related common metrics without treating them as semantic labels.

Review all 16 representative sample IDs across both arms and all 12 conditions
(384 output judgments). Show each arm on a readable sheet. Select four tail
outputs per donor arm, ranking clipped_fraction descending then rms_change
descending then sample ID/radius/alpha ascending, requiring distinct content IDs
within each arm. These eight selected outputs are the separate tail review.

## Procedure and handoff

Preflight -> implement/test -> generate -> measure -> visually inspect ->
recompute deterministically -> integrity verification -> handoff. Training and
GPU usage both remain zero. Respect the common finite resource cap. Keep donor
arms separate in every aggregation. Deliver the common summary/result schema,
source/config, raw outputs, display gallery, metrics, review judgments and proof.
