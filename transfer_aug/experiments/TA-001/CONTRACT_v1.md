# TA-001: Intra-class Color Transfer — contract revision 1

Follow `../../MASTER.md` and frozen `../../shared/PROTOCOL_V1.md`. Only this
experiment directory is writable by the worker. The master ledger pins this
contract, common protocol and bundle before dispatch.

## Question and falsifiable hypothesis

Can whole-image color moments be transferred between same-class Pokémon at
64×64 with recognizable identity and visible markings retained? Hypothesis:
moment changes are controllable with strength, but background-dominated donor
statistics and gamut clipping can disrupt appearance. Failure to change moments,
strong clipping, lost markings, or identity uncertainty are informative results.
Do not infer lower FID or training usefulness from image-space observations.

## Implementation and inputs

No learned parameters; training is not applicable. Reference: Reinhard color
transfer and community `jrosebr1/color_transfer` commit
`6724ccf927f2e3838ccd3f3da8b2542b79c9bf2f`. Use a documented NumPy float64
CIELAB D65 implementation with standard sRGB decoding/encoding; do not install
OpenCV. This is an adaptation, not bitwise replication of the community code's
uint8 Lab convention or default inverse ratio.

For each Lab channel, compute population standard deviations. Define
T=(c-mu_c)*sigma_d/sigma_c+mu_d when sigma_c>=1e-6; otherwise use
T=c-mu_c+mu_d (mean shift only). Interpolate Lab: (1-alpha)*c+alpha*T.
Convert to raw sRGB without clipping before diagnostics. For negative linear
RGB, use the linear branch of sRGB encoding; avoid fractional powers of negative
values. Export clipped RGB as specified in the common protocol.

Use 604 content/intra pairs in the frozen bundle with alpha={0.25,0.5,1.0}:
**1,812 transformed outputs** plus 604 explicit unchanged controls. Save raw
float arrays and every display PNG. Self-donor checks run on all 604 sources
at all strengths. No cross-class color arm or segmentation/mask adaptation.

## Mechanism checks and EDA

In addition to common checks, verify sRGB/Lab round trips on black/white/primary
colors and cohort inputs, using max absolute RGB tolerance 1e-6. Test moment
alignment in pre-inverse Lab at alpha=1 for nondegenerate channels, with mean
and standard-deviation error <=1e-8. Test expected interpolated moments at
weaker strengths and the documented zero-variance exception. Record moment
errors both before RGB gamut clipping and after converting clipped RGB back
to Lab, separating mathematical transfer from realizable image appearance.

Use common descriptive metrics and 16 representative sample sheets. For eight
tails, rank outputs by descending clipped_fraction then descending rms_change
then ascending sample ID/alpha, taking distinct content IDs. Review every
representative transformed output (48) and the eight selected tail outputs,
even if a tail overlaps a representative (mark its group separately).

## Procedure and handoff

Preflight -> implement/test -> generate -> measure -> visually inspect ->
recompute deterministically -> integrity verification -> handoff. Training and
GPU usage both remain zero. Respect the common finite resource cap. Report
all strengths; do not choose a winner. Deliver the common summary/result schema,
source/config, raw outputs, display gallery, metrics, review judgments and proof.
