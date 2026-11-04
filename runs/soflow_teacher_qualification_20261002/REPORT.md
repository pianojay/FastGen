# SoFlow U-Net160 teacher qualification — 2026-10-02

Evaluation-only probe authorized by the user and run on idle, exclusively leased physical GPU 3. No model parameters were updated. Existing AFM, SoFlow and SiD jobs were preserved.

## Verdict

**Numerically usable, but unsupported as a stronger score teacher than the current RF model.** SoFlow's diagonal field completed all numerical and gradient checks, produced recognizable class-dependent samples, and achieved FID **38.3894** at 32 Euler calls versus RF **37.3267** with the same draw. Its endpoint generator achieved **26.5761** at four calls. Thus the strong endpoint result does not transfer to the diagonal field that SiD would consume.

Do **not** replace the current RF teacher on the premise that the existing SoFlow FID is ~26.5. The matched draw shows no teacher advantage, and SoFlow has higher prediction error under the existing SiD time distribution (+3.39% velocity MSE; +7.55% denoiser MSE). One generation seed does not prove a universal FID ranking, nor do these diagnostics prove either model would distill better.

Keep the currently running pilot unchanged. SoFlow remains a useful *generator initialization* asset, distinct from claiming it is a stronger score teacher. Testing such an initialization with a fixed teacher would be a separate, newly specified training experiment; none was launched here. No result here justifies an expensive new teacher campaign by itself.

In the reviewed 16-class panel, both fields produce class-dependent forms with anatomical distortions; SoFlow diagonal samples often have stronger contrast and more saturated backgrounds. Endpoint samples generally look cleaner. This is a small visual audit, not exhaustive recognition/diversity or memorization certification.

## Matched generation results

One common initial-noise draw, 20 generated images per class, 3,020 total; identical official clean-FID function and development references. These are local development results, not official challenge scores. The endpoint sampler additionally uses three independently fixed re-noising draws. All methods use FP16 autocast with FP32 parameters and states.

| Teacher/sampler | FID ↓ | Actual NFE | Generation + FID seconds | Raw pixels outside [-1,1] |
|---|---:|---:|---:|---:|
| soflow_diagonal32 | 38.3894 | 32 | 405.7 | 16.9241% |
| rf32 | 37.3267 | 32 | 449.1 | 1.7694% |
| soflow_endpoint4 | 26.5761 | 4 | 82.2 | 11.6102% |

[Matched class-reference gallery](comparison_gallery.png). Each row has one development reference, four SoFlow diagonal32 samples, four RF32 samples and four SoFlow endpoint4 samples. Initial noise is shared columnwise. Clamping occurs only during PNG conversion, never inside the diagonal Euler solver or differentiable teacher.

Historical SoFlow FID 26.5204 averaged four generation seeds under its original FP32 sampler; historical RF FID 37.2526 used selection seed 1729. This probe uses a new shared CPU draw seed 2026100213 and FP16 SoFlow inference, so its matched comparisons are the appropriate comparison here. Generation-seed variation is not training replication.

## Field qualification

604 fixed development images (first four/class in the existing manifest order), one shared corruption noise per image, nine time conditions. Targets are conditional flow-matching targets `noise − clean`; there is no oracle conditional score. Comparisons estimate teacher prediction risk on this finite, previously reused development set. Confidence intervals resample 151 paired class means (2,000 bootstrap draws), conditional on selected images/noise; they do not quantify training-seed uncertainty or account for multiple time-bin comparisons.

| Time | SoFlow velocity MSE ↓ | RF velocity MSE ↓ | Paired delta SoFlow−RF [95% class-bootstrap CI] |
|---|---:|---:|---|
| 0.01 | 0.470980 | 0.446246 | +0.024734 [+0.023132, +0.026422] |
| 0.05 | 0.197810 | 0.194890 | +0.002919 [+0.002536, +0.003296] |
| 0.1 | 0.137289 | 0.139312 | -0.002023 [-0.002270, -0.001778] |
| 0.25 | 0.095707 | 0.100150 | -0.004443 [-0.004778, -0.004123] |
| 0.5 | 0.092861 | 0.097431 | -0.004570 [-0.005002, -0.004177] |
| 0.75 | 0.125734 | 0.125696 | +0.000038 [-0.000683, +0.000713] |
| 0.9 | 0.187709 | 0.174920 | +0.012789 [+0.011285, +0.014276] |
| 0.99 | 0.415848 | 0.340024 | +0.075824 [+0.067446, +0.084878] |
| sid_logit_normal | 0.146054 | 0.141267 | +0.004786 [+0.002802, +0.006924] |

SiD times use `sigmoid(N(log(2), 1.6²))`, clamped to [0.01,0.99], matching the running pilot. At those draws, converted-denoiser MSE is 0.080139 versus 0.074511. Denoiser MSE is exactly t²-weighted velocity MSE under the same corruption, **not independent confirmation**. Measured per-image conversion identity maximum error: 2.1e-07.

[Time-resolved error plot](teacher_field_errors.png).

The adapted teacher is `predict_velocity(x,t)` = SoFlow network queried at `(t,s=t)` with zero augmentation labels. Its SiD denoiser is `D=x−t*v`. The existing generator uses a different query `(t,s=0)` and cannot be substituted as the conditional-mean denoiser solely because it has lower FID.

## Input gradients and SiD path

Teacher parameters frozen; input Jacobians retained. Both teachers were evaluated on the same 30 class-spanning generated inputs using the completed S1 EMA generator and the **same** completed S1 fake-score network. This is a numerical path/scale diagnostic, not a fair estimate of each teacher's eventual distillation loss: that fake network was trained with RF and may favor it. Full SiD surrogate with alpha=1 and weighting `(1−t)` is used. The scalar sums per-image mean-pixel losses, so reported input gradients have per-image normalization. A final check applies loss scale 65,536 and unscales the input gradient, matching the pilot's FP16 backward strategy. An earlier unscaled pass is preserved separately; use `gradient_scaled_metrics.json` for the final gradient figures.

| Time | SoFlow SiD input-gradient RMS | RF RMS | SoFlow denoiser VJP mean norm | RF VJP mean norm |
|---|---:|---:|---:|---:|
| 0.05 | 3.82e-07 | 3.6e-07 | 0.865 | 0.867 |
| 0.25 | 1.68e-06 | 9.74e-07 | 1.011 | 1.018 |
| 0.5 | 2.45e-06 | 1.12e-06 | 1.067 | 1.062 |
| 0.75 | 2.29e-06 | 7.67e-07 | 1.143 | 1.047 |
| 0.95 | 8.19e-07 | 1.51e-07 | 1.357 | 0.960 |

All recorded outputs/gradients were finite; teacher parameter `.grad` values remained absent. Random-vector denoiser VJPs and surrogate gradients were nonzero. Numerical viability is established on tested examples, not convergence or gradient superiority. Velocity-to-score convention is `score = -(x+(1−t)*v)/t` for positive t; inference integrates velocity directly and does not divide by t. A float64 algebra check against `score=((1−t)*D−x)/t²` over t=0.01…1 had maximum absolute error 2.61e-12.

## Cost, identity and boundaries

- GPU: physical 3, UUID `GPU-c651701d-9ecd-e23e-13a9-2bc13b838b5d`, NVIDIA TITAN RTX. Lease `/tmp/fastgen-gpu3.lock`, no fallback and no concurrent training on that device.
- Main probe 1023.0s; initial smoke 13.2s; scaled-gradient numerical check 35.5s; total measured process occupancy **1071.7s (0.298 GPU-hours)**. CPU preparation/reporting is additional wall time.
- Peak allocated CUDA memory 5.034 GB; peak reserved 5.323 GB. Main-process peak host RSS 4.206 GB (not a summed process-tree peak).
- Finite watchdog declared before full evaluation: 2,400 seconds. Separate numerical checks bounded to 300 seconds each. No training horizon, extra seed sweep, teacher training or student pair was launched.
- SoFlow generator/teacher: 61,934,243 parameters. RF teacher: 61,823,875. These are separate complete learned frameworks.
- SoFlow completed S2 checkpoint SHA256 `32bf7e04d68750aff505546266d97ba9a4077e17fd68ba911254850473e75a79`; weights loaded from EMA. RF selected EMA checkpoint SHA256 `8bafcc5cb0e5d1e0f7a644fe6789640ccda9b513813633bc27a40eb81d17b3cf`. Fixed completed S1 generator/fake checkpoint SHA256 `db96bdd657a3e7f6f6819e8320efa19c6153ae71e52e7cb18ea09d99a1c72495`.
- Partition SHA256 `825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`; 14,059 optimization/3,020 development. Shared images read-only. Both projects' development reference PNG manifests were verified byte-identical: `1c926031c13a015673774e50991e881d5a76a63e5b84e44abbc628f9ecc82f71`.
- Runtime used copied, hashed source modules under `snapshot/`; original active experiment code was not imported or changed. Official `compute_fid` is extracted unchanged from the pinned snapshot using Python AST to avoid importing unrelated mutable model/data modules.
- Precision smoke: relative FP16-vs-FP32 forward RMS <0.0006 on a 16-example batch. This is a numerical smoke, not proof FIDs are identical across precisions.

## Reproduction and artifacts

Source and checkpoint identities: [source manifest](source_manifest.json), [verification](verification.json), [protocol](protocol.json). Exact script: [probe.py](probe.py), [gradient_scaled_probe.py](gradient_scaled_probe.py). Metrics: [field](field_metrics.json), [per-image arrays](field_per_image.npz), [scaled gradients](gradient_scaled_metrics.json), [FIDs](fid_metrics.json), [complete resource record](completed.json).

Commands (run only after checking GPU 3 remains idle/unclaimed and taking its lease; do not overwrite the preserved artifact directory):

```bash
flock -n /tmp/fastgen-gpu3.lock env CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/user/miniconda3/envs/fastgen_sid_dit/bin/python -u probe.py smoke
flock -n /tmp/fastgen-gpu3.lock env CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/user/miniconda3/envs/fastgen_sid_dit/bin/python -u probe.py full
flock -n /tmp/fastgen-gpu3.lock env CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/user/miniconda3/envs/fastgen_sid_dit/bin/python -u gradient_scaled_probe.py
```

Report plotting used the already installed `fastgen_soflow` Matplotlib on CPU with `CUDA_VISIBLE_DEVICES=""`, `PYTHONNOUSERSITE=1`, and `MPLCONFIGDIR=/tmp/teacher_probe_matplotlib`. No packages were installed or environments modified.

No Slack/Notion writes or training launch occurred. The next training decision remains separate from this diagnostic.
