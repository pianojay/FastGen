from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/home/user/jw_soflow/FastGen-SiD-DiT/runs/soflow_teacher_qualification_20261002')
r=json.loads((O/'completed.json').read_text());f=json.loads((O/'field_metrics.json').read_text());g=json.loads((O/'gradient_scaled_metrics.json').read_text());gc=json.loads((O/'gradient_scaled_completed.json').read_text());sm=json.loads((O/'smoke_start.json').read_text());ver=json.loads((O/'verification.json').read_text())
ts=[float(k) for k in f if k!='sid_logit_normal']
fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
for key,label in [('soflow_diagonal','SoFlow U-Net160 diagonal'),('rf_teacher','Current RF teacher')]:
 axes[0].plot(ts,[f[str(t)][key]['velocity_mse'] for t in ts],marker='o',label=label)
 axes[1].plot(ts,[f[str(t)][key]['denoiser_mse'] for t in ts],marker='o',label=label)
axes[0].set_ylabel('Velocity MSE');axes[1].set_ylabel('Denoiser MSE = t² × velocity MSE');axes[1].set_xlabel('Flow time t (0=data, 1=noise)');axes[0].legend();axes[0].set_title('Matched 604-image development diagnostic, 4 images/class')
for ax in axes:ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(O/'teacher_field_errors.png',dpi=160);plt.close(fig)
rows=[]
for t,rr in f.items():
 d=rr['paired_velocity_delta_soflow_minus_rf'];lo,hi=d['class_bootstrap_95ci'];rows.append(f"| {t} | {rr['soflow_diagonal']['velocity_mse']:.6f} | {rr['rf_teacher']['velocity_mse']:.6f} | {d['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
grows=[]
for t,rr in g['results'].items():
 a=rr['soflow_diagonal'];b=rr['rf_teacher'];grows.append(f"| {t} | {a['sid_gradient_rms']:.3g} | {b['sid_gradient_rms']:.3g} | {a['denoiser_vjp_norm_mean']:.3f} | {b['denoiser_vjp_norm_mean']:.3f} |")
fr=[]
for name,rr in r['fids'].items():fr.append(f"| {name} | {rr['fid']:.4f} | {rr['neural_calls_per_sample']} | {rr['total_seconds']:.1f} | {rr['raw_out_of_range_fraction']:.4%} |")
# Elementwise algebraic conversion check from measured errors.
a=np.load(O/'field_per_image.npz');identity={}
for t in ts:
 for key in ['soflow_diagonal','rf_teacher']:
  err=np.max(np.abs(a[str(t)+'_'+key+'_d']-t*t*a[str(t)+'_'+key+'_v']));identity[str(t)+'_'+key]=float(err)
crng=np.random.default_rng(19);cx=crng.normal(size=(10,3,8,8));cv=crng.normal(size=cx.shape);ct=np.array([.01,.05,.1,.25,.5,.75,.9,.95,.99,1.])[:,None,None,None]
cd=cx-ct*cv;sc1=-(cx+(1-ct)*cv)/ct;sc2=((1-ct)*cd-cx)/(ct*ct)
score_error=float(np.max(np.abs(sc1-sc2)));assert score_error<1e-9
(O/'conversion_check.json').write_text(json.dumps({'max_abs_per_image_mse_identity_error':max(identity.values()),'score_denoiser_velocity_identity_max_abs_error_float64':score_error,'by_bin':identity},indent=2)+'\n')
total=r['elapsed_seconds']+sm['elapsed_seconds']+gc['elapsed_seconds']
text=f'''# SoFlow U-Net160 teacher qualification — 2026-10-02

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
{chr(10).join(fr)}

[Matched class-reference gallery](comparison_gallery.png). Each row has one development reference, four SoFlow diagonal32 samples, four RF32 samples and four SoFlow endpoint4 samples. Initial noise is shared columnwise. Clamping occurs only during PNG conversion, never inside the diagonal Euler solver or differentiable teacher.

Historical SoFlow FID 26.5204 averaged four generation seeds under its original FP32 sampler; historical RF FID 37.2526 used selection seed 1729. This probe uses a new shared CPU draw seed {r['fids']['rf32']['noise_seed']} and FP16 SoFlow inference, so its matched comparisons are the appropriate comparison here. Generation-seed variation is not training replication.

## Field qualification

604 fixed development images (first four/class in the existing manifest order), one shared corruption noise per image, nine time conditions. Targets are conditional flow-matching targets `noise − clean`; there is no oracle conditional score. Comparisons estimate teacher prediction risk on this finite, previously reused development set. Confidence intervals resample 151 paired class means (2,000 bootstrap draws), conditional on selected images/noise; they do not quantify training-seed uncertainty or account for multiple time-bin comparisons.

| Time | SoFlow velocity MSE ↓ | RF velocity MSE ↓ | Paired delta SoFlow−RF [95% class-bootstrap CI] |
|---|---:|---:|---|
{chr(10).join(rows)}

SiD times use `sigmoid(N(log(2), 1.6²))`, clamped to [0.01,0.99], matching the running pilot. At those draws, converted-denoiser MSE is {f['sid_logit_normal']['soflow_diagonal']['denoiser_mse']:.6f} versus {f['sid_logit_normal']['rf_teacher']['denoiser_mse']:.6f}. Denoiser MSE is exactly t²-weighted velocity MSE under the same corruption, **not independent confirmation**. Measured per-image conversion identity maximum error: {max(identity.values()):.3g}.

[Time-resolved error plot](teacher_field_errors.png).

The adapted teacher is `predict_velocity(x,t)` = SoFlow network queried at `(t,s=t)` with zero augmentation labels. Its SiD denoiser is `D=x−t*v`. The existing generator uses a different query `(t,s=0)` and cannot be substituted as the conditional-mean denoiser solely because it has lower FID.

## Input gradients and SiD path

Teacher parameters frozen; input Jacobians retained. Both teachers were evaluated on the same 30 class-spanning generated inputs using the completed S1 EMA generator and the **same** completed S1 fake-score network. This is a numerical path/scale diagnostic, not a fair estimate of each teacher's eventual distillation loss: that fake network was trained with RF and may favor it. Full SiD surrogate with alpha=1 and weighting `(1−t)` is used. The scalar sums per-image mean-pixel losses, so reported input gradients have per-image normalization. A final check applies loss scale 65,536 and unscales the input gradient, matching the pilot's FP16 backward strategy. An earlier unscaled pass is preserved separately; use `gradient_scaled_metrics.json` for the final gradient figures.

| Time | SoFlow SiD input-gradient RMS | RF RMS | SoFlow denoiser VJP mean norm | RF VJP mean norm |
|---|---:|---:|---:|---:|
{chr(10).join(grows)}

All recorded outputs/gradients were finite; teacher parameter `.grad` values remained absent. Random-vector denoiser VJPs and surrogate gradients were nonzero. Numerical viability is established on tested examples, not convergence or gradient superiority. Velocity-to-score convention is `score = -(x+(1−t)*v)/t` for positive t; inference integrates velocity directly and does not divide by t. A float64 algebra check against `score=((1−t)*D−x)/t²` over t=0.01…1 had maximum absolute error {score_error:.3g}.

## Cost, identity and boundaries

- GPU: physical 3, UUID `{r['gpu_uuid']}`, NVIDIA TITAN RTX. Lease `/tmp/fastgen-gpu3.lock`, no fallback and no concurrent training on that device.
- Main probe {r['elapsed_seconds']:.1f}s; initial smoke {sm['elapsed_seconds']:.1f}s; scaled-gradient numerical check {gc['elapsed_seconds']:.1f}s; total measured process occupancy **{total:.1f}s ({total/3600:.3f} GPU-hours)**. CPU preparation/reporting is additional wall time.
- Peak allocated CUDA memory {max(r['peak_allocated_bytes'],gc['peak_allocated_bytes'])/1e9:.3f} GB; peak reserved {max(r['peak_reserved_bytes'],gc['peak_reserved_bytes'])/1e9:.3f} GB. Main-process peak host RSS {r['host_peak_rss_bytes']/1e9:.3f} GB (not a summed process-tree peak).
- Finite watchdog declared before full evaluation: 2,400 seconds. Separate numerical checks bounded to 300 seconds each. No training horizon, extra seed sweep, teacher training or student pair was launched.
- SoFlow generator/teacher: 61,934,243 parameters. RF teacher: 61,823,875. These are separate complete learned frameworks.
- SoFlow completed S2 checkpoint SHA256 `{r['soflow_checkpoint_sha256']}`; weights loaded from EMA. RF selected EMA checkpoint SHA256 `{r['rf_checkpoint_sha256']}`. Fixed completed S1 generator/fake checkpoint SHA256 `{g['source_sha256']}`.
- Partition SHA256 `{ver['partition_sha256']}`; 14,059 optimization/3,020 development. Shared images read-only. Both projects' development reference PNG manifests were verified byte-identical: `{ver['reference_image_manifest_sha256']}`.
- Runtime used copied, hashed source modules under `snapshot/`; original active experiment code was not imported or changed. Official `compute_fid` is extracted unchanged from the pinned snapshot using Python AST to avoid importing unrelated mutable model/data modules.
- Precision smoke: relative FP16-vs-FP32 forward RMS <0.0006 on a 16-example batch. This is a numerical smoke, not proof FIDs are identical across precisions.

## Reproduction and artifacts

Source and checkpoint identities: [source manifest](source_manifest.json), [verification](verification.json), [protocol](protocol.json). Exact script: [probe.py](probe.py), [gradient_scaled_probe.py](gradient_scaled_probe.py). Metrics: [field](field_metrics.json), [per-image arrays](field_per_image.npz), [scaled gradients](gradient_scaled_metrics.json), [FIDs](fid_metrics.json), [complete resource record](completed.json).

Commands (run only after checking GPU 3 remains idle/unclaimed and taking its lease; do not overwrite the preserved artifact directory):

```bash
flock -n /tmp/fastgen-gpu3.lock env CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
  /home/user/miniconda3/envs/fastgen_sid_dit/bin/python -u probe.py smoke
flock -n /tmp/fastgen-gpu3.lock env CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
  /home/user/miniconda3/envs/fastgen_sid_dit/bin/python -u probe.py full
flock -n /tmp/fastgen-gpu3.lock env CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
  /home/user/miniconda3/envs/fastgen_sid_dit/bin/python -u gradient_scaled_probe.py
```

Report plotting used the already installed `fastgen_soflow` Matplotlib on CPU with `CUDA_VISIBLE_DEVICES=\"\"`, `PYTHONNOUSERSITE=1`, and `MPLCONFIGDIR=/tmp/teacher_probe_matplotlib`. No packages were installed or environments modified.

No Slack/Notion writes or training launch occurred. The next training decision remains separate from this diagnostic.
'''
(O/'REPORT.md').write_text(text)
print('Report generated',O/'REPORT.md')
