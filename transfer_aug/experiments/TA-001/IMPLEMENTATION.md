# TA-001 implementation and reproduction

The frozen contract is implemented in `run_experiment.py`. All image transforms
and numerical metrics use NumPy float64. Pillow only reads/writes display PNGs
and renders labels. No model, training framework, classifier, or GPU is loaded.
The runner binds each BLAS/OpenMP thread setting to two and disables user-site
packages. The environment is recorded in `environment.json`.

The pinned community source is
`transfer_aug/audit_sources/color/color_transfer/__init__.py`, revision
`6724ccf927f2e3838ccd3f3da8b2542b79c9bf2f` (SHA-256 recorded in integrity
evidence). This implementation adapts its whole-image channel-moment concept;
it does not execute that source or reproduce its OpenCV uint8 Lab convention,
Lab clipping, or default reciprocal ratio. Here the multiplier is donor std /
content std, with the specified mean-shift fallback below 1e-6.

Standard sRGB decoding uses the 0.04045 breakpoint; encoding uses 0.0031308.
Negative linear RGB values use the encoding's linear branch. XYZ uses the
documented RGB matrix and D65 2-degree white in `config.json`. The inverse
matrix is calculated from the forward coefficients, giving matched numerical
round trips. CIELAB uses the conventional delta=6/29 branches. Moment reduction
first subtracts one pixel to reduce cancellation for near-constant fixtures;
the population mean and std definitions are unchanged. Weak-strength expected
std is `(1-alpha)*std_content + alpha*std_donor` for nondegenerate channels.
For degenerate channels it remains `std_content`, while the mean interpolates.

Metrics use clipped float RGB before quantization. Raw out-of-gamut values
remain in `arrays/raw_rgb.npy`, indexed by alpha then frozen cohort order.
PNG values use `np.rint(clip(raw,0,1)*255)`. The high-frequency metric pools
energy over RGB channels, excludes DC separately per channel, and selects
the exact max(abs(fx),abs(fy))>=16 mask. All pre-Lab versus postclip-Lab
moment distances are stored separately. Control moment-to-donor distances
are descriptive distances, not expected transfer errors.

Run from `/home/user/jw_soflow/exploration`:

```bash
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-001/run_experiment.py generate > transfer_aug/experiments/TA-001/generate.log 2>&1
bash transfer_aug/run_python.sh transfer_aug/experiments/TA-001/run_experiment.py recompute > transfer_aug/experiments/TA-001/recompute.log 2>&1
```

Generation refuses to overwrite an existing raw array. Recompute is the
repeatable validation command for this completed directory: it recalculates
every control and transformed array, every PNG, CSV/aggregate metrics, ranked
tails, contact sheets and gallery, and compares exact arrays/bytes or parsed
JSON without rewriting the successful artifacts. Its own verification/status
records are refreshed. Human/AI visual judgments cannot be deterministically
recomputed; their evidence is the saved review record and tool-inspected sheets.

The first generation attempt failed before producing output arrays because
unshifted mean reduction on the near-constant synthetic fixture produced a
3.20e-8 mean error. The original implementation, traceback, and diagnosis are
preserved under `verification/failed_attempt_1*`; the fixed implementation
retains the same formulas and passes the frozen tolerance.
