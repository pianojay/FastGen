# Adversarial integration smoke results — October 4, 2026

**Passed.** SoFlow S3 EMA + class-conditioned StyleGAN2 + RpGAN/R1/R2 completed
the bounded integration check. No sample-quality experiment or FID evaluation
was performed. The existing S3 results and checkpoints remain the incumbents.

| Measurement | Result |
|---|---:|
| Contract tests | 35 passed |
| S3 generator parameters | 89,140,419 |
| StyleGAN2 discriminator parameters | 28,930,048 |
| Actual disposable D / G updates, including replay | 4 / 3 |
| Unique post-recovery state: D / G updates | 3 / 2 |
| Peak combined CUDA allocation | 12,111,966,208 bytes (12.11 GB) |
| Peak CUDA reservation | 12,842,958,848 bytes (12.84 GB) |
| Peak S3 worker host RSS | 3,693,711,360 bytes (3.69 GB) |
| Successful full verification wall time | 85.60 seconds |
| Earlier failed contract-check attempt | 15.35 seconds; no S3 updates |
| Total leased verification wall time | 100.94 seconds |
| Independent inference loading | 89,140,419 parameters; exactly 1 / 4 calls; no D imports |

See [machine-readable evidence](smoke-evidence-20261004.json),
[the integration contract](INTEGRATION.md), and
[the preserved theory report](soflow_adversarial_theory.md).
Raw logs and disposable checkpoints are in
`runs/adversarial_integration_smoke_20261004_retry1/`; the earlier test failure
is retained in `runs/adversarial_integration_smoke_20261004/`.

## Verified behavior

- Source and protected hashes, unchanged partition, and upstream StyleGAN2
  forward/first-/second-derivative parity passed. Its 23 vendored source files
  retain their original bytes and license.
- RpGAN signs, the real discriminator branch, generator-only adversarial
  gradients, analytic batch-coupled R penalties, isolated R2, and flip second
  derivatives passed. Accumulation preserves groups of four.
- The zero-adversarial small-model control measured maximum gradient difference
  **0** and maximum weight difference **0** on the passing GPU run. Declared
  tolerances are grad atol=2e-6/rtol=2e-5 and weights atol=1e-7/rtol=1e-6.
- Both online G and its new EMA reproduced source EMA at initialization.
  D-only updates preserved both bitwise and did not advance G clocks or moments.
- Saving and restoring reproduced the next D/G update **bitwise**, including
  online/EMA/D states, both optimizers, named RNG streams and counters, on this
  same deterministic GPU setup.
- Independent CPU processes loaded G-only checkpoints through the official
  interface, without importing D, and verified exactly one/four backbone calls.

The first contract pass stopped at a bitwise comparison with a first reported
weight difference of 1.65e-24. S3 was not launched. The fixture was corrected to
request deterministic cuDNN and declare numerical tolerances; no loss or model
code was changed to obtain the passing result. That failed attempt remains
included in the cost and logs.

## Observations and limits

The initial four-image one-call panel had **17.69%** raw pixels outside [-1,1].
Across the two unique G updates, clipping fractions were **23.59%** and
**12.36%**. The mean per-group shares of visible-image gradient energy masked
by clipping were **20.56%** and **9.53%**. These use different small noise draws;
they are not an improvement trajectory or a broad class estimate.

Adversarial gradient norms were **3.243 / 3.053**, compared with SCM norms
**0.938 / 0.580**. Detached balancing coefficients were **0.02892 / 0.01901**;
SCM/adv gradient cosines were **-0.0534 / 0.1013**. The configured rho=.1 thus
provided a finite, nonzero adversarial contribution. This establishes working
gradient wiring, not stable adversarial learning or improved image quality.

S3 D updates took about 3.0–3.4 seconds and G updates 1.3–1.6 seconds for the
B=8 smoke fixture. This native-reference, FP32, exact-R1/R2 path was selected
for correctness. These timings are not a qualified production recipe or an
extrapolation to effective batch 128. The peak host RSS above is for the S3
worker, not an aggregate of its supervisor and independent loading processes.

Still unmeasured: learning stability, discriminator overfit, class fidelity,
diversity/memorization, FID, useful loss strength, EMA policy, update ratio,
warmup/ramp, training horizon, and optimized production throughput. None is
resolved by these few disposable updates. The branch's old autoresearch program
is historical and inactive; no follow-on training was launched.
