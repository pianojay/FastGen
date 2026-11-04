# TA-001: completed, contract v1

Executed the frozen 604-pair A1 intra-class grid at alpha 0.25/0.5/1.0:
**1,812 transformed outputs plus 604 exact controls**. No training, learned
parameters, model loading, or GPU use. All 24 required sheets were actually
opened with image tools; 48 representative-output and eight tail judgments
are recorded. All frozen/protected hashes, selected A1 source hashes, and
20,099 source-file size/mtime records pass before/after verification.

NumPy float64 CIELAB transfer passes round trips, all 1,812 self-donor checks,
exact alpha-zero bypass, constant/near-constant fixtures, and expected moments.
Maximum cohort reconstruction error is 1.16e-14 RGB; self-donor exports differ
by zero LSB. Expected pre-inverse Lab mean/std errors are below 5.4e-12.
One nearly achromatic b channel uses the contracted mean-only fallback,
so matching that donor variance is intentionally impossible. The threshold
is sensitive near numerical achromatic values; details are preserved.

| Alpha | RMS change median / p95 | Clipped channels median / p95 |
|---|---:|---:|
| 0.25 | 0.0619 / 0.1468 | 0.59% / 17.93% |
| 0.50 | 0.1156 / 0.2848 | 2.71% / 20.76% |
| 1.00 | 0.2180 / 0.5336 | 9.36% / 33.96% |

Mathematical moments interpolate as specified, while RGB gamut clipping
prevents their full visual realization. At alpha 1, postclip Lab mean-distance
to donor has median/p95 1.59/8.24; std-distance is 2.28/11.81.
Maximum RGB excursion has median/p95 0.426/3.333 at alpha 1.

Strict clipping frequency is **not visible damage**: the top-ranked Mewtwo
tail clips 61.95% of channels but changes little (RMS 0.00649, maximum excursion
0.01239, a small real overshoot rather than mere roundoff). Dewgong clips
45.77% with RMS 0.384 and visibly loses white-body detail. Frozen tail ranking
was unchanged.

Unblinded AI review judged identity retained in 15/16, 15/16, and 12/16
representative outputs as strength increased; remaining cases were uncertain.
At alpha 1, anatomy/markings were retained/uncertain/disrupted in 7/7/2 cases.
Whole-field background shifts were widespread. Primeape lost eye contrast;
Tentacool developed flat cyan/magenta regions. Source blur and card clutter
limited judgments. These are relative appearance observations, not measured
class correctness, discriminator behavior, training utility, or FID evidence.

Exact recomputation matched every raw array, PNG, metric table, selected tail,
sheet and gallery. A failed synthetic test and its fix are preserved: shifted
moment reduction removed cancellation without changing formulas.

See `result.json` for pinned hashes, elapsed/host-memory accounting, limitations,
and proof paths; `gallery.html`, `visual_review.csv`, and `condition_metrics.json`
contain the main evidence. No strength is selected as a winning recipe.
