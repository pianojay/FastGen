# AFM stability search — October 2, 2026

The user started the bounded campaign on October 2. The historical baseline is closed and was audited read-only. GPU 0 retains UUID `GPU-e9d30df5-3b84-1a7e-c505-c6fff0a0ce7f`; no sibling project was modified. No FID will be used in this campaign.

## Frozen audit

[Machine-readable protocol](../resources/stability_audit_protocol.json). Each scheduled audit covers all 151 classes, raw G and EMA, with eight common and eight fresh independent Gaussian inputs per class. Common seeds are `18001 + class`; fresh seeds are `audit_seed + G_updates + 100000 + class`. Saved tensors preserve actual draws and unclamped outputs. Galleries include four same-class references and generated/nearest optimization pairs. Input perturbation sensitivity uses Gaussian standard deviation .05 on four draws per class. These diagnostics preserve training RNG and data state.

The quantitative gate combines translation-tolerant, color-normalized 16×16 structure (median pair distance ≥0.377035), low-frequency image variation (8×8 pooled median pair MSE ≥0.0770924), and median nearest optimization pixel MSE >0.0110560. Thresholds derive from 151-class real/repeated/noisy/recolored/shifted/small-set-copy controls. All nuisance groups fail. Real examples pass in 120/151 classes; the remaining groups fail the conservative novelty gate amid substantial dataset duplication. The historical baseline passes in **0/151 raw and 0/151 EMA**. Calibration did not assume all real examples were unique.

Acceptance requires at least **114/151 classes** (75%, chosen before candidate 1 with the real-control coverage in view), the same class intersection in raw G and EMA across two consecutive audits. All classes must be reviewed: at least six of eight recognizable samples in each draw half, at least three pose/shape/composition variants across 16, visible variation on fresh draws, and no texture/color/translation-only or near-copy explanation. Ambiguous classes fail acceptance. Near-neighbor distance cannot establish absence of memorization. Quantitative metrics alone cannot establish recognizable diversity.

After OT decay, two consecutive audits with at most 15 quantitative-pass classes in **both** raw G and EMA, plus visual review of at least 30 classes showing persistent templates or noise-only variation, permit early termination. A first 6k check cannot trigger that rule. Incomplete or ambiguous learning continues to the declared horizon. Numerical/resource failures may stop immediately. No acceptance thresholds change after candidate 1.

## Engineering and accounting

The new search runner uses the host-wide `/tmp/fastgen-gpu0.lock`, device identity and occupancy checks; all GPU stages hold the lease. Architecture, objective signs and source-noise OT remain fixed. Separate G/D AdamW settings and the upstream normalization **target scale** are exposed. A target scale multiplies the gradient after normalization; a loss coefficient can be canceled by normalization. Seven objective tests include this distinction.

Baseline audit: 149.74 GPU-job seconds, peak allocated 2.748 GB. A diagnostic eight-class batch gave G parameter-gradient norms 0.8856 adversarial versus 0.03445 OT at coefficient .005 (~25.7×); D norms were 6.415 adversarial, .613 R1, 1.261 R2, .188 centering. This is diagnostic evidence, not a population estimate or causal proof. Raw G was also collapsed, so EMA lag alone does not explain the result.

One initial disposable qualification executed 65 cycles before refusing recovery because source files changed during the probe. Its failure, work and elapsed cost are retained. The second bounded probe passed 64 cycles plus an original/replayed next cycle after source freeze: exact recovery and duplicate-device refusal passed, peak allocated 7.524 GB, reserved 8.147 GB, sampled device 8.376 GB, host RSS 2.143 GB/PSS 2.129 GB. Mean warm cycle .677 seconds; last16 mean approximately .75 seconds. Probe job cost61.08 seconds. No probe is a quality trial. [Campaign ledger](../runs/stability_campaign/record.json) includes all overhead and failed/replayed work; original baseline cost is separately identified.

## Candidate record

| Candidate / seed | Hypothesis / change / reference | Source + config | G/D updates; presentations | GPU hours / allowance | Numerical status | Diversity + recognizable classes | Decision / reason | Evidence |
|---|---|---|---|---|---|---|---|---|
| Historical / 20261001 | Original recipe; read-only reference | `ee74486`, baseline config | 10,294 each; 329,408 real/stream | 2.396 train historical | Finite | Quantitative gate 0/151 raw and EMA | Closed; collapse reference | [Audit](../artifacts/stability/calibration/summary.json) |
| C1 / 2026100201 | Stronger retained transport; normalized adversarial target .5; G LR 5e-5, D LR 2e-5; reference historical | `c72b62f`; [recipe](../configs/stability_candidate1.json) | 6,000/6,000; 192,000 real/stream | ~1.42h / 8h | finite; no recovery | Quantitative141 raw/146 EMA; recognizable0/30 inspected | Continue: first audit, incomplete OT phase; no recognizable-diversity claim | [6k audit](../runs/stability_campaign/candidate1/audits/006000/decision.json) |

C1 retains batch32/micro32, D:G=1:1, EMA .999, other regularization/augmentation unchanged. Schedules use completed G updates. Maximum two exact infrastructure recoveries within the original 8-hour watchdog; no reset on relaunch. No automatic retry after a user stop. Full state is saved every500 updates and before audits. The first qualifying recipe gets exactly one independent-seed confirmation at the same assessment points, then the campaign stops. Otherwise stop after six substantive candidates; failed candidates consume slots. Subsequent recipes will be chosen from observed evidence, not a prefilled sweep.
