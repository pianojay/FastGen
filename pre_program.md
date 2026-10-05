# Fixed pre-program audit — October 4, 2026

**Historical, completed procedure; no current execution authorization.** The user
clarified October 5 that the intended assignment was to write this protocol;
the other agent also executed it. The former statement that execution was
authorized was mistaken. Preserve all completed facts, reports, frozen manifests
and artifacts. The 12 aggregate GPU-job-hour ceiling below describes the executed
procedure, not a remaining allowance. **Never launch `program.md` as a successor.**
The user later separately authorized the fixed matched pilot; see its
[protocol](docs/adversarial/MATCHED_PILOT_DRAFT.md). This does not reopen this audit.

## Completion record — October 5, 2026

**Completed once; do not rerun automatically.** All A0–A4 stages finished in
3.9786 aggregate GPU-job hours with no failed stages. See the
[final report](docs/adversarial/PRE_PROGRAM_RESULTS.md),
[frozen manifest](docs/adversarial/pre-program-manifest-20261004.json), and
[compact evidence](docs/adversarial/pre-program-evidence-20261005.json).
The historical procedure and limits below describe that completed audit.
`program.md` remains tentative and inactive.

## Frozen question and sources

Determine whether exact-clamp S3 endpoint gradients, discriminator learnability,
and measured execution costs support later local recipe search. This audit does
not determine a useful generator learning horizon or demonstrate FID improvement.

Keep the mathematical contract in [INTEGRATION](docs/adversarial/INTEGRATION.md).
Initialize G and its EMA from S3 U192 final EMA, checkpoint SHA256
`9b95652fb5e503b2f877b26fd3652129abcce115bdea526db00ffe1f0daefd4a`.
Keep the canonical partition SHA256
`825bde8601ca141e992f004ceb13a4fb989e818ec401e335955f70ef6a17677e`.
No inference changes, additional teacher, FM, composition loss, surrogate
training gradient, or architecture search. The range penalty remains diagnostic.

## Fixed experiments

| ID | Procedure | Evidence |
|---|---|---|
| A0 | Source/protected/vendor hashes, numerical contracts, device UUID and lease checks | Valid implementation and ownership |
| A1 | Frozen S3: 20 samples/class for each seed 6201/6202, 6,040 images; compare all 14,059 optimization images | Per-class clipping, overshoot, spatial maps, galleries, real boundary statistics |
| A2 | Fresh D seeds 20261004/20261005 on GPUs 1/2; 512 updates each at B32; audit steps 0/32/128/512 | Fit-pool versus held-out logits, pair accuracy, wrong-label sensitivity, R1/R2, throughput |
| A3 | At each A2 checkpoint, four mixed-group samples/class/noise seed, 604 per seed | Actual and clamp-blocked parameter gradients; comparison with SCM and squared overshoot gradients |
| A4 | Disposable complete updates at B8/B32, phase boundaries and exact replay | Time, memory, accumulation contracts, schedule/data/RNG recovery |

D fixture: class-conditioned StyleGAN2, group four, native FP32 operators,
Adam lr=2e-5, betas=(0,.99), beta=.1 for R1+R2, paired flip probability .25.
These are audit settings, not a qualified post-training recipe. D sees fresh
noise and uniformly sampled classes, then uniformly sampled fit images.

Reserve four byte-unique optimization images per class, ordered by
SHA256(`pre-audit:{class_id}:{relative_image_path}`), for D diagnostics only.
Exclude all 604 from D fit draws. Freeze their original indices and paths in
the manifest. The official partition is not edited; neither development nor
official validation is used for fitting. Fit-pool diagnostic images are not
guaranteed to have been presented to D; label the comparison accordingly.

A3 uses a fixed mixed-class permutation and no D flips during measurement.
Wrong labels use the fixed class permutation `(class+37)%151`, for diagnosis
only. Measurements never advance training RNG streams. G is fixed throughout
A2/A3; unchanged raw/EMA state hashes are checked at every checkpoint.

For visible image gradient u and raw-output Jacobian J, report actual
J^T(mask*u), blocked J^T((1-mask)*u), and their sum. The latter is a diagnostic
vector, never a generator update. Norm ratios need not be fractions of learning
lost because parameter contributions can cancel. SCM and range gradients are
cached per fixed draw set because G never changes across D checkpoints.
The range term is mean(relu(abs(raw)-1)^2), training coefficient zero.

A4 uses an explicit disposable phase fixture: two D-only steps; G rho values
0, .05, .1, with replay of the .1 step. Fresh optimizers, source EMA, unchanged
mature gap, smoke EMA=.99 and smoke G lr=2e-5. Each B8/B32 fixture attempts six
D and four G updates (12 D / 8 G overall); hard ceiling 16/16 includes failures
and replay. These weights are discarded and never evaluated as quality models.

## Execution and failure rules

Use `/home/user/miniconda3/envs/fastgen_soflow/bin/python` and
`PYTHONNOUSERSITE=1`. The supervisor exposes only the assigned GPU to each worker
(`CUDA_VISIBLE_DEVICES=1` or `2`, addressed internally as `cuda:0`). Use existing
GPU1 `runs/gpu_1.lock` and GPU2 `/tmp/fastgen-gpu2.lock`; check UUID and occupancy.
Leave all other devices/projects alone. No distributed training or extra agents.

Prepare with `python -m scripts.pre_audit prepare --run-dir <fresh absolute path>`;
execute with `python -m scripts.pre_audit supervise --run-dir <same path>`.
The prepared manifest freezes sources and all inputs before GPU execution.
Workers reject changed source hashes. Preparation and orchestration do not
create an environment or install packages.

Run A0 first, then A1 on GPU1 and A4 on GPU2, then the two independent A2/A3 jobs.
A2/A3 each have a three-hour watchdog including checkpoint and diagnostic work.
Other GPU jobs each have a one-hour watchdog. The supervisor sums overlapping
job durations separately under the shared 43,200-second ceiling, without
double-counting child work. G and D each remain <100M parameters; entire workload
on each GPU remains <20 billion allocated bytes. Record reservation and host RSS.

A failed correctness check blocks dependent stages. Weak D learning is an
observation, not permission for LR search or longer training. Timeout means
incomplete, never a quietly shortened successful run. No automatic retries.
If a repair/recovery is required, retain the old ledger and count its cost and
attempts before any replacement work; never reset consumed allowance.

## Completion

Keep large outputs/checkpoints ignored under the dedicated run directory.
Commit compact evidence, report and code. Report completed/incomplete checks,
cost, saturation, D learnability, clipping-gradient findings, galleries and
limitations. Update the tentative program with measured facts; keep its search
ranges, trial horizon and total campaign allowance unresolved where unsupported.
No Slack messages, Notion edits, remote push or autoresearch launch is included.
