# Frozen campaign protocol — October 2, 2026

This protocol was declared before substantive teacher training. The machine-readable
recipe is [`configs/campaign.json`](../configs/campaign.json); source hashes are
in [`resources/campaign_source.json`](../resources/campaign_source.json).
See [implementation and deviations](IMPLEMENTATION.md) for equations and gradient policy.

| Stage | Updates | Effective batch | Presentations | Selection checkpoints | Solver / sampler |
|---|---:|---:|---:|---|---|
| Scratch teacher T | 30,000 | 64 | 1,920,000 optimization draws | 10k, 20k, 30k | 32-call uniform Euler |
| Each of S1/A1/S2/A2/S3/A3/S4/A4 | 4,000 | 64 per phase | 256,000 G + 256,000 fake-score draws | 1k, 2k, 4k | Exactly the arm's K calls |
| Each adversarial arm, additional real D images | 4,000 | 64 | 256,000 optimization draws | Same as its paired SiD arm | Shared fake encoder |

The teacher horizon gives about 136.6 optimization-set-sized draws. This is a
finite first teacher fit, not a teacher search. Each student's exposure is
fixed at 256k generated images per phase, with 40k images of G adversarial warmup
(625 updates) followed by a 625-update linear ramp to coefficient 0.01. D
training begins at the first fake update; SiD weight stays 1 throughout.
These are local research choices, not claims of reproducing unavailable
external settings. Poor but valid teacher quality does not permit a new teacher.

Teacher LR warms up for 500 updates; G/fake LR warm up for 100. After warmup
each rate remains constant to its finite horizon. No quality-dependent schedule,
early stopping, fake reset, architecture change or horizon extension is allowed.
EMA half-life caps are 100k teacher / 50k student presentations, with a 0.05×
current-presentation ramp-up; this makes the final effective half-lives 96k and
12.8k respectively. Every arm initializes G, fake and EMA freshly from the same
selected teacher, with fresh optimizer and RNG state. K-specific seeds differ;
both methods at the same K share all common draw streams and initialization.

Selection uses seed 1729, exactly 20 samples in each of 151 classes, the fixed
3,020-image development reference and unchanged official clean-FID functions.
Lowest FID at the declared checkpoints wins; earliest wins exact ties. Each
selected student then receives seed 271828 confirmation without reselection.
Checkpoint hash T* is frozen once after teacher selection and gallery review.
There is no use of official validation and no sampling-seed search.

Nonfinite teacher output blocks distillation. Persistent severe teacher collapse
means at least 30 classes flagged at each of the final two teacher checkpoints
and at the selected checkpoint. A class flag is relative 16×16 pixel diversity
below 0.05 or fewer than 10 exact unique outputs among 20. Galleries separately
check recognition and structure; pixel diagnostics alone are not semantic
coverage. FID by itself is not an invalidity threshold. Failures remain in the
nine-run result table rather than being replaced.

Qualification measured about 1.0 second per teacher update and 6.0 seconds per
full 4-NFE adversarial update, at microbatch 16 / accumulation 4. This projects
roughly 8.3 hours of teacher optimization and at most about 6.7 hours of
optimization per student at the profiled longest path; lower K and SiD may take
less, and actual stage cost will be reported. The student horizon is exposure
matched within pairs, not claimed to be compute matched across K.

Finite whole-run watchdogs are **20 hours for T** and **16 hours per student**,
including evaluation, checkpoint I/O, failed attempts and exact replay. They
provide roughly a factor of two around measured optimization cost plus
evaluation/recovery allowance. Qualification/setup is capped at four hours;
the sum of all campaign component allowances is 152 hours. The GPU probe budget
is 96 attempted complete updates; 64 were consumed before final checks, with
all probe weights discarded. The final ledger records any remaining checks.
No-progress watchdog: one hour (evaluation is a distinct phase). The measured
time is specific to this GPU; these watchdogs do not inherit any sibling cap.

Checkpoint every 250 complete updates. At most two recoveries per run may replay
from the last exact checkpoint after external interruption, charging the entire
attempt. Numerical, data, source and ownership failures surface as blockers;
no silent skipped updates or filtered samples. The allowance ledger exists
outside checkpoint state and cannot rewind. Physical GPU 2 never falls back to
another device. The full simultaneous workload must remain below
20,000,000,000 allocated bytes; reserved/device/host measurements are also saved.

Execute T, review/select T*, then S1,A1,S2,A2; report those four student outcomes.
Continue automatically to S3,A3,S4,A4. Export all eight selected students,
report all outcomes and shared-teacher pipeline cost, then stop. No extra
student, independent training replication, adversarial-only arm or second
teacher is included.
