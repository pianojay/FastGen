# <experiment ID>: <question> — contract revision <N>

Master fills every applicable field before freezing. Mark inapplicable stages
explicitly. Place the contract in the assigned experiment directory and record
its SHA-256 in the master ledger. Workers do not edit frozen contracts.

## Question and hypothesis

- Intended transferable factor and operational definition:
- Required invariants (identity, anatomy, markings, layout, etc.):
- Observable falsification/failure conditions:
- Scope of claims and explicitly unmeasured outcomes:

## Implementation

- Primary paper, upstream repository, pinned commit and file hashes:
- Exact algorithm/configuration and all adaptations:
- Learned components, initialization/data provenance, parameter count per
  component and containing framework (include frozen/auxiliary/EMA modules):
- Environment/dependencies, CPU/GPU device, exclusive writable directory:
- No unapproved external weights, package changes, or shared-file changes.

## Inputs

- Frozen cohort/manifests and hashes; allowed partitions:
- Preprocessing, color/numeric spaces, export/quantization convention:
- Content/donor sampling policy, seed, duplicate handling:
- Exact grid, unchanged/self-donor/comparator controls:
- Representative review cohort and deterministic diagnostic-tail rule:

## Deterministic procedure

1. Preflight: verify contract, bundle, protected inputs, environment and resources.
2. Implement and verify numerical/mechanism properties against declared checks.
3. Train/optimize only if applicable: initialization, optimizer, batch size,
   update schedule, loss weights, seed, checkpoint cadence/selection, recovery.
   Otherwise record `training_applicable=false`; do not fabricate training.
4. Generate every declared condition and control; preserve unclipped numeric
   results plus display images, provenance, order, and hashes.
5. Compute declared EDA; inspect the complete required representative/tail
   review set. Distinguish measurements from qualitative judgments.
6. Verify reproducibility and input integrity; produce handoff and final status.

## Resources, monitoring and stopping

- Finite work ceiling, wall-time/GPU-time budget, cumulative retry accounting:
- UTC budget-start timestamp persisted before dispatch; GPU usage measured
  separately from CPU development/reporting wall time:
- CPU threads and memory limits; GPU UUID, existing lease, VRAM qualification:
- Status record: current stage, owned PID, completed/expected items, elapsed
  time, last update, checkpoint/recovery location; update at least per minute
  during sustained work. Send the master short milestone/blocker messages.
- Fail on input/contract mismatch, nonfinite output, resource overrun or
  boundary violation. Debug implementation faults within the original scope;
  stop and report if fixing requires changing the scientific contract.
- Never extend time, change the grid, use another partition/device, install
  packages, or start a successor experiment autonomously.

## EDA and acceptance

- Exact metrics and definitions, aggregation units, visual rubric:
- Mechanism/correctness tests and numeric tolerances:
- Meaningful success/failure criteria, with uncertainty and limitations:
- Accepted scientific status: completed / budget_limited / blocked / failed.
  Completion requires evidence completeness, not favorable image quality.

## Deliverables and master interface

- `SUMMARY.md`: <=500 words; question, key findings, evidence references,
  limitations and proposed next question. No raw log dumps.
- `result.json`: ID/revision/status; contract/cohort/source hashes; training
  applicability; completed/expected counts; parameter/device/time/memory
  accounting; check results; proof paths; limitations.
- `status.json`, commands/run configuration, source and output hash manifest,
  verification record, per-output/condition metrics and visual review records.
- Representative contact sheets and separately labeled diagnostic tails;
  numeric outputs and display images needed to reproduce the findings.
- Worker inspects low-level artifacts. Master checks the summary and handoff
  metadata; discrepancies receive a targeted worker follow-up.
