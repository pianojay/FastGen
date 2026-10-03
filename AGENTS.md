# FastGen-AFM work instructions

## Current assignment and history

- October 3, 2026: the user-started bounded stability campaign is **complete**. Six candidates failed qualification; no confirmation was triggered. Read [final stability report](docs/STABILITY_REPORT.md). Do not restart candidates or launch further training without a new assignment.

- Read [program.md](program.md) first, then the [closed baseline report](docs/EXPERIMENT_REPORT.md) and [implementation mapping](docs/IMPLEMENTATION.md).
- October 2, 2026: the user approved preparation of a stability-search handoff. Preparation creates documentation only; it does not launch training. When explicitly started, the campaign may implement necessary controls, qualify and run up to six fresh recipes plus one independent confirmation without intermediate approval waits. Stop at the program's boundary.
- The October 1 single-baseline assignment is closed by user-directed termination. Its recipe, checkpoint and report remain historical evidence; do not resume it. Older `docs/HANDOFF.md`, `docs/BASELINE_PLAN.md` and preparation provenance are dated records, not current launch instructions.
- `program.md` is the current campaign policy. No automatic capacity sweep, extra seeds, FID search, teacher, CAFM or few-NFE extension. No extra agents unless requested.

## Machine and isolation

- Work only in `/home/user/jw_soflow/FastGen-AFM`; use **physical GPU 0** with `CUDA_VISIBLE_DEVICES=0`. Recheck GPU UUID and occupancy before GPU work. Historical GPU 0 UUID: `GPU-e9d30df5-3b84-1a7e-c505-c6fff0a0ce7f`; do not silently substitute another GPU after a hardware change.
- Use a host-wide device-specific lease (`/tmp/fastgen-gpu0.lock`) for future GPU jobs, together with PID/start-time identity and occupancy checks. The completed campaign implemented and qualified this host-wide lease in `afm_fastgen/common.py`; retain it for any separately authorized future GPU work.
- GPU 1 and all SoFlow jobs belong to the sibling project. **Physical GPU 2 belongs to FastGen-SiD-DiT**; its independent experiment may run concurrently. Never auto-select the least-used GPU or terminate an unknown process.
- Preserve both sibling repositories, their branches, files, environments and processes. No runtime Python imports from mutable sibling checkouts. The closed CIFAR experiment must not restart.
- Retain `/home/user/miniconda3/envs/fastgen_afm/bin/python`; set `PYTHONNOUSERSITE=1` for execution and installation. Include pip in any explicitly needed environment recreation and use that interpreter's `python -m pip`. No changes to other environments.
- Course dependencies govern the implementation; upstream requirements are reference material, not an installation instruction. Do not contact the course or silently add an essential dependency exception.

## Data and resource boundaries

- Keep G and D architecture fixed at the baseline; each new recipe starts fresh weights, optimizers, EMA, RNG and data state. The official FID Inception is evaluation-only; no external pretrained training networks.
- Preserve official protected files and base `Model`, pinned AFM snapshot and partition in [original provenance](resources/provenance.json). Count each learned model's full framework, including frozen parameters, separately: **at most 100,000,000 parameters per model**. Export only inference G.
- Entire simultaneous training workload must remain **below 20,000,000,000 peak allocated bytes**. Report reserved/device and host memory too. Record device identity and measured throughput; do not translate timings to KCloud/A100.
- Use the copied 14,059 optimization / 3,020 development partition. Official validation is never tuning, calibration, training or checkpoint-selection input. Shared images under `/home/user/jw_soflow/FastGen-SoFlow/data/pokemon-generation-one-22k` are read-only. Do not invoke their downloader, conversion or partition writer.
- Keep artifacts, caches, datasets, generated images, logs and environments out of Git. Use modest CPU/data-loader thread counts for coexistence.
- One-call inference executes G once, with no discriminator, guide, decoder or teacher in sampling. Arithmetic and random-noise generation are not learned calls.
- Predeclare finite horizons/watchdogs, audits and bounded exact recovery. Carry all failures/replayed work and overhead into cumulative accounting. No inherited 12-hour cap and no cost reset by renaming a run.

## Reporting and external actions

- Keep concise incremental tables and machine-readable evidence as described in the program. Preparation is complete after source/document verification and push; saying “launched” is not experimental completion.
- Source commits/pushes to private `pianojay/fastgen-afm` are approved. No Slack messages, Notion edits or collaborator invitations are part of this assignment.
