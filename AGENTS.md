# FastGen-AFM work instructions

## Read first and scope

- Read [handoff](docs/HANDOFF.md) and [baseline assignment](docs/BASELINE_PLAN.md). This is a prepared source-and-context workspace, not a working AFM integration.
- Preparation is complete when sources, documents and provenance are verified and pushed. Do not interpret preparation, opening this repository or reading these files as an instruction to train.
- Once the user explicitly starts the baseline, implement, qualify, train and evaluate **one** U-Net-first, teacher-free, one-call pixel-space discrete AFM baseline. Do not wait for approval between those stages. Short disposable qualification checks and exact recovery are included; additional quality trials, architecture sweeps, second training seeds and few-step experiments are not.
- After baseline evaluation, report results and proposed next directions, then stop. No campaign loop, mandatory essays or inherited 12-hour cap. No additional agents unless requested.

## Machine and isolation

- Work in `/home/user/jw_soflow/FastGen-AFM`. This is the local TITAN server, not KCloud.
- Use physical GPU 0 only: `CUDA_VISIBLE_DEVICES=0`. Verify device UUID and ownership before work; do not terminate an unknown process. GPU 1 is running SoFlow. GPU 0's closed CIFAR experiment must not be restarted.
- Do not change either sibling repository, its branch, environment or jobs. Read their source at the pinned commit in provenance; never runtime-import from their mutable working trees.
- Create a separate `fastgen_afm` Conda environment during implementation, including pip. Set `PYTHONNOUSERSITE=1`; use that interpreter's `python -m pip`, never bare pip. Do not install into `fastgen_soflow`, `soflow` or `soflow_eval`.
- Prefer the official supplied dependencies. Upstream AFM requirements are reference material, not an approved install list. Port the needed operations using supplied libraries; report an essential dependency exception to the user rather than contacting the course or installing it silently.
- All run outputs, caches, references and checkpoints belong here, outside Git. The sibling image directory is read-only input: do not run its downloader, transparency conversion or partition writer. Limit CPU/data-loader threads reasonably to coexist with GPU 1; measure resulting throughput.

## Training and evaluation boundaries

- Generator and discriminator begin with fresh weights, optimizer, EMA, RNG and data state. No teacher, pretrained VAE, pretrained discriminator/features, classifier guidance or CAFM in this baseline. The official FID feature extractor is evaluation-only.
- Preserve official protected files and base `Model`; hashes and source pins are in [provenance](resources/provenance.json). Every learned model used in training or sampling must individually have at most 100,000,000 parameters, including its full framework and frozen parameters. Check generator and discriminator separately; export only the inference generator.
- Enforce peak training allocation below 20,000,000,000 bytes for the complete simultaneous training workload. Log reserved/device memory and host memory too. TITAN measurements are not A100 timing predictions.
- Use the copied fixed optimization/development partition. Official validation images are never training, checkpoint-selection or tuning input. Do not regenerate the partition.
- One-call sampling executes the generator exactly once; no trained auxiliary network or guidance at inference. Arithmetic and noise generation do not add NFE. Preserve the official loader and check a self-contained export.
- Choose a finite full horizon and schedule before training from measured cost and a written rationale. Account for G and D updates, image presentations, qualification, overhead, evaluation and failures. Exact resume keeps recipe/horizon unchanged and carries accounting forward. Do not silently extend a completed run or rename failures to reset its allowance.
- Minimal durable process identity, GPU ownership, logs and recoverable state should prevent duplicate jobs and support hours-long supervision. An agent saying “launched” is not completion. Report real blockers; do not fill unknown metrics with invented values.

## Communication and external actions

- Log incrementally in the baseline document's tables, linking configurations and machine-readable metrics; short progress updates suffice.
- The user approved the private repository `pianojay/fastgen-afm` and source pushes. Exclude credentials, datasets and large artifacts. No Slack messages, collaborator invitations or Notion writes are part of this assignment.
- The authorization for GPU 0 here supersedes historical CIFAR ownership wording in sibling documents; it does not authorize changes to those projects. Their older budgets, launch commands and broader autonomous loops are historical/contextual, not this experiment's policy.
