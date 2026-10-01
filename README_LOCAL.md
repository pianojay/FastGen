# FastGen-AFM

**Current state, October 2, 2026:** the first baseline was implemented and stopped for strongly suspected within-class collapse. A bounded stability-search assignment is now prepared, **not started**. Read [program.md](program.md) and [AGENTS.md](AGENTS.md) for current instructions.

The stopped baseline completed 10,294 G/D cycles; the complete 10k checkpoint and logs are preserved. Trained development FID was unmeasured. See the [experiment report](docs/EXPERIMENT_REPORT.md), [frozen recipe](configs/baseline.json) and [historical stage accounting](docs/BASELINE_PLAN.md). Do not resume that run.

Available: working baseline G/D/trainer, recoverable state, data cache, panels and official evaluation/export utilities. Still to implement when the campaign is started: calibrated collapse/diversity diagnostics, configurable search controls, device-wide lease and bounded campaign orchestration. This documentation preparation did not change training code or create GPU work.

The new campaign holds the architecture fixed, searches up to six evidence-driven training recipes, and independently repeats the first qualifying result once. Its goal is recognizable, persistent within-class variation; FID is not its search objective. Use **physical GPU 0** and the existing `fastgen_afm` environment with `PYTHONNOUSERSITE=1`. SiD-DiT uses physical GPU 2; GPU 1 remains assigned to SoFlow.

Private remote: <https://github.com/pianojay/fastgen-afm>. [Campaign preparation provenance](resources/campaign_handoff.json) separates the new instructions from the [original source provenance](resources/provenance.json).

## Launch instruction for a fresh agent

> Read AGENTS.md and program.md. Start the bounded AFM stability-search campaign on physical GPU 0. Audit the preserved baseline, implement only the required controls and diagnostics, then run up to six fresh training recipes and one independent confirmation under the program's stopping rules. Proceed without intermediate approval waits inside that scope. Preserve SoFlow and the GPU 2 project, keep concise incremental evidence, and stop with the campaign result.
