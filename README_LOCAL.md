# FastGen-AFM

**Current state, October 3, 2026:** the bounded stability-search campaign is complete. All six fresh recipes failed the frozen recognizable-diversity criteria; no confirmation was triggered. Training is stopped. See the [final report](docs/STABILITY_REPORT.md), [incremental evidence](docs/STABILITY_SEARCH.md), [program.md](program.md) and [AGENTS.md](AGENTS.md).

The stopped baseline completed 10,294 G/D cycles; the complete 10k checkpoint and logs are preserved. Trained development FID was unmeasured. See the [experiment report](docs/EXPERIMENT_REPORT.md), [frozen recipe](configs/baseline.json) and [historical stage accounting](docs/BASELINE_PLAN.md). Do not resume that run.

Available: working baseline G/D/trainer, recoverable state, data cache, panels and official evaluation/export utilities. The campaign adds calibrated diagnostics, separate optimizer/gradient-normalization controls, a device-wide lease and a finite search runner. The historical preparation provenance remains unchanged.

The completed campaign held the architecture fixed and tested six evidence-driven recipes. Its goal was recognizable, persistent within-class variation; no FID benchmarking was performed. Use **physical GPU 0** and the existing `fastgen_afm` environment with `PYTHONNOUSERSITE=1`. SiD-DiT uses physical GPU 2; GPU 1 remains assigned to SoFlow.

Private remote: <https://github.com/pianojay/fastgen-afm>. [Campaign preparation provenance](resources/campaign_handoff.json) separates the new instructions from the [original source provenance](resources/provenance.json).

## Next assignment

The campaign and historical baseline are closed. Preserve their artifacts; do not relaunch or extend either run. Proposed next directions are in the final report and require a new assignment.
