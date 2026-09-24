# SoFlow CIFAR-10 documentation

- [Few-NFE training-cost literature audit](few-nfe-training-cost-audit.md): Shortcut, MeanFlow, iCT, distillation, pretrained consistency tuning, and LD3/BézierFlow sampler optimization; teacher-cost accounting and implications for local budgets.

- **[Formal experiment report and stop decision](EXPERIMENT_REPORT.md)**: completed trainability/throughput findings, one-day feasibility analysis, preserved evidence, and checkpoint limitations. The full training run was stopped on 2026-09-24.

- [CIFAR-10 training scale and runtime](cifar10-training-cost.md): paper schedule, epoch calculations, TITAN RTX throughput measurements, and interpretation of the approximately four-month estimate.
- [Training run record](TRAINING_RUN.md): setup, commands, implementation changes, and operational history.

This folder is the canonical home for documents produced by the SoFlow CIFAR-10 work. They live in Astra's FastGen checkout so both agents can use them; both repositories' AGENTS.md files link here. Add future CIFAR documents here and maintain this index and those references instead of creating duplicate documents in the SoFlow checkout.

The experiment described here runs in `/home/user/jw_soflow/SoFlow` on GPU 0. Commands, configurations, and runtime artifacts refer to that checkout unless explicitly stated otherwise. The FastGen/Pokémon experiment has separate settings and records elsewhere in `FastGen-SoFlow/docs/`.

The documents were relocated here on 2026-09-24. Code, scripts, datasets, checkpoints, and logs remain in the SoFlow checkout. Links to those files assume the two checkouts are siblings; measured summaries remain readable when only the FastGen repository is available.
