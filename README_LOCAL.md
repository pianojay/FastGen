# FastGen-AFM handoff

Prepared October 1, 2026 (Asia/Seoul). **Materials only: no integrated Pokémon AFM trainer, environment, trained model or GPU qualification exists here.** Root `model.py` remains the official unimplemented starter. The upstream [README](README.md) documents the challenge, not completed local capabilities.

Read [AGENTS.md](AGENTS.md), then [context and source map](docs/HANDOFF.md), then [one-baseline assignment](docs/BASELINE_PLAN.md). [Provenance](resources/provenance.json) records source revisions and checksums. [Attribution](THIRD_PARTY.md) records source licenses.

This independent checkout belongs to `pianojay/fastgen-afm`. GPU 0 is allocated to AFM; the separate SoFlow job uses GPU 1. Preparation does not launch experiments. A future agent starts only when the user explicitly assigns it the baseline below.

Suggested launch instruction:

> Read AGENTS.md, docs/HANDOFF.md and docs/BASELINE_PLAN.md. Implement, qualify, train and evaluate the single teacher-free, one-call, U-Net-first pixel-space AFM baseline on GPU 0. Choose and record the complete recipe and finite runtime allowance from source inspection and measured throughput. Proceed without intermediate approval waits within that assignment. Keep concise incremental records, preserve SoFlow, and stop after the baseline evaluation with results and proposed next directions.

Private remote: https://github.com/pianojay/fastgen-afm
