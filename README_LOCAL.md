# FastGen-SiD-DiT: flow-SiDA with a class-conditioned U-Net

**October 2, 2026: source-and-context handoff prepared; experiments not started.** This repository retains the official FastGen starter and pinned upstream references. Its root `model.py` is still the official unimplemented skeleton. There is no integrated Pokémon trainer, environment, qualified GPU or trained model here.

Read [AGENTS.md](AGENTS.md), [context and source map](docs/HANDOFF.md), then [experiment assignment](docs/EXPERIMENT_PLAN.md). The project name records the method source: the planned backbone is a roughly 62M DDPM++ **U-Net**, not a DiT.

Assignment: train one eligible pixel-space rectified-flow teacher, then eight separate students: matched SiD/SiDA pairs at 1, 2, 3 and 4 NFE. Complete/report the 1–2 NFE pairs before continuing to 3–4. Use **physical GPU 2**, independent future environment `fastgen_sid_dit`, and the existing 14,059/3,020 optimization/development partition. GPU 0 belongs to AFM; GPU 1 and SoFlow remain untouched.

Private remote: <https://github.com/pianojay/fastgen-sid-dit>. [Source provenance](resources/provenance.json) · [Third-party notices](THIRD_PARTY.md).

## Launch instruction for a fresh agent

> Read AGENTS.md, docs/HANDOFF.md and docs/EXPERIMENT_PLAN.md. Start the flow-SiDA assignment on physical GPU 2: implement and qualify the class-conditioned U-Net adaptation, train one rectified-flow teacher, and complete the eight matched SiD/SiDA students at 1/2/3/4 NFE. Freeze recipes and finite watchdogs from source evidence and measured throughput before substantive training. Report the 1–2 NFE pairs, then continue to 3–4 without intermediate approval waits. Preserve GPU 0 AFM and GPU 1 SoFlow. Keep concise incremental records and stop after the assigned results.
