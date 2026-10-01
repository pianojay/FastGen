# FastGen-SiD-DiT work instructions

## State and authorization

- Read [HANDOFF](docs/HANDOFF.md) and [EXPERIMENT_PLAN](docs/EXPERIMENT_PLAN.md). This is a source-and-context handoff, not a runnable integration. Root `model.py` is unchanged starter code; `vendor/` contains reference sources only.
- Preparation ends after documentation/source verification, commit and push. Do not install dependencies, qualify a GPU or train merely because these files were opened. No environment or GPU qualification was performed during preparation.
- Once explicitly started, implement, qualify, train one scratch flow teacher and eight teacher-initialized students (SiD/SiDA at 1, 2, 3, 4 NFE), evaluate and export without intermediate approval waits. Report 1–2 NFE results before automatically continuing to 3–4. No ninth student, independent training-seed replication, architecture sweep or second teacher search is authorized by this assignment. Bounded engineering probes and exact recovery belong to it.
- Stop after final evaluation/report. Genuine resource, data or implementation blockers must be reported; do not invent results or silently replace the scientific assignment. No extra agents unless requested.

## Machine, environment and data

- Work in `/home/user/jw_soflow/FastGen-SiD-DiT`. **Physical GPU 2 only: `CUDA_VISIBLE_DEVICES=2`.** The selected device is then process-local `cuda:0`, not `cuda:2`. Verify physical UUID, model, capability and ownership at qualification; no UUID/hardware type was assumed during preparation. If GPU 2 is unavailable, report/wait for that device; never choose another GPU automatically.
- Use a host-wide physical-device lease, recommended `/tmp/fastgen-gpu2.lock`, with `flock`, process start identity and occupancy checks, held for every GPU stage. AFM independently uses GPU 0 and `/tmp/fastgen-gpu0.lock`. Both experiments may run concurrently. Do not copy AFM's hardcoded GPU 0 UUID or its project-local lock as a cross-project solution.
- Preserve GPU 1 and all SoFlow jobs, both sibling repositories and their environments. Do not kill unknown processes, reset devices, change clocks or runtime-import from mutable sibling checkouts.
- Create independent Conda environment `fastgen_sid_dit` only during implementation, including Python and pip. Set `PYTHONNOUSERSITE=1` for installation/execution; use its interpreter's `python -m pip` and verify package locations. Choose the supported precision after measuring GPU 2; do not assume native BF16 from upstream defaults.
- Use official supplied dependencies. Upstream environment files are an audit list, not bulk-install instructions. Report an essential unapproved dependency instead of silently adding it or contacting the course.
- Shared images under `/home/user/jw_soflow/FastGen-SoFlow/data/pokemon-generation-one-22k` are read-only. Use the copied partition; never regenerate it, run the shared downloader/compositor, or tune on official validation. Keep separate caches, logs and checkpoints here, outside Git. Bound CPU/data-loader threads for coexistence.

## Scientific and challenge boundaries

- Preserve protected starter files, the base `Model`, upstream snapshots and partition identity in [provenance](resources/provenance.json). Implement outside `vendor/`, with copied-source attribution where needed.
- All training weights originate from our scratch-trained teacher on the optimization partition. Initializing students and fake-score networks from that eligible teacher is the explicitly authorized distillation procedure; it does not permit external pretrained weights.
- No text encoder, VAE, CLIP/perceptual training network, external DiT weights, classifier guidance or classifier-free guidance in this initial class-conditioned recipe. The official FID feature extractor is evaluation-only.
- Each learned model's entire framework, including frozen parameters and any attached discriminator head, must separately have **at most 100,000,000 parameters**. Record teacher, student and fake-score/discriminator counts; shared parameters count once within their containing framework. Do not hide a large model by splitting its count.
- Enforce **peak allocated memory below 20,000,000,000 bytes** for the full simultaneous workload (teacher, student, fake-score/D, optimizer state, EMA and activations). Report reserved/device and host memory too. Gradient accumulation/checkpointing can adapt execution; silent architecture changes cannot preserve an architecture-matched claim.
- Inference exports contain only the selected student plus required sampler metadata. Each neural evaluation counts: verify exactly N calls for each NFE student, including partially executed networks. No hidden teacher/fake-score/decoder/guide at inference.
- Predeclare finite horizons, watchdogs, recovery limits and selection rules, with all failed/replayed work charged. No inherited 12-hour cap, automatic 40k horizon, or reset by renaming an interrupted run.

## Communication and external actions

- Use concise incremental metrics/tables, including stage cost and all eight outcomes. Teacher cost is incurred once and included in the pipeline total. Sampling-seed checks do not constitute training replication.
- Source commits/pushes to private `pianojay/fastgen-sid-dit` are approved. Keep large artifacts and credentials outside Git. No Slack messages, Notion edits or collaborator invitations are part of this work.
