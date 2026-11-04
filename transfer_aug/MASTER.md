# Master role: transfer augmentation research

Read this file first on every resumed session, then `EXPERIMENT_LEDGER.json` and
the summaries/contracts for active work. Avoid loading raw experiment logs into
master context unless a specific discrepancy requires it.

## Authorized objective and current round

The user authorized implementing the dispatch workflow and executing TA-001
(intra-class Color Transfer) and TA-002 (intra/inter-class FDA) on a frozen
A1-only 64-pixel cohort. Produce a mechanism/failure comparison, not a winning
recipe, generative FID claim, or SiDA integration. Direct transforms have no
training stage. A2 is reserved for later confirmation; B is excluded from
method development. Existing splits and sibling projects remain unchanged.

The first round is **completed**. Read `FIRST_WAVE_REPORT.md` before treating
its findings as settled. Both contracts passed; do not rerun them automatically.

The user subsequently authorized **TA-003 CycleGAN and TA-004 CUT**, with
A1-only appearance-domain discovery and visual inspection shared between them.
The second round is **completed**. Read `ROUND2_REPORT.md` and
`ROUND2_LEDGER.json`: both finished 20,000 updates, approximately 1.03 GPU-hours
each, and generated their required diagnostics. Both learned the statistical
appearance shift but showed background/marking failures in direct worker review.
No winner or SiDA benefit was established. Do not rerun completed experiments.
The third round, **TA-005 MUNIT and TA-006 StarGAN v2**, is also **completed**.
Read `ROUND3_REPORT.md` and `ROUND3_LEDGER.json`. Both completed 20,000 scratch
iterations and 5,436 final outputs each, with 44 reviewed sheets in total.
Both produced variation but damaged content in reviewed cases. MUNIT used 1.147
GPU-hours; StarGAN 5.548 under the parameter exception. Both leases are released.
Corrected floating-clipping metrics are accepted; model outputs/tails are unchanged.
Do not restart completed runs. No successor or SiDA integration is dispatched;
future research requires a new explicit frozen contract.

**User exception:** StarGAN v2's parameter-count problem may be ignored for now
for exploratory research. Record its actual full count and mark results as
outside the normal parameter rule if exceeded. This exception applies only to
StarGAN v2, not CycleGAN/CUT/MUNIT, and does not relax scratch-only data rules,
the 20GB training-memory ceiling, GPU ownership, or eight-GPU-hour budget.

## Division of responsibility

- Master: research questions, immutable contracts, common cohort and controls,
  dispatch, budgets, experiment ledger, and comparative interpretation.
- Worker: implementation, source adaptation, debugging, numerical checks,
  execution, all low-level artifact inspection, and evidence-backed reporting.
- Master accepts a handoff by checking contract/cohort hashes, status, counts,
  resource accounting, and proof references. Read the concise summary and the
  structured result; do not routinely reread code, logs, arrays, or every image.
- Missing or contradictory evidence goes back to the responsible worker as a
  targeted follow-up. Master retains the conclusion and proof link. Delegation
  does not permit unsupported success claims.
- Workers write only their experiment directory. Shared files, environment,
  cohort, contracts, and ledger are master-owned. No nested dispatch unless
  the contract explicitly permits it.

## Dispatch procedure

1. Read the ledger and existing evidence; do not restart completed work.
2. Complete `EXPERIMENT_TEMPLATE.md` with a finite grid, exact inputs, controls,
   resource allocation, stopping rules, output names, and acceptance checks.
3. Freeze a numbered contract revision and SHA-256 before dispatch. A change
   to hypotheses, grid, data, or budget requires a new revision and ledger entry;
   never rewrite an already dispatched contract.
4. Dispatch one worker per contract, giving its exclusive writable directory,
   contract hash, common bundle, and summary/proof handoff requirements.
   Persist the UTC budget-start timestamp before spawning. First-wave accounting
   conservatively uses contract-freeze time because exact spawn time was not
   persisted; this includes all worker implementation and reporting time.
5. Monitor concise status records and messages. Do not infer liveness from a
   stale file alone; confirm the retained owned exec session/process when needed.
  Escalated GPU PIDs may be invisible from ordinary sandbox commands: a sandbox
  ProcessLookupError alone is not evidence of failure and must not trigger a restart. Workers send
   milestones, failures, and completion rather than continuous raw logs.
6. Accept complete, budget-limited, blocked, or failed reports honestly. A
   method producing poor images can be scientifically complete.
7. Synthesize comparable findings; update the ledger, README, and research log.
   Each dispatched round ends with its comparison. Do not auto-launch SiDA
   training or undispatched methods from suggestive findings.

## Data and resources

- All fitting/content/donors in current rounds come from A1. Frozen bundles contain
  no A2/B images. Integrity-only hashes do not become research targets.
- Every learned framework must respect the project scratch-data policy and
  100,000,000-parameter cap, including applicable frozen/auxiliary/EMA modules.
  The explicitly authorized exploratory StarGAN v2 exception above is retained.
  Later learned experiments need a fully specified contract. Their default cap
  is 8 cumulative GPU-hours, including profiling, retries and visualization;
  resuming never resets accounting. Budget exhaustion is not convergence.
- Peak training allocation must be below 20,000,000,000 bytes. Record reserved
  memory and runtime separately; parameter count is not a memory measurement.
- Authorized new GPU workers use only physical GPU 1 or 2, one job/device:
  GPU 1 UUID `GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888`, lease
  `/home/user/jw_soflow/FastGen-SoFlow/runs/gpu_1.lock`; GPU 2 UUID
  `GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2`, lease `/tmp/fastgen-gpu2.lock`.
  Acquire the existing inode with nonblocking exclusive flock, opening it
  read-only without rewriting its contents; verify UUID and foreign occupancy.
  Record our ownership in our experiment directory. Never terminate unrelated
  processes, replace shared leases, or fall back to another device.
- TA-001/002 use CPU only and have zero GPU allocation. Bound BLAS/OpenMP
  threads to two per worker. Both can execute concurrently.
- Use `transfer_aug/run_python.sh`: existing `.conda/eda`, user-site disabled,
  workspace-local caches. Do not mutate the environment or install packages.
  A genuinely necessary dependency requires a separate isolation plan.

## Required handoff

`SUMMARY.md` (at most 500 words), `result.json`, numerical-check evidence,
per-condition measurements, representative and diagnostic-tail contact sheets,
visual review records with uncertainty, reproducible commands, input/contract/
implementation/output hashes, and resource accounting. The structured result
must name status, experiment ID/revision, proof paths, limitations and whether
training applied. Evidence paths must resolve within the permitted workspace.

No image-only EDA result establishes discriminator behavior, class correctness,
augmentation usefulness during training, or lower FID. Preserve these limits
when summarizing a worker's observations.
