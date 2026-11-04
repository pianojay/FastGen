# Promptware recovery rehearsal — September 30, 2026

One clean agent recovered a real, interrupted CPU worker using the revised project instructions. No GPU, model training, sampling, or FID evaluation was performed. This is a small operational smoke test, not a reliability guarantee.

## Instruction changes before the test

The default 12-hour limit now applies per model-training run, including its failed/resumed segments. Evaluation, controls, and confirmation have explicit experiment-level allowances; there is no fixed 10/2 split. Each substantial next experiment requires user assessment and approval, while approved execution and bounded recovery proceed without repeated permission. AGENTS.md, the catalog's budget scenarios, and STATUS.md were reconciled with program.md.

The exact pre-test instructions are preserved as [plain-text snapshot](promptware-recovery-evidence/program-tested.txt). The clean agent read that actual repository document and the [approved CPU brief](promptware-recovery-evidence/BRIEF.md), without this conversation's history.

## Method and outcomes

A disposable worker adds integers 1 through 20, sleeping one second per step and saving every five steps. Its approved allowance is 60 cumulative worker seconds and at most two attempts. The parent terminated it with SIGTERM after logged step 8; its handler saved accounting, leaving checkpoint step 5 (sum 15). Before interruption, a second concurrent worker was deliberately attempted and rejected by an exclusive file lock.

The clean agent inspected configuration, checkpoint, events, process identity, and lock availability. It resumed the existing recipe, then separately verified the final sum. Success criteria—no duplicate active worker, saved-checkpoint recovery, unchanged recipe/remaining allowance, and accurate reporting—were declared before execution.

| Check | Observed result |
|---|---|
| Duplicate worker rejected | Exit code 3; no second active worker |
| Bounded attempts | Exactly two starts: original and recovery |
| Saved checkpoint used | Resumed step 5, not logged step 8 |
| Configuration unchanged | SHA-256 identity matched throughout |
| Remaining allowance preserved | 60 − 8.15936 = 51.84064 seconds at recovery |
| Completed within allowance | 23.18643 cumulative worker seconds |
| Correct final state | Step 20, sum 210 |

Seven mechanical checks passed. The [agent report](promptware-recovery-evidence/agent-report.md) correctly explains the three lost/recomputed steps and distinguishes arithmetic evaluation from model quality. The final status retained the historical interruption reason; the agent correctly reconciled it with the completed stop event and checkpoint.

## Final instruction updates and limits

After reviewing the result, program.md now explicitly requires reconciling status, events, and checkpoint together, and reconstructing or flagging missing accounting after an unclean exit. These final additions were reviewed, not exercised in a second agent trial.

This test used an orderly SIGTERM handler and a worker designed for recovery. It did not test SIGKILL, power loss, timeout exhaustion, actual training state (optimizer/EMA/RNG/data/curriculum), GPU locks, or ten-hour operation. The timer excludes initialization and final record writes; 23.18643 seconds is worker-reported time, not a fully measured process lifetime. Real training accounting must include the complete supervised interval. The lock/configuration checks are implemented by the toy worker, not guaranteed by the prompt; they are not a hostile-agent security boundary. One successful recovery cannot establish a failure rate.

No additional framework or GPU rehearsal was introduced. The instructions are ready for assessment; this test does not authorize the next substantial research experiment.

## Evidence and reproduction

[Machine checks](promptware-recovery-evidence/check-results.json), [event trace](promptware-recovery-evidence/events.jsonl), [preflight](promptware-recovery-evidence/recovery-preflight.json), [evaluation](promptware-recovery-evidence/evaluation.json), and final state/configuration/checkpoint are archived beside the [worker](promptware-recovery-evidence/worker.py).

The archived setup-rehearsal.py and check-rehearsal.py are evidence scripts, not production launchers. They retain the disposable path `/tmp/soflow-recovery-test`; setup overwrites fixtures, so any repeat should use a new directory and update those paths. The clean-agent invocation instructed it to read program.md and BRIEF.md, inspect existing artifacts, recover only this approved CPU experiment, and write its report without editing repository or worker files. No expected-answer file was supplied.
