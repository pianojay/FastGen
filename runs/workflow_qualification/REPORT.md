# Pokémon SoFlow local experiment

Environment: local TITAN RTX; KCloud qualification and fresh training are separate.

Status: {"state": "training_complete", "step": 1, "schedule_steps": 1, "reason": "schedule_complete", "elapsed_budget_seconds": 23.07673840224743}

Logged updates: 1

Development FID (official training-derived holdout; not TA scores):

| Checkpoint | NFE | Seed | FID |
|---|---:|---:|---:|
| step_000001 | 1 | 42 | 469.2630 |
| step_000001 | 2 | 42 | 464.7993 |
| step_000001 | 3 | 42 | 462.5511 |
| step_000001 | 4 | 42 | 460.8107 |

Peak training allocation: 8.287 GB.
Peak parent RSS: 1.707 GB.
Peak sampled process-tree PSS (including loader workers): 1.808 GB.

![Training curves](curves.png)

## Latest labeled grids

![000001, nfe1](samples/000001/nfe1.png)

![000001, nfe2](samples/000001/nfe2.png)

![000001, nfe3](samples/000001/nfe3.png)

![000001, nfe4](samples/000001/nfe4.png)
