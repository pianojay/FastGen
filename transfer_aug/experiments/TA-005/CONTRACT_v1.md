# TA-005 MUNIT — final contract revision1

Execute the complete shared/round3/TRAINING_PROTOCOL_v1.md, SHA256 `75937769254ab5634618d104d97b0e51c4b1bf20128e02c1566eeffa32d6a8f2`, incorporated by reference, with this experiment's SETUP_CONTRACT_v1.md SHA256 `045526507a7364844b31b5aefcadf363ee6639b1368286f8c0f97b3ff366f1f9`. This final contract authorizes GPU qualification, scratch training, final reference/latent generation, EDA, direct visual review and handoff. It supersedes setup-only GPU/training restriction.

Assigned physical GPU1, UUID `GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888`, existing read-only-opened lease `/home/user/jw_soflow/FastGen-SoFlow/runs/gpu_1.lock`. Exclusive write directory experiments/TA-005/. Batch8, seed20261008, max20,000 complete iterations, cumulative28,800GPU seconds including qualification/retries/inference, at least900s reserved for final generation; peak training allocation<20,000,000,000 bytes. Method-specific losses/phases and optimizer schedule are in pinned setup contract.

MUNIT count46,594,716; framework must remain below100M. No EMA/VGG.

Frozen shared/round3 evaluation:604diagnostic sources,4reference+4latent translations each,604self-controls:5,436final outputs. Gallery, diversity/range/preservation checks and fixed representative/tail review per common protocol. All data A1; noA2/B/SiDA integration, external weights or packages. Protocol/setup grids immutable absent revised contract.
