# TA-006 StarGAN v2 — final contract revision1

Execute the complete shared/round3/TRAINING_PROTOCOL_v1.md, SHA256 `75937769254ab5634618d104d97b0e51c4b1bf20128e02c1566eeffa32d6a8f2`, incorporated by reference, with this experiment's SETUP_CONTRACT_v1.md SHA256 `88e8d644d6c75d723bd44896c926713a787486264693d4bd9a1743922c11fdef`. This final contract authorizes GPU qualification, scratch training, final reference/latent generation, EDA, direct visual review and handoff. It supersedes setup-only GPU/training restriction.

Assigned physical GPU2, UUID `GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2`, existing read-only-opened lease `/tmp/fastgen-gpu2.lock`. Exclusive write directory experiments/TA-006/. Batch8, seed20261008, max20,000 complete iterations, cumulative28,800GPU seconds including qualification/retries/inference, at least900s reserved for final generation; peak training allocation<20,000,000,000 bytes. Method-specific losses/phases and optimizer schedule are in pinned setup contract.

StarGAN v2 count131,065,096 including55,400,963 EMA parameters; trainable75,664,133. Explicit user exception waives100M cap for this framework only. Preserve EMA and full count; all other limits remain. Use EMA final outputs; w_hpf=0, noFAN.

Frozen shared/round3 evaluation:604diagnostic sources,4reference+4latent translations each,604self-controls:5,436final outputs. Gallery, diversity/range/preservation checks and fixed representative/tail review per common protocol. All data A1; noA2/B/SiDA integration, external weights or packages. Protocol/setup grids immutable absent revised contract.
