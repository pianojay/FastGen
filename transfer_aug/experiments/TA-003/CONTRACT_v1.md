# TA-003 CycleGAN training contract — revision 1

Execute the complete frozen shared/round2/TRAINING_PROTOCOL_v1.md (SHA256 `ae9323b66e4969f87f000d08d0227cc6770e9f1377bc7f62deb6c92e39409c94`), incorporated here in full by reference, with accepted implementation `SETUP_CONTRACT_v1.md` (SHA256 `419a8e3372082d6b48ed657f80ec3f5886a5d57ee857bfcda4fc8d657978df3c`). This final contract authorizes GPU qualification, scratch training, generation and EDA; supersedes setup-only launch restriction.

Assigned physical GPU 1, UUID `GPU-dd35c186-88f4-ac1e-26ae-028adbd8e888`, existing lease `/home/user/jw_soflow/FastGen-SoFlow/runs/gpu_1.lock`. Exclusive write directory: experiments/TA-003/. Shared inputs/documentation are read-only. Budget28,800 cumulative GPU seconds; parameters<100M; peak allocated memory<20,000,000,000 bytes. Source/data/seed/losses/procedure/review/stopping/handoff requirements are those pinned above.

Direction: D0->D1 and D1->D0. CPU framework count: 17,001,352; confirm on GPU. Local convolution/filter discriminator receptive field is 34 pixels; instance normalization couples spatial statistics globally. No unapproved model/grid changes.
