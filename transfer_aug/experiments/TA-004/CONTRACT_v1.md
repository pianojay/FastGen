# TA-004 CUT training contract — revision 1

Execute the complete frozen shared/round2/TRAINING_PROTOCOL_v1.md (SHA256 `ae9323b66e4969f87f000d08d0227cc6770e9f1377bc7f62deb6c92e39409c94`), incorporated here in full by reference, with accepted implementation `SETUP_CONTRACT_v2.md` (SHA256 `48d2c28533af0a7e591605f459c7c71bcbd5e4d8c7bb9ef8be5df8b81a365a82`). This final contract authorizes GPU qualification, scratch training, generation and EDA; supersedes setup-only launch restriction.

Assigned physical GPU 2, UUID `GPU-1dc48c25-2861-508e-c83b-a2a4fab02ad2`, existing lease `/tmp/fastgen-gpu2.lock`. Exclusive write directory: experiments/TA-004/. Shared inputs/documentation are read-only. Budget28,800 cumulative GPU seconds; parameters<100M; peak allocated memory<20,000,000,000 bytes. Source/data/seed/losses/procedure/review/stopping/handoff requirements are those pinned above.

Direction: D0->D1. CPU framework count: 9,061,060; confirm on GPU. Local convolution/filter discriminator receptive field is 40 pixels; instance normalization couples spatial statistics globally. No unapproved model/grid changes.
