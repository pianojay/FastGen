# TA-003 preparation completed

Implemented the proposed CycleGAN RGB64 adaptation using exact selected pinned-upstream network definitions and preserved license notices. The framework has **17,001,352 registered parameters**: 7,837,699 for each six-block generator and 662,977 for each two-layer discriminator. The 34px convolutional PatchGAN receptive field and 14×14 output fit the small input; InstanceNorm still couples spatial statistics globally.

Both directions use standard LSGAN, cycle weight 10, identity weight 5, independent 50-image replay pools, scratch normal initialization, and Adam (0.0002; 0.5/0.999). The prepared sampler chooses classes uniformly, then unpaired same-class D0/D1 optimization images with independent optional flips. Diagnostic images are excluded and inputs/hashes validated. No labels enter model losses.

**29 CPU synthetic checks passed.** Proof: `proof/cpu_checks.json`, `proof_cpu.log`, and `test_setup.py`. Exact upstream forward equality, finite outputs/losses/nonzero gradients, replay behavior, normalization, schedule endpoints, data rejection, and registered counts were checked. Full-width networks completed three batch1 synthetic optimizer calls. Atomic checkpoint recovery restored both Adam states, all network parameters, both replay pools, Python/NumPy/torch RNGs, and sampler state; subsequent outputs, losses and state digests matched exactly. The nominal batch16 sampler was tested; GPU and batch16 optimization are not yet qualified.

All work remained CPU-only: zero GPU seconds, no research-cohort training, external weights, or package installation. `README.md` documents attribution, equations, assumptions and the reproducible wrapper command. `setup_hashes.json` binds implementation and evidence.

Training remains intentionally blocked on the master’s final contract/domain freeze and authorized GPU preflight. The provisional manifest schema and the LR endpoint convention (zero at update 20,000) need final alignment. GPU lease/budget orchestration, checkpoint cadence, shared evaluation protocol and GPU recovery qualification await that contract. No image-quality, class-preservation, usefulness or FID claims are made.
