# TA-005 MUNIT — CPU preparation complete

The pinned MUNIT architecture works at RGB64 without shrinking its discriminator: the three scales produce 4×4, 2×2 and 1×1 logits. Registered framework size is **46,594,716 parameters**: each domain generator 15,026,187 and each three-scale discriminator 8,271,171. No EMA, VGG or pretrained component is instantiated.

`networks.py` is byte-identical to the pinned upstream file. The modernized engine preserves its Kaiming/Gaussian initialization, LSGAN, image/style/content reconstruction weights 10/1/1 and zero explicit cycle/VGG losses. Changes are device-independent random styles, safe checkpointing and removal of unused discriminator gradients during generator updates. The exact pinned initialization helper was fetched into this experiment directory and tested against the adapted initializer.

Full-width CPU synthetic training checks passed: finite losses and gradients in all four components, content/style reconstruction shapes, nonzero style-code effect and **bit-exact next-step recovery** of parameters, Adam, schedulers, sampler and Python/NumPy/Torch RNG. Three optimizer iterations were executed on synthetic batch-one tensors. Batch eight GPU memory/time qualification remains required. StepLR is constant over the proposed 20,000-update pilot.

All 8,554 frozen A1 pixel hashes passed. A 1,024-pair sampler test exercised all 151 classes, verified same-class unpaired optimization-only draws, independent flips and exact sampler replay. No dataset optimization or GPU use occurred.

Final runner interface: `TRANSFER_AUG_GPU=1 bash transfer_aug/run_python.sh transfer_aug/experiments/TA-005/runner.py --stage all` after the final contract is frozen. It will expose qualification, training, deterministic reference/Gaussian-style generation and CPU reporting stages. Reference translation encodes the source through its domain content encoder and a target donor through the other domain style encoder; sampled translation substitutes the frozen eight-dimensional Gaussian style. Self-reconstruction decodes the source's own content/style.

Proof: `verification.json`, `data_verification.json`, `setup_checks.log`, `verify_setup.py`, `verify_data.py`. There are no setup blockers; GPU qualification is pending authorization through the final contract. These checks establish implementation readiness, not semantic separation or augmentation usefulness.
