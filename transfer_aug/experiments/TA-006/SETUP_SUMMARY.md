# TA-006 StarGAN v2 — CPU setup complete

The scratch-only 64×64 implementation is ready for GPU qualification after the master freezes its training contract. No GPU or dataset optimization occurred.

The full framework contains **131,065,096 parameters**, including EMA: generator32,634,883; mapping2,438,272; style encoder20,327,808; discriminator20,263,170; and EMA copies of G/F/E totaling55,400,963. Trainable total75,664,133. The complete framework exceeds the normal100M cap and relies on the user's StarGAN-specific waiver.

Pinned upstream architectural classes, loss functions, and EMA operation pass exact AST parity. FAN/pretrained metric imports are excluded; no downloaded weights, classifiers, VGG, or dependency installation. The unavailable Munch package is replaced by a local attribute dictionary. Initialization matches the pinned He-normal helper; EMA starts from the initialized trainable models, a documented upstream initialization-order correction.

Full-width synthetic CPU checks passed for RGB64 tensor shapes and domain heads, reference/latent sensitivity, detached second diversity branch, finite four-phase losses including second-order R1, exact optimizer participation, EMA formula, and bit-exact next-step checkpoint recovery for trainable/EMA models, losses, RNG and sampler. CPU verification took72 seconds using batch1; planned GPU batch8 memory/time remains unqualified.

The generator has **unbounded linear RGB outputs**; raw-output preservation and clipping diagnostics are required during EDA. Scratch synthetic sensitivity does not establish learned diversity or identity preservation.

Evidence: `verification.json`, `verify_setup.log`, `source_adaptation.json`, `setup_hashes.json`, `README.md`. API: `StarGAN64`, `Sampler`, four-phase `step`, atomic `checkpoint`/`restore`, and explicit `.nets`/`.ema`. No setup blockers remain.
