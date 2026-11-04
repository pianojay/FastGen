# Method interpretations and experimental consequences

October 7, 2026. These distinctions reconcile the user's impressions with the
primary studies and guide the authorized first round. They do not constitute
new evidence about performance on Pokémon.

| Method | Interpretation and consequence |
| --- | --- |
| Color Transfer | Intra-class donors are conservative, but global moments also carry background and lighting. Class restriction alone cannot ensure suitable colors or retained markings. First round uses intra-class transfer only. |
| [FDA](https://arxiv.org/abs/2004.05498) | Low-frequency amplitude transfer preserves source phase, not a semantic guarantee. At 64 pixels both the integer window and gamut clipping need explicit examination. Compare intra/inter-class donors separately. |
| [MixStyle](https://arxiv.org/abs/2104.02008) | Domain labels are optional; mixing occurs in feature statistics rather than RGB. Intra-class pairing is a possible adaptation, but statistics may encode more than color. It needs an eligible host representation and a distinct experimental contract. |
| [Gatys](https://arxiv.org/abs/1508.06576) | Optimizes pixels for each pair using content and feature-Gram losses. It does not train a reusable image generator. The standard pretrained feature network conflicts with our retained scratch-only policy. |
| [AdaIN](https://arxiv.org/abs/1703.06868) | Aligns channel moments in an encoded space and uses a learned decoder. Similar content/style motivation to Gatys, different computation and training requirements. A scratch feature replacement needs separate justification. |
| [WCT](https://arxiv.org/abs/1705.08086) | Whitening/coloring transfers feature covariance. It is not a boundary-discovery or segmentation method; masks are optional spatial controls. A verified foreground mask could localize transfer but is not presently available. |
| [CycleGAN](https://arxiv.org/abs/1703.10593) | Learns mappings between supplied distributions with adversarial/cycle losses. It is not a per-image Gatys optimizer, and cycle consistency does not prove identity preservation. Appearance domains must be defined first. |
| [CUT](https://arxiv.org/abs/2007.15651) | Contrastive patch correspondence constrains the mapping but does not decide permissible semantic changes. Horse-to-zebra translation intentionally changes category. Domain and identity choices remain our responsibility. |
| [MUNIT](https://arxiv.org/abs/1804.04732) | Multimodality means multiple valid outputs for one input, not multiple class IDs. Separate content/style encoders address within-class variation that class labels do not specify. Complexity can justify deferral; labels do not make it redundant. |
| [StarGAN v2](https://arxiv.org/abs/1912.01865) | Shared facial structure helps its demonstrations but is not a universal architectural prerequisite. Domains could be rendering types rather than species. Unknown domains, diverse geometry and audited framework size justify lower priority here. |
| [Swapping Autoencoder](https://arxiv.org/abs/2007.00653) | Automatically swaps learned codes without manually paired training data or style-domain labels. Its costs are training, 64-pixel redesign, and uncertain structure/texture disentanglement—not manual editing per output. |
| [DiffAugment](https://arxiv.org/abs/2006.10738) | An augmentation policy, not a metric. Forward Gaussian noising and differentiable color/geometric augmentation impose different transformations. Their interaction needs a later objective-specific comparison. |

## FDA artifacts and the actual SiDA objective

The local pixel-flow adaptation presents noisy real and generated images to its
discriminator, with corruption `(1-t)x + t*epsilon`; the discriminator shares
the fake-score encoder. It also uses a time-dependent generator adversarial
weight. See [corruption/objective code](/home/user/jw_soflow/FastGen-SiD-DiT/flow_sid/flow.py:15)
and [training loop](/home/user/jw_soflow/FastGen-SiD-DiT/flow_sid/engine.py:214).
These local choices must be distinguished from the [SiDA paper](https://arxiv.org/html/2410.14919v4).

The [original SiDA implementation](https://github.com/mingyuanzhou/SiD/tree/sida)
supports an augmentation pipeline before Gaussian noise. This is also visible
in the [vendored loss](/home/user/jw_soflow/FastGen-SiD-DiT/vendor/sid-sida/sida_training/sida_loss.py:38).
Thus diffusion noise does not establish that DiffAugment is redundant.

**Inference, not measurement:** if FDA-modified images become positive real
targets, systematic artifacts may be rewarded as target-distribution features.
Noising may obscure some artifacts at high noise, while lower-noise cases remain.
An image-only audit cannot tell us what the trained discriminator would do.
The first round therefore records artifacts and failure cases without claiming
discriminator rejection, training improvement, or lower generative FID.
