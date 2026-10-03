# Sources and attribution

FastGen starter: https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen
Pinned revision e3f5120000cd91037017d86f23993492195101a5. Protected interfaces and manifests are preserved.

SoFlow: Tianze Luo, Haotian Yuan, and Zhuang Liu, *SoFlow: Solution Flow Models for One-Step Generative Modeling*, ICLR 2026.
Paper: https://arxiv.org/abs/2512.15657
Repository: https://github.com/zlab-princeton/SoFlow
Pinned revision cc47661a0b0843f8f1b6cff3a17107a052d18b21.
Adapted models.py (scheduler and objectives), unet.py, and augment.py reside in soflow_fastgen/.
Changes: class conditioning, configurable compact U-Net, relative imports, removal of DiT/latent dependencies, FastGen sampling wrapper and new training/evaluation tooling.

The vendored U-Net and augmentation code retain NVIDIA's original copyright and Creative Commons Attribution-NonCommercial-ShareAlike 4.0 notices.
License: https://creativecommons.org/licenses/by-nc-sa/4.0/
EDM paper: Tero Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, NeurIPS 2022. https://arxiv.org/abs/2206.00364

Dataset: bhawks, Pokémon Generation One – 22k, Kaggle version 5, downloaded through the official FastGen utility.
https://www.kaggle.com/datasets/bhawks/pokemon-generation-one-22k

FID: the unchanged official evaluator uses clean-fid and its pretrained Inception feature network. This network is used only for evaluation.

The challenge explicitly allows cited open-source implementations. The general course page has broader wording against Internet-code copying; the Notion challenge notes record that tension. Preserve attribution and raise any remaining interpretation question through the team's normal course process; this implementation sends no course messages.

StyleGAN2 discriminator: NVIDIA's official StyleGAN2-ADA PyTorch implementation,
https://github.com/NVlabs/stylegan2-ada-pytorch, pinned revision
`d72cc7d041b42ec8e806021a205ed9349f87c6a4`. Unmodified selected files, license and
file hashes are in `vendor/stylegan2_ada/`. The runtime adapter selects native
reference operators and checks top-level import origins. No pretrained weights,
ADA trainer, StyleGAN2 generator or original GAN loss is used. NVIDIA Source Code
License is retained in `vendor/stylegan2_ada/LICENSE.txt`.

Adversarial objective: AYF Appendix F.2/F.3 (https://arxiv.org/abs/2506.14603),
and the higher-is-real convention in https://github.com/brownvc/R3GAN.
Our code implements RpGAN + R1/R2 around retained SoFlow SCM; it is an adaptation,
not an AYF reproduction. See `docs/adversarial/INTEGRATION.md`.
