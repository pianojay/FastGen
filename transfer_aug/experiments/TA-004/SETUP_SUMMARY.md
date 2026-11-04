# TA-004 CUT64 preparation

CPU setup passed under SETUP_CONTRACT_v2.md. No dataset training, GPU allocation, external weights, package installation, or sibling edits occurred. The final data/training contract remains required before profiling or training.

The standalone implementation extracts the relevant network classes directly from pinned CUT commit `b3ac297708dfb6f7589d04662277e53c0d579c27`; unrelated StyleGAN dependencies are excluded. Full upstream license notices are retained in `LICENSE.upstream`. CUT64 uses six residual blocks, width64, instance normalization without affine parameters/running statistics, no dropout, upstream antialiased sampling, 256-dimensional patch MLPs, LSGAN, and identity PatchNCE. No image pool, pretrained feature network, EMA, or FastCUT branch is included.

The complete initialized framework contains **9,061,060 parameters**: generator 7,837,699; discriminator 662,977; five patch MLPs 560,384. Feature indices 0/4/8/12/16 yield per-image shapes 3×70×70, 128×64×64, 256×32×32, 256×16×16, and 256×16×16. These correspond to reflection padding, the two encoder convolutions, and residual blocks 1 and 5. RGB64 output and the discriminator's 14×14 map are finite. The discriminator's local convolution/filter receptive field is 40px, as corrected in revision 2; instance normalization additionally couples spatial positions.

`verification.json` records explicit per-image PatchNCE reference agreement, invariance to keys in another image, stopped key gradients, and nonzero generator/MLP gradients from NCE alone. Synthetic batch2 optimizer steps populate all three Adam states. Reloading the atomic checkpoint reproduces the next sampler draws, Python/NumPy/Torch random values, losses, and complete model parameter hash exactly. Recovery retains update count and derives the scheduled learning rate from it.

The sampler implements uniform class selection, independent same-class D0/D1 draws, and independent horizontal flips. Dataset loading and the authorized GPU launch wrapper await the final training contract. CPU checks do not qualify batch16 GPU memory or throughput, nor establish image quality or convergence.

Reproduce: `bash transfer_aug/run_python.sh transfer_aug/experiments/TA-004/verify_setup.py`.
