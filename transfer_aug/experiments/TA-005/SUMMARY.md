# TA-005 MUNIT — completed pilot; preservation failed in review

MUNIT tests whether separate learned content/style codes support multiple appearances of the same input. Both reference-guided and Gaussian-style outputs were evaluated using the frozen A1-only appearance groups and same-class donors.

The scratch model completed **20,000 iterations** with **46.59M parameters**, **1.15 GPU-hours**, and **1.97GB peak training allocation**. GPU1's lease is released. We generated all **5,436 final images**, retained 320 milestone translations, and directly inspected all 22 required sheets. All 512 repeated translations reproduced quantized pixels exactly; protected inputs are unchanged.

**Diversity increased without reliable identity preservation.** Mean pairwise RGB RMS among four variants was **0.307 / 0.263** for reference outputs and **0.450 / 0.314** for Gaussian styles (D0→D1 / D1→D0, nominal RGB scale 0–1). All variants were distinct, but this is pixel variation, not evidence of useful semantic diversity.

Reference outputs often follow the donor's background tone and palette while retaining only coarse pose or silhouette. Fine facial, wing, shell and body markings are blurred, erased or replaced by streaked textures. Sampled styles frequently produce unrelated colors, hollow outlines or distorted blobs. Self-reconstruction is much closer to the source (mean RGB RMS **0.054 / 0.062**) than translation (**0.410–0.449**), so a functioning autoencoder does not establish safe cross-domain transfer.

The appearance axis shifts in the intended directions: reference **+1.17 / −3.87**, Gaussian **+3.13 / −3.94**. These statistical movements coexist with visible damage. Selected largest-change tails are dominated by background inversion; selected low-diversity tails still lose identity. No raw output exceeded [−1,1], so clipping does not explain these failures.

**Guidance:** do not mix these outputs into SiDA's real-image targets as validated augmentation. The next research question is whether transfer can be constrained to an explicitly permitted factor while protecting foreground structure and markings. This run does not establish convergence or reject MUNIT under other settings.

Evidence: [gallery](gallery.html), [measurements](eda/aggregate.json), [direct review](visual_review.json), [verification](generation_verification.json). Diagnostics participated in domain discovery; the visual review is small and unblinded. No A2/B, pretrained metric, classifier, FID or SiDA experiment was used.
