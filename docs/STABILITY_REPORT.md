# AFM bounded stability-search report

**October 2–3, 2026 · Closed · Six candidates, no confirmation**

The campaign did **not** establish sustained, recognizable within-class diversity. All six fresh recipes failed the frozen acceptance criteria. Candidates 3, 5 and 6 met the persistent-collapse criterion; the other recipes retained more variation but produced inadequate structure. Training has stopped at the program boundary. No candidate qualified for independent confirmation, and no FID benchmarking was performed.

## Method

The teacher-free, 64×64, 151-class AFM architecture remained fixed: U-Net generator **39,664,771** parameters and projection discriminator **9,918,657** parameters. Each recipe started with fresh weights, optimizer/EMA state, RNG and data cursors. Training used the fixed **14,059-image optimization partition**; official validation was unused. Sampling and exported models use one generator call.

Before candidate 1, diagnostics were calibrated on real images, the historical collapsed checkpoint, repeated images, noise, recolorings, translations and small-set copies. Scheduled audits evaluated all 151 classes in raw G and EMA, with eight common and eight fresh inputs per class. Acceptance required recognizable structural variation in the same ≥114 classes across both models and two consecutive audits. Quantitative variation alone could not qualify a recipe. The [frozen protocol](../resources/stability_audit_protocol.json) and [incremental record](STABILITY_SEARCH.md) retain thresholds, seeds, hypotheses and exact recipes.

## Results

Quantitative counts below are terminal classes passing the variation/novelty gate, out of 151; they are **not** counts of recognizable diverse classes.

| Candidate | Main intervention / reference | G/D cycles | Raw / EMA | Job hours | Outcome |
|---|---|---:|---:|---:|---|
| C1 | Stronger OT retention, normalized adversarial target .5, slower optimizers / historical | 18,000 each | 99 / 138 | 4.341 | Noisy, malformed structure; horizon |
| C2 | D learning rate 2e-5→1e-4 / C1 | 18,000 each | 112 / 140 | 7.652 | Better critic separation; structure still inadequate |
| C3 | OT schedule .5→.1 reduced to .1→.02 / C2 | 14,000 each | 0 / 0 | 3.412 | Persistent collapse; early stop |
| C4 | OT floor .05 and finite-difference penalty 1 / C2 | 18,000 each | 19 / 26 | 4.367 | Deteriorating variation; horizon |
| C5 | Batch 64 and Adam beta2 .99 / C4 | 18,000 each | 1 / 4 | 7.970 | Persistent collapse at horizon |
| C6 | Penalty .25 and paired augmentation disabled / C5 | 18,000 each | 0 / 0 | 7.923 | Persistent collapse at horizon |

No recognizable diversity was established in the **40 classes visually reviewed at each terminal audit**, in either raw G or EMA. The remaining 111 classes were quantitatively audited but not visually scored; 40 failures already preclude the 114-class acceptance threshold. Gallery review used eight common and four displayed fresh samples, with all sixteen draws retained for quantitative analysis. No full visual acceptance assessment was claimed.

C6 declined from **115 raw / 151 EMA quantitative passes at 6k**, to **8/47 at 10k**, then **0/0 at both 14k and 18k**. Charizard repeatedly produced one distorted frontal composition; Ditto a purple blob; Diglett a brown stump; and Magikarp a distorted red silhouette. Raw and EMA both collapsed, ruling out EMA lag as the sole explanation. Nearest-image galleries repeatedly retrieved a few optimization examples, but pixel neighbors neither prove nor exclude memorization. [Final raw gallery](../runs/stability_campaign/candidate6/audits/018000/raw_019.png), [EMA gallery](../runs/stability_campaign/candidate6/audits/018000/ema_019.png).

## Qualification, resources and cost

Seven objective tests passed. Bounded qualification verified exact next-cycle recovery, conditioning and duplicate-device refusal. All **104,000 substantive G/D cycles** had finite logged values and contiguous counters. Qualification additionally executed **301 cycles per model**, including failed and replayed work. One initial probe refused recovery after a source edit; its estimated elapsed cost is retained. C5 received one permitted export-only recovery after an unexplained SIGTERM following its terminal audit: **zero training updates or audits replayed**.

On physical GPU 0, NVIDIA TITAN RTX (`GPU-e9d30df5-3b84-1a7e-c505-c6fff0a0ce7f`), sustained optimizer cycles averaged **0.813–0.828 s** at batch 32 and **1.538–1.543 s** at batch 64. Campaign peak allocation was **14.472 GB** including qualification, below 20 GB; peak reserved memory **15.915 GB**, sampled device memory **16.146 GB**, host RSS **2.966 GB** and PSS **2.951 GB**. No hardware-normalized timing estimate is made.

Total charged GPU-job time was **35.804 h**: **30.901 h optimizer work**, **0.536 h candidate audits**, **4.227 h other candidate overhead**, and **0.139 h calibration/qualification**, including failed/replayed work. C2's prolonged terminal review wait and C5's detection/recovery gap remain charged. Historical baseline cost (~2.454 h) is separate. Candidates consumed **4,480,000 real presentations per stream**, generated **8,960,000 training images**, and generated **138,920 audit plus 4,992 panel images**. Detailed counters and export hashes are in the [machine-readable summary](../runs/stability_campaign/summary.json).

Protected source, pinned AFM snapshot and partition hashes passed verification. All six generator-only exports contain finite tensors and load strictly on CPU; checkpoint hashes match their records. Each export also contains 192 non-parameter buffer elements. GPU 0 had no compute process at closure, and its host lease was available. Sibling jobs remained untouched.

## Interpretation and proposed next direction

The strongest working hypothesis is an optimization tradeoff: stronger pixel-space source transport preserves variation without coherent structure, while weaker retention permits class-template collapse. Faster critic optimization, stronger regularization, larger batches and reduced augmentation did not resolve it within these recipes. Bundled interventions and one seed per recipe prevent causal attribution; this negative result does not prove an architectural limit or that AFM cannot preserve modes.

A subsequent, separately authorized study should first measure post-normalization adversarial/OT gradient magnitudes **and alignment over time**, together with critic separation on held-out development examples. Then isolate a scheduled normalized-adversarial target or normalization ablation at a fixed transport schedule. Capacity expansion and FID ranking are premature before recognizable diversity is established. No further training was launched.
