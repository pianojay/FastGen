# SoFlow endpoint adversarial post-training: mathematical foundation

Independent audit, 2026-10-03. This is a proposed implementation contract, not an implemented trainer or a launched experiment. No active project files, checkpoints, processes or GPU allocations were changed. Only small CPU algebra/autograd checks were run.

Reconciled with all 14 visible messages of the exploratory [SoFlow applicability discussion](https://chatgpt.com/share/6ac05371-cd0c-83e8-bb10-639a4bc90923), including its final retraction of mandatory Flow Matching. Section 13 records adopted insights, disagreements, and changes. That conversation is an input to this assessment, not an authoritative source or an instruction to add its proposed components.

## Conclusion

A coherent experiment is possible: initialize the generator from the completed U-Net160 EMA checkpoint, retain the exact existing data-driven SoFlow consistency gradient at its terminal gap, and add a conditional relativistic adversarial objective on the deployed one-call endpoint. The critic is freshly initialized. There is no external teacher, fake-score network, distillation from four-call samples, or new generator architecture.

The important risks are not merely GAN hyperparameters. They are inadvertently restarting the consistency curriculum; combining gradients using a misleading loss scalar; changing the augmentation-conditioned distribution; using a different generator function during GAN training than during inference; and treating EMA or optimizer history as interchangeable with model weights.

This tests whether direct endpoint distribution supervision improves the useful checkpoint. It does not establish that consistency guarantees coverage, or that an adversarially modified map retains the original transport trajectory.

After the reconciliation, the selected architecture remains a **class-conditioned, unguided U160 generator plus a separate scratch projection critic with a shared trunk**. No explicit Flow Matching loss or teacher feature network is added. The main addition is a small deterministic raw-map composition diagnostic, distinct from the deployed re-noising sampler. Class conditioning is retained deliberately; few examples per class alone do not demonstrate that an unconditional critic is preferable.

## 1. What AYF actually supplies

AYF Appendix F.2 uses a StyleGAN2 discriminator and the RpGAN objective, with adversarial examples from the full noise-to-data map. Its non-adversarial distillation term remains active. Algorithm 2 fixes the mature tangent coefficient at 0.99. Reported tuning is roughly 3,000 updates at batch 1,024 on 32 A100s, about four hours; this is not a four-hour TITAN recipe.

The source contains a real ambiguity: prose weights the adversarial term, while Algorithm 2 weights the EMD term. The adaptive-weight function is unspecified. The apparent sign reversal is not necessarily an error: Algorithm 2 is consistent with a critic whose larger output means “fake.” We instead declare larger=real throughout. These observations apply to the published PDF as well as arXiv. Official code remains marked TBA.

Sources: [published AYF](https://proceedings.neurips.cc/paper_files/paper/2025/file/d79ac139911df25d27f14bbf008deaee-Paper-Conference.pdf), Appendix F.2–F.3; [project](https://research.nvidia.com/labs/toronto-ai/AlignYourFlow/).

Our method should be called **SoFlow endpoint adversarial post-training inspired by AYF**, not an AYF reproduction. In particular, our retained objective learns from optimization images and samplewise interpolation velocities; it is not the AYF external-teacher EMD objective.

RpGAN names the adversarial objective, not a required network architecture. The [official R3GAN trainer](https://github.com/brownvc/R3GAN/blob/main/R3GAN/Trainer.py) provides a clear primary reference for the higher=real convention, both input-gradient penalties, and same-condition real/fake evaluation. Its code also makes clear that the regularizer differentiates through preprocessing to original input leaves. None of its convergence claims automatically applies to our finite-data, conditioned, combined-objective experiment.

## 2. The actual SoFlow gradient

Let n=3×64×64=12,288. Distinguish the neural-network output N_theta from the solution operator M_theta:

    M_theta(x,t,s,c,a) = x + (s-t) N_theta(x,t,s,c,a).

The exact boundary M_theta(x,t,t,c,a)=x follows from this parameterization. The one-call raw generator is

    g_theta(z,c) = M_theta(z,1,0,c,0) = z - N_theta(z,1,0,c,0),
    z ~ N(0,I).

The existing deployed sampler applies C(u)=clamp(u,-1,1) only after the final map. For one call its visible generator is y_theta=C(g_theta). It does not use the instantaneous diagonal velocity, and it does not use an intermediate re-noising draw.

For the retained consistency branch, draw optimization image x and its existing conditioning augmentation a; let x_a=T_a(x). Draw z independently and form

    x_t=(1-t)x_a+t z,
    v=z-x_a,
    x_l=x_t+(l-t)v=(1-l)x_a+l z.

The two online-network evaluations share their dropout realization. The second is detached; it is not an EMA teacher. With Delta=t-l>0 and d=t-s>0,

    q = stopgrad((M_theta(x_t,t,s,c,a)-M_theta(x_l,l,s,c,a))/Delta),
    w = stopgrad((mean_pixels(q^2)+0.001)^(-0.75)),
    L_C = mean_examples[ -mean_pixels(N_theta(x_t,t,s,c,a) * w*q) ].

This is the current no-guidance branch. Its scalar value is a surrogate and need not be nonnegative. The actual gradient is

    grad_theta L_C = - E[ J_N^T(w*q)/n ]
                   =   E[ J_M^T(w*q)/(n*d) ].

Holding q's target branch, w and sampled times fixed, it equals the gradient of

    E[ w/(2*d*Delta) * mean_pixels((M_theta(x_t,t,s)-stopgrad(M_theta(x_l,l,s)))^2) ].

Thus neither “consistency MSE” nor the signed scalar surrogate is the correct quantity for balancing GAN gradients. In particular, adding a constant to the surrogate changes its value without changing its gradient; any scalar-loss ratio would respond incorrectly.

A float64 CPU check verified the equivalence above to maximum absolute gradient error 3.47e-18. This is an algebra check, not validation of an integrated trainer.

Local source: [models.py:136](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/models.py:136), especially 163–177; [training.py:22](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/training.py:22); [sampling.py:51](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/sampling.py:51). Audit checkout HEAD was b3ba0de8bdb178b81a91ab5dd8e4f799ece5b5a8; the future project must pin its chosen source and checkpoint hashes independently.

## 3. Do not restart the model's maturity schedule

The completed S2 recipe uses

    r(k) = 1.0 * (0.0003125/1.0)^(k/40000),
    t = max(sigmoid(N(-0.9,1.6^2)), 1e-4),
    s = min(sigmoid(N(-4,1.6^2)), t-1e-4),
    l = min(t-r(k)*(t-s), t-1e-4).

For the proposed post-training, fix r=0.0003125 and preserve the clamp, so

    Delta = max(0.0003125*(t-s), 1e-4),   l=t-Delta.

This is not the same as setting Delta=0.0003125 or merely l=t-r(t-s) without the floor. In 100,000 CPU draws using the configured distributions, 58.48% hit the 1e-4 floor; median gap was 1e-4. These are synthetic sampling diagnostics, not model measurements.

At a naive restart, r=1 gives l=s. The exact boundary then gives M(x_l,s,s)=x_l; q=v-N_theta. Consequently the consistency gradient reduces to a residual-weighted regression of N_theta toward the samplewise velocity v. That is a real change from the mature local-consistency regime. It could erase endpoint behavior while being misdescribed as harmless warmup. The sampled median restarted gap was 0.2454, more than 2,000 times the terminal median.

Keep the original t/s distribution, p, epsilon, and augmentation for this first comparison. Changing endpoint oversampling or tangent scheduling is a separate intervention. The post-training optimizer clock starts at zero; the consistency maturity parameter does not.

Local source: [models.py:192](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/models.py:192), [training.py:30](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/training.py:30), [S2 configuration](/home/user/jw_soflow/FastGen-SoFlow/configs/research_s2_train.json).

## 4. Conditional relativistic objectives, with all signs explicit

Use a scalar critic D_psi(y,c), where larger means more real. Draw a real image x conditional on c, Gaussian z, and a discriminator transform A_xi. Pair the generated image with a real image of the same class:

    y_r=x,
    y_f=C(g_theta(z,c)),
    d_r=D_psi(A_xi(y_r),c),
    d_f=D_psi(A_xi(y_f),c).

Use the same transform parameters within each real/fake pair. Independent transforms with the same marginal kernel are another possible formulation, but change the joint pair distribution and generally the expected nonlinear softplus objective, not merely its variance. The first implementation fixes shared per-pair transforms as part of the declared objective. There is no pixel correspondence between real and fake, and no regression toward the paired real example.

Same-class pairing defines a conditional game E_c E_{x|c,z}. It removes arbitrary class-specific critic offsets from each difference. Pairing different classes can define a joint-distribution game, but is a different objective with cross-class comparisons. The real minibatch's labels also set the fake labels, so the priors match. Retain the existing empirical image/class sampling for the control and treatment; do not silently introduce balanced-class reweighting only in the treatment.

Minimize, on discriminator updates,

    L_D,adv = mean softplus(d_f-d_r).

The derivatives with respect to real/fake logits are -sigmoid(d_f-d_r) and +sigmoid(d_f-d_r), respectively. Gradient descent raises the real logit and lowers the fake logit. Both branches must backpropagate into D. Generated images are detached from G.

On generator updates, freeze discriminator parameters and minimize

    L_A = mean softplus(d_r-d_f).

Here dL_A/dd_f=-sigmoid(d_r-d_f), so gradient descent moves fake images toward higher critic scores. The real score is still required in the difference. It can be detached computationally because neither the real image nor its score depends on theta. Removing its value altogether changes RpGAN into another loss. Do not detach the fake branch or put that D forward under no_grad.

A scalar CPU check at equal logits obtained D derivatives (-0.5,+0.5) and generator fake-logit derivative -0.5.

## 5. Exact R1/R2 and reduction conventions

Define the penalties with respect to critic-input images y BEFORE discriminator augmentation:

    R1 = E[ || grad_{y_r} D_psi(A_xi(y_r),c) ||_2^2 ],
    R2 = E[ || grad_{y_f} D_psi(A_xi(y_f),c) ||_2^2 ],
    L_D = L_D,adv + gamma/2 * (R1+R2).

In the D update, first generate/clamp/detach y_f, then create a fresh requires_grad leaf from that visible image. R2 regularizes D's visible-input gradient; it must not backpropagate through the generator or the output clamp. R1 similarly uses a detached real-image leaf.

The squared norm SUMS all 12,288 coordinates per image, then averages images. The adversarial term averages one scalar per pair. Compute input gradients from logits.sum(), not logits.mean(); otherwise a hidden factor 1/B^2 enters the penalty. For independent per-image D, autograd of the sum returns the desired per-image gradients. This is one reason to omit batch-statistic layers.

Use both penalties every D update initially. Lazy regularization would require declared cadence and scaling and changes optimizer dynamics; it is unnecessary for the first correctness contract. Use FP32 input gradients and penalty arithmetic. Source convention gamma/2 differs from a source that writes beta*(R1+R2): gamma=0.2 corresponds to beta=0.1 under these exact norm reductions. It does not establish that this strength is empirically suitable for our critic or image scale.

Finite differences are not exact R1/R2. For u~N(0,I), the small-noise expression

    E_u[(D(A(y+sigma*u),c)-D(A(y),c))^2 / sigma^2]

approximates the gradient norm in original y coordinates. Its two evaluations MUST share augmentation randomness. Without that coupling, augmentation variance divided by sigma^2 contaminates the penalty. Unit-normalized u changes the expected scale by roughly 1/n and needs an n factor. Omitting sigma^-2 also changes strength. The proposed first experiment uses exact autograd penalties, not this approximation.

## 6. Two augmentation distributions, not one switch

The existing consistency augmentation draws x-flip plus intermittent scaling, anisotropy and translation, and sends six transform coordinates to BOTH SoFlow branches. With xflip=1e8 and p=0.12, the x-flip selection is always enabled and the actual flip is Bernoulli(0.5). Scale/aniso/translation are individually selected with probability 0.12. Setting p=0.12 does not mean “12% horizontal flips.” Inference uses the zero augmentation vector.

Retain that pipeline unchanged for L_C. For the GAN branch, generate with a=0 and compare to unaugmented real images before a separate symmetric discriminator transform. Do not compare metadata-augmented real images against zero-metadata fake images. Do not feed discriminator transform metadata into G: these transforms regularize the observation made by D, rather than define which conditional image distribution G should produce.

Recommended first D transform: horizontal flip with probability 0.25, identical for real and fake within a pair; otherwise identity. Do not initially copy scale/anisotropy/interpolation into this branch. This is deliberately minimal, not a claim that richer D augmentation is incompatible.

There are two reasons for this choice. First, horizontal flip is an isometry, so R1/R2 measured before or after it have equal gradient norms. Second, the single-image marginal augmentation operator K_p=(1-p)I+pR has eigenvalues 1 and 1-2p for reflection-symmetric/antisymmetric components. At p=0.5 the latter vanishes, so these observation marginals lose orientation-asymmetric information. At p=0.25 this particular marginal degeneracy is absent. This is a reason not to transfer the consistency branch's 50% conditional flip blindly to an unconditioned D observation. It does not by itself prove identifiability of the RpGAN game with shared pair transforms, whose joint pair distribution also matters.

The current geometric AugmentPipe uses torch grid_sample. A CPU check in the existing fastgen_soflow environment (torch 2.7.1+cu126) confirmed that exact R1 backpropagation through this original-image pipeline throws “derivative for aten::grid_sampler_2d_backward is not implemented.” Existing SoFlow training does not encounter this because augmentation is applied to real images before the learnable network and needs no such second derivative. New GPU/custom-op qualification would be required to change this conclusion for another implementation.

An alternative is to apply geometric augmentation first, detach, and measure R1/R2 in augmented coordinates. That bypasses the unsupported derivative but changes the regularizer's metric: it omits the transform Jacobian. It must be declared as an adaptation, not quietly treated as the same objective. Another alternative is a custom second-order-compatible augmentation implementation. Neither is needed initially.

Source: [training.py:18](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/training.py:18), [augment.py:187](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/augment.py:187), [augment.py:285](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/augment.py:285).

## 7. Visible output range and dropout are part of the objective

Recommended contract: D sees C(g)=clamp(g,-1,1), matching current deployed one-call samples. The consistency branch continues to operate on the raw M_theta. Never move clamp into solution_operator: that would change the exact flow-map boundary for noisy inputs and the consistency algebra.

Clamping gives zero GAN gradient for raw pixels outside (-1,1), apart from boundary conventions. Shared parameters and L_C can still move those pixels, but the adversarial branch provides no direct recovery gradient there. This is a real qualification concern, not an implementation mistake. Measure the one-call raw fraction outside range, overshoot magnitude, per-class saturation, and the share of GAN image-gradient energy masked by clamp. Do not assume earlier four-call overshoot percentages describe one-call generation. Flag numerically zero or effectively lost adversarial gradients across classes as a qualification failure; do not silently repair them with straight-through clipping or tanh.

Using raw g in D is another coherent objective but lets D exploit a range discrepancy hidden by deployed clipping. A tanh or soft clamp changes the existing generator function, while a straight-through clamp supplies a biased gradient unrelated to the stated differentiable objective. Neither is a silent drop-in fix. If clamping leaves unusable gradients, report that obstacle and revise the declared experiment before a substantive launch.

Consistency calls remain in training mode with the original shared dropout realization. GAN endpoint calls run the generator in eval mode, with autograd ENABLED during G updates; this targets the deterministic inference function. D updates generate in eval+no_grad and detach. Switching eval mode does not itself freeze parameters. Restore model mode before consistency calls. The selected projection D has no dropout or batch-dependent state, so freezing its parameters suffices during G updates.

Source: [unet.py:172](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/unet.py:172), [models.py:163](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/models.py:163), [sampling.py:65](/home/user/jw_soflow/FastGen-SoFlow/soflow_fastgen/sampling.py:65).

## 8. One explicit gradient-balancing contract

Let g_C and g_A be the COMPLETE effective-batch parameter gradients of L_C and L_A, respectively. Compute them separately before the optimizer update. Choose

    rho(k) = 0.1 * min((k+1)/256, 1),
    lambda(k) = stopgrad(rho(k) * min(10000, ||g_C||_2/(||g_A||_2+1e-12))),
    g_total = g_C + lambda(k)*g_A.

Set lambda=0 if the adversarial gradient is exactly zero; record that diagnostic. Do not invent a fallback that creates a huge update from numerical noise. The coefficient is outside all existing per-example SoFlow weighting. It therefore actually changes the parameter update rather than being canceled by another normalization.

This is a chosen **full-generator gradient balancing adaptation**, not a claim about AYF's unspecified adaptive_weight. The [VQGAN implementation](https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py) supplies precedent for a detached gradient ratio on the last layer; our choice uses the full parameter vector so the initial interpretation is unambiguous.

Ignoring the cap and epsilon, the adversarial gradient has norm rho times the consistency gradient. By Cauchy–Schwarz, g_C dot g_total >= (1-rho)||g_C||^2 for rho<1. This bounds a raw-gradient relationship only. It is not a convergence, preservation, FID or optimizer-step guarantee: RAdam preconditioning/momentum and moving targets change the actual trajectory. Log component norms, lambda, cosine similarity, total gradient norm and actual parameter-update norm. Parameter-wise preconditioning can amplify different components differently.

Microbatching must compute g_C=sum_m (B_m/B) g_C,m and g_A=sum_m (B_m/B) g_A,m BEFORE taking norms. Averaging ratios or separately normalizing each microbatch defines a different objective and changes with microbatch size. Two FP32 gradient buffers cost about 496 MB for 61.93M parameters; qualify memory, but this is a simpler correctness contract than recomputation with unstable per-microbatch weights. Accumulate each branch sequentially so activation graphs need not coexist. The ratio need not itself have an autograd graph.

The no-adversarial control simply uses g_total=g_C and performs no D work. Initial settings 0.1, 256 and 10000 are transparent engineering defaults for qualification, not theoretically identified optima. If qualification changes them, freeze the revised values before the paired comparison rather than tuning silently during it.

## 9. Recommended initial component and phase specification

This table selects one internally consistent design. The numerical post-training defaults are unmeasured recommendations; they must be frozen after correctness/throughput qualification and before the bounded paired experiment.

| Component | Selected design |
|---|---|
| G initialization | Copy the selected S2 U160 **EMA** weights into both new online G and new G-EMA. This reproduces the useful incumbent initially. Record source checkpoint, state key and hash. |
| G optimizer | Fresh RAdam, betas=(0.9,0.999), weight decay 0; maximum lr=2e-5, linear warmup over 64 G updates. Same in control. Do not attach raw-model optimizer moments to EMA weights. |
| Consistency maturity | Fixed terminal ratio 0.0003125 with the exact 1e-4 floor. Preserve original time distributions, residual weighting and augmentation. |
| G EMA | Initialize from the same source EMA; fixed half-life 32,768 G image presentations, beta=2^(-B/32768). At B=128, beta=0.9972960561. No scratch-EMA ramp and no EMA updates during D-only warmup. Same control policy; report raw and EMA. |
| D architecture | Reuse the existing **architecture only** of the AFM class-conditioned projection critic, freshly initialized: down blocks with widths 64,128,256,512, a 512-channel final block, global spatial mean, scalar head plus class projection. No batch statistics, dropout, pretrained features, or AFM objective. The existing instance has 9,918,657 parameters; remeasure after integration. |
| D source | Local AFM models.py at inspected commit 262203e6a59d47caeba0235eb5979ba6c1145822. Copy under its applicable source/license/provenance, not by importing a mutable sibling tree. This is an architecture adaptation from AYF's StyleGAN2. |
| D optimizer | Fresh Adam, lr=1e-4, betas=(0,0.99), weight decay 0. This faster scratch-critic rate is a hypothesis; verify stable learning before G exposure. |
| D warmup | 256 D-only updates at fixed G and fixed EMA; no G optimizer, curriculum or EMA clock advancement. Count its real/fake exposure and cost separately. Fresh z/data each update, no persistent generated-image cache. |
| Update ratio after warmup | One D update then one G update. Count both explicitly. No DMD-style five-fake-score rule: this is a different network/objective. |
| GAN generation | Full endpoint M(z,1,0,c,a=0), generator eval mode; visible clamp before D; no intermediate trajectory, no guidance. |
| D augmentation | Per-pair shared Bernoulli(0.25) horizontal flip, otherwise identity. Independent from consistency augmentation. |
| D loss | RpGAN higher=real plus exact R1+R2 every D update. Initial gamma=0.2 under gamma/2 convention; norm=sum pixels, mean examples. |
| G combination | Full-effective-batch gradient ratio above; rho ramps 0→0.1 over 256 G updates. The mature consistency phase remains fixed throughout. |
| Batch | Effective 128 for both G and D initially; choose fitting microbatches after profiling, preserving the effective means. |
| Precision | Retain FP32 consistency arithmetic and forward evaluations initially. Delta~1e-4 makes naive FP16 subtraction hazardous. Keep input-gradient penalties FP32. AMP is a separate qualification, not a free optimization. |
| Randomness | Independent named streams for consistency data, augmentation, times/noise/dropout; GAN z; D augmentation; D initialization. Extra GAN draws must not shift matched control consistency draws. Save every stream for recovery. |

Source for the selected D: [AFM models.py:24](/home/user/jw_soflow/FastGen-AFM/afm_fastgen/models.py:24). Reusing architecture does not imply reusing its prior collapsed checkpoint; the old failure did not isolate the architecture as the cause.

Why new EMA rather than the inherited runtime rule? The current training rule is min(0.9999,(1+step)/(10+step)). At S2's final update, its pre-increment step was 39,999, giving beta=40000/40009, approximately 0.999775, or a half-life of about 3,081 updates; S2 had not reached the 0.9999 cap. Resetting step restarts a scratch EMA policy. Using the cap alone would instead give a roughly 6,931-update half-life, which could mask a short post-training response. The proposed explicit shorter half-life is a new evaluation policy, shared by both arms and disclosed. Its optimality is unestablished.

Why EMA rather than raw checkpoint? The reported incumbent samples/FID use EMA. Starting there gives the intended asset as step zero. Fresh optimizer state is then the coherent choice. Starting from raw weights with old moments is a valid different continuation experiment but would not reproduce the qualified EMA incumbent at initialization. These choices should not be mixed.

## 10. Pseudocode and separation of clocks

    load identical source EMA into G and G_ema for both arms
    initialize fresh G optimizer; define fixed mature gap policy
    independently initialize D and its optimizer in treatment only
    evaluate and archive step-zero raw/EMA equivalence

    treatment only:
      repeat exactly H_D=256 D warmup updates:
        freeze G; do not change G clocks or EMA
        draw real/class and GAN noise; generate eval-mode visible fake
        compute paired RpGAN + exact R1/R2; step D

    for k in 0 .. H_G-1:                # H_G fixed before launch
      treatment only:
        compute one fresh D update, with generated fake detached
      compute complete-batch consistency gradient g_C
        training-mode G; paired dropout; terminal gap; original augmentation
      treatment only:
        compute complete-batch endpoint GAN gradient g_A
        eval-mode G WITH gradients; zero augmentation metadata
        D parameters frozen, real score detached, fake path differentiated
        compute detached lambda from aggregated gradients
      set G.grad = g_C + lambda*g_A      # lambda=0 in control
      step G optimizer using the declared LR clock
      update G_ema using G-image exposure only
      increment G counter; save/evaluate only at declared boundaries

D's warmup counter, D optimizer-step counter, G counter, total exposures, and elapsed-budget counter must be separate. Recovery restores them exactly, including all optimizers, G-EMA, paired-data position, random streams and ramp phase. A recovered job is the same recipe, not another candidate.

The existing get_consistency_loss saves/restores the default CUDA RNG to match dropout in its two calls. A new trainer must provide explicit RNG isolation around that branch rather than assume extra GAN/D calls leave the next control draw unchanged. No elaborate orchestration framework is needed, but this semantic contract is necessary.

## 11. Qualification and experiment decision

Cheap correctness checks before GPU training:

1. Scalar logit tests prove the declared D/G signs; real D branch participates in D updates and no real-data gradient is accidentally expected for G.
2. Verify consistency surrogate versus detached-target gradient equivalence, including sign and all pixel/batch factors. A constant scalar offset must not affect gradient balancing.
3. Verify zero adversarial coefficient yields the exact control update, with matched RNG; D-only updates leave G and EMA bitwise unchanged.
4. Verify R2 changes D parameters but never G; penalties use logits.sum and pixel-summed norms. On a linear critic, compare exact R1/R2 against an analytic value. Test the chosen augmentation's second-order path.
5. Compare full batch with accumulated microbatches using identical sampled draws, including paired dropout masks; compare accumulated gradient vectors before norms, not just total scalar loss.
6. Verify endpoint eval-mode output before/after cloning equals the source inference function; verify no-grad is absent on the generator training path; verify clipping only occurs in the visible GAN view.
7. Test the actual terminal gap, FP32 quotient, new EMA decay and LR/adv clocks around boundaries; no inherited scratch schedule should be reachable accidentally.

GPU qualification should then measure complete update throughput and peak simultaneous memory, independent loading, finite component gradients, saturation/gradient retention, critic train-versus-held-out-real discrimination, raw/EMA drift and the fake/real logit gap. “Held out” here means a declared optimization-only diagnostic holdout, not official validation and not a new training source. Discriminator accuracy alone is not a warmup success condition: a saturated range detector is easy to train and can be useless to G.

If the declared 256-step warmup and initial generator updates produce invalid or unusable gradients, stop qualification and revise the manifest; do not spend an arbitrary long horizon hoping it repairs itself. Qualification updates must be discarded or consistently incorporated into the frozen recipe for both accounting and initialization; do not give the treatment unreported extra generator training.

Before substantive launch, profile full control and treatment updates and declare one finite G horizon, audit checkpoints, evaluation allocation and watchdog. No 12-hour inherited cap, no automatic 3,000-update AYF transplant, no automatic candidate sweep. A worthwhile horizon must include a meaningful interval after LR/adversarial ramp and enough EMA half-lives to observe a response, with the corresponding time explicitly costed.

Run exactly one matched control and one treatment. Evaluate one-call FID on the same development references and random draws, with separate declared confirmation draws; preserve prior use of the development set in reporting. Four-call FID is a secondary preservation diagnostic, not a promise. Track fixed and fresh class panels, broad recognizability, structural diversity, duplicates and nearest optimization examples; do not accept pixel jitter as diversity. Report both raw and EMA trajectories and the untouched initial checkpoint.

Interpretation: this comparison measures the addition of the specified warm-start adversarial machinery at matched generator exposure. It is not matched compute: the treatment costs extra D updates and G forwards. Report incremental time plus inherited pretraining separately. With one training seed, apparent superiority is preliminary. A new independent seed is a later decision, not an automatic expansion.

## 12. What remains empirical

The architecture family and probability semantics can be made exact now. Critic capacity adequacy, gamma, critic learning speed, adversarial strength, EMA half-life, useful horizon and eventual FID cannot be derived from AYF or from warm-starting alone. The recommended defaults expose those choices and provide measurements that would falsify their suitability.

The existing consistency branch is a data-driven stabilizing hypothesis. Its detached second evaluation tracks the current model, so it is not a frozen teacher or a fixed anchor to U160. However, the transition x_l=x_t+(l-t)(z-x_a) continues to inject real-data information on every update; calling the entire objective purely self-referential would be wrong. Adversarial changes can co-adapt both network evaluations despite that external information. If destructive drift persists despite well-scaled gradients, a fixed-checkpoint function anchor would be a separate future intervention with an extra term, cost and preservation target. Do not add one silently to the first paired experiment.

A valid initial result can be modest: a stable, repeatable improvement over the no-adversarial continuation, with credible structural diversity. One-call FID below 30 is an ambition, not a theoretical consequence or a sensible binary requirement for a first bounded probe.

## 13. Reconciliation with “SoFlow 적용성 검토”

### What was reviewed and independently checked

All 14 chronological messages were read, including the final answer that retracts earlier insistence on an explicit FM term and changes its discriminator recommendation. The reference is the [public shared conversation](https://chatgpt.com/share/6ac05371-cd0c-83e8-bb10-639a4bc90923); the full local transcript is [/tmp/soflow_shared_conversation.md](/tmp/soflow_shared_conversation.md). Its citations and numerical claims were not accepted automatically.

The [SoFlow paper](https://arxiv.org/html/2512.15657v2), especially Section 4.1 and Appendix A, and local sources independently confirm the relevant distinction: the paper's sufficient differential condition uses the marginal velocity and the identity boundary; its CIFAR recipe has zero explicit FM data. The local original [cifar.yaml:9](/home/user/jw_soflow/SoFlow/cifar.yaml:9) likewise sets velocity_loss_ratio=0. Our Pokémon trainer calls only the consistency branch. No new training or inference experiments were needed for this reconciliation.

### Adopted insights

**Keep SCM without introducing FM.** The final correction is right for this checkpoint. SCM already contains the data-derived displacement z-x_a; its second network evaluation moves with theta, but its information source is not exclusively the model. FM would be a new diagonal-velocity objective, adding cost and a second simultaneous intervention. It is optional if later evidence specifically identifies a tangent-field problem. A weak diagonal teacher score by itself does not establish that FM will improve the one-call endpoint.

**The endpoint signal is complementary, with a narrower causal claim.** As r decreases, l approaches t, not s. This changes the separation between consistency evaluations; the prediction still spans t→s. Moreover, our logit-normal t/s sampler has no intended point mass at exactly (1,0). GAN training directly supervises this deployed endpoint. That is a good hypothesis for addressing the one-call gap. It does not establish that the curriculum caused the gap or that training literally forgot the full map.

**Retain the mature phase, and distinguish distributional from paired supervision.** Adversarial real/fake pairs are not corresponding image targets. The data-derived interpolation state x_l is also not generally the exact marginal-ODE state associated with a particular x_t. Our loss uses the existing local construction; we must not extend it into an unjustified long-step pixel reconstruction target.

**Add finite-interval composition as a diagnostic.** It can reveal changes missed by a local consistency metric, but it is not a preservation guarantee and need not be a training loss. This is the principal addition to the previous specification.

### Conditional versus unconditional critic: deliberate disagreement

The shared discussion ultimately favors D(x) because each class has few examples and broad styles. This is a reasonable hypothesis about estimation difficulty, but the argument treats conditional discrimination too much like 151 separate critics.

Our chosen architecture has one shared image trunk and one shared scalar realism head:

    D(x,c) = b + w^T h(x) + e_c^T h(x).

Every example trains h and w. Only the embedding table is explicitly class-specific: 151×512=77,312 parameters, about 0.78% of the 9,918,657-parameter critic. Setting all e_c=0 recovers its unconditional subfamily. Conditional inputs can still encourage memorization through the shared trunk, so the small embedding does not guarantee generalization. But few examples per class does not imply an automatic 151-fold fragmentation of all discriminator learning.

Diverse rendering styles do not make class identity irrelevant. A projection critic can share style and structural features while checking whether the generated object fits the requested category. Removing c discards that check; continued conditional SCM may preserve semantics in practice, but does not guarantee it.

There is also a precise distinction between an unconditional critic and a conditional pair sampler. If we retain same-class pairing but remove labels from D, the objective is

    E_c E_{x~p_data(.|c),z}[softplus(D(G(z,c))-D(x))].

This is not the same joint pair law as independently drawing real and fake labels from their marginal priors. Nevertheless, same-class pairing alone cannot replace conditioning the critic. If real and generated IMAGE marginals match, for every label-blind D their expected logit difference is zero; Jensen gives

    E softplus(D(fake)-D(real)) >= softplus(0)=log(2).

A constant critic attains that bound, with zero R1/R2, even if the conditional distributions differ. The argument still holds with matched augmentation marginals. Class swaps under equal priors are a simple example. Thus label-blind discrimination has a real semantic blind spot, regardless of the conditional pair construction.

**Decision:** keep the shared projection critic for the first paired experiment; measure broad class fidelity and per-class overfit. An unconditional critic is a later alternative if class-specific critic fitting is the observed bottleneck. It is not proved worse, but neither the conversation nor available local results establish that it is the better first choice. Do not introduce a separate frozen classifier loss merely to compensate for removing class conditioning; that would add another model and objective without evidence that the simpler projection option fails.

“Guided discriminator” is not the useful distinction here. The generator is **class-conditioned but unguided**: the local trainer passes cfg_scale=1 and uses label_dropout=0. The critic is **class-conditioned**, with no CFG operation. Guidance strength and discriminator conditioning are different axes. The shared discussion's guided-flow-versus-real-data tension is valid as a general warning, but does not apply to the current unguided initialization. Do not add CFG to this experiment.

### Generator and critic feature architecture

| Candidate | Decision and reason |
|---|---|
| Existing U160 map as generator | Keep unchanged, including both time embeddings and augmentation metadata. The endpoint uses t=1,s=0,a=0; consistency continues exercising other times. No DiT conversion or new generator head. |
| Scratch shared projection critic | Selected. It already contains a common realism head plus inexpensive class modulation, supports exact per-image R penalties, and has no batch-statistic coupling. Use its architecture/code only, not AFM weights. |
| Unconditional version of that critic | Plausible later ablation for demonstrated class-specific overfit. Removes class-sensitive adversarial supervision; requires an explicit pair law. |
| Literal StyleGAN2 discriminator | Applicable, but introduces another architecture/implementation choice and potentially minibatch statistics/custom operators. It is not required by RpGAN or by functional compatibility with SoFlow. |
| Critic initialized from a copied SoFlow encoder | Challenge-trained features are potentially reusable, but not an established improvement. Generator features depend on noisy input and time/augmentation conditioning; their clean-image critic interface must be defined. Copying compatible shapes is not enough to establish meaningful features. |
| Frozen SoFlow feature extractor plus small critic head | Lower trainable capacity, but potentially blind to artifacts outside its representation; a fixed learned feature bottleneck may constrain what distributional differences can be detected. It still costs forward/backward computation through the extractor for generator training. Not included initially. |
| Shared trainable G/D encoder | Reject for the first experiment: D updates would change G representation directly, invalidating the intended alternating game and the no-adv control. If considered later it needs a different objective and update contract. |
| External pretrained classifier/perceptual/VAE features | Excluded by the challenge constraints and unnecessary for this pixel-space experiment. |

The shared conversation's VAE-decoding-gap concern is valid for latent generators but inapplicable here: both real and fake images are 64×64 RGB pixels. Conversely, our real implementation has an output-clipping and dropout distribution issue that the conversation does not resolve; Sections 6–7 remain part of the contract.

### The composition diagnostic, correctly scoped

At declared checkpoints, use a small frozen, class-stratified noise panel and u in {0.25,0.5,0.75}. Put the evaluated model in eval mode, set augmentation metadata to zero, and evaluate raw maps:

    R_comp(u) = E_{z,c} mean_pixels[
        (M(z,1,0,c,0) - M(M(z,1,u,c,0),u,0,c,0))^2
    ].

No clipping, re-noising, dropout, or gradient update occurs in this diagnostic. Report the initial value and changes for raw and EMA models. Also record outlier magnitudes so a few unstable raw outputs do not obscure typical behavior.

This is NOT the current two-call sampler. That sampler predicts an endpoint at each call and then adds fresh noise between calls. Its quality must be evaluated separately using the unchanged deployed implementation. Nor does the existing four-call FID establish that deterministic composed maps produce better targets for a one-call regression.

A rising residual shows reduced agreement between the measured direct and composed maps. It does not prove mode collapse, infer which branch is more correct, or invalidate useful one-call specialization. A low residual likewise does not establish transport correctness: the identity map M(x,t,s)=x satisfies both the identity boundary and exact composition while failing to turn Gaussian noise into Pokémon. Keep FID, recognizable diversity, class conditioning, and nearest examples as separate evidence.

Semigroup losses are differentiable, as the discussion suggests, but an online/EMA composed target still introduces model-generated supervision and does not establish that the target is better. A stop-gradient target controls gradient routing; it does not prove that only the direct branch is wrong. The first experiment therefore measures R_comp without adding L_SG. A fixed-checkpoint or EMA composition anchor is optional only after a specific diagnosed failure and a separately defined goal.

### Claims retained only with qualifications

- Exact compatibility belongs to an idealized setting: an unguided true flow with the assumed regularity transports the intended marginals and is compatible with endpoint distribution matching. Finite-gap stochastic SCM, residual-dependent weighting, finite networks, clipping and GAN optimization do not inherit a joint-optimum or convergence guarantee from that statement. Regularity at image/noise endpoints also requires care.
- A fixed teacher supplies a fixed reference field; that does not make it correct. Online SCM supplies moving network targets plus data-derived displacements; that does not leave it unanchored. Neither establishes protection from mode loss.
- A finite neural map x+(s-t)N is not guaranteed invertible. The identity boundary alone does not make it an ODE flow, and final clipping is explicitly many-to-one. Latent sensitivity is useful evidence, not a proof of semantic diversity or an instruction to penalize all contractions.
- AYF's published adversarial recipe does not specify an image-augmentation operator or explicit class input to D. This is insufficient to establish that the actual implementation uses neither. Our augmentation and conditional projection critic are declared choices, not facts attributed to AYF.
- The shared conversation's “ImageNet-64 less dramatic” characterization is arithmetically misleading: 2.98→1.32 is a larger relative reduction than 3.32→1.92. Those published gains motivate the idea but do not predict the same relative improvement on Pokémon.
- Gradient-norm balancing controls magnitude, not agreement or preservation. The current specification already logs cosine similarity and limits the adversarial norm; RAdam updates and the moving objective still require observation.
- Interval GANs, multi-NFE adversarial mixtures, FM, semantic feature losses and composition penalties are separate interventions. Their theoretical plausibility is not a reason to stack them into the first implementation.

### Actual changes and remaining decisions

**Changes to the previous recommendation:** explicitly acknowledge the data-derived anchor in SCM; explain the absence of exact-endpoint atoms in the current time sampler; add the raw-map composition diagnostic; justify the shared conditional critic against the unconditional alternative; and prohibit shared trainable G/D feature parameters in this first game. Strengthen statements separating conditional generation from CFG, and unspecified AYF details from confirmed absence.

**Unchanged:** U160 EMA initialization, fresh optimizers, mature fixed gap with floor, no new FM term, no external teacher or feature network, scratch projection critic, full-endpoint RpGAN, exact R1/R2, distinct conditioning/discriminator augmentation, gradient-based balancing, explicit EMA/clocks/precision, and one matched no-adv control plus treatment.

**Still empirical:** whether the selected critic overfits, how useful its conditional gradient is, whether clipping leaves adequate signal, the appropriate critic warmup and regularization strength, the useful post-training horizon, and the eventual quality/diversity tradeoff. These remain qualification questions; the discussion does not settle them. The next implementation should resolve these within the minimal declared recipe rather than expand into an architecture or loss sweep.
