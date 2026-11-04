# Audit: transferring appearance for augmentation

October 7, 2026. Literature and static source audit of 12 method families and 14 repositories. No weights were downloaded, upstream programs executed, images augmented, or models instantiated, trained, or evaluated. Parameter totals below are source-derived arithmetic, not runtime measurements.

Subsequent execution is tracked separately: [round2 results](/home/user/jw_soflow/exploration/transfer_aug/ROUND2_REPORT.md) and [round3 ledger](/home/user/jw_soflow/exploration/transfer_aug/ROUND3_LEDGER.json). The user later authorized bounded scratch-trained pilots and waived the parameter cap for exploratory StarGAN v2 only. That exception does not alter the historical configuration counts below or other resource/data limits.

## Finding and proposed direction

**“Appearance-factor transfer” is a useful working description.** Color moments, Fourier amplitudes, feature statistics, and learned texture codes are different interventions. Calling all of them “style” would imply a separation from identity that our data has not established.

Start with A1-only factor discovery and inexpensive donor-driven color/Fourier demonstrations. If recurring appearance groups emerge across categories, CUT is a relatively small learned comparison. Swapping Autoencoder is a closer conceptual match to discovering transferable factors without domain labels, but its released setup exceeds our framework parameter cap and needs architectural changes for 64 × 64. Neither is ready to adopt.

AdaIN and WCT fit the parameter cap in the inspected configurations; their more immediate obstacle is reliance on pretrained visual features. MixStyle does not generate augmented RGB images. No audited method guarantees Pokémon identity preservation or lower generative FID.

## Our setting and the comparison boundary

| Property | Working constraint or observation |
| --- | --- |
| Dataset | Kaggle Pokémon Generation One: 20,099 available images, 151 category folders. This is the actual local inventory, despite “22k” in the dataset name. |
| Development data | A = 17,079; fixed A1 = 14,059 for discovery/fitting/donors; fixed A2 = 3,020 for later validation. B = 3,020 remains outside method development. |
| Resolution | Native images have heterogeneous sizes/aspect ratios. The task input is 64 × 64 RGB with the official white transparency composition and resize. Larger source images do not remove the 64 × 64 deployment constraint. |
| Distribution | A contains different rendering/media types, backgrounds, poses, and framing. There are no verified appearance-domain annotations or paired same-content/different-appearance targets. A1 has only 26–163 images per category. |
| Rules | Each learned model framework ≤100M parameters, including registered auxiliary/frozen/EMA components; distinct models are not automatically summed. Learned models used in training/sampling must be trained from scratch on allowed data. Peak training VRAM ≤20 GB. |
| Environment | Existing FastGen-SoFlow-aligned EDA environment: Python 3.10 / PyTorch 2.7.1+cu126. GPU 0 is available by authorization; this audit used no GPU. Runtime compatibility and VRAM remain unmeasured. |

Sources: [project rules](/home/user/jw_soflow/exploration/README.md:188), [dataset preparation](/home/user/jw_soflow/exploration/eda/README.md), [historical EDA](/home/user/jw_soflow/exploration/eda/REPORT.md). Historical B-specific findings do not supply donor policies, transformation targets, or hyperparameters. Previously fitted full-A PCA/clusters are not A1-only development transforms.

## What data supported the original methods?

These are representative published experiments or explicit released configurations, not a claim that every listed dataset was used in every experiment. A small benchmark domain can still depend on features pretrained on a much larger external dataset.

| Method | Original data, scale, and characteristics | Image resolution used |
| --- | --- | --- |
| [Reinhard color transfer](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf) | Individual source/reference pairs; no learned training corpus. Demonstrations transfer global color appearance. | Variable; no architecture-imposed size. |
| [FDA](https://arxiv.org/abs/2004.05498) | Driving segmentation: GTA5 24,966 synthetic frames or SYNTHIA 9,400; Cityscapes 2,975 target training images, 500 validation. | GTA5 native 1914 × 1052, resized 1280 × 720 then cropped 1024 × 512; SYNTHIA native 1280 × 760, cropped 1024 × 512; Cityscapes resized 1024 × 512. |
| [MixStyle](https://arxiv.org/abs/2104.02008) | PACS: 9,991 images total, seven classes, four domains (photo, art painting, cartoon, sketch). Domain-generalization protocol trains on three domains and holds out one. | Released PACS SSDG/MSDA configs specify 224 × 224; do not treat this as verification of every paper configuration. |
| [Gatys neural style transfer](https://arxiv.org/abs/1508.06576) | One content/style pair per optimization; ImageNet-pretrained VGG supplies the representation. No learned transfer generator. | Author notebook uses 512, then an 800-pixel higher-resolution stage; variable-size operation. |
| [AdaIN](https://arxiv.org/abs/1703.06868) | Approximately 80k MS-COCO content images and 80k mostly WikiArt style images; natural-photo/art cross-domain training. | Short side 512, random 256 × 256 training crops. |
| [WCT](https://arxiv.org/abs/1705.08086) | COCO reconstruction data for decoders plus pretrained VGG encoders. Exact selected training-image count and training crop policy were not established from inspected sources. | Paper timing at 256 × 256; official demo defaults to minimum side 768. Neither establishes a 64 × 64 training recipe. |
| [CycleGAN](https://arxiv.org/abs/1703.10593) | Photo/art experiments: 6,853 photographs; 1,074 Monet, 584 Cézanne, 401 Van Gogh, 1,433 ukiyo-e paintings. Domains explicitly identified, pairs unaligned. | 256 × 256; paper uses nine residual blocks at 256 and six at 128. |
| [CUT / FastCUT](https://arxiv.org/abs/2007.15651) | Paper reports 2,403 training images combined for horse↔zebra; also Cityscapes with 2,975 train / 500 validation. These are domain translation tasks, often changing semantic category. | 256 × 256 training. |
| [MUNIT](https://arxiv.org/abs/1804.04732) | Unpaired two-domain multimodal translation. High-resolution Yosemite set: 3,253 summer + 2,385 winter photographs. Released edges↔handbags config supplies the parameter reference below. | Yosemite short side 1024; counted edges↔handbags config uses 256 crops. These are different configurations. |
| [StarGAN v2](https://arxiv.org/abs/1912.01865) | Paper AFHQ: approximately 15k aligned animal faces across cat/dog/wildlife; 500 per domain held out. Roughly 13.5k training by that description; later dataset revisions can differ. | Dataset 512 × 512, model training 256 × 256. |
| [Swapping Autoencoder](https://arxiv.org/abs/2007.00653) | LSUN Church 126,227; Bedroom 3,033,042; FFHQ 70k; AFHQ about 15k combined without domain labels. Mostly coherent image collections. | Church/Bedroom/AFHQ 256; FFHQ initially 512 then 1024. |
| [DiffAugment](https://arxiv.org/abs/2006.10738) | GAN regularization: CIFAR-10/100 with 50k training images and reduced-data settings; extreme examples include 100 images each of Obama, grumpy cat, or panda. | CIFAR 32 × 32; those 100-image demonstrations 256 × 256. |

Implication: published success at 256–1024 pixels or on aligned faces does not establish that identity and appearance can be separated across 151 heterogeneous categories at 64 pixels. Conversely, our total image count alone does not rule out learned translation: some original domains were smaller, but more coherently defined.

## Parameter and implementation audit

M = 1,000,000 parameters. G = generator, D = discriminator, E = encoder. Counts include biases and affine normalization; exclude buffers and optimizer state. Source revisions, individual component counts, and formulas are in [parameter_audit.json](/home/user/jw_soflow/exploration/transfer_aug/parameter_audit.json) and [count_audit_parameters.py](/home/user/jw_soflow/exploration/transfer_aug/count_audit_parameters.py). These are configuration-specific counts, not blanket compliance certificates. Extra registered components must be included in any eventual implementation.

| Method and codebase | Audited parameter scope | Applicability here |
| --- | --- | --- |
| [Color transfer](https://github.com/jrosebr1/color_transfer) — community implementation | **0** learned parameters | Cheap direct image-space candidate; implementation uses OpenCV Lab, not the original paper's lαβ color space. |
| [FDA](https://github.com/YanchaoYang/FDA) — official | **0** in transfer operator; original segmentation network is separate and unnecessary for transfer | Cheap image-space candidate using A1 donors only. |
| [MixStyle](https://github.com/KaiyangZhou/mixstyle-release) — official | **0 additional**; standard ResNet-18 with seven-class head ≈**11.18M** | Feature-space regularizer; no RGB decoder. Original pretrained-backbone recipe cannot be imported unchanged. |
| [Gatys](https://github.com/leongatys/PytorchNeuralStyleTransfer) — author notebook | VGG-19 convolution object **20.02M**; prefix through conv5_1 **12.94M**. No classifier head needed. | Parameter fit, but pretrained feature dependency and per-image optimization. |
| [AdaIN original](https://github.com/xunhuang1995/AdaIN-style) / [PyTorch port](https://github.com/naoto0804/pytorch-AdaIN) | Counted community port: active E **3.506M** + decoder **3.505M** = **7.011M** | Small enough; standard pretrained encoder/checkpoints conflict with scratch-only augmentation. |
| [WCT original](https://github.com/Yijunmaverick/UniversalStyleTransfer) / [PyTorch port](https://github.com/sunshineatnoon/PytorchWCT) | Counted community port's five separate encoders + five decoders: **34.24M** | Pretraining and low-resolution covariance are larger issues than parameter count.                          |
| [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — official | RGB/width64/instance norm: G9 **11.38M**, D **2.765M**; two G + two D **28.29M** | Scratch-capable core; requires meaningful appearance domains, which we have not established. |
| [CUT / FastCUT](https://github.com/taesungp/contrastive-unpaired-translation) — official | G **11.38M** + D **2.765M** + five patch MLPs **0.560M** = **14.70M** | Smaller learned baseline if domains emerge; no external perceptual encoder required by core objective. |
| [MUNIT](https://github.com/NVlabs/MUNIT) — official | Edges↔handbags: each G **15.03M**, each multiscale D **8.271M**; total **46.59M**, `vgg_w=0` | Scratch-capable variant; explicit style code and multimodal output, but more objectives and domain assumptions. |
| [StarGAN v2](https://github.com/clovaai/stargan-v2) — official | AFHQ/256/3 domains/`w_hpf=0`: G+mapping+E+D **78.95M**; released Solver with registered EMA copies **137.06M** | **Released Solver exceeds 100M.** Smaller/adapted framework required. |
| [Swapping Autoencoder](https://github.com/taesungp/swapping-autoencoder-pytorch) — official | Church/default: E **30.58M**, G **25.06M**, image D **28.86M**, patch D **24.52M**; total **109.03M** | **Released framework exceeds 100M.** Conceptually relevant, but needs reduced capacity and a 64-pixel redesign. |
| [DiffAugment](https://github.com/mit-han-lab/data-efficient-gans) — official | **0 additional**; base GAN still counts | Useful later regularization control; does not transfer donor appearance or establish a valid expanded real-image distribution. |

The table counts full adversarial frameworks where applicable; it does not add an independent transfer model to the SiDA teacher/student allowance. Frozen external encoders remain a separate scratch-training obstacle even when small. Low-rank adapters or freezing a large network would not make its full registered parameter count disappear.

## What would actually transfer?

### Cheap, interpretable interventions

**Color moments:** transfer per-channel location/scale statistics, not rendering geometry or texture. The original [Reinhard method](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf) uses decorrelated lαβ; the [audited code](https://github.com/jrosebr1/color_transfer/blob/6724ccf927f2e3838ccd3f3da8b2542b79c9bf2f/color_transfer/__init__.py) is an OpenCV Lab variant. Color can encode Pokémon identity, and whole-image statistics can be dominated by background. Specify color space, strength, near-zero-variance handling, and clipping before a demo. A foreground-restricted variant would require its own mask evidence; a white-pixel threshold is not verified segmentation.

**FDA:** replace a small low-frequency Fourier-amplitude region using a donor while retaining the source phase. This is an appearance intervention, not a proof of semantic invariance. The original method uses target-domain images in domain adaptation; our adaptation would use **only A1 donors**, with no B reference distribution. At 64 pixels the inspected [NumPy implementation](https://github.com/YanchaoYang/FDA/blob/b9a0cdf0bcc9b787c00e39df73eda5673706f219/utils/__init__.py) uses radius `floor(64 × L)`: `L=0.01` changes only DC, and even `L=0` replaces DC. Use an explicit bypass for the unchanged control. The torch branch uses obsolete FFT APIs and needs a port.

**MixStyle:** mixes feature-channel means/standard deviations between samples. Its [implementation](https://github.com/KaiyangZhou/mixstyle-release/blob/16f7cf1fe2c7b1b3c660c72b32817ebb0545397a/README.md) is useful for thinking about donor-conditioned statistics, but does not produce images. Applying it to a later generative discriminator would be a new integration question; recognition robustness does not demonstrate generative distribution fidelity.

### Feature-based image stylization

**Gatys, AdaIN, WCT** impose different definitions of appearance: feature Gram statistics, channelwise moments, and full feature covariance, respectively. Content-feature matching is a constraint, not a category guarantee. See the [Gatys paper](https://arxiv.org/abs/1508.06576), [AdaIN paper](https://arxiv.org/abs/1703.06868), and [WCT paper](https://arxiv.org/abs/1705.08086).

Their standard learned feature spaces are external pretrained priors. Substituting a randomly initialized or A1-trained encoder changes the method's evidence base; we would first need an objective that learns useful appearance/content features. Reducing their already-small active backbones does not resolve this issue.

Resolution also matters. AdaIN's relu4_1 map is only 8 × 8 on a 64-pixel image. At WCT's relu5_1, 512-channel covariance is estimated from 4 × 4 locations, with centered rank at most 15. This is a geometry-derived limitation, not a measured failure on our dataset. Eigenvalue regularization cannot recover missing spatial evidence. The [WCT port](https://github.com/sunshineatnoon/PytorchWCT) also relies on the removed `load_lua` interface.

### Learned image translation and factor swapping

**CycleGAN / CUT:** learn mappings between explicitly supplied domains. Cycle consistency or patch correspondence can preserve some structure without preserving category. In our task, domains should describe evidence-supported appearance groups—not Pokémon species. CUT shares generator features for its contrastive objective and is the smaller starting comparison. The [released CycleGAN architecture](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2a7afba2895d52556dd5dfe07e8555ef657ced6f/models/networks.py) has a 70-pixel PatchGAN receptive field, already larger than our image; a 64-pixel adaptation changes the discriminator's spatial role. [CUT implementation](https://github.com/taesungp/contrastive-unpaired-translation/blob/b3ac297708dfb6f7589d04662277e53c0d579c27/models/cut_model.py).

**MUNIT:** explicitly separates a spatial content code and an eight-dimensional style code in the counted configuration. Multimodality is relevant, but a style code can still carry identity. The [audited configuration](https://github.com/NVlabs/MUNIT/blob/a82e222bc359892bd0f522d7a0f1573f3ec4a485/configs/edges2handbags_folder.yaml) disables the VGG loss; recipes that enable an external pretrained perceptual network need separate rejection/adaptation. This is a more involved follow-on if a simple domain translation baseline is too restrictive.

**StarGAN v2:** reference-guided transfer across labeled domains, with domain-specific mapping/encoder heads. Using 151 species as domains both changes the research question and raises the mapping network alone to **124.74M** in the inspected design. Simply changing resolution to 64 leaves the three-domain non-EMA components at **76.52M**, because the initial channel width increases as resolution decreases. Its [registered EMA copies](https://github.com/clovaai/stargan-v2/blob/875b70a150609e8a678ed8482562e7074cdce7e5/core/solver.py) also count under our rules. Keep this as a lower-priority reference until domains and a compliant architecture exist.

**Swapping Autoencoder:** learns spatial structure and global texture codes from one collection, without requiring appearance-domain labels. That matches our uncertainty better than forcing two style domains. However, disentanglement is learned from adversarial/reconstruction constraints, not category labels. With the [released encoder](https://github.com/taesungp/swapping-autoencoder-pytorch/blob/6baa180f1184ee79a6b967f9d80ee0e02a979ac7/models/networks/encoder.py), 64-pixel input becomes a 4 × 4 spatial map, then 1 × 1 after the first unpadded global downsampling; the next 3 × 3 convolution cannot operate. The default patch discriminator also resizes small sampled regions to 128 pixels. Reducing downsampling, channels, and patch scale is substantive adaptation, not simply resizing input. Its [README](https://github.com/taesungp/swapping-autoencoder-pytorch/blob/6baa180f1184ee79a6b967f9d80ee0e02a979ac7/README.md) reports 14 days on four V100s for Church/256 and uses custom CUDA kernels. That is evidence of a costly reference setup, not a runtime prediction for a redesigned 64-pixel model.

### A necessary non-transfer comparison

**DiffAugment:** applies differentiable transformations consistently in GAN training to regularize learning from limited data. It does not need donor images. Later, it can distinguish benefits of ordinary regularization from benefits of transferring a particular factor. Saving cutout/color-distorted images as additional “real” targets is a different intervention from the [paper's training objective](https://arxiv.org/abs/2006.10738).

## Next research sequence

1. **Discover a candidate factor within A1.** Compare intra-class and inter-class color, frequency, texture, and framing statistics with category-balanced and duplicate-aware controls. Shared statistical variation is a candidate, not proof of an editable semantic factor.
2. **Specify a narrow intervention.** State the donor statistic, intended change, preserved anatomy/markings, and failure criteria. Contrast within-category and cross-category donors.
3. **Begin with color moments and small Fourier interventions.** Compare unchanged input, donor transfer, and a donor-free perturbation of comparable strength. Inspect all outcomes at 64 × 64; measure actual factor change alongside identity/artifact failures.
4. **Escalate only if the factor cannot be tested adequately with explicit transforms.** Consider CUT for discovered domains; a reduced Swapping Autoencoder for learned factorization without domains. Scratch training, actual registered counts, memory, and runtime must then be verified before replication claims.
5. **Consider SiDA only after augmentation quality is established.** Adding transformed real targets changes the target distribution; discriminator regularization and teacher-data augmentation change different objectives. Preserve these distinctions and use A2 for later development decisions.

No method is selected or scheduled for training by this audit. The open question is whether the desired appearance change can be demonstrated while retaining identity—not whether an implementation can output more images.

## Evidence and reproducibility

- [Source registry](/home/user/jw_soflow/exploration/transfer_aug/SOURCE_REGISTRY.md): code, pinned commits, license-file labels, paper URLs, and provenance limitations.
- [Source manifest](/home/user/jw_soflow/exploration/transfer_aug/source_manifest.json): pinned file URLs and SHA-256 hashes, including recorded unsuccessful lookups.
- [Parameter arithmetic](/home/user/jw_soflow/exploration/transfer_aug/count_audit_parameters.py) and [results](/home/user/jw_soflow/exploration/transfer_aug/parameter_audit.json): no model construction. Exact integers mean exact arithmetic for the stated configuration, not measured runtime counts.
- Cached source files and papers are local under `audit_sources/` and excluded from version control. [Restore script](/home/user/jw_soflow/exploration/transfer_aug/restore_audit_sources.py) retrieves hash-checked source files at the recorded revisions; it does not install or execute them.

This is an applicability audit, not a cross-paper performance ranking. Unverified items include actual 64-pixel transfer quality, identity preservation, current runtime compatibility, peak memory, throughput, and generative FID impact. WCT's exact reconstruction-corpus count remains unverified rather than inferred from another method.
