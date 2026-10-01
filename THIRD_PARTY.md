# Source attribution

- Official FastGen starter: KAIST Visual AI Group, https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen, pinned in [provenance](resources/provenance.json). The starter's files are preserved; this handoff does not assign them a new license.
- Adversarial Flow Models: Shanchuan Lin, Ceyuan Yang, Zhijie Lin, Hao Chen and Haoqi Fan, ByteDance Seed. Official repository https://github.com/ByteDance-Seed/Adversarial-Flow-Models. The complete tracked source snapshot is preserved at `vendor/adversarial-flow-models`, including [MIT license](vendor/adversarial-flow-models/LICENSE.txt) and embedded notices. It includes reference CAFM files, but those are not the baseline scope. No pretrained training weights are used; official evaluation-only FID weights are recorded separately below.
- AFM paper: https://arxiv.org/abs/2511.22475. CAFM context: https://arxiv.org/abs/2604.11521. AYF context: https://arxiv.org/abs/2506.14603.
- SoFlow adaptation: https://github.com/pianojay/fastgen-soflow, pinned for source navigation only. The original handoff copied no implementation; the later baseline U-Net adaptation is recorded below. The copied development partition is that project's generated metadata. Its U-Net carries NVIDIA CC BY-NC-SA 4.0 notices; retain and review applicable notices when copying components during implementation. Do not relabel borrowed code under the AFM snapshot's MIT license.

The upstream AFM snapshot has external-code placeholders and model-asset dependencies. Preserving its source is not a claim that all optional models or training configurations are runnable here.


## Integrated baseline

`afm_fastgen/unet.py` is adapted from the exact pinned SoFlow `soflow_fastgen/unet.py` source, itself NVIDIA's EDM/Song U-Net implementation under CC BY-NC-SA 4.0. The original header remains and the full license is in `licenses/NVIDIA-CC-BY-NC-SA-4.0.txt`. AFM endpoint modifications are listed in `docs/IMPLEMENTATION.md`. No runtime sibling imports exist.

`afm_fastgen/losses.py` ports the AFM loss and gradient-normalization formulas from the MIT-licensed pinned ByteDance snapshot; its copyright and license remain in `vendor/adversarial-flow-models/LICENSE.txt`. The discriminator and local runner are new code. The evaluation-only Inception model is downloaded through the official supplied clean-fid package and its artifact SHA-256 is recorded with every FID.
