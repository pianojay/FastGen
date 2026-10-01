# Source attribution

- Official FastGen starter: KAIST Visual AI Group, https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen, pinned in [provenance](resources/provenance.json). The starter's files are preserved; this handoff does not assign them a new license.
- Adversarial Flow Models: Shanchuan Lin, Ceyuan Yang, Zhijie Lin, Hao Chen and Haoqi Fan, ByteDance Seed. Official repository https://github.com/ByteDance-Seed/Adversarial-Flow-Models. The complete tracked source snapshot is preserved at `vendor/adversarial-flow-models`, including [MIT license](vendor/adversarial-flow-models/LICENSE.txt) and embedded notices. It includes reference CAFM files, but those are not the baseline scope. No pretrained weights were downloaded.
- AFM paper: https://arxiv.org/abs/2511.22475. CAFM context: https://arxiv.org/abs/2604.11521. AYF context: https://arxiv.org/abs/2506.14603.
- SoFlow adaptation: https://github.com/pianojay/fastgen-soflow, pinned for source navigation only. No SoFlow implementation code is copied into this handoff. The copied development partition is that project's generated metadata. Its U-Net carries NVIDIA CC BY-NC-SA 4.0 notices; retain and review applicable notices when copying components during implementation. Do not relabel borrowed code under the AFM snapshot's MIT license.

The upstream AFM snapshot has external-code placeholders and model-asset dependencies. Preserving its source is not a claim that all optional models or training configurations are runnable here.
