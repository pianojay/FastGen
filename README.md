# FastGen submission — EDM teacher + adversarial Score identity Distillation

Both submitted models are EDM-preconditioned DDPM++ U-Nets (`networks.py`) trained from scratch on
`train_split.txt` only (no pretrained weights, no validation images):

* `ModelOneNFE` (128 channels, 61.8M parameters): one-step generator `x = D(2.5 z; sigma=2.5, class)` (exactly 1 NFE).
* `ModelFewNFE` (160 channels, 96.6M parameters): two-step generator on the noise ladder `(2.5, 0.4)`:
  `x_1 = D(2.5 z_1; 2.5, class)`, `x = D(x_1 + 0.4 z_2; 0.4, class)` (2 NFEs).

Checkpoints: `checkpoints/one_nfe.ckpt`, `checkpoints/few_nfe.ckpt` (state dicts incl. the noise
ladder buffer). Architecture hyper-parameters are fixed in `model.py`; sampling uses a fixed seed.

## Method

1. **Teachers** (`train_edm.py`): class-conditional EDM diffusion models (DDPM++, dropout 0.25, x-flip,
   no augmentation conditioning) trained on the full train split for 40M images; the EMA with a
   1M-image half-life is used as teacher. Two sizes: 128 channels (61.8M, for the one-step chain) and
   160 channels (96.6M, for the two-step model). Every model, teacher or student, has at most 96.6M parameters.
2. **Few-NFE student** (`train_sida.py --sigmas 2.5,0.4`): adversarial Score identity Distillation
   (SiDA) of the 96.6M teacher into a 96.6M generator initialised from it. The fake-score network is
   initialised from the teacher and its encoder doubles as the discriminator. Uniform-step matching
   trains every step of the 2-step ladder. Learning rate 2e-5 (generator and fake score), linearly
   decayed to 2e-6 between 1.5M and 3M images; 3M images, EMA with a 1M-image half-life (rampup 0.15).
3. **One-NFE student** (`train_sida.py`, 1 step), 61.8M chain: a 61.8M two-step generator is first
   distilled from the 61.8M teacher exactly as in step 2 (3M images), and the one-step generator is
   initialised from it (EMA 1M). Adversarial weight 0.03 with a multi-scale discriminator
   (channel-mean logit maps of the fake-score encoder at 8x8, 16x16 and 32x32), learning rate 2e-5,
   and a **restart every 1.5M images**: the EMA generator becomes the raw generator, the fake score is
   re-initialised from the teacher and the optimiser state is cleared. 3.25M images; EMA with a
   1M-image half-life (rampup 0.15).

Model selection (model size, training lengths, EMA) was done on a 20-image-per-class hold-out of the
**train** split (`prep_data.py`) with development runs that never see the hold-out, additionally
scored on a de-duplicated subset of the hold-out (Inception-feature cosine similarity < 0.90 to every
remaining training image, `fid_holdout_clean` in the logs). A change was adopted only if it improved
both hold-out scores by more than 0.3: the 96.6M two-step model did (dev hold-out 17.90 / 27.25 vs
18.24 / 27.67 for 61.8M), the 96.6M one-step model did not (18.54 / 28.31 vs 19.01 / 28.29, mean of
3 sampling seeds and 4 runs), so the one-step model stays at 61.8M. The final models reuse the
selected schedules on the full train split.

## Files

| File | Role |
|---|---|
| `model.py` | `ModelOneNFE` / `ModelFewNFE` (base `Model` class unchanged) |
| `networks.py` | DDPM++ U-Net + EDM preconditioning (adapted from NVlabs/edm); optional multi-resolution encoder features for the discriminator |
| `augment.py` | Non-leaky ADA-style augmentation pipeline (adapted from NVlabs/edm) |
| `prep_data.py` | Caches the train split as uint8 tensors; creates a 20-per-class dev hold-out from the TRAIN split and its clean-fid statistics (`pk_holdout`, used by the in-training hold-out FID) |
| `train_edm.py` | Teacher (EDM) training |
| `train_sida.py` | SiD / SiDA distillation (1-step and few-step, restarts, lr schedule, multi-scale discriminator) |
| `export.py` | Converts a distillation snapshot into a submission checkpoint |
| `fid_utils.py`, `eval_holdout.py`, `eval_teacher.py` | clean-fid helpers and hold-out evaluation (model selection uses the train-split hold-out only) |
| `plot_logs.py` | Training-curve plots for the write-up |
| `dataset.py`, `download_dataset.py`, `evaluate.py`, `src/` | Provided files, unmodified |

## Reproduction

```bash
python download_dataset.py
python prep_data.py                          # data_cache/train64.npz (train split only) + hold-out FID statistics
E="--subset all --dropout 0.25 --augment_p 0 --ema_halflives_kimg 250,500,1000,2000,4000 --ema_rampup 0.1 --duration_mimg 40"
# 1) teachers (single GPU each, peak VRAM 14.0 / 18.0 GB); teacher = ema-040000.pt, EMA index 2 (1M half-life)
python train_edm.py --out runs/teacher62 --channels 128 $E
python train_edm.py --out runs/teacher97 --channels 160 $E
S="--augment_p 0 --subset all --extra_ema_kimg 1000@0.15,3000@0.3 --batch 256 --sigmas 2.5,0.4 --lr 2e-5 --glr 2e-5 \
   --lr_decay_start_kimg 1500 --lr_decay_end_kimg 3000 --lr_final_ratio 0.1 --duration_mimg 3"
# 2) few-NFE: 96.6M two-step student (peak VRAM 15.0 GB)
python train_sida.py --out runs/few --teacher runs/teacher97/ema-040000.pt --teacher_ema 2 $S --micro 16 --micro_f 32
# 3) one-NFE: 61.8M two-step generator (peak VRAM 19.0 GB), then the one-step student initialised from it (19.0 GB)
python train_sida.py --out runs/two62 --teacher runs/teacher62/ema-040000.pt --teacher_ema 2 $S --micro 32 --micro_f 64
python train_sida.py --out runs/one --teacher runs/teacher62/ema-040000.pt --teacher_ema 2 --augment_p 0 --subset all \
    --extra_ema_kimg 1000@0.15,3000@0.3 --micro 32 --micro_f 64 --batch 128 --lr 2e-5 --glr 2e-5 --lsg_gan 0.03 \
    --disc_scales 8,16,32 --fake_reset_kimg 1500 --restart_from_ema 1 --reset_extra_emas 0 \
    --init_generator runs/two62/gen-003000.pt --init_ema 1000 --snap_kimg 250 --duration_mimg 3.25
# 4) export
python export.py --snapshot runs/one/gen-003250.pt --mode one_nfe --ema_kimg 1000 --out checkpoints/one_nfe.ckpt
python export.py --snapshot runs/few/gen-003000.pt --mode few_nfe --ema_kimg 1000 --out checkpoints/few_nfe.ckpt
# 5) official evaluation
python evaluate.py --model_checkpoint ./checkpoints/one_nfe.ckpt --evaluate_mode one_nfe
python evaluate.py --model_checkpoint ./checkpoints/few_nfe.ckpt --evaluate_mode few_nfe
```

Peak training VRAM is logged as `peak_mem_gb` (`torch.cuda.max_memory_allocated()` / 1e9) in every
`log.jsonl`. All scripts use only the provided environment (torch, torchvision, numpy, Pillow,
clean-fid); `plot_logs.py` draws the curves with Pillow.

## References

* T. Karras et al., Elucidating the Design Space of Diffusion-Based Generative Models (EDM), NeurIPS 2022. Code: https://github.com/NVlabs/edm
* M. Zhou et al., Score identity Distillation, ICML 2024; Adversarial Score identity Distillation, ICLR 2025; Few-Step Diffusion via Score identity Distillation, 2025. Code: https://github.com/mingyuanzhou/SiD
* T. Yin et al., Improved Distribution Matching Distillation (DMD2), NeurIPS 2024.
* A. Sauer et al., Adversarial Diffusion Distillation, ECCV 2024; Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation, SIGGRAPH Asia 2024 (multi-level discriminator heads).
* T. Karras et al., Training Generative Adversarial Networks with Limited Data (ADA), NeurIPS 2020.
* T. Karras et al., Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2), CVPR 2024 (EMA length).
* Y. Song et al., Score-Based Generative Modeling through SDEs (DDPM++), ICLR 2021.
* G. Parmar et al., On Aliased Resizing and Surprising Subtleties in GAN Evaluation (clean-fid), CVPR 2022.
