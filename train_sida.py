"""Distil the EDM teacher into a few-step generator with (adversarial) Score identity Distillation.

Implements SiD (Zhou et al., ICML 2024) and its adversarial variant SiDA
(Zhou et al., ICLR 2025), following the official implementation
(https://github.com/mingyuanzhou/SiD, Apache-2.0): the fake-score network is
initialised from the teacher and doubles as the discriminator through the
channel-mean of its encoder bottleneck.

Multi-step extension (``--steps K``): the generator is a denoiser evaluated at
a fixed decreasing noise ladder sigma_1 > ... > sigma_K.  Step k re-noises the
previous output to sigma_k and denoises it; during training one step index is
sampled per micro-batch and its input is produced by running the previous
steps without gradient (backward simulation, as in DMD2, Yin et al. 2024).
K = 1 recovers one-step SiDA with sigma_1 = init_sigma.
"""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import AugmentPipe
from networks import EDMPrecond
from train_edm import AUG_DIM, GPUData, GuidedDenoiser, build_net, parse_list


def set_dropout(net, p):
    for m in net.modules():
        if hasattr(m, "dropout") and isinstance(m.dropout, float):
            m.dropout = p


class Generator:
    """Few-step sampler around a denoiser network (shared by training and model.py)."""

    def __init__(self, sigmas):
        self.sigmas = [float(s) for s in sigmas]

    def run(self, net, labels, z, upto=None, grad_last=False):
        """Run steps 0..upto (inclusive). Only the last step keeps its graph if grad_last."""
        upto = len(self.sigmas) - 1 if upto is None else upto
        x = None
        for k in range(upto + 1):
            s = self.sigmas[k]
            inp = z[k] * s if k == 0 else x + z[k] * s
            with torch.set_grad_enabled(grad_last and k == upto):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = net(inp, torch.full((len(labels),), s, device=labels.device), labels)
        return x


class DiscHead(torch.nn.Module):
    """DMD2-style classifier on the 8x8 encoder bottleneck of the fake-score network (Yin et al., 2024)."""

    def __init__(self, c):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(c, c, 4, stride=2, padding=1), torch.nn.GroupNorm(32, c), torch.nn.SiLU(),
            torch.nn.Conv2d(c, c, 4, stride=4, padding=0), torch.nn.GroupNorm(32, c), torch.nn.SiLU(),
            torch.nn.Conv2d(c, 1, 1))

    def forward(self, feats):
        return self.net(feats)


def bce(logits, target_one):
    t = torch.ones_like(logits) if target_one else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, t)


def main(args):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    ck = torch.load(args.teacher, map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    data = GPUData(args.data, args.subset, device, xflip=args.xflip, class_power=args.class_power)
    teacher_sd = ck["emas"][args.teacher_ema] if "emas" in ck else ck["state_dict"]

    true_score = build_net(targs, data.num_classes).to(device)
    true_score.load_state_dict(teacher_sd)
    true_score.eval().requires_grad_(False)
    fake_score = copy.deepcopy(true_score).train().requires_grad_(True)
    gargs = targs
    if args.generator_teacher:
        # student architecture/initialisation from a (smaller) diffusion model; the score networks
        # (true + fake) still come from --teacher, so the student may differ in size from the teacher
        gck = torch.load(args.generator_teacher, map_location="cpu", weights_only=False)
        gargs = argparse.Namespace(**gck["args"])
        G = build_net(gargs, data.num_classes).to(device)
        G.load_state_dict(gck["emas"][args.generator_teacher_ema] if "emas" in gck else gck["state_dict"])
    else:
        G = copy.deepcopy(true_score)
    if args.init_generator:
        gsd = torch.load(args.init_generator, map_location="cpu", weights_only=False)
        if args.init_ema:
            gsd = gsd["G_ema_extra"][args.init_ema]  # one of the extra EMAs instead of the main one
        G.load_state_dict(gsd["G_ema"] if "G_ema" in gsd else gsd)
    set_dropout(fake_score, args.dropout)
    set_dropout(G, args.dropout)
    G.train().requires_grad_(True)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    # extra EMA spec: "HL" or "HL@R" (half-life in kimg, optional rampup ratio R overriding --ema_rampup)
    extra_spec = [s.split("@") for s in args.extra_ema_kimg.split(",")] if args.extra_ema_kimg else []
    extra_hl = [float(s[0]) for s in extra_spec]
    extra_ramp = [float(s[1]) if len(s) > 1 else args.ema_rampup for s in extra_spec]
    extra_emas = [copy.deepcopy(G).eval().requires_grad_(False) for _ in extra_hl]
    G_ref, ref_gen = None, None
    if args.regress_to:
        rs = torch.load(args.regress_to, map_location="cpu", weights_only=False)
        G_ref = build_net(argparse.Namespace(**(rs.get("generator_args") or rs["teacher_args"])), data.num_classes).to(device)
        G_ref.load_state_dict(rs["G_ema"])
        G_ref.eval().requires_grad_(False)
        ref_gen = Generator(rs["sigmas"])
    ref = None
    if args.guide == "auto":
        rck = torch.load(args.guide_ref, map_location="cpu", weights_only=False)
        ref = build_net(argparse.Namespace(**rck["args"]), data.num_classes).to(device)
        ref.load_state_dict(rck["emas"][args.guide_ref_ema])
        ref.eval().requires_grad_(False)
    guided_true = GuidedDenoiser(true_score, ref, args.guide_w, args.guide)

    augment = AugmentPipe(p=args.augment_p, xflip=0, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1) if args.augment_p > 0 else None
    zero_aug = lambda n: torch.zeros(n, AUG_DIM, device=device) if targs.augment_p > 0 else None
    # D-only augmentation (ADA / DiffAugment style, Karras et al. 2020; Zhao et al. 2020): applied to every image the
    # discriminator sees (real, fake and the generator's adversarial term) but never to the fake-score denoising loss,
    # so it also works when the score networks are not augmentation-conditioned (e.g. the no-aug teacher T8).
    d_augment = None
    if args.d_augment_p > 0:
        color = dict(brightness=1, contrast=1, saturation=1) if args.d_augment_color else {}
        d_augment = AugmentPipe(p=args.d_augment_p, xflip=0, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1, **color)
    daug = (lambda x: d_augment(x)[0]) if d_augment is not None else (lambda x: x)

    head = DiscHead(fake_score.model.bottleneck_channels).to(device) if args.disc_head == "dmd" else None
    # Discriminator logit maps from the fake-score encoder. Default: channel mean of the 8x8 bottleneck (SiDA).
    # --disc_scales adds the channel means of the last encoder activation at further resolutions (multi-level
    # heads as in ADD/LADD); the per-scale adversarial losses are averaged.
    disc_scales = parse_list(args.disc_scales, int)
    multiscale = disc_scales != [8]
    assert not (multiscale and head is not None), "--disc_scales needs the parameter-free mean head"
    enc_mode, both_mode = ("encoder_ms", "both_ms") if multiscale else ("encoder", "both")
    if head is not None:
        disc = lambda f: [head(f).float()]
    elif multiscale:
        disc = lambda fs: [f.mean(dim=1, keepdim=True) for f in fs if f.shape[-1] in disc_scales]
    else:
        disc = lambda f: [f.mean(dim=1, keepdim=True)]
    # bce() is the batch/position mean, so with one logit map these reproduce the original SiDA losses exactly;
    # with several scales the per-scale losses are averaged.
    def d_loss(real_maps, fake_maps, weight):
        return sum((weight * (bce(r, True) + bce(f, False)) / 2).sum() for r, f in zip(real_maps, fake_maps)) / len(real_maps)

    def g_gan_loss(maps, wf):
        return sum((bce(m.clamp(-10, 10), True) / wf).mean(dim=[1, 2, 3]) for m in maps) * CHW / len(maps)

    f_params = list(fake_score.parameters()) + (list(head.parameters()) if head is not None else [])
    opt_f = torch.optim.Adam(f_params, lr=args.lr, betas=(0.0, 0.999), eps=1e-8)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.glr, betas=(args.g_beta1, 0.999), eps=1e-8)

    gen = Generator(parse_list(args.sigmas, float) if args.sigmas else [args.init_sigma])
    K = len(gen.sigmas)
    nimg = 0
    state_path = out / "training-state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)  # CPU: avoids a transient VRAM spike
        G.load_state_dict(st["G"]); G_ema.load_state_dict(st["G_ema"]); fake_score.load_state_dict(st["fake"])
        for e, hl in zip(extra_emas, extra_hl):
            e.load_state_dict(st["G_ema_extra"][hl])
        if head is not None:
            head.load_state_dict(st["head"])
        opt_g.load_state_dict(st["opt_g"]); opt_f.load_state_dict(st["opt_f"]); nimg = st["nimg"]
        print(f"resumed at {nimg/1e6:.3f} Mimg", flush=True)

    f_fwd = torch.compile(fake_score) if args.compile else fake_score
    t_fwd = torch.compile(guided_true) if args.compile else guided_true
    g_fwd = torch.compile(G) if args.compile else G

    sigma_min, sigma_max, rho = 0.002, 80.0, 7.0
    min_inv, max_inv = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    accum = args.batch // args.micro
    assert accum * args.micro == args.batch
    micro_f = args.micro_f or args.micro
    accum_f = args.batch // micro_f
    assert accum_f * micro_f == args.batch
    log = open(out / "log.jsonl", "a")
    t_start = t_last = time.time()
    n_last = nimg
    total = int(args.duration_mimg * 1e6)
    ev = int(args.eval_kimg * 1e3)
    sn = int(args.snap_kimg * 1e3)
    next_eval = (nimg // ev + 1) * ev
    next_snap = (nimg // sn + 1) * sn
    stats = {k: torch.zeros((), device=device) for k in ("loss_f", "loss_d", "loss_g", "loss_g_gan", "logit_r", "logit_f", "loss_reg")}
    n_stats = 0
    CHW = 3 * 64 * 64

    def gen_input(labels, step_k):
        z = [torch.randn(len(labels), 3, 64, 64, device=device) for _ in range(step_k + 1)]
        return z

    g_freeze_until = args.g_start_kimg * 1e3

    def lr_factor(n):
        # constant, then linear decay to lr_final_ratio between lr_decay_start_kimg and lr_decay_end_kimg
        if args.lr_decay_end_kimg <= args.lr_decay_start_kimg or n <= args.lr_decay_start_kimg * 1e3:
            return 1.0
        t = min(1.0, (n - args.lr_decay_start_kimg * 1e3) / ((args.lr_decay_end_kimg - args.lr_decay_start_kimg) * 1e3))
        return 1.0 + t * (args.lr_final_ratio - 1.0)

    while nimg < total:
        lf = lr_factor(nimg)  # set every step, so it also overrides learning rates restored from a training state
        for group in opt_f.param_groups:
            group["lr"] = args.lr * lf
        for group in opt_g.param_groups:
            group["lr"] = args.glr * lf
        # ---------------- fake score / discriminator update ----------------
        for _fs in range(args.fake_steps):
          fake_score.train().requires_grad_(True)
          if head is not None:
              head.requires_grad_(True)
          G.eval()
          opt_f.zero_grad(set_to_none=True)
          for _ in range(accum_f):
            real, labels = data.sample(micro_f)
            k = int(torch.randint(0, K, ()).item())
            with torch.no_grad():
                images = gen.run(g_fwd, labels, gen_input(labels, k), upto=k)
            rnd = torch.randn([micro_f, 1, 1, 1], device=device)
            sigma = (rnd * args.P_std + args.P_mean).exp()
            weight = (sigma ** 2 + 0.25) / (sigma * 0.5) ** 2
            y, aug = augment(images) if augment is not None else (images, zero_aug(micro_f))
            n = torch.randn_like(y) * sigma
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if d_augment is None:
                    y_fake, logit_fake = f_fwd(y + n, sigma.flatten(), labels, augment_labels=aug, mode=both_mode)
                    logit_fake = disc(logit_fake)
                else:
                    y_fake = f_fwd(y + n, sigma.flatten(), labels, augment_labels=aug)
                    if args.use_gan:
                        logit_fake = disc(f_fwd(daug(images) + n, sigma.flatten(), labels, augment_labels=zero_aug(micro_f), mode=enc_mode))
            loss_f = (weight * (y_fake - y) ** 2).sum() * args.ls / args.batch
            loss = loss_f
            if args.use_gan:
                if d_augment is None:
                    yr, aug_r = augment(real) if augment is not None else (real, zero_aug(micro_f))
                else:
                    yr, aug_r = daug(real), zero_aug(micro_f)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit_real = disc(f_fwd(yr + n, sigma.flatten(), labels, augment_labels=aug_r, mode=enc_mode))
                loss_d = d_loss(logit_real, logit_fake, weight) * args.lsd / args.batch
                loss = loss + loss_d
                stats["loss_d"] += loss_d.detach()
                stats["logit_r"] += sum(m.detach().mean() for m in logit_real) / len(logit_real) / accum_f
                stats["logit_f"] += sum(m.detach().mean() for m in logit_fake) / len(logit_fake) / accum_f
            loss.backward()
            stats["loss_f"] += loss_f.detach()
          for p in f_params:
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
          if args.use_gan:
            torch.nn.utils.clip_grad_value_(f_params, 1)
          opt_f.step()
          opt_f.zero_grad(set_to_none=True)

        # ---------------- generator update ----------------
        fake_score.eval().requires_grad_(False)
        if head is not None:
            head.requires_grad_(False)
        G.train().requires_grad_(True)
        opt_g.zero_grad(set_to_none=True)
        for _ in range(accum):
            _, labels = data.sample(args.micro)
            k = int(torch.randint(0, K, ()).item())
            zs = gen_input(labels, k)
            images = gen.run(g_fwd, labels, zs, upto=k, grad_last=True)
            rnd_t = torch.rand([args.micro, 1, 1, 1], device=device) * args.tmax / 1000
            sigma = (max_inv + (1 - rnd_t) * (min_inv - max_inv)) ** rho
            y = images
            n = torch.randn_like(y) * sigma
            if args.gloss == "sid":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y_real = t_fwd(y + n, sigma.flatten(), labels, augment_labels=zero_aug(args.micro))
                    if args.use_gan and d_augment is None:
                        y_fake, feats_f = f_fwd(y + n, sigma.flatten(), labels, augment_labels=zero_aug(args.micro), mode=both_mode)
                        y_D = disc(feats_f)
                    else:
                        y_fake = f_fwd(y + n, sigma.flatten(), labels, augment_labels=zero_aug(args.micro))
                        if args.use_gan:  # discriminator sees the (differentiably) augmented generator output
                            y_D = disc(f_fwd(daug(y) + n, sigma.flatten(), labels, augment_labels=zero_aug(args.micro), mode=enc_mode))
                with torch.no_grad():
                    wf = (y - y_real).abs().mean(dim=[1, 2, 3], keepdim=True).clamp(min=1e-5)
                loss_g = ((y_real - y_fake) * ((y_real - y) - args.alpha * (y_real - y_fake)) / wf).sum() * args.lsg / args.batch
            else:  # DMD: gradient (D_fake - D_real) / |x - D_real| applied to x, no backprop through score nets
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    y_real = t_fwd(y.detach() + n, sigma.flatten(), labels, augment_labels=zero_aug(args.micro)).float()
                    y_fake = f_fwd(y.detach() + n, sigma.flatten(), labels, augment_labels=zero_aug(args.micro)).float()
                if args.use_gan:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y_D = disc(f_fwd(daug(y) + n, sigma.flatten(), labels, augment_labels=zero_aug(args.micro), mode=enc_mode))
                wf = (y.detach() - y_real).abs().mean(dim=[1, 2, 3], keepdim=True).clamp(min=1e-5)
                g = torch.nan_to_num((y_fake - y_real) / wf)
                loss_g = (y * g).sum() * args.lsg / args.batch
            total_g = loss_g
            if args.use_gan:
                loss_gan = g_gan_loss(y_D, wf)
                loss_gan = loss_gan.sum() * args.lsg_gan / args.batch
                stats["loss_g_gan"] += loss_gan.detach()
                if nimg > args.gan_start_kimg * 1e3:
                    total_g = 0.5 * loss_g + 0.5 * loss_gan
            if G_ref is not None and K == 1 and args.regress_w > 0:
                # progressive-distillation-style target: the multi-step reference generator evaluated
                # with the same first-step noise (later-step noise is a fixed rotation of it)
                with torch.no_grad():
                    z_ref = [zs[0]] + [torch.rot90(zs[0], j, dims=(2, 3)) for j in range(1, len(ref_gen.sigmas))]
                    x_ref = ref_gen.run(G_ref, labels, z_ref).float()
                loss_reg = ((images.float() - x_ref) ** 2).sum() * args.regress_w / args.batch
                stats["loss_reg"] += loss_reg.detach()
                total_g = total_g + loss_reg
            total_g.backward()
            stats["loss_g"] += loss_g.detach()
        for p in G.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
        if nimg >= g_freeze_until:
            opt_g.step()
        opt_g.zero_grad(set_to_none=True)
        n_stats += 1

        nimg += args.batch
        period = int(args.fake_reset_kimg * 1e3)
        # resets at absolute multiples of the period, so the schedule survives resuming from a training state
        if period > 0 and nimg // period > (nimg - args.batch) // period:
            if args.reset_fake_score:
                # re-initialise the fake score / discriminator from the teacher and let it re-adapt
                # to the current generator before the generator is updated again
                fake_score.load_state_dict(teacher_sd)
                opt_f.state.clear()
                g_freeze_until = nimg + args.g_start_kimg * 1e3
            if args.restart_from_ema:
                # an EMA generator becomes the raw generator, optimiser state is dropped
                # (with the fake-score reset: same as launching a new run initialised from that EMA)
                src = extra_emas[extra_hl.index(args.restart_ema)] if args.restart_ema else G_ema
                G.load_state_dict(src.state_dict())
                if src is not G_ema:
                    G_ema.load_state_dict(src.state_dict())
                opt_g.state.clear()
                if args.reset_extra_emas:
                    for e in extra_emas:
                        e.load_state_dict(src.state_dict())
            print(f"reset at {nimg/1e3:.0f} kimg (fake={args.reset_fake_score}, G<-EMA={args.restart_from_ema})", flush=True)
        with torch.no_grad():
            g_params = [p.detach() for p in G.parameters()]
            for e, hl_k, ramp in [(G_ema, args.ema_halflife_kimg, args.ema_rampup)] + list(zip(extra_emas, extra_hl, extra_ramp)):
                hl = min(hl_k * 1e3, nimg * ramp) if ramp > 0 else hl_k * 1e3
                beta = 0.5 ** (args.batch / max(hl, 1e-8))
                torch._foreach_lerp_(list(e.parameters()), g_params, 1 - beta)

        if nimg % (args.batch * args.log_every) == 0:
            dt = time.time() - t_last
            rec = {k: v.item() / max(n_stats, 1) for k, v in stats.items()}
            rec.update(kimg=nimg / 1e3, img_per_s=(nimg - n_last) / dt,
                       peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                       elapsed_h=(time.time() - t_start) / 3600, lr_factor=lf)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            t_last, n_last = time.time(), nimg
            stats = {k: torch.zeros((), device=device) for k in stats}
            n_stats = 0

        if nimg >= next_snap or nimg >= total:
            next_snap += sn
            torch.save(dict(G=G.state_dict(), G_ema=G_ema.state_dict(), fake=fake_score.state_dict(),
                            head=head.state_dict() if head is not None else None,
                            G_ema_extra={hl: e.state_dict() for e, hl in zip(extra_emas, extra_hl)},
                            opt_g=opt_g.state_dict(), opt_f=opt_f.state_dict(), nimg=nimg), str(state_path) + ".tmp")
            os.replace(str(state_path) + ".tmp", state_path)
            torch.save(dict(G_ema=G_ema.state_dict(), sigmas=gen.sigmas, teacher_args=vars(targs), generator_args=vars(gargs), nimg=nimg,
                            ema_halflife_kimg=args.ema_halflife_kimg,
                            G_ema_extra={hl: e.state_dict() for e, hl in zip(extra_emas, extra_hl)}),
                       out / f"gen-{nimg//1000:06d}.pt")

        if args.eval_kimg > 0 and (nimg >= next_eval or nimg >= total):
            next_eval += ev
            from fid_utils import clean_holdout, color_stats, color_stats_dir, fid_of_sampler, fid_subset, save_grid_from_dir
            clean_pos = clean_holdout(args.data) if args.fid_stats == "pk_holdout" else None
            if "ref_color" not in locals():
                ref_color = color_stats(data.images.permute(0, 2, 3, 1).cpu().numpy())
            t0 = time.time()
            for kk in sorted({K - 1} | ({0} if args.eval_all_steps else set())):
                for net_e, hl_k in [(G_ema, args.ema_halflife_kimg)] + (list(zip(extra_emas, extra_hl)) if kk == K - 1 else []):
                    g = torch.Generator(device=device).manual_seed(1234)
                    fn = lambda c: gen.run(net_e, c, [torch.randn(len(c), 3, 64, 64, device=device, generator=g) for _ in range(kk + 1)], upto=kk).clamp(-1, 1)
                    score = fid_of_sampler(fn, data.holdout_labels, args.fid_stats, batch_size=128, keep_dir=str(out / "fid_tmp"))
                    if hl_k == args.ema_halflife_kimg:
                        save_grid_from_dir(out / "fid_tmp", out / "samples" / f"{nimg//1000:06d}_nfe{kk + 1}.png")
                    cs = color_stats_dir(out / "fid_tmp")
                    rec = dict(kimg=nimg / 1e3, nfe=kk + 1, ema_kimg=hl_k, fid_holdout=score, eval_s=time.time() - t0,
                               contaminated=args.subset != "dev",
                               contrast_ratio=cs["contrast"] / ref_color["contrast"], sat_ratio=cs["sat"] / ref_color["sat"])
                    if clean_pos is not None:  # de-duplicated hold-out subset (no near-copies of dev-train images)
                        rec["fid_holdout_clean"] = fid_subset(out / "fid_tmp", clean_pos, "pk_holdout_clean")
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(json.dumps(rec), flush=True)
            t_last = time.time()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--teacher_ema", type=int, default=1)
    p.add_argument("--init_generator", default="")
    p.add_argument("--init_ema", type=float, default=0, help="with --init_generator: half-life key of an extra EMA to load (0: main EMA)")
    p.add_argument("--generator_teacher", default="", help="diffusion snapshot defining the student architecture/init (default: --teacher)")
    p.add_argument("--generator_teacher_ema", type=int, default=0)
    p.add_argument("--regress_to", default="", help="multi-step generator snapshot used as a regression target (1-step only)")
    p.add_argument("--regress_w", type=float, default=0.0)
    p.add_argument("--data", default="data_cache/train64.npz")
    p.add_argument("--subset", choices=["dev", "all"], default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--gloss", choices=["sid", "dmd"], default="sid")
    p.add_argument("--fake_steps", type=int, default=1, help="fake-score updates per generator update (TTUR)")
    p.add_argument("--guide", choices=["none", "auto", "cfg"], default="none", help="guidance applied to the teacher score")
    p.add_argument("--guide_w", type=float, default=1.0)
    p.add_argument("--guide_ref", default="", help="weaker teacher snapshot for autoguidance")
    p.add_argument("--guide_ref_ema", type=int, default=1)
    p.add_argument("--xflip", type=int, default=1)
    p.add_argument("--class_power", type=float, default=1.0, help="class sampling prob ~ count**power")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--micro", type=int, default=32)
    p.add_argument("--micro_f", type=int, default=0, help="micro-batch of the fake-score step (0 = --micro)")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--glr", type=float, default=5e-6)
    p.add_argument("--lr_decay_start_kimg", type=float, default=0, help="linear lr decay (both networks) from here ...")
    p.add_argument("--lr_decay_end_kimg", type=float, default=0, help="... to here (0: constant lr)")
    p.add_argument("--lr_final_ratio", type=float, default=0.1, help="lr multiplier reached at --lr_decay_end_kimg")
    p.add_argument("--g_beta1", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tmax", type=float, default=800)
    p.add_argument("--init_sigma", type=float, default=2.5)
    p.add_argument("--sigmas", default="", help="comma list for multi-step generation, e.g. 80,1.5")
    p.add_argument("--P_mean", type=float, default=-1.2)
    p.add_argument("--P_std", type=float, default=1.2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--augment_p", type=float, default=0.15)
    p.add_argument("--d_augment_p", type=float, default=0, help="discriminator-only augmentation probability (not used for the denoising loss)")
    p.add_argument("--d_augment_color", type=int, default=0, help="add brightness/contrast/saturation to the discriminator augmentation")
    p.add_argument("--use_gan", type=int, default=1)
    p.add_argument("--disc_head", choices=["mean", "dmd"], default="mean", help="SiDA channel-mean logits or DMD2 conv head")
    p.add_argument("--disc_scales", default="8", help="encoder resolutions used as discriminator logit maps, e.g. 8,16,32")
    p.add_argument("--ls", type=float, default=1.0)
    p.add_argument("--lsd", type=float, default=1.0)
    p.add_argument("--lsg", type=float, default=100.0)
    p.add_argument("--lsg_gan", type=float, default=0.01)
    p.add_argument("--gan_start_kimg", type=float, default=200)
    p.add_argument("--g_start_kimg", type=float, default=100)
    p.add_argument("--fake_reset_kimg", type=float, default=0, help="periodically re-initialise the fake score from the teacher")
    p.add_argument("--restart_from_ema", type=int, default=0, help="at each reset also copy G_ema into G and clear the G optimiser")
    p.add_argument("--reset_fake_score", type=int, default=1, help="at each reset re-initialise the fake score from the teacher (0: EMA switch only)")
    p.add_argument("--reset_extra_emas", type=int, default=1, help="with --restart_from_ema: also restart the extra EMAs from the main EMA")
    p.add_argument("--restart_ema", type=float, default=0, help="with --restart_from_ema: half-life key of the extra EMA used as restart source (0: main EMA)")
    p.add_argument("--ema_halflife_kimg", type=float, default=500)
    p.add_argument("--ema_rampup", type=float, default=0.05)
    p.add_argument("--extra_ema_kimg", default="100,2000", help="additional generator EMAs, evaluated too: comma list of HL or HL@R (half-life kimg, rampup ratio)")
    p.add_argument("--duration_mimg", type=float, default=100)
    p.add_argument("--eval_kimg", type=float, default=1000)
    p.add_argument("--eval_all_steps", type=int, default=0)
    p.add_argument("--fid_stats", default="pk_holdout")
    p.add_argument("--snap_kimg", type=float, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
