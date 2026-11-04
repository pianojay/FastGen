"""Train the class-conditional EDM teacher (auxiliary network, not used at sampling time).

Recipe follows EDM (Karras et al., NeurIPS 2022) for 64x64 small datasets
(DDPM++ backbone, lognormal noise sampling, non-leaky ADA-style augmentation,
EMA of weights), re-implemented for single-GPU training with gradient
accumulation, bf16 autocast and a GPU-resident uint8 dataset.
Only images from ``train_split.txt`` are used (see prep_data.py).
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

from augment import AugmentPipe
from networks import EDMPrecond


def parse_list(s, typ=int):
    return tuple(typ(v) for v in s.split(",") if v != "")


class GuidedDenoiser(torch.nn.Module):
    """Autoguidance (Karras et al., NeurIPS 2024) or classifier-free guidance around an EDM denoiser.

    D = D_ref + w (D_main - D_ref), where D_ref is a weaker model (autoguidance) or the
    unconditional prediction of the same model (CFG, requires label dropout at training time).
    """

    def __init__(self, main, ref=None, w=1.0, mode="none"):
        super().__init__()
        self.main, self.ref, self.w, self.mode = main, ref, w, mode
        self.img_channels, self.img_resolution = main.img_channels, main.img_resolution

    def forward(self, x, sigma, labels, augment_labels=None):
        d = self.main(x, sigma, labels, augment_labels=augment_labels)
        if self.mode == "none" or self.w == 1.0:
            return d
        if self.mode == "auto":
            d_ref = self.ref(x, sigma, labels, augment_labels=augment_labels)
        else:
            d_ref = self.main(x, sigma, None, augment_labels=augment_labels)
        return d_ref + self.w * (d - d_ref)


@torch.no_grad()
def edm_sampler(net, labels, num_steps=18, sigma_min=0.002, sigma_max=80.0, rho=7.0, generator=None):
    """Deterministic 2nd-order Heun sampler of EDM (Algorithm 1)."""
    device = labels.device
    latents = torch.randn([len(labels), net.img_channels, net.img_resolution, net.img_resolution],
                          device=device, generator=generator)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    def denoise(x, t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net(x, t, labels).to(torch.float64)

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        d_cur = (x_cur - denoise(x_cur.float(), t_cur.float())) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next.float(), t_next.float())) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.float().clamp(-1, 1)


class GPUData:
    """uint8 images resident on the GPU with random x-flip.

    ``class_power`` p samples class c with probability proportional to n_c ** p
    (p = 1: natural frequencies, p = 0: uniform over classes).
    """

    def __init__(self, npz_path, subset, device, xflip=True, class_power=1.0):
        d = np.load(npz_path)
        idx = d["dev_train"] if subset == "dev" else np.arange(len(d["labels"]))
        self.images = torch.from_numpy(d["images"][idx]).to(device)
        self.labels = torch.from_numpy(d["labels"][idx].astype(np.int64)).to(device)
        self.num_classes = int(d["num_classes"])
        self.holdout_labels = torch.from_numpy(d["labels"][d["holdout"]].astype(np.int64)).to(device)
        self.xflip = xflip
        self.device = device
        self.weights = None
        if class_power != 1.0:
            counts = torch.bincount(self.labels, minlength=self.num_classes).float()
            self.weights = counts[self.labels] ** (class_power - 1.0)

    def __len__(self):
        return len(self.labels)

    def sample(self, n, generator=None):
        if self.weights is None:
            i = torch.randint(0, len(self.labels), (n,), device=self.device, generator=generator)
        else:
            i = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        x = self.images[i].to(torch.float32) / 127.5 - 1
        if self.xflip:
            flip = torch.rand(n, device=self.device, generator=generator) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x, self.labels[i]


AUG_DIM = 8  # yflip, scale, rotate (2), aniso (2), translate (2); x-flip is a plain data augmentation


def build_net(args, num_classes):
    return EDMPrecond(
        img_resolution=64, img_channels=3, label_dim=num_classes,
        augment_dim=AUG_DIM if args.augment_p > 0 else 0,
        model_channels=args.channels, channel_mult=parse_list(args.channel_mult),
        num_blocks=args.num_blocks, attn_resolutions=parse_list(args.attn),
        dropout=args.dropout, label_dropout=args.label_dropout,
    )


def fid_eval(net, labels, stats_name, steps, tmpdir):
    from fid_utils import fid_of_sampler
    gen = torch.Generator(device=labels.device).manual_seed(1234)
    return fid_of_sampler(lambda c: edm_sampler(net, c, num_steps=steps, generator=gen), labels, stats_name,
                          batch_size=128, keep_dir=tmpdir)


def main(args):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    data = GPUData(args.data, args.subset, device, xflip=args.xflip, class_power=args.class_power)
    net = build_net(args, data.num_classes).to(device).train().requires_grad_(True)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params:,}  train images: {len(data)}", flush=True)

    augment = None
    if args.augment_p > 0:
        augment = AugmentPipe(p=args.augment_p, xflip=0, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, fused=True)
    halflives = parse_list(args.ema_halflives_kimg, float)
    emas = [copy.deepcopy(net).eval().requires_grad_(False) for _ in halflives]

    nimg = 0
    state_path = out / "training-state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)  # CPU: avoids a transient VRAM spike
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        for i, e in enumerate(emas):
            # EMAs added after the state was saved start from the longest saved EMA (not the raw weights)
            e.load_state_dict(st["emas"][min(i, len(st["emas"]) - 1)])
        nimg = st["nimg"]
        print(f"resumed at {nimg/1e6:.2f} Mimg", flush=True)

    fwd = torch.compile(net) if args.compile else net
    accum = args.batch // args.micro
    assert accum * args.micro == args.batch
    log = open(out / "log.jsonl", "a")
    t_start = time.time()
    t_last, n_last = time.time(), nimg
    total = int(args.duration_mimg * 1e6)
    next_eval = (nimg // int(args.eval_kimg * 1e3) + 1) * int(args.eval_kimg * 1e3)
    next_snap = (nimg // int(args.snap_kimg * 1e3) + 1) * int(args.snap_kimg * 1e3)
    torch.cuda.reset_peak_memory_stats()
    loss_acc, loss_n = torch.zeros((), device=device), 0
    while nimg < total:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = data.sample(args.micro)
            rnd = torch.randn([args.micro, 1, 1, 1], device=device)
            sigma = (rnd * args.P_std + args.P_mean).exp()
            weight = (sigma ** 2 + 0.25) / (sigma * 0.5) ** 2
            if augment is not None:
                x, aug = augment(x)
            else:
                aug = None
            noisy = x + torch.randn_like(x) * sigma
            with torch.autocast("cuda", dtype=torch.bfloat16):
                D = fwd(noisy, sigma.flatten(), y, augment_labels=aug)
            loss = (weight * (D - x) ** 2).sum() / args.batch
            loss.backward()
            loss_acc += loss.detach() * (args.batch / args.micro / (3 * 64 * 64))
            loss_n += 1
        lr = args.lr * min(1.0, (nimg + args.batch) / max(args.lr_warmup_kimg * 1e3, 1e-8))
        if args.lr_decay_kimg > 0:
            lr = lr / math.sqrt(max((nimg + args.batch) / (args.lr_decay_kimg * 1e3), 1.0))
        if args.cooldown_end_kimg > args.cooldown_start_kimg and nimg > args.cooldown_start_kimg * 1e3:
            # linear cooldown to cooldown_ratio x lr between cooldown_start_kimg and cooldown_end_kimg
            t = min(1.0, (nimg - args.cooldown_start_kimg * 1e3) / ((args.cooldown_end_kimg - args.cooldown_start_kimg) * 1e3))
            lr = lr * (1.0 + t * (args.cooldown_ratio - 1.0))
        for g in opt.param_groups:
            g["lr"] = lr
        for p in net.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
        opt.step()
        nimg += args.batch
        with torch.no_grad():
            params = [p.detach() for p in net.parameters()]
            for e, hl in zip(emas, halflives):
                hl_nimg = min(hl * 1e3, nimg * args.ema_rampup)
                beta = 0.5 ** (args.batch / max(hl_nimg, 1e-8))
                torch._foreach_lerp_(list(e.parameters()), params, 1 - beta)

        if nimg % (args.batch * args.log_every) == 0:
            dt = time.time() - t_last
            rec = dict(kimg=nimg / 1e3, loss=loss_acc.item() / max(loss_n, 1), lr=lr,
                       img_per_s=(nimg - n_last) / dt, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                       peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30, elapsed_h=(time.time() - t_start) / 3600)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            t_last, n_last = time.time(), nimg
            loss_acc, loss_n = torch.zeros((), device=device), 0

        if nimg >= next_snap or nimg >= total:
            next_snap += int(args.snap_kimg * 1e3)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), emas=[e.state_dict() for e in emas],
                            nimg=nimg, args=vars(args)), str(state_path) + ".tmp")
            os.replace(str(state_path) + ".tmp", state_path)
            torch.save(dict(emas=[e.state_dict() for e in emas], halflives=halflives, nimg=nimg, args=vars(args)),
                       out / f"ema-{nimg//1000:06d}.pt")

        if args.eval_kimg > 0 and (nimg >= next_eval or nimg >= total):
            next_eval += int(args.eval_kimg * 1e3)
            for e, hl in zip(emas, halflives):
                t0 = time.time()
                score = fid_eval(e, data.holdout_labels, args.fid_stats, args.eval_steps, str(out / "fid_tmp"))
                from fid_utils import clean_holdout, fid_subset, save_grid_from_dir
                save_grid_from_dir(out / "fid_tmp", out / "samples" / f"{nimg//1000:06d}_ema{int(hl)}.png")
                rec = dict(kimg=nimg / 1e3, ema_halflife_kimg=hl, fid_holdout=score, steps=args.eval_steps,
                           eval_s=time.time() - t0, contaminated=args.subset != "dev")
                clean_pos = clean_holdout(args.data) if args.fid_stats == "pk_holdout" else None
                if clean_pos is not None:  # de-duplicated hold-out subset (no near-copies of dev-train images)
                    rec["fid_holdout_clean"] = fid_subset(out / "fid_tmp", clean_pos, "pk_holdout_clean")
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            t_last = time.time()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_cache/train64.npz")
    p.add_argument("--subset", choices=["dev", "all"], default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--channel_mult", default="1,2,2,2")
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--attn", default="16")
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--label_dropout", type=float, default=0.0)
    p.add_argument("--augment_p", type=float, default=0.15)
    p.add_argument("--xflip", type=int, default=1)
    p.add_argument("--class_power", type=float, default=1.0, help="class sampling prob ~ count**power")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--micro", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_warmup_kimg", type=float, default=10000)
    p.add_argument("--lr_decay_kimg", type=float, default=0)
    p.add_argument("--cooldown_start_kimg", type=float, default=0, help="linear lr cooldown from here ...")
    p.add_argument("--cooldown_end_kimg", type=float, default=0, help="... to here (0: none)")
    p.add_argument("--cooldown_ratio", type=float, default=0.05, help="lr multiplier reached at the end of the cooldown")
    p.add_argument("--P_mean", type=float, default=-1.2)
    p.add_argument("--P_std", type=float, default=1.2)
    p.add_argument("--ema_halflives_kimg", default="250,500,1000")
    p.add_argument("--ema_rampup", type=float, default=0.05)
    p.add_argument("--duration_mimg", type=float, default=200)
    p.add_argument("--eval_kimg", type=float, default=5000)
    p.add_argument("--eval_steps", type=int, default=18)
    p.add_argument("--fid_stats", default="pk_holdout")
    p.add_argument("--snap_kimg", type=float, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
