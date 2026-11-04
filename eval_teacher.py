"""Hold-out FID of a teacher snapshot with the EDM Heun sampler (optionally guided)."""

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from fid_utils import clean_holdout, fid_of_sampler, fid_subset
from train_edm import GuidedDenoiser, build_net, edm_sampler


def load(path, ema_idx, num_classes, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(argparse.Namespace(**ck["args"]), num_classes)
    net.load_state_dict(ck["emas"][ema_idx])
    return net.to(device).eval().requires_grad_(False)


def main(args):
    device = torch.device("cuda")
    d = np.load(args.data)
    labels = torch.from_numpy(d["labels"][d["holdout"]].astype(np.int64)).to(device)
    nc = int(d["num_classes"])
    main_net = load(args.snapshot, args.ema, nc, device)
    ref = load(args.ref, args.ref_ema, nc, device) if args.guide == "auto" else None
    clean_pos = clean_holdout(args.data)
    for w in [float(v) for v in args.w.split(",")]:
        net = GuidedDenoiser(main_net, ref, w, args.guide)
        g = torch.Generator(device=device).manual_seed(args.seed)
        keep = tempfile.mkdtemp(prefix="evt_", dir=os.environ.get("FID_TMP", "/tmp"))
        score = fid_of_sampler(lambda c: edm_sampler(net, c, num_steps=args.steps, generator=g), labels,
                               "pk_holdout", batch_size=args.batch, keep_dir=keep)
        rec = dict(snapshot=args.snapshot, ema=args.ema, guide=args.guide, ref=args.ref, ref_ema=args.ref_ema, w=w,
                   steps=args.steps, fid_holdout=score)
        if clean_pos is not None:
            rec["fid_holdout_clean"] = fid_subset(keep, clean_pos, "pk_holdout_clean")
        shutil.rmtree(keep)
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(f"{args.out}/results.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--guide", choices=["none", "auto", "cfg"], default="none")
    p.add_argument("--ref", default="")
    p.add_argument("--ref_ema", type=int, default=1)
    p.add_argument("--w", default="1.0")
    p.add_argument("--steps", type=int, default=18)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--data", default="data_cache/train64.npz")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="", help="optional run dir; results are appended to OUT/results.jsonl")
    main(p.parse_args())
