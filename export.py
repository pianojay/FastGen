"""Export a distilled generator snapshot (train_sida.py) to the submission checkpoint format."""

import argparse
from pathlib import Path

import torch

import model as M


def main(args):
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    cls = M.ModelOneNFE if args.mode == "one_nfe" else M.ModelFewNFE
    net = cls(device="cpu")
    sigmas = torch.tensor(snap["sigmas"], dtype=torch.float32)
    if sigmas.shape != net.sigmas.shape:
        raise ValueError(f"snapshot has {len(sigmas)} steps but model.py expects {len(net.sigmas)}")
    if args.mode == "one_nfe" and len(sigmas) != 1:
        raise ValueError("one_nfe must use exactly one step")
    if args.mode == "few_nfe" and not 2 <= len(sigmas) <= 4:
        raise ValueError("few_nfe must use 2-4 steps")
    sd = snap["G_ema"]
    if args.ema_kimg:
        sd = snap["G_ema_extra"][float(args.ema_kimg)] if float(args.ema_kimg) != snap.get("ema_halflife_kimg") else sd
    net.backbone.load_state_dict(sd)
    net.sigmas.copy_(sigmas)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict()}, out)
    # round-trip through the official loader: the noise ladder must survive
    loaded = M.Model.load_checkpoint(str(out), evaluate_mode=args.mode, device="cpu")
    if not torch.equal(loaded.sigmas, sigmas):
        raise RuntimeError(f"sigma ladder mismatch after reload: {loaded.sigmas.tolist()} vs {sigmas.tolist()}")
    print(f"{args.mode}: {net.count_parameters():,} params, sigmas={loaded.sigmas.tolist()} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--ema_kimg", default="", help="pick an extra EMA half-life stored in the snapshot")
    p.add_argument("--mode", choices=["one_nfe", "few_nfe"], required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
