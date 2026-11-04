"""Score a submission checkpoint on the TRAIN-split hold-out (model selection only).

Uses Model.load_checkpoint + Model.sample exactly as evaluate.py does, but with
the hold-out class composition and the ``pk_holdout`` clean-fid statistics.
"""

import argparse

import numpy as np
import torch

from fid_utils import fid_of_sampler
from model import Model


def main(args):
    model = Model.load_checkpoint(args.ckpt, evaluate_mode=args.mode, device="cuda")
    d = np.load(args.data)
    labels = torch.from_numpy(d["labels"][d["holdout"]].astype(np.int64))
    scores = []
    for r in range(args.repeats):
        model._generator = torch.Generator(device="cuda").manual_seed(args.seed + r)
        fn = lambda c: model.sample((len(c), 3, 64, 64), device="cuda", category=c)
        scores.append(fid_of_sampler(fn, labels, "pk_holdout", batch_size=32))
    print(f"{args.ckpt} {args.mode} params={model.count_parameters():,} fid_holdout={np.mean(scores):.3f} "
          f"(runs: {', '.join(f'{s:.3f}' for s in scores)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", choices=["one_nfe", "few_nfe"], required=True)
    p.add_argument("--data", default="data_cache/train64.npz")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
