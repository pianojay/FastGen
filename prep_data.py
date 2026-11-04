"""Cache the Pokemon splits as uint8 tensors for fast GPU-resident training.

Images are loaded through the provided ``PokemonDataset`` (same resize and
normalisation as the evaluation reference), then mapped back to exact uint8
values.  The training cache contains ONLY ``train_split.txt`` images.  A
separate validation cache is written for final FID reporting only; no training
or selection script reads it.

A development hold-out (20 images per class, drawn from the TRAIN split) is
also stored so that hyper-parameters can be selected without touching the
validation split.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PokemonDataModule


def to_uint8(loader):
    images, labels = [], []
    for x, y in loader:
        u8 = ((x + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
        images.append(u8)
        labels.append(y)
    return torch.cat(images).numpy(), torch.cat(labels).numpy()


def main(args):
    module = PokemonDataModule(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=256,
        num_workers=16,
        return_category=True,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(module.train_dataset, batch_size=256, shuffle=False, num_workers=16)
    x_train, y_train = to_uint8(train_loader)
    num_classes = len(module.category_to_id)

    rng = np.random.default_rng(args.seed)
    holdout = []
    for c in range(num_classes):
        idx = np.flatnonzero(y_train == c)
        take = min(args.holdout_per_class, len(idx) // 3)
        holdout.extend(rng.choice(idx, size=take, replace=False).tolist())
    holdout = np.sort(np.array(holdout, dtype=np.int64))
    is_holdout = np.zeros(len(y_train), dtype=bool)
    is_holdout[holdout] = True

    np.savez(
        out / "train64.npz",
        images=x_train,
        labels=y_train,
        holdout=holdout,
        dev_train=np.flatnonzero(~is_holdout),
        num_classes=num_classes,
    )
    counts = np.bincount(y_train, minlength=num_classes)
    print(f"train: {x_train.shape} classes={num_classes} per-class min={counts.min()} max={counts.max()} "
          f"holdout={len(holdout)} dev_train={int((~is_holdout).sum())}")

    # clean-fid reference statistics of the hold-out, used by the in-training hold-out FID of train_edm / train_sida
    from fid_utils import ensure_custom_stats, save_uint8_images
    with tempfile.TemporaryDirectory() as tmp:
        save_uint8_images(x_train[holdout], tmp)
        ensure_custom_stats("pk_holdout", tmp)

    if args.val:
        val_loader = DataLoader(module.val_dataset, batch_size=256, shuffle=False, num_workers=16)
        x_val, y_val = to_uint8(val_loader)
        np.savez(out / "val64_REPORT_ONLY.npz", images=x_val, labels=y_val)
        print(f"val: {x_val.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--out_dir", default="./data_cache")
    parser.add_argument("--holdout_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val", action="store_true", help="also cache val split (FID reporting only)")
    main(parser.parse_args())
