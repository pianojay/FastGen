"""FID helpers that reproduce the official ``evaluate.py`` pipeline.

Images are written as PNG exactly like ``evaluate.generate_samples`` (float in
[-1, 1] -> ``to_pil_image`` truncation) and scored with clean-fid folder mode.
Reference statistics can be cached as clean-fid custom stats; the numbers are
identical to folder-vs-folder evaluation because the same features are used.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def save_float_images(samples, out_dir, start=0):
    """Save [-1, 1] tensors the same way evaluate.py does."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        image = ((sample.float().cpu() + 1) / 2).clamp(0, 1)
        to_pil_image(image).convert("RGB").save(out_dir / f"{start + i:05d}.png")


def save_uint8_images(images, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(np.ascontiguousarray(img.transpose(1, 2, 0))).save(out_dir / f"{i:05d}.png")


def ensure_custom_stats(name, ref_dir):
    if not fid.test_stats_exists(name, mode="clean"):
        fid.make_custom_stats(name, str(ref_dir), mode="clean", num_workers=8)


def fid_against_stats(gen_dir, name, batch_size=64):
    return float(fid.compute_fid(str(gen_dir), dataset_name=name, mode="clean",
                                 dataset_split="custom", batch_size=batch_size, num_workers=8))


def fid_dirs(dir_a, dir_b, batch_size=64):
    return float(fid.compute_fid(str(dir_a), str(dir_b), batch_size=batch_size, num_workers=4))


@torch.no_grad()
def fid_of_sampler(sample_fn, labels, stats_name, batch_size=256, keep_dir=None, device="cuda"):
    """Generate one image per entry of ``labels`` with ``sample_fn(labels)`` and score it."""
    tmp = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="fidgen_", dir=os.environ.get("FID_TMP", "/tmp")))
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    count = 0
    for start in range(0, len(labels), batch_size):
        batch = labels[start:start + batch_size].to(device)
        save_float_images(sample_fn(batch), tmp, start=count)
        count += len(batch)
    score = fid_against_stats(tmp, stats_name)
    if not keep_dir:
        shutil.rmtree(tmp)
    return score


def fid_subset(gen_dir, positions, stats_name, batch_size=64):
    """FID of the images ``{pos:05d}.png`` (pos in ``positions``) of ``gen_dir`` against ``stats_name``.

    Used for the de-duplicated hold-out subset: the files are generated in hold-out order, so the
    subset keeps the matching class composition without generating new images."""
    sub = Path(tempfile.mkdtemp(prefix="fidsub_", dir=os.environ.get("FID_TMP", "/tmp")))
    try:
        for p in positions:
            os.symlink(Path(gen_dir).resolve() / f"{int(p):05d}.png", sub / f"{int(p):05d}.png")
        return fid_against_stats(sub, stats_name, batch_size=batch_size)
    finally:
        shutil.rmtree(sub)


def clean_holdout(data_path="data_cache/train64.npz"):
    """Positions of the de-duplicated hold-out subset (None if not prepared on this host)."""
    path = Path(data_path).with_name("holdout_clean.npy")
    if path.exists() and fid.test_stats_exists("pk_holdout_clean", mode="clean"):
        return np.load(path)
    return None


def save_grid_from_dir(img_dir, out_png, n=16, cols=8):
    """Tile the first ``n`` PNGs of ``img_dir`` (e.g. an evaluation folder) into one grid image."""
    files = sorted(Path(img_dir).glob("*.png"))
    files = files[::max(1, len(files) // n)][:n]  # spread over classes (files are class-sorted)
    if not files:
        return
    tiles = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    h, w, _ = tiles[0].shape
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_png)


def color_stats(images_u8):
    """Brightness / saturation / contrast of an (N, H, W, 3) uint8 array (drift diagnostic vs training data)."""
    x = images_u8.astype(np.float32) / 255
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return dict(bright=float(x.mean()), sat=float(sat.mean()), contrast=float(x.std((1, 2, 3)).mean()))


def color_stats_dir(img_dir):
    files = sorted(Path(img_dir).glob("*.png"))
    return color_stats(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]))


def background_stats(images_u8):
    """Fraction of images whose border is (near-)white / dark: matches the dataset's white-background convention."""
    x = images_u8.astype(np.float32) / 255
    border = np.concatenate([x[:, :4].reshape(len(x), -1), x[:, -4:].reshape(len(x), -1),
                             x[:, :, :4].reshape(len(x), -1), x[:, :, -4:].reshape(len(x), -1)], 1)
    m = border.mean(1)
    return dict(white_bg=float((m > 0.9).mean()), dark_bg=float((m < 0.25).mean()))
