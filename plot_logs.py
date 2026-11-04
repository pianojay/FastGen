"""Plot training curves (loss, hold-out FID) from log.jsonl files for the write-up.

Uses only Pillow (part of the provided environment) so no extra package is needed.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if "fid_holdout" not in r], [r for r in rows if "fid_holdout" in r]


def panel(series, title, size=(520, 360), ylim=None):
    """series: list of (label, xs, ys). Returns a PIL image with axes, ticks and a legend."""
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 56, 12, 28, 40
    xs_all = [x for _, xs, _ in series for x in xs] or [0, 1]
    ys_all = [y for _, _, ys in series for y in ys] or [0, 1]
    x0, x1 = min(xs_all), max(xs_all)
    y0, y1 = ylim if ylim else (min(ys_all), max(ys_all))
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1
    sx = lambda x: left + (x - x0) / (x1 - x0) * (w - left - right)
    sy = lambda y: h - bottom - (min(max(y, y0), y1) - y0) / (y1 - y0) * (h - top - bottom)
    d.rectangle([left, top, w - right, h - bottom], outline=(0, 0, 0))
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        d.line([left, sy(yv), w - right, sy(yv)], fill=(225, 225, 225))
        d.text((4, sy(yv) - 6), f"{yv:.3g}", fill=(0, 0, 0))
        xv = x0 + (x1 - x0) * i / 4
        d.text((sx(xv) - 10, h - bottom + 6), f"{xv:.3g}", fill=(0, 0, 0))
    d.text((w // 2 - 20, h - 16), "Mimg", fill=(0, 0, 0))
    d.text((left, 8), title, fill=(0, 0, 0))
    for k, (label, xs, ys) in enumerate(series):
        c = COLORS[k % len(COLORS)]
        pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            d.line(pts, fill=c, width=2)
        for p in pts if len(pts) < 60 else []:
            d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=c)
        d.text((left + 8, top + 6 + 12 * k), label[:60], fill=c)
    return img


def main(args):
    runs = [Path(r) for r in args.runs]
    loss_series, fid_series = [], []
    for run in runs:
        train, fid = load(run / "log.jsonl")
        if args.max_mimg:
            train = [r for r in train if r.get("kimg", 0) <= args.max_mimg * 1e3]
            fid = [r for r in fid if r["kimg"] <= args.max_mimg * 1e3]
        for key in args.keys.split(","):
            pts = [(r["kimg"] / 1e3, r[key]) for r in train if key in r]
            if pts:
                ys = [p[1] for p in pts]
                if args.smooth > 1:  # trailing moving average over logged points
                    ys = [sum(ys[max(0, i - args.smooth + 1):i + 1]) / len(ys[max(0, i - args.smooth + 1):i + 1]) for i in range(len(ys))]
                loss_series.append((f"{run.name}:{key}", [p[0] for p in pts], ys))
        groups = {}
        for r in fid:
            groups.setdefault(r.get("ema_halflife_kimg", r.get("ema_kimg", r.get("nfe"))), []).append(r)
        for g, rs in groups.items():
            if args.ema and float(g) != args.ema:
                continue
            fid_series.append((f"{run.name} [EMA {g}]", [r["kimg"] / 1e3 for r in rs], [r["fid_holdout"] for r in rs]))
    ylim = tuple(map(float, args.ylim.split(","))) if args.ylim else None
    vals = sorted(v for _, _, ys in loss_series for v in ys)
    loss_lim = (vals[int(0.02 * (len(vals) - 1))], vals[int(0.98 * (len(vals) - 1))]) if vals else None
    a = panel(loss_series, "training loss" + (f" (moving average {args.smooth})" if args.smooth > 1 else ""), ylim=loss_lim)
    b = panel(fid_series, args.fid_title, ylim=ylim)
    out = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    out.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--keys", default="loss")
    p.add_argument("--ylim", default="")
    p.add_argument("--smooth", type=int, default=1, help="moving-average window (logged points) for the losses")
    p.add_argument("--fid_title", default="FID (train-split hold-out)")
    p.add_argument("--max_mimg", type=float, default=0, help="plot only up to this many Mimg (0: everything)")
    p.add_argument("--ema", type=float, default=0, help="plot only the hold-out FID of this EMA half-life (0: all)")
    p.add_argument("--out", default="curves.png")
    main(p.parse_args())
