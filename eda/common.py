"""Shared utilities for dataset-only EDA."""
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/pokemon-generation-one-22k"
RES = ROOT / "eda/resources"
OUT = ROOT / "eda/reports/analysis"
CACHE = ROOT / "eda/features"
GPU_UUID = "GPU-e9d30df5-3b84-1a7e-c505-c6fff0a0ce7f"
SEED = 20261005


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    temp.replace(path)


def save_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def inputs():
    a = (RES / "A.txt").read_text().splitlines()
    b = (RES / "B.txt").read_text().splitlines()
    return a+b, len(a)


def verify():
    provenance = json.loads((RES / "provenance.json").read_text())
    for name, expected in provenance["protected_sha256"].items():
        assert sha(ROOT/name) == expected, name
    assert sha(RES/"development_split.json") == provenance["partition_sha256"]
    for line in (RES/"soflow-reference-requirements.txt").read_text().splitlines():
        if "==" in line and not line.startswith("--"):
            import importlib.metadata
            name, expected = line.split("==")
            assert importlib.metadata.version(name) == expected, name
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert Path(sys.prefix).resolve() == ROOT/".conda/eda"


@contextmanager
def gpu_lease():
    import torch
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    with Path("/tmp/fastgen-gpu0.lock").open("r") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        assert uuid == GPU_UUID
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        assert not any(GPU_UUID in row for row in apps.splitlines()), apps
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        save_json(OUT/"gpu_job.json", {"pid": os.getpid(), "gpu_uuid": uuid,
            "start_ticks": Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]})
        yield


def load_features():
    meta = json.loads((CACHE/"metadata.json").read_text())
    paths, na = inputs()
    assert meta["paths_sha256"] == hashlib.sha256("\n".join(paths).encode()).hexdigest()
    assert meta["feature_sha256"] == sha(CACHE/"inception.npy")
    x = np.load(CACHE/"inception.npy", mmap_mode="r")
    assert x.shape == (len(paths),2048) and np.isfinite(x).all()
    return x, paths, na


def quantiles(values):
    return {name: float(x) for name,x in zip(["min","p25","median","p75","p90","p95","p99","max"],
        np.quantile(values,[0,.25,.5,.75,.9,.95,.99,1]))}


def sheet(rows, filename, title):
    """Contact sheet: each cell shows original thumbnail alongside actual 64px input."""
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    import textwrap
    paths,_ = inputs(); pixels = np.load(CACHE/"pixels64.npy", mmap_mode="r")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",12)
    cols = max(len(r) for r in rows); w,h = 280,205
    canvas = Image.new("RGB", (cols*w,60+len(rows)*h), "#f1f5f9")
    draw = ImageDraw.Draw(canvas); draw.text((12,10),title,fill="black",font=font)
    draw.text((12,30),"Each cell: original aspect ratio (left); actual 64x64 input enlarged (right)",fill="black",font=font)
    for y,row in enumerate(rows):
        for x,(index,label) in enumerate(row):
            ox,oy = x*w,60+y*h
            with Image.open(DATA/paths[index]) as im:
                thumb=ImageOps.contain(im.convert("RGB"),(128,128))
            canvas.paste(thumb,(ox+4+(128-thumb.width)//2,oy+(128-thumb.height)//2))
            canvas.paste(Image.fromarray(pixels[index]).resize((128,128),Image.Resampling.NEAREST),(ox+142,oy))
            text=label+"\n"+paths[index]
            lines=[]
            for part in text.splitlines(): lines.extend(textwrap.wrap(part,39))
            draw.multiline_text((ox+4,oy+132),"\n".join(lines[:5]),font=font,fill="black",spacing=1)
    target=OUT/filename; target.parent.mkdir(parents=True,exist_ok=True); canvas.save(target)
