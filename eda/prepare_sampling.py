"""Save deterministic sampling indices for the proposed FID comparisons.

Indices address resources/A.txt. This creates analysis subsets only, never
changes the fixed A1/A2 partition, and computes no features or model scores.
"""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def main():
    paths = (ROOT / "resources/A.txt").read_text().splitlines()
    b_paths = (ROOT / "resources/B.txt").read_text().splitlines()
    m = len(b_paths)
    categories = defaultdict(list)
    b_counts = defaultdict(int)
    for index, path in enumerate(paths):
        categories[Path(path).parent.name].append(index)
    for path in b_paths:
        b_counts[Path(path).parent.name] += 1
    repeats = []
    for seed in range(20261005, 20261025):
        rng = random.Random(seed)
        sampled = rng.sample(range(len(paths)), 2*m)
        uniform = {"U": sampled[:m], "V": sampled[m:]}
        rng = random.Random(seed)
        balanced = {"U": [], "V": []}
        for category in sorted(categories):
            count = b_counts[category]
            sampled = rng.sample(categories[category], 2*count)
            balanced["U"].extend(sampled[:count])
            balanced["V"].extend(sampled[count:])
        for pair in (uniform, balanced):
            assert len(pair["U"]) == len(pair["V"]) == m
            assert len(set(pair["U"] + pair["V"])) == 2*m
        repeats.append({"seed": seed, "uniform": uniform, "balanced": balanced})
    design = {
        "A_manifest_sha256": hashlib.sha256((ROOT / "resources/A.txt").read_bytes()).hexdigest(),
        "B_manifest_sha256": hashlib.sha256((ROOT / "resources/B.txt").read_bytes()).hexdigest(),
        "index_base": 0, "index_manifest": "resources/A.txt", "m": m,
        "comparisons": ["FID(A, B)", "FID(U, B)", "FID(U, V)"],
        "note": "Path-disjoint U/V; duplicate-group sensitivity is separate. B is always fixed.",
        "repeats": repeats,
    }
    target = ROOT / "resources/fid_sampling.json.gz"
    # mtime=0 makes repeated output byte-reproducible.
    target.write_bytes(gzip.compress(json.dumps(design, separators=(",", ":")).encode(), mtime=0))
    print(f"Saved {len(repeats)} repeats × 2 sampling schemes: {target}")


if __name__ == "__main__":
    main()
