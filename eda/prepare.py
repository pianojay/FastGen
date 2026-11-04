"""Audit the local Kaggle images and preserve the existing A/A1/A2/B partition.

No model imports, training, GPU use, or writes outside this checkout.
Run from any directory: python3 eda/prepare.py [--extract]
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/pokemon-generation-one-22k"
RESOURCES = ROOT / "eda/resources"
REPORTS = ROOT / "eda/reports"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def check_protected(provenance):
    for name, expected in provenance["protected_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Protected file changed: {name}")
    if digest(RESOURCES / "development_split.json") != provenance["partition_sha256"]:
        raise ValueError("Saved development partition changed")


def extract(archive):
    target = DATA / "PokemonData"
    if target.exists():
        raise FileExistsError("Refusing to extract over an existing PokemonData tree")
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist() if not m.is_dir()]
        for member in members:
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "PokemonData":
                raise ValueError(f"Unexpected archive path: {path}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink: {path}")
        # ZipFile validates CRC as each member is read. Extract only into this checkout.
        source.extractall(DATA)
    sys.path.insert(0, str(ROOT))
    from download_dataset import composite_transparency_in_tree, NORMALIZED_MARKER
    count = composite_transparency_in_tree(target)
    (DATA / NORMALIZED_MARKER).touch()
    save_json(RESOURCES / "download.json", {
        "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
        "archive_files": len(members), "transparency_composited_count": count,
        "source": "https://www.kaggle.com/api/v1/datasets/download/bhawks/pokemon-generation-one-22k",
        "preprocessing": "Supplied download_dataset.py compositor, on independent local files",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    provenance = json.loads((RESOURCES / "provenance.json").read_text())
    check_protected(provenance)
    if args.extract:
        extract(ROOT / ".cache/downloads/pokemon-generation-one-22k.zip")
    if not (DATA / ".transparency_composited_on_white").exists():
        raise ValueError("Local white-composited dataset is not ready")
    train = (DATA / "train_split.txt").read_text().splitlines()
    val = (DATA / "val_split.txt").read_text().splitlines()
    mapping = json.loads((DATA / "category_to_id.json").read_text())
    partition = json.loads((RESOURCES / "development_split.json").read_text())
    assert partition["source_train_sha256"] == digest(DATA / "train_split.txt")
    assert partition["category_mapping_sha256"] == digest(DATA / "category_to_id.json")
    a1, a2 = partition["optimization_indices"], partition["development_indices"]
    assert len(a1) == len(set(a1)) and len(a2) == len(set(a2))
    assert not set(a1) & set(a2)
    assert set(a1) | set(a2) == set(range(len(train)))
    assert len(train) == len(set(train)) and len(val) == len(set(val))
    assert not set(train) & set(val)
    sets = {"A": train, "A1": [train[i] for i in a1],
            "A2": [train[i] for i in a2], "B": val}
    for label, paths in sets.items():
        (RESOURCES / f"{label}.txt").write_text("\n".join(paths) + "\n")
    subpartition = {p: label for label in ("A1", "A2", "B") for p in sets[label]}
    actual = {str(p.relative_to(DATA)) for p in (DATA / "PokemonData").rglob("*") if p.is_file()}
    expected = set(train + val)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    for number, relative in enumerate(train + val):
        path = DATA / relative
        try:
            assert path.resolve().is_relative_to(DATA.resolve())
            with Image.open(path) as image:
                image.load()
                rgb = image.convert("RGB")
                rgb64 = rgb.resize((64, 64), Image.Resampling.BILINEAR)
                row = {"path": relative, "split": "A" if subpartition[relative] != "B" else "B",
                       "partition": subpartition[relative], "category": path.parent.name,
                       "category_id": mapping[path.parent.name], "width": image.width,
                       "height": image.height, "mode": image.mode, "format": image.format,
                       "bytes": path.stat().st_size, "sha256": digest(path),
                       "rgb_sha256": hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(),
                       "rgb64_sha256": hashlib.sha256(rgb64.tobytes()).hexdigest(),
                       "has_transparency": "A" in image.getbands() or "transparency" in image.info}
                rows.append(row)
        except Exception as error:
            errors.append({"path": relative, "error": str(error)})
        if (number + 1) % 2000 == 0:
            print(f"Audited {number + 1}/{len(train) + len(val)} images", flush=True)
    if not rows:
        raise ValueError("No readable images")
    with (REPORTS / "inventory.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    duplicate_summary = {}
    for key in ("sha256", "rgb_sha256", "rgb64_sha256"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        repeated = [g for g in groups.values() if len(g) > 1]
        cross_ab = [g for g in repeated if {r["split"] for r in g} == {"A", "B"}]
        cross_a12 = [g for g in repeated if {"A1", "A2"} <= {r["partition"] for r in g}]
        duplicate_summary[key] = {
            "duplicate_groups": len(repeated), "redundant_files": sum(len(g)-1 for g in repeated),
            "cross_A_B_groups": len(cross_ab),
            "B_files_with_A_match": sum(r["split"] == "B" for g in cross_ab for r in g),
            "cross_A1_A2_groups": len(cross_a12)}
        save_json(REPORTS / f"duplicates_{key}.json", [
            [{k: r[k] for k in ("path", "split", "partition", "category")} for r in g]
            for g in repeated])
    categories = []
    for name in sorted(mapping):
        counts = {label: sum(Path(p).parent.name == name for p in paths) for label, paths in sets.items()}
        categories.append({"category": name, "category_id": mapping[name], **counts,
                           "p_A": counts["A"]/len(train), "p_B": counts["B"]/len(val),
                           "p_B_minus_p_A": counts["B"]/len(val)-counts["A"]/len(train)})
    with (REPORTS / "category_composition.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(categories[0])); writer.writeheader(); writer.writerows(categories)
    summary = {
        "counts": {k: len(v) for k,v in sets.items()}, "categories": len(mapping),
        "missing": missing, "unlisted_files": extra, "decode_errors": errors,
        "readable_images": len(rows), "remaining_transparency": sum(r["has_transparency"] for r in rows),
        "modes": dict(Counter(r["mode"] for r in rows)),
        "formats": dict(Counter(r["format"] for r in rows)), "duplicates": duplicate_summary,
        "per_category_count_range": {k: [min(r[k] for r in categories), max(r[k] for r in categories)] for k in sets},
        "category_total_variation_A_B": sum(abs(r["p_B_minus_p_A"]) for r in categories)/2,
        "largest_category_differences": sorted(categories, key=lambda r: abs(r["p_B_minus_p_A"]), reverse=True)[:10],
        "protected_files_unchanged": True,
    }
    check_protected(provenance)
    save_json(REPORTS / "audit.json", summary)
    print(json.dumps(summary, indent=2))
    if missing or errors:
        raise SystemExit("Audit failed: missing or unreadable images")


if __name__ == "__main__":
    main()
