"""Plot measured category composition from preparation outputs, on CPU only."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with (ROOT / "reports/category_composition.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    ordered = sorted(rows, key=lambda r: float(r["p_B_minus_p_A"]))
    selected = ordered[:8] + ordered[-8:]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    x = list(range(len(rows)))
    axes[0].plot(x, [100*float(r["p_A"]) for r in ordered], label="A: official training pool", color="#2563eb")
    axes[0].plot(x, [100*float(r["p_B"]) for r in ordered], label="B: evaluation reference", color="#d97706")
    axes[0].set(xlabel="Pokémon categories ordered by A frequency", ylabel="Share of split (%)",
                title="B is balanced; A has unequal category frequencies")
    axes[0].legend(frameon=False)
    differences = [100*float(r["p_B_minus_p_A"]) for r in selected]
    axes[1].barh([r["category"] for r in selected], differences,
                 color=["#2563eb" if d < 0 else "#d97706" for d in differences])
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set(xlabel="B share − A share (percentage points)", title="Largest category proportion differences")
    fig.suptitle("Dataset audit • A = 17,079 images • B = 3,020 images • 151 categories")
    target = ROOT / "reports/category_composition.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
