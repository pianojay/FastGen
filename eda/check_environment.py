"""Qualify the isolated EDA runtime and frozen feature extractor on GPU 0."""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UUID = "GPU-e9d30df5-3b84-1a7e-c505-c6fff0a0ce7f"


def main():
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert not site.ENABLE_USER_SITE
    assert Path(sys.prefix).resolve() == ROOT / ".conda/eda"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    reference = ROOT / "eda/resources/soflow-reference-requirements.txt"
    matched_versions = {}
    for line in reference.read_text().splitlines():
        if "==" in line and not line.startswith("--"):
            package, expected = line.split("==", 1)
            actual = importlib.metadata.version(package)
            assert actual == expected, (package, actual, expected)
            matched_versions[package] = actual
    import numpy as np
    import scipy
    import sklearn
    import matplotlib
    import torch
    import torchvision
    import cleanfid
    from sklearn.decomposition import PCA
    from cleanfid.inception_torchscript import InceptionV3W
    modules = [np, scipy, sklearn, matplotlib, torch, torchvision, cleanfid]
    for module in modules:
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix)), module.__file__
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cpu_shape = PCA(n_components=2).fit_transform(np.arange(24.).reshape(6, 4)).shape
    # Reuse the same advisory lock as the sibling project, without changing its contents.
    with Path("/tmp/fastgen-gpu0.lock").open("r") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = subprocess.check_output([
            "nvidia-smi", "-i", "0", "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader"], text=True).strip()
        assert identity.split(",")[0] == EXPECTED_UUID, identity
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        assert not any(EXPECTED_UUID in row for row in processes.splitlines()), processes
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.cuda.reset_peak_memory_stats()
        weights = ROOT / ".cache/clean-fid/inception-2015-12-05.pt"
        extractor = InceptionV3W(str(weights.parent), download=False).to("cuda").eval()
        with torch.inference_mode():
            features = extractor(torch.zeros(2, 3, 299, 299, device="cuda"))
        torch.cuda.synchronize()
        assert features.shape == (2, 2048) and torch.isfinite(features).all()
        report = {
            "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "user_site_enabled": site.ENABLE_USER_SITE,
            "reference_project": "FastGen-SoFlow",
            "matched_reference_versions": matched_versions,
            "packages": {p: importlib.metadata.version(p) for p in
                         ["torch", "torchvision", "clean-fid", "kagglehub", "numpy", "scipy",
                          "scikit-learn", "matplotlib", "Pillow", "pip"]},
            "module_paths": {m.__name__: m.__file__ for m in modules},
            "gpu_identity": identity, "cuda_runtime": torch.version.cuda,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cpu_pca_shape": list(cpu_shape), "feature_smoke_shape": list(features.shape),
            "extractor_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "scope": "Environment qualification using synthetic inputs only; no dataset metrics or project models.",
        }
        (ROOT / "eda/resources/qualified_environment.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
