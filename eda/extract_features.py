"""Extract frozen clean-fid features; never loads a project model."""
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
from cleanfid.resize import build_resizer
from cleanfid.inception_torchscript import InceptionV3W

from common import ROOT, DATA, RES, OUT, CACHE, inputs, verify, gpu_lease, save_json, sha


class Images(Dataset):
    def __init__(self):
        sys.path.insert(0,str(ROOT))
        import dataset
        # Use the actual fixed transform; suppress the downloader so this is strictly read-only.
        with patch.object(dataset,"download_dataset",lambda *a,**k: None):
            self.parts=[dataset.PokemonDataset(DATA,RES/f"{name}.txt") for name in ["A","B"]]
        self.paths,self.na=inputs()
        self.resize=build_resizer("clean")
        import csv
        with (ROOT/"eda/reports/inventory.csv").open() as f:
            self.hashes={r["path"]:r["sha256"] for r in csv.DictReader(f)}

    def __len__(self): return len(self.paths)

    def tensor(self,i):
        return self.parts[0][i] if i<self.na else self.parts[1][i-self.na]

    def __getitem__(self,i):
        assert sha(DATA/self.paths[i])==self.hashes[self.paths[i]],self.paths[i]
        pixels=np.asarray(to_pil_image(((self.tensor(i).float()+1)/2).clamp(0,1)).convert("RGB")).copy()
        clean=torch.from_numpy(self.resize(pixels).transpose(2,0,1).copy())
        return clean,pixels


def validate_export(images):
    # Execute only the supplied export function AST: do not import evaluate.py or model.py.
    tree=ast.parse((ROOT/"evaluate.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="prepare_reference_set")
    namespace={"Path":Path}; exec(compile(ast.Module(body=[fn],type_ignores=[]),"evaluate.py:prepare_reference_set","exec"),namespace)
    indices=list(range(12))+[images.na+i for i in range(12)]
    import csv
    with (ROOT/"eda/reports/inventory.csv").open() as f:
        special={r["path"] for r in csv.DictReader(f) if r["mode"]!="RGB"}
    indices+= [i for i,p in enumerate(images.paths) if p in special]
    class Reference:
        def val_dataloader(self):
            yield torch.stack([images.tensor(i) for i in indices])
    folder=CACHE/"preprocessing_check"
    namespace["prepare_reference_set"](Reference(),folder)
    for j,i in enumerate(indices):
        _,pixels=images[i]
        with Image.open(folder/f"{j:04d}.png") as im: assert np.array_equal(pixels,np.asarray(im))
    # Verify clean-fid's PNG loader yields the same float resized tensor.
    from cleanfid.utils import ResizeDataset
    check=ResizeDataset([str(folder/f"{j:04d}.png") for j in range(len(indices))],mode="clean")
    for j,i in enumerate(indices): assert torch.equal(check[j],images[i][0])
    return {"samples":len(indices),"pixel_exact_to_official_export":True,"tensor_exact_to_cleanfid_loader":True}


def main():
    verify(); CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    if (CACHE/"metadata.json").exists():
        from common import load_features
        load_features(); print("Verified completed feature cache; no extraction needed."); return
    images=Images(); check=validate_export(images)
    print("Preprocessing verified:",check,flush=True)
    with gpu_lease():
        start=time.monotonic()
        weights=ROOT/".cache/clean-fid/inception-2015-12-05.pt"
        assert sha(weights)==json.loads((RES/"qualified_environment.json").read_text())["extractor_sha256"]
        extractor=InceptionV3W(str(weights.parent),download=False).to("cuda").eval()
        extractor.requires_grad_(False)
        feats=np.lib.format.open_memmap(CACHE/"inception.npy",mode="w+",dtype="float32",shape=(len(images),2048))
        pixels=np.lib.format.open_memmap(CACHE/"pixels64.npy",mode="w+",dtype="uint8",shape=(len(images),64,64,3))
        loader=DataLoader(images,batch_size=64,num_workers=2,shuffle=False,pin_memory=True)
        offset=0
        with torch.inference_mode():
            for batch,pix in loader:
                features=extractor(batch.to("cuda",non_blocking=True)).cpu().numpy()
                assert np.isfinite(features).all()
                end=offset+len(batch); feats[offset:end]=features; pixels[offset:end]=pix.numpy(); offset=end
                if offset%1024==0 or offset==len(images):
                    print(f"Features {offset}/{len(images)} elapsed={time.monotonic()-start:.1f}s",flush=True)
        feats.flush();pixels.flush();verify()
        save_json(CACHE/"metadata.json",{
            "count":offset,"dimensions":2048,"dtype":"float32","mode":"clean","model_name":"inception_v3",
            "paths_sha256":hashlib.sha256("\n".join(images.paths).encode()).hexdigest(),
            "feature_sha256":sha(CACHE/"inception.npy"),"pixels_sha256":sha(CACHE/"pixels64.npy"),
            "inventory_sha256":sha(ROOT/"eda/reports/inventory.csv"),"extractor_sha256":sha(weights),
            "environment":json.loads((RES/"qualified_environment.json").read_text()),
            "preprocessing":check,"seconds":time.monotonic()-start,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"source_sha256":sha(__file__),
            "input_pipeline":"official dataset transform -> official PNG export uint8 -> clean-fid PIL float bicubic 299x299",
        })
        print("Feature extraction complete",flush=True)


if __name__=="__main__":main()
