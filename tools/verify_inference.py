#!/usr/bin/env python3
"""CPU-only loader and actual network-call checks against restored inference packages."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = {
    'soflow-one': ('FastGen-SoFlow/runs/research_s3_unet192_40k_seed42/export', 'one_nfe', 1),
    'soflow-few': ('FastGen-SoFlow/runs/research_s3_unet192_40k_seed42/export', 'few_nfe', 4),
    'peer-one': ('fastgen_zip/fastgen_sida_1007', 'one_nfe', 1),
    'peer-few': ('fastgen_zip/fastgen_sida_1007', 'few_nfe', 2),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--case', choices=list(CASES))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = args.legacy_root.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    if args.case is None:
        results = []
        for case in CASES:
            command = [sys.executable, str(Path(__file__).resolve()), '--legacy-root', str(root), '--case', case]
            result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=300)
            row = json.loads(result.stdout)
            results.append(row)
            print(json.dumps(row), flush=True)
        report = {'state':'pass','completed_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'cases':results,'limits':'CPU inference/loading only; no GPU, FID, training, or scientific-result reproduction.'}
        if args.report:
            args.report.write_text(json.dumps(report,indent=2)+'\n')
        return
    start = time.monotonic()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    relative, mode, expected = CASES[args.case]
    package = root/relative
    if not package.is_relative_to(root):
        raise ValueError('Package outside restored root')
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location('restored_model', package/'model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(1729)
    model = module.Model.load_checkpoint(str(package/'checkpoints'/f'{mode}.ckpt'), evaluate_mode=mode, device='cpu')
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters <= 100000000
    assert all(torch.isfinite(p).all() for p in model.parameters())
    network = model.backbone.network if args.case.startswith('soflow') else model.backbone.model
    calls = []
    hook = network.register_forward_pre_hook(lambda *_: calls.append(1))
    with torch.inference_mode():
        image = model.sample((2,3,64,64), device='cpu', category=torch.tensor([0,150],dtype=torch.long))
    hook.remove()
    assert len(calls) == expected, (args.case, len(calls), expected)
    assert tuple(image.shape) == (2,3,64,64) and torch.isfinite(image).all()
    assert float(image.min()) >= -1 and float(image.max()) <= 1
    assert all(p.device.type == 'cpu' for p in model.parameters())
    print(json.dumps({'case':args.case,'state':'pass','package':str(package),'parameters':parameters,'actual_backbone_calls':len(calls),'batch_shape':list(image.shape),'output_min':float(image.min()),'output_max':float(image.max()),'device':'cpu','seconds':time.monotonic()-start,'torch':torch.__version__}))

if __name__ == '__main__':
    main()
