"""Verify hashes and use the official loader in the exported package's own directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from soflow_fastgen.common import digest


def verify(package, device='cpu'):
    package = Path(package).resolve()
    for name, expected in json.loads((package/'SHA256SUMS.json').read_text()).items():
        if digest(package/name) != expected:
            raise ValueError(f'Export checksum mismatch: {name}')
    code = '''
import json, sys, time
import torch
from model import Model
torch.set_num_threads(4)
device = sys.argv[1]
results = {}
for mode in ('one_nfe','few_nfe'):
    model = Model.load_checkpoint('checkpoints/'+mode+'.ckpt',evaluate_mode=mode,device=device)
    model.eval()
    count = model.count_parameters()
    assert count < 100_000_000
    assert not any('ema' in k or 'teacher' in k for k in model.state_dict())
    calls = []
    hook = model.backbone.network.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        output = model.sample((3,3,64,64),device=device,category=torch.tensor([0,30,150],device=device))
    hook.remove()
    assert output.shape == (3,3,64,64) and torch.isfinite(output).all() and output.abs().max() <= 1
    assert len(calls) == int(model.sampling_nfe)
    results[mode] = {'parameters':count,'backbone_calls':len(calls),'shape':list(output.shape)}
    del model, output
print(json.dumps(results))
'''
    result = subprocess.run([sys.executable,'-c',code,device],cwd=package,
                            check=True,capture_output=True,text=True)
    return json.loads(result.stdout.splitlines()[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package',required=True)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    print(json.dumps(verify(args.package,args.device),indent=2))
