import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from flow_sid.common import ROOT, StageCost, atomic_json
from flow_sid.gpu import GPU2Lease


def main():
    signal.alarm(3600)
    with StageCost('qualification_evaluation_export'), GPU2Lease():
        import torch
        from flow_sid.engine import setup_torch, memory_metrics
        from flow_sid.networks import FlowNet
        from flow_sid.evaluation import evaluate_model, exported_state
        from model import Model
        setup_torch()
        torch.manual_seed(1729)
        net = FlowNet()
        result = {'purpose': 'Untrained plumbing only; not a model-quality result', 'exports': {}}
        with tempfile.TemporaryDirectory(prefix='fastgen_export_', dir=ROOT / 'artifacts') as temp:
            temp = Path(temp)
            for relative in ['model.py', 'src/__init__.py', 'src/utils.py', 'flow_sid/__init__.py',
                             'flow_sid/unet.py', 'flow_sid/networks.py', 'flow_sid/flow.py', 'scripts/check_export.py']:
                target = temp / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            for nfe in [1, 2, 3, 4]:
                checkpoint = temp / 'probe.ckpt'
                torch.save(exported_state(net.state_dict(), nfe), checkpoint)
                env = os.environ.copy()
                env['PYTHONPATH'] = str(temp)
                env['CUDA_VISIBLE_DEVICES'] = ''
                process = subprocess.run([sys.executable, 'scripts/check_export.py', str(checkpoint), '--nfe', str(nfe)],
                    cwd=temp, env=env, capture_output=True, text=True, timeout=120)
                assert process.returncode == 0, process.stderr
                result['exports'][nfe] = json.loads(process.stdout)
                # Loaded GPU sampler also verifies exact network call count.
                model = Model.load_checkpoint(str(checkpoint), 'one_nfe' if nfe == 1 else 'few_nfe', device='cuda:0')
                calls = []
                hook = model.backbone.register_forward_pre_hook(lambda *a: calls.append(1))
                samples = model.sample((2, 3, 64, 64), category=torch.tensor([0, 1], device='cuda:0'), device='cuda:0')
                hook.remove()
                assert len(calls) == nfe and torch.isfinite(samples).all()
                if nfe == 1:
                    result['evaluation'] = evaluate_model(model, ROOT / 'artifacts/plumbing_evaluation',
                                                          seed=271828, diagnostics_enabled=False)
                del model
                torch.cuda.empty_cache()
        result['memory'] = memory_metrics()
        atomic_json(ROOT / 'runs/evaluation_export_smoke.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
