"""Run from an isolated directory with only inference sources and one checkpoint."""
import argparse
from pathlib import Path
import json
import torch
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--nfe', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    mode = 'one_nfe' if args.nfe == 1 else 'few_nfe'
    model = Model.load_checkpoint(args.checkpoint, evaluate_mode=mode, device='cpu')
    assert model.count_parameters() == 61823875
    assert set(model._modules) == {'backbone'}
    calls = []
    handle = model.backbone.register_forward_pre_hook(lambda *a: calls.append(1))
    image = model.sample((1, 3, 64, 64), category=torch.tensor([25]), device='cpu')
    handle.remove()
    assert len(calls) == args.nfe and torch.isfinite(image).all()
    print(json.dumps({'independent_loading': True, 'nfe': len(calls),
                      'parameters': model.count_parameters(), 'finite': True,
                      'source_directory': str(Path.cwd())}))


if __name__ == '__main__':
    main()
