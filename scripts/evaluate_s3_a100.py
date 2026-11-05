"""Run the S3 paired development metric on a local inference export or training EMA."""
import argparse
import json
import os
from pathlib import Path

import torch

from model import Model
from soflow_fastgen.common import atomic_json, digest, source_provenance
from soflow_fastgen.evaluation import prepare_reference
from soflow_fastgen.models import Soflow
from soflow_fastgen.paired_evaluation import evaluate_schedule, fingerprint, image_manifest


def main(args):
    archived = json.loads(Path('configs/research_s3_20261002.json').read_text())
    config = json.loads(args.config.read_text())
    config['run_dir'] = str(args.run_dir)
    torch.set_num_threads(config['torch_threads'])
    partition_hash = digest(config['partition'])
    if partition_hash != archived['partition_sha256']:
        raise ValueError('Expected the archived S3 development partition')
    reference = prepare_reference(config)
    reference_hash = fingerprint(image_manifest(reference))
    if reference_hash != archived['reference_sha256']:
        raise ValueError(f'S3 reference PNGs changed: {reference_hash}')
    checkpoint_hash = digest(args.checkpoint)
    if args.export:
        mode = 'one_nfe' if args.nfe == 1 else 'few_nfe'
        model = Model.load_checkpoint(str(args.checkpoint), mode, device='cpu',
                                      **config['architecture'])
        if int(model.sampling_nfe) != args.nfe:
            raise ValueError('Export does not encode the requested NFE')
        backbone = model.backbone
    else:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if state['extra']['partition_sha256'] != partition_hash:
            raise ValueError('Training checkpoint belongs to a different partition')
        backbone = Soflow(**state['config']['architecture'])
        backbone.load_state_dict(state['ema'])
        del state
    if sum(parameter.numel() for parameter in backbone.parameters()) != 89_140_419:
        raise ValueError('Expected the archived S3 backbone architecture')
    backbone.eval().requires_grad_(False).cuda()
    provenance = source_provenance()
    atomic_json(args.run_dir / 'evaluation_source.json', provenance)
    protocol = {
        'run_dir': str(args.run_dir), 'checkpoint_sha256': checkpoint_hash,
        'partition_sha256': partition_hash, 'nfe': args.nfe,
        'weight_source': 'ema_export' if args.export else 'ema',
        'eval_batch_size': archived['eval_batch_size'],
    }
    schedule = {'name': 'uniform',
                'intermediate_times': [1 - index / args.nfe for index in range(1, args.nfe)]}
    for seed in args.seeds:
        evaluate_schedule(backbone, protocol, schedule, seed, reference,
                          fingerprint(provenance['source_sha256']), reference_hash)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--nfe', type=int, choices=(1, 4), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('PYTHONNOUSERSITE') != '1':
        parser.error('Set CUDA_VISIBLE_DEVICES=0 and PYTHONNOUSERSITE=1')
    main(args)
