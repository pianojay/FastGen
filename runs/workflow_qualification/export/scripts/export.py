"""Create a self-contained FastGen package with inference-only checkpoints."""
import argparse
import json
from pathlib import Path
import shutil

import torch
from model import ModelOneNFE, ModelFewNFE
from soflow_fastgen.common import ROOT, atomic_json, atomic_save, digest, verify_protected


def export(checkpoint, output, few_nfe=4):
    verify_protected()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    output=Path(output)
    if output.exists():raise FileExistsError(f'Export must use a fresh destination: {output}')
    output.mkdir(parents=True)
    for name in ('model.py','dataset.py','download_dataset.py','evaluate.py','README.md',
                 'README_LOCAL.md','AGENTS.md','requirements.txt','requirements-fastgen-lock.txt','THIRD_PARTY.md'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,output/name)
    for name in ('src','soflow_fastgen','scripts','configs','docs','tests'):
        shutil.copytree(ROOT/name,output/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('train_split.txt','val_split.txt','category_to_id.json'):
        target=output/'data/pokemon-generation-one-22k'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'data/pokemon-generation-one-22k'/name,target)
    shutil.copy2(ROOT/'train.py',output/'train.py')
    atomic_json(output/'inference_config.json',state['config']['architecture'])
    counts={}
    for cls,mode,nfe in ((ModelOneNFE,'one_nfe',1),(ModelFewNFE,'few_nfe',few_nfe)):
        model=cls(**state['config']['architecture'])
        model.backbone.load_state_dict(state['ema'])
        model.sampling_nfe.fill_(nfe)
        count=model.count_parameters()
        if count>=100_000_000:raise ValueError('Loaded sampling model is not below 100M')
        atomic_save(output/'checkpoints'/f'{mode}.ckpt',{'state_dict':model.state_dict()})
        counts[mode]={'parameters':count,'nfe':nfe}
    metadata={'source_checkpoint_sha256':digest(checkpoint),'step':state['step'],
              'architecture':state['config']['architecture'],'models':counts,'official_score':False,
              'training_provenance':state.get('extra',{})}
    atomic_json(output/'export_metadata.json',metadata)
    hashes={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    atomic_json(output/'SHA256SUMS.json',hashes)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--few-nfe',type=int,choices=(2,3,4),default=4);a=p.parse_args()
    print(json.dumps(export(a.checkpoint,a.output,a.few_nfe),indent=2))
