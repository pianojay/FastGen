import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from soflow_fastgen.common import seed_all, atomic_json
from soflow_fastgen.training import Trainer

torch.set_num_threads(4)
seed_all(42)
images=torch.randn(8,3,64,64)
labels=torch.arange(8)
config=json.loads(Path('configs/pokemon.json').read_text())
config.update(architecture={'model_channels':32,'num_blocks':1,'dropout':.1},batch_size=8,microbatch_size=5)
rows=[]
for explicit_zero in (False,False,True):
    seed_all(91)
    trainer=Trainer(dict(config,**({'fm_fraction':0.} if explicit_zero else {})))
    initial={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
    stats=trainer.update(images,labels)
    rows.append({'model':{k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()},'ema':{k:v.detach().cpu().clone() for k,v in trainer.ema.state_dict().items()},'initial':initial,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'stats':stats})
    del trainer
out=[]
for index in (1,2):
    a,b=rows[0],rows[index]
    differences={k:float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'] if a['model'][k].dtype.is_floating_point}
    all_close=True
    for k in a['model']:
        try: torch.testing.assert_close(a['model'][k],b['model'][k],rtol=1e-6,atol=1e-7)
        except AssertionError: all_close=False
    out.append({'comparison':'default versus repeat default' if index==1 else 'default versus explicit zero', 'initial_equal':all(torch.equal(a['initial'][k],b['initial'][k]) for k in a['initial']), 'cpu_rng_equal':torch.equal(a['cpu_rng'],b['cpu_rng']), 'cuda_rng_equal':torch.equal(a['cuda_rng'],b['cuda_rng']), 'max_weight_abs_difference':max(differences.values()), 'changed_parameter_tensors':sum(v>0 for v in differences.values()), 'all_close_rtol1e-6_atol1e-7':all_close, 'stats': [{k:r['stats'][k] for k in ('loss_surrogate','consistency_mse','weight','grad_norm')} for r in (a,b)]})
atomic_json('runs/research_c3_diagnostic_20261004/result.json',out)
print(json.dumps(out,indent=2))
