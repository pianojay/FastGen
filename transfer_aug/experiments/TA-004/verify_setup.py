"""CPU-only synthetic contract checks. No dataset is opened."""
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from cut64 import CUT64, PairSampler, LAYERS, patch_nce, seed_all

root=Path(__file__).resolve().parent
start=time.monotonic()
torch.set_num_threads(2)
seed_all()
assert not torch.cuda.is_initialized()
model=CUT64()
a,b=torch.rand(2,3,64,64)*2-1,torch.rand(2,3,64,64)*2-1
with torch.no_grad():
    output=model.G(a)
    features=model.G(a,LAYERS,True)
    d=model.D(output)
assert output.shape==a.shape and d.shape==(2,1,14,14)
assert torch.isfinite(output).all() and torch.isfinite(d).all()
shapes=[list(f.shape) for f in features]
assert shapes==[[2,3,70,70],[2,128,64,64],[2,256,32,32],[2,256,16,16],[2,256,16,16]]
counts={name:sum(p.numel() for p in getattr(model,name).parameters()) for name in ('G','D','F')}
assert sum(counts.values())<100_000_000
# Independent explicit per-image reference excludes all cross-image keys.
q=torch.randn(8,7,requires_grad=True); k=torch.randn(8,7,requires_grad=True)
loss=patch_nce(q,k,2)
ref=[]
for image in range(2):
    for patch in range(4):
        idx=image*4+patch
        logits=[(q[idx]*k[idx].detach()).sum()]
        for other in range(4):
            logits.append(torch.tensor(-10.) if patch==other else (q[idx]*k[image*4+other].detach()).sum())
        ref.append(torch.nn.functional.cross_entropy(torch.stack(logits)[None]/.07,torch.tensor([0])))
assert torch.allclose(loss,torch.stack(ref),atol=1e-5)
changed=k.detach().clone(); changed[4:]+=10
assert torch.equal(loss[:4],patch_nce(q,changed,2)[:4])
loss.mean().backward()
assert k.grad is None and q.grad is not None and q.grad.abs().sum()>0
# Real G/F NCE gradient flow, separately from GAN.
model.zero_grad(set_to_none=True)
nce=model.nce(a,model.G(a)); nce.backward()
grads={name:float(sum(p.grad.abs().sum() for p in getattr(model,name).parameters() if p.grad is not None)) for name in ('G','F')}
assert all(np.isfinite(v) and v>0 for v in grads.values())
domains={'c0':{'D0':['a0','a1'],'D1':['b0']},'c1':{'D0':['a2'],'D1':['b1','b2']}}
sampler=PairSampler(domains)
for _ in range(7):sampler.draw()
first=model.step(a,b)
checkpoint=root/'synthetic_checkpoint.pt'
model.checkpoint(checkpoint,sampler)
def continuation(m,s):
    draws=[s.draw() for _ in range(8)]
    rng=[random.random(),float(np.random.rand()),float(torch.rand(()))]
    losses=m.step(a,b)
    digest=hashlib.sha256()
    for t in m.state_dict().values():digest.update(t.detach().cpu().numpy().tobytes())
    return {'draws':draws,'rng':rng,'losses':losses,'parameter_sha256':digest.hexdigest()}
expected=continuation(model,sampler)
restored=CUT64(); restored_sampler=PairSampler(domains)
restored.restore(checkpoint,restored_sampler)
actual=continuation(restored,restored_sampler)
assert expected==actual, (expected,actual)
assert restored.updates==2
assert all(len(o.state)>0 for o in restored.optimizers.values())
assert not torch.cuda.is_initialized()
report={'status':'passed','device':'cpu','torch_version':torch.__version__,
'generator_output_shape':list(output.shape),'discriminator_output_shape':list(d.shape),
'feature_mapping':[{'index':i,'module':type(model.G.model[i]).__name__,'shape':shape} for i,shape in zip(LAYERS,shapes)],
'parameter_counts':counts,'framework_parameters':sum(counts.values()),'nce_only_gradient_abs_sums':grads,
'checks':{'finite_outputs':True,'per_image_negatives_reference':True,'other_image_key_invariance':True,'stopped_keys':True,'G_F_nonzero_nce_gradients':True,'all_optimizer_states_present':True,'checkpoint_exact_next_step_rng_sampler_weights':True,'no_cuda_initialization':True},
'initial_synthetic_losses':first,'recovery_reference':expected,'recovery_actual':actual,
'discriminator_nominal_local_receptive_field':40,'receptive_field_note':'Conv/filter support only; instance normalization couples spatial sites.',
'elapsed_seconds':time.monotonic()-start,'peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu_seconds':0}
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','framework_parameters','parameter_counts','elapsed_seconds','peak_host_rss_bytes')},indent=2))
