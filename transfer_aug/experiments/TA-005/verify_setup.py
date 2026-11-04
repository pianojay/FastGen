import hashlib,json,random,time,gc,ast,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import init
from engine import Trainer,seed_all,weights_init
ROOT=Path(__file__).resolve().parent
class SyntheticSampler:
    def __init__(self):self.rng=np.random.default_rng(20261008);self.draws=0
    def draw(self):
        self.draws+=1
        a=self.rng.uniform(-1,1,(1,3,64,64)).astype('float32');b=self.rng.uniform(-1,1,(1,3,64,64)).astype('float32')
        return torch.from_numpy(a),torch.from_numpy(b)
    def state_dict(self):return dict(rng=self.rng.bit_generator.state,draws=self.draws)
    def load_state_dict(self,s):self.rng.bit_generator.state=s['rng'];self.draws=s['draws']
def digest(t):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(x.detach().cpu().numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():h.update(str(k).encode());visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    for x in [t.net.state_dict(),t.opt_g.state_dict(),t.opt_d.state_dict(),t.sch_g.state_dict(),t.sch_d.state_dict(),t.update]:visit(x)
    return h.hexdigest()
start=time.monotonic();seed_all();t=Trainer();s=SyntheticSampler();a,b=s.draw();checks={}
counts=t.net.counts();assert counts['total']<100000000
checks['parameter_cap']=True
# Compare copied selected architecture bytes and exact initializer helper semantics.
upstream=ROOT.parents[1]/'audit_sources/munit/networks.py'
assert upstream.read_bytes()==(ROOT/'networks.py').read_bytes();checks['upstream_networks_byte_identical']=True
module=ast.parse((ROOT/'upstream_utils.py').read_text());node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='weights_init');scope={'init':init,'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'initializer','exec'),scope)
for kind in ['kaiming','gaussian']:
    m=torch.nn.Conv2d(3,4,3);seed_all();m.apply(weights_init(kind));expected=m.weight.detach().clone();seed_all();m.apply(scope['weights_init'](kind));assert torch.equal(m.weight,expected)
checks['initializer_matches_pinned_helper']=True
with torch.no_grad():
    c,style=t.net.G0.encode(a);z0=torch.zeros(1,8,1,1);z1=torch.ones(1,8,1,1)
    o0=t.net.G1.decode(c,z0);o1=t.net.G1.decode(c,z1);rec=t.net.G0.decode(c,style)
    shapes=[list(x.shape) for x in t.net.D0(a)]
    assert shapes==[[1,1,4,4],[1,1,2,2],[1,1,1,1]]
    assert c.shape==(1,256,16,16) and style.shape==(1,8,1,1)
    assert o0.shape==rec.shape==a.shape and torch.isfinite(rec).all()
    effect=float((o0-o1).abs().mean());assert effect>1e-6
    c2,s2=t.net.G1.encode(o1);assert c2.shape==c.shape and s2.shape==style.shape
checks.update(three_scale_64_valid=True,content_style_decode_reencode_shapes=True,style_code_effect_mean_absolute=effect)
loss=t.step(a,b);assert all(np.isfinite(v) for v in loss.values());checks['finite_full_width_D_G_backward_and_optimizer_step']=True
for m in (t.net.G0,t.net.G1,t.net.D0,t.net.D1):assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
checks['all_four_components_receive_nonzero_gradients']=True
checkpoint=ROOT/'setup_checkpoint.pt';t.save(checkpoint,s,'synthetic_contract');a2,b2=s.draw();expectedloss=t.step(a2,b2);expectedhash=digest(t);expectedrng=(random.random(),np.random.rand(),torch.rand(1).item());del t;gc.collect()
t=Trainer();s=SyntheticSampler();t.load(checkpoint,s,'synthetic_contract');aa,bb=s.draw();assert torch.equal(aa,a2) and torch.equal(bb,b2);actual=t.step(aa,bb);assert actual==expectedloss and digest(t)==expectedhash;assert (random.random(),np.random.rand(),torch.rand(1).item())==expectedrng
checks['checkpoint_next_step_parameters_optimizer_scheduler_sampler_rng_bitexact']=True
assert t.opt_g.param_groups[0]['lr']==.0001 and t.sch_g.last_epoch==2
checks['stepLR_constant_in_pilot']=True
try:t.load(checkpoint,s,'wrong');raise AssertionError('accepted wrong contract')
except ValueError:checks['checkpoint_contract_rejection']=True
checkpoint.unlink()
record=dict(status='passed',checks=checks,parameters=counts,discriminator_shapes=shapes,synthetic_optimizer_steps=3,batch_size_verified=1,gpu_used=False,dataset_training=False,cpu_seconds=time.monotonic()-start,loss_step1=loss)
(ROOT/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
