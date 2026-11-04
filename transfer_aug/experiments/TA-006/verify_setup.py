import ast,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from stargan64 import StarGAN64,Sampler,seed_all
from losses import compute_g_loss,moving_average,r1_reg
R=Path(__file__).resolve().parent
start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed_all()
checks={}
def check(k,v):
    checks[k]=bool(v);assert v,k
    print(k,flush=True)
def digest(m):
    h=hashlib.sha256()
    for collection in (m.nets,m.ema):
        for net in collection.values():
            for p in net.state_dict().values():h.update(p.detach().numpy().tobytes())
    return h.hexdigest()
m=StarGAN64();counts=m.counts();print(counts,flush=True)
check('ema_initial_exact',all(torch.equal(p,q) for k,n in m.ema.items() for p,q in zip(m.nets[k].parameters(),n.parameters())))
x=torch.randn(1,3,64,64);y=torch.zeros(1,dtype=torch.long);yt=1-y;refs=[torch.randn_like(x),torch.randn_like(x)];z=[torch.randn(1,16),torch.randn(1,16)]
with torch.no_grad():
    s=m.nets.mapping_network(z[0],yt);out=m.nets.generator(x,s);enc=m.nets.style_encoder(x,y);d=m.nets.discriminator(x,y)
    check('tensor_shapes',s.shape==(1,64) and out.shape==(1,3,64,64) and enc.shape==(1,64) and d.shape==(1,))
    h=m.nets.mapping_network.shared(z[0]);check('mapping_domain_head',torch.equal(s,m.nets.mapping_network.unshared[1](h)))
    h=m.nets.style_encoder.shared(x).flatten(1);check('encoder_domain_head',torch.equal(enc,m.nets.style_encoder.unshared[0](h)))
    h=m.nets.discriminator.main(x).flatten(1);check('discriminator_domain_head',torch.equal(d,h[:,0]))
    out2=m.nets.generator(x,m.nets.mapping_network(z[1],yt));check('latent_sensitivity',float((out-out2).abs().mean())>1e-6)
    sr=[m.nets.style_encoder(r,yt) for r in refs];check('reference_style_sensitivity',float((m.nets.generator(x,sr[0])-m.nets.generator(x,sr[1])).abs().mean())>1e-6)
# Detached second latent receives no gradient; first branch is differentiable.
zz=[q.clone().requires_grad_() for q in z];loss,_=compute_g_loss(m.nets,m.args,x,y,yt,z_trgs=zz);loss.backward()
check('diversity_second_branch_detached',zz[1].grad is None)
check('first_style_branch_gradient',zz[0].grad is not None and torch.isfinite(zz[0].grad).all() and zz[0].grad.abs().sum()>0)
for o in m.optimizers.values():o.zero_grad(set_to_none=True)
# Small exact EMA formula, independent of full architecture.
a=torch.nn.Linear(2,2);b=torch.nn.Linear(2,2);old=[p.detach().clone() for p in b.parameters()];moving_average(a,b,.999)
check('ema_formula',all(torch.equal(q,torch.lerp(p,oldv,.999)) for p,q,oldv in zip(a.parameters(),b.parameters(),old)))
sampler=Sampler({'synthetic':{'D0':[0,1,2],'D1':[3,4,5]}})
old_steps={k:0 for k in m.optimizers};phase_records=[]
def hook(label,participants):
    steps={k:int(next(iter(o.state.values()))['step']) if o.state else 0 for k,o in m.optimizers.items()}
    expected={k:old_steps[k]+int(k in participants) for k in steps};check('optimizer_participation_'+label,steps==expected)
    old_steps.update(steps);phase_records.append({'phase':label,'steps':steps})
loss=m.step(x,y,refs,yt,z,hook)
check('all_four_phase_losses_finite',all(np.isfinite(list(loss.values()))))
check('r1_finite',loss['D_latent/reg']>=0 and loss['D_reference/reg']>=0)
m.checkpoint(R/'synthetic_checkpoint.pt',sampler)
draw1=sampler.draw();loss1=m.step(x,y,refs,yt);dig1=digest(m);rng1=(random.random(),float(np.random.rand()),float(torch.rand(())))
m.restore(R/'synthetic_checkpoint.pt',sampler);draw2=sampler.draw();loss2=m.step(x,y,refs,yt);dig2=digest(m);rng2=(random.random(),float(np.random.rand()),float(torch.rand(())))
check('exact_next_step_model_ema',dig1==dig2);check('exact_next_step_losses',loss1==loss2);check('exact_sampler_recovery',draw1==draw2);check('exact_rng_recovery',rng1==rng2)
# AST equality of inherited model/loss functions.
up=R.parents[1]/'audit_sources/starganv2/core'
for fname,local in [('model.py','model.py'),('solver.py','losses.py')]:
    src={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((up/fname).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    copied={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((R/local).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    check('upstream_ast_exact_'+local,all(src[k]==v for k,v in copied.items()))
rec={'status':'passed','checks':checks,'counts':counts,'total_parameters':sum(counts.values()),'trainable_parameters':sum(v for k,v in counts.items() if not k.endswith('_ema')),'outside_normal_100M_limit':sum(counts.values())>100_000_000,'optimizer_phase_evidence':phase_records,'cpu_seconds':time.monotonic()-start,'gpu_seconds':0,'cpu_batch_size':1,'synthetic_complete_iterations':3,'losses':loss,'recovery_digests':[dig1,dig2]}
(R/'verification.json').write_text(json.dumps(rec,indent=2)+'\n')
print('DONE',rec['cpu_seconds'],flush=True)
