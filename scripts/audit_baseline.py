import copy,json,time
from pathlib import Path
import torch
from afm_fastgen.common import *
from afm_fastgen.audit import audit,calibrate
from afm_fastgen.models import Generator,Discriminator
from afm_fastgen.losses import GradientNormalization,generator_losses,discriminator_losses

def main():
    c=json.loads((ROOT/'configs/baseline.json').read_text()); configure(c)
    out=ROOT/'artifacts/stability/calibration'; out.mkdir(parents=True,exist_ok=True)
    with GpuLease():
        start=time.monotonic(); atomic_json(out/'start.json',{'process':process_identity(),'source':source_identity(),'protected':verify_protected(),'start_unix':time.time()})
        state=torch.load(ROOT/'runs/afm_unet128_50k_seed20261001/checkpoints/latest.pt',map_location='cpu',weights_only=False)
        g=Generator(**c['generator']).cuda().eval(); g.load_state_dict(state['g'])
        ema=Generator(**c['generator']).cuda().eval(); ema.load_state_dict(state['ema'])
        d=Discriminator(**c['discriminator']).cuda().eval(); d.load_state_dict(state['d'])
        norm=GradientNormalization().cuda(); norm.load_state_dict(state['norm'])
        del state
        audit(g,ema,c,out/'baseline',10000,seed=19001)
        calibrate(c,out/'controls')
        from afm_fastgen.audit import real_groups
        _,refs=real_groups(c); y=torch.arange(8,device='cuda'); z=torch.randn(8,3,64,64,device='cuda'); real=torch.stack([r[0] for r in refs[:8]]).cuda()
        fake=g(z,y); d.requires_grad_(False)
        losses=generator_losses(d(real,y),d(norm(fake),y),fake,z,.005)
        gradients={}
        for key,loss in losses.items():
            grad=torch.autograd.grad(loss,tuple(g.parameters()),retain_graph=True,allow_unused=True)
            gradients[key+'_parameter_l2']=float(sum(v.square().sum() for v in grad if v is not None).sqrt())
            gradients[key+'_loss']=float(loss)
        # Finite-difference and centering magnitudes in D parameter space.
        d.requires_grad_(True); f=fake.detach(); eps=.01
        logits=d(torch.cat([real,f,real+eps*torch.randn_like(real),f+eps*torch.randn_like(f)]),y.repeat(4))
        a,b,ar,br=logits.chunk(4)
        for key,loss in discriminator_losses(a,b,ar,br,torch.arange(8,device='cuda'),eps,.25,.01).items():
            grad=torch.autograd.grad(loss,tuple(d.parameters()),retain_graph=True)
            gradients[key+'_parameter_l2']=float(sum(v.square().sum() for v in grad).sqrt()); gradients[key+'_loss']=float(loss)
        atomic_json(out/'summary.json',{'seconds':time.monotonic()-start,'memory':memory(),'gradients':gradients,'historical_checkpoint_read_only':True})
        print(json.dumps(gradients),flush=True)
if __name__=='__main__': main()
