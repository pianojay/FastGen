"""Scratch MUNIT adaptation of NVlabs/MUNIT a82e222. See LICENSE.upstream.
Preserves upstream networks, initialization and losses; device/RNG/checkpoint handling modernized.
"""
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from networks import AdaINGen, MsImageDis
SEED=20261008
GCONFIG=dict(dim=64,mlp_dim=256,style_dim=8,activ='relu',n_downsample=2,n_res=4,pad_type='reflect')
DCONFIG=dict(dim=64,norm='none',activ='lrelu',n_layer=4,gan_type='lsgan',num_scales=3,pad_type='reflect')
CONFIG=dict(generator=GCONFIG,discriminator=DCONFIG,batch_size=8,max_updates=20000,lr=.0001,betas=(.5,.999),weight_decay=.0001,step_size=100000,gamma=.5,seed=SEED)
def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
def weights_init(kind):
    def init(m):
        name=m.__class__.__name__
        if (name.startswith('Conv') or name.startswith('Linear')) and hasattr(m,'weight'):
            if kind=='kaiming': nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in')
            elif kind=='gaussian': nn.init.normal_(m.weight.data,0.,.02)
            else: raise ValueError(kind)
            if hasattr(m,'bias') and m.bias is not None: nn.init.constant_(m.bias.data,0.)
    return init
class MUNIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.G0=AdaINGen(3,GCONFIG);self.G1=AdaINGen(3,GCONFIG)
        self.D0=MsImageDis(3,DCONFIG);self.D1=MsImageDis(3,DCONFIG)
        self.apply(weights_init('kaiming'));self.D0.apply(weights_init('gaussian'));self.D1.apply(weights_init('gaussian'))
    def counts(self):
        out={k:sum(p.numel() for p in m.parameters()) for k,m in self.named_children()}
        out['total']=sum(p.numel() for p in self.parameters());return out
class Trainer:
    def __init__(self,device='cpu'):
        self.device=torch.device(device);self.net=MUNIT().to(device);self.update=0
        self.opt_g=torch.optim.Adam(list(self.net.G0.parameters())+list(self.net.G1.parameters()),lr=.0001,betas=(.5,.999),weight_decay=.0001)
        self.opt_d=torch.optim.Adam(list(self.net.D0.parameters())+list(self.net.D1.parameters()),lr=.0001,betas=(.5,.999),weight_decay=.0001)
        self.sch_g=torch.optim.lr_scheduler.StepLR(self.opt_g,100000,.5);self.sch_d=torch.optim.lr_scheduler.StepLR(self.opt_d,100000,.5)
    def styles(self,n):return torch.randn(n,8,1,1,device=self.device),torch.randn(n,8,1,1,device=self.device)
    @staticmethod
    def finite(opt,loss):
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        for g in opt.param_groups:
            for p in g['params']:
                if p.grad is None or not torch.isfinite(p.grad).all():raise FloatingPointError('missing/nonfinite gradient')
    def step(self,a,b):
        if self.update>=20000:raise RuntimeError('update cap')
        if a.shape!=b.shape or tuple(a.shape[1:])!=(3,64,64):raise ValueError('RGB64 batch required')
        if any(not torch.isfinite(x).all() or x.min() < -1 or x.max()>1 for x in (a,b)):raise ValueError('invalid input')
        a=a.to(self.device);b=b.to(self.device);n=self.net;n.train()
        s0,s1=self.styles(len(a));self.opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            c0,_=n.G0.encode(a);c1,_=n.G1.encode(b)
            f0=n.G0.decode(c1,s0);f1=n.G1.decode(c0,s1)
        losses={'d0':n.D0.calc_dis_loss(f0,a),'d1':n.D1.calc_dis_loss(f1,b)}
        loss_d=losses['d0']+losses['d1'];loss_d.backward();self.finite(self.opt_d,loss_d);self.opt_d.step()
        # Freezing D during G preserves the G derivatives and avoids unused D gradients.
        n.D0.requires_grad_(False);n.D1.requires_grad_(False);self.opt_g.zero_grad(set_to_none=True)
        s0,s1=self.styles(len(a));c0,own0=n.G0.encode(a);c1,own1=n.G1.encode(b)
        r0=n.G0.decode(c0,own0);r1=n.G1.decode(c1,own1)
        f0=n.G0.decode(c1,s0);f1=n.G1.decode(c0,s1)
        cr1,sr0=n.G0.encode(f0);cr0,sr1=n.G1.encode(f1)
        l1=nn.functional.l1_loss
        losses.update(recon0=10*l1(r0,a),recon1=10*l1(r1,b),style0=l1(sr0,s0),style1=l1(sr1,s1),content0=l1(cr0,c0),content1=l1(cr1,c1),gan0=n.D0.calc_gen_loss(f0),gan1=n.D1.calc_gen_loss(f1))
        loss_g=sum(v for k,v in losses.items() if k not in ('d0','d1'));loss_g.backward();self.finite(self.opt_g,loss_g);self.opt_g.step()
        n.D0.requires_grad_(True);n.D1.requires_grad_(True)
        self.sch_g.step();self.sch_d.step();self.update+=1
        return {**{k:float(v.detach()) for k,v in losses.items()},'update':self.update,'lr':self.opt_g.param_groups[0]['lr']}
    def save(self,path,sampler,contract):
        state=dict(config=CONFIG,contract=contract,net=self.net.state_dict(),opt_g=self.opt_g.state_dict(),opt_d=self.opt_d.state_dict(),sch_g=self.sch_g.state_dict(),sch_d=self.sch_d.state_dict(),update=self.update,sampler=sampler.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None)
        path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
    def load(self,path,sampler,contract):
        s=torch.load(path,map_location='cpu',weights_only=False)
        if s['config']!=CONFIG or s['contract']!=contract:raise ValueError('checkpoint mismatch')
        self.net.load_state_dict(s['net'])
        for name in ('opt_g','opt_d','sch_g','sch_d'):getattr(self,name).load_state_dict(s[name])
        self.update=s['update'];sampler.load_state_dict(s['sampler']);random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'])
        if s['cuda_rng'] is not None:torch.cuda.set_rng_state_all(s['cuda_rng'])
