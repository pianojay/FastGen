"""Scratch-only 64-pixel StarGAN v2, exact upstream losses and four phase order."""
import copy,os,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from munch import Munch
from model import Generator,MappingNetwork,StyleEncoder,Discriminator
from losses import compute_d_loss,compute_g_loss,moving_average

def seed_all(seed=20261008):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def initialize(m):
    if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
        if m.bias is not None:torch.nn.init.constant_(m.bias,0)

class Sampler:
    def __init__(self,domains,seed=20261008):
        self.domains=domains;self.classes=sorted(domains);self.rng=random.Random(seed)
    def draw(self):
        c=self.rng.choice(self.classes);d=self.rng.randrange(2);target=1-d
        refs=self.rng.sample(self.domains[c][f'D{target}'],2)
        return {'category':c,'source_domain':d,'target_domain':target,'source':self.rng.choice(self.domains[c][f'D{d}']),'references':refs,'flips':[self.rng.random()<.5 for _ in range(3)]}
    def state_dict(self):return self.rng.getstate()
    def load_state_dict(self,s):self.rng.setstate(s)

class StarGAN64:
    def __init__(self,device='cpu'):
        self.device=torch.device(device);self.updates=0
        self.nets=Munch(generator=Generator(64,64,512,0),mapping_network=MappingNetwork(16,64,2),style_encoder=StyleEncoder(64,64,2,512),discriminator=Discriminator(64,2,512))
        for n in self.nets.values():n.apply(initialize);n.to(self.device)
        self.ema=Munch({k:copy.deepcopy(v).eval().requires_grad_(False) for k,v in self.nets.items() if k!='discriminator'})
        self.optimizers={k:torch.optim.Adam(v.parameters(),lr=1e-6 if k=='mapping_network' else 1e-4,betas=(0.,.99),weight_decay=1e-4) for k,v in self.nets.items()}
        self.args=SimpleNamespace(lambda_reg=1.,lambda_sty=1.,lambda_cyc=1.,lambda_ds=2.,w_hpf=0)
    def counts(self):
        d={k:sum(p.numel() for p in v.parameters()) for k,v in self.nets.items()};d.update({k+'_ema':sum(p.numel() for p in v.parameters()) for k,v in self.ema.items()});return d
    def step(self,x,y,refs,yt,zs=None,phase_hook=None):
        if zs is None:zs=[torch.randn(len(x),16,device=self.device) for _ in range(2)]
        self.args.lambda_ds=2*max(0.,1-self.updates/20000)
        result={'lambda_ds':self.args.lambda_ds}
        for label,loss_fn,kw,participants in [
            ('D_latent',compute_d_loss,{'z_trg':zs[0]},['discriminator']),
            ('D_reference',compute_d_loss,{'x_ref':refs[0]},['discriminator']),
            ('G_latent',compute_g_loss,{'z_trgs':zs},['generator','mapping_network','style_encoder']),
            ('G_reference',compute_g_loss,{'x_refs':refs},['generator'])]:
            for o in self.optimizers.values():o.zero_grad(set_to_none=True)
            loss,terms=loss_fn(self.nets,self.args,x,y,yt,**kw)
            if not torch.isfinite(loss):raise FloatingPointError(label)
            loss.backward()
            for k in participants:
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.nets[k].parameters()):raise FloatingPointError(k+' gradient')
                self.optimizers[k].step()
            if phase_hook:phase_hook(label,participants)
            result.update({label+'/'+k:float(v) for k,v in terms.items()})
        for k,m in self.ema.items():moving_average(self.nets[k],m,.999)
        self.updates+=1
        return result
    def checkpoint(self,path,sampler=None):
        obj={'nets':{k:v.state_dict() for k,v in self.nets.items()},'ema':{k:v.state_dict() for k,v in self.ema.items()},'optimizers':{k:v.state_dict() for k,v in self.optimizers.items()},'updates':self.updates,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None},'sampler':sampler.state_dict() if sampler else None}
        p=Path(path);tmp=p.with_name(p.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,p)
    def restore(self,path,sampler=None):
        q=torch.load(path,map_location=self.device,weights_only=False)
        for k,v in self.nets.items():v.load_state_dict(q['nets'][k])
        for k,v in self.ema.items():v.load_state_dict(q['ema'][k])
        for k,v in self.optimizers.items():v.load_state_dict(q['optimizers'][k])
        self.updates=q['updates'];random.setstate(q['rng']['python']);np.random.set_state(q['rng']['numpy']);torch.set_rng_state(q['rng']['torch'].cpu())
        if q['rng']['cuda'] is not None:torch.cuda.set_rng_state_all([t.cpu() for t in q['rng']['cuda']])
        if sampler is not None:sampler.load_state_dict(q['sampler'])
