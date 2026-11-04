"""Standalone CUT64 setup; training launch requires a separate frozen contract.
PatchNCE and optimization adapted from Park et al., CUT (ECCV 2020), pinned
b3ac297708dfb6f7589d04662277e53c0d579c27; see LICENSE.upstream.
"""
import copy
import functools
import random
import numpy as np
import torch
from torch import nn
import networks

LAYERS = [0,4,8,12,16]

def seed_all(seed=20261008):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def patch_nce(q, k, batch_size, temperature=.07):
    k = k.detach()
    n, dim = q.shape
    pos = (q * k).sum(1, keepdim=True)
    q = q.reshape(batch_size, -1, dim)
    k = k.reshape(batch_size, -1, dim)
    patches = q.shape[1]
    neg = torch.bmm(q, k.transpose(1,2))
    neg = neg.masked_fill(torch.eye(patches, dtype=torch.bool, device=q.device)[None], -10.)
    logits = torch.cat((pos, neg.reshape(n, patches)), 1) / temperature
    return nn.functional.cross_entropy(logits, torch.zeros(n, dtype=torch.long, device=q.device), reduction='none')


class PairSampler:
    """Uniform class then independent D0/D1 draws and flips; no image loading."""
    def __init__(self, domains, seed=20261008):
        self.domains = copy.deepcopy(domains)
        self.classes = sorted(domains)
        assert self.classes and all(domains[c][d] for c in self.classes for d in ('D0','D1'))
        self.rng = random.Random(seed)
        self.draws = 0

    def draw(self):
        c = self.rng.choice(self.classes)
        self.draws += 1
        return {'class': c, 'D0': self.rng.choice(self.domains[c]['D0']),
                'D1': self.rng.choice(self.domains[c]['D1']),
                'flip_D0': self.rng.random() < .5, 'flip_D1': self.rng.random() < .5}

    def state_dict(self):
        return {'domains': self.domains, 'rng': self.rng.getstate(), 'draws': self.draws}

    def load_state_dict(self, state):
        if state['domains'] != self.domains:
            raise ValueError('Sampler domain mismatch')
        self.rng.setstate(state['rng']); self.draws = state['draws']


class CUT64(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        norm = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.G = networks.ResnetGenerator(3,3,64,norm,False,6)
        self.D = networks.NLayerDiscriminator(3,64,2,norm,no_antialias=False)
        self.F = networks.PatchSampleF(use_mlp=True, nc=256)
        networks.init_weights(self.G); networks.init_weights(self.D)
        self.to(device)
        # Data-dependent channel discovery with synthetic zeros; no dataset access.
        with torch.no_grad():
            feats = self.G(torch.zeros(1,3,64,64,device=device),LAYERS,True)
        self.F.create_mlp(feats)
        self.F.to(device)
        self.optimizers = {name: torch.optim.Adam(getattr(self,name).parameters(),lr=.0002,betas=(.5,.999)) for name in ('G','D','F')}
        self.updates = 0

    def nce(self, source, target):
        q = self.G(target,LAYERS,True)
        k = self.G(source,LAYERS,True)
        k, ids = self.F(k,256)
        q, _ = self.F(q,256,ids)
        return torch.stack([patch_nce(a,b,source.shape[0]).mean() for a,b in zip(q,k)]).mean()

    def step(self, a, b):
        if self.updates >= 20000:
            raise RuntimeError('Maximum update count reached')
        # First 10,000 optimizer updates at full LR, subsequent 10,000 linearly decay.
        u = self.updates + 1
        factor = 1. if u <= 10000 else (20000-u)/10000
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups: group['lr'] = .0002*factor
        fake, identity = self.G(torch.cat((a,b),0)).split(a.shape[0])
        self.D.requires_grad_(True)
        self.optimizers['D'].zero_grad(set_to_none=True)
        d = .5*((self.D(b)-1).square().mean()+self.D(fake.detach()).square().mean())
        if not torch.isfinite(d): raise FloatingPointError('Nonfinite discriminator loss')
        d.backward(); self.assert_finite_gradients(('D',)); self.optimizers['D'].step()
        self.D.requires_grad_(False)
        for name in ('G','F'): self.optimizers[name].zero_grad(set_to_none=True)
        gan = (self.D(fake)-1).square().mean()
        nce_a, nce_b = self.nce(a,fake), self.nce(b,identity)
        g = gan + .5*(nce_a+nce_b)
        if not torch.isfinite(g): raise FloatingPointError('Nonfinite generator loss')
        g.backward()
        self.assert_finite_gradients(('G','F'))
        for name in ('G','F'): self.optimizers[name].step()
        self.updates += 1
        return {k:float(v.detach()) for k,v in {'D':d,'G':g,'GAN':gan,'NCE':nce_a,'NCE_identity':nce_b}.items()}

    def assert_finite_gradients(self, names):
        checks = [torch.isfinite(p.grad).all() for n in names for p in getattr(self,n).parameters() if p.grad is not None]
        if not torch.stack(checks).all(): raise FloatingPointError('Nonfinite gradient')

    def checkpoint(self, path, sampler):
        state = {'model':self.state_dict(),'optimizers':{k:v.state_dict() for k,v in self.optimizers.items()},
                 'updates':self.updates,'sampler':sampler.state_dict(), 'python_rng':random.getstate(),
                 'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if next(self.parameters()).is_cuda else None}
        torch.save(state,str(path)+'.tmp')
        import os
        os.replace(str(path)+'.tmp',path)

    def restore(self,path,sampler):
        state = torch.load(path,map_location=next(self.parameters()).device,weights_only=False)
        self.load_state_dict(state['model'])
        for k,v in self.optimizers.items(): v.load_state_dict(state['optimizers'][k])
        self.updates=state['updates']; sampler.load_state_dict(state['sampler'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
