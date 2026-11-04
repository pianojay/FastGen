"""CycleGAN RGB64 preparation engine. No training CLI until final authorization.
Network/loss adaptation attribution and license: networks.py, UPSTREAM_LICENSE.
"""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from networks import ResnetGenerator, NLayerDiscriminator, get_norm_layer, init_weights

SEED = 20261008
CONFIG = dict(rgb_size=64, ngf=64, residual_blocks=6, discriminator_layers=2,
              batch_size=16, learning_rate=0.0002, betas=(0.5, 0.999),
              cycle_weight=10., identity_factor=.5, replay_size=50,
              constant_updates=10000, max_updates=20000, seed=SEED)

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def lr_for_update(update):
    """One-based update index: 1..10000 constant; 20000 reaches zero."""
    if not 1 <= update <= 20000:
        raise ValueError('update outside frozen proposed schedule')
    return .0002 * (1 - max(0, update - 10000) / 10000)

class ReplayPool:
    def __init__(self, size=50, seed=SEED):
        self.size = size; self.images = []; self.rng = random.Random(seed)
    def query(self, batch):
        out = []
        for image in batch.detach():
            image = image.unsqueeze(0).clone()
            if len(self.images) < self.size:
                self.images.append(image.cpu()); out.append(image)
            elif self.rng.random() > .5:
                i = self.rng.randrange(self.size)
                out.append(self.images[i].to(image.device).clone()); self.images[i] = image.cpu()
            else:
                out.append(image)
        return torch.cat(out)
    def state_dict(self):
        return dict(size=self.size, images=self.images, rng=self.rng.getstate())
    def load_state_dict(self, state):
        if state['size'] != self.size: raise ValueError('replay size mismatch')
        self.images = [x.clone() for x in state['images']]; self.rng.setstate(state['rng'])

class ManifestSampler:
    """JSON {records:[{path,sha256,partition:'A1',class_id,domain,role}]}.
    Paths are absolute. Only optimization records enter the sampler. No resize.
    Final domain-manifest adapter must be frozen by master before real use.
    """
    def __init__(self, path, expected_sha256, flips=True):
        self.path = Path(path); self.digest = sha256(path); self.flips = flips
        if self.digest != expected_sha256: raise ValueError('manifest hash mismatch')
        self.records = json.loads(self.path.read_text())['records']; self.groups = {}
        seen = set()
        for row in self.records:
            if row['partition'] != 'A1': raise ValueError('non-A1 input')
            if row['domain'] not in ('D0','D1') or row['role'] not in ('optimization','diagnostic'):
                raise ValueError('invalid domain or role')
            p = Path(row['path'])
            if not p.is_absolute() or str(p.resolve()) in seen: raise ValueError('path invalid or duplicate')
            seen.add(str(p.resolve()))
            if sha256(p) != row['sha256']: raise ValueError('image hash mismatch')
            with Image.open(p) as im:
                if im.size != (64,64): raise ValueError('RGB64 requires native 64px input')
            if row['role'] == 'optimization':
                self.groups.setdefault(str(row['class_id']), {'D0':[], 'D1':[]})[row['domain']].append(row)
        self.classes = sorted(self.groups)
        if not self.classes or any(not g[d] for g in self.groups.values() for d in ('D0','D1')):
            raise ValueError('every optimization class must populate both domains')
        self.rng = random.Random(SEED); self.draws = 0
    def draw(self, count=16):
        a,b,provenance = [],[],[]
        for _ in range(count):
            cls = self.rng.choice(self.classes); pair=[]
            for domain, dest in [('D0',a), ('D1',b)]:
                row = self.rng.choice(self.groups[cls][domain]); flip = self.flips and self.rng.random() < .5
                if sha256(row['path']) != row['sha256']: raise ValueError('input changed since preflight')
                with Image.open(row['path']) as im: arr = np.asarray(im.convert('RGB'), dtype=np.float32).copy()
                if flip: arr = arr[:,::-1].copy()
                dest.append(torch.from_numpy(arr.transpose(2,0,1)).div(127.5).sub(1))
                pair.append(dict(path=row['path'], flip=flip, domain=domain))
            provenance.append(dict(class_id=cls, pair=pair)); self.draws += 1
        return torch.stack(a),torch.stack(b),provenance
    def state_dict(self):
        return dict(manifest_sha256=self.digest, flips=self.flips, draws=self.draws, rng=self.rng.getstate())
    def load_state_dict(self,state):
        if state['manifest_sha256'] != self.digest or state['flips'] != self.flips: raise ValueError('sampler mismatch')
        self.draws=state['draws']; self.rng.setstate(state['rng'])

class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__(); norm=get_norm_layer('instance')
        self.G01=ResnetGenerator(3,3,64,norm_layer=norm,n_blocks=6)
        self.G10=ResnetGenerator(3,3,64,norm_layer=norm,n_blocks=6)
        self.D0=NLayerDiscriminator(3,64,n_layers=2,norm_layer=norm)
        self.D1=NLayerDiscriminator(3,64,n_layers=2,norm_layer=norm)
        init_weights(self,'normal',.02)
    def counts(self):
        counts={name:sum(p.numel() for p in net.parameters()) for name,net in self.named_children()}
        counts['total']=sum(p.numel() for p in self.parameters()); return counts

class Trainer:
    def __init__(self, device='cpu'):
        self.device=torch.device(device); self.net=CycleGAN().to(self.device)
        self.opt_g=torch.optim.Adam(list(self.net.G01.parameters())+list(self.net.G10.parameters()),lr=.0002,betas=(.5,.999))
        self.opt_d=torch.optim.Adam(list(self.net.D0.parameters())+list(self.net.D1.parameters()),lr=.0002,betas=(.5,.999))
        self.pool0=ReplayPool(seed=SEED+1); self.pool1=ReplayPool(seed=SEED+2); self.update=0
    def step(self,a,b):
        if self.update >= 20000: raise RuntimeError('update ceiling')
        if a.shape != b.shape or a.ndim != 4 or tuple(a.shape[1:]) != (3,64,64): raise ValueError('invalid batch')
        if any(not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1 for x in (a,b)): raise ValueError('invalid input')
        a=a.to(self.device); b=b.to(self.device); n=self.net; n.train()
        lr=lr_for_update(self.update+1)
        for opt in [self.opt_g,self.opt_d]:
            for group in opt.param_groups: group['lr']=lr
        for d in (n.D0,n.D1): d.requires_grad_(False)
        self.opt_g.zero_grad(set_to_none=True)
        f1=n.G01(a); f0=n.G10(b)
        losses=dict(gan01=((n.D1(f1)-1)**2).mean(),gan10=((n.D0(f0)-1)**2).mean(),
                    cycle0=10*nn.functional.l1_loss(n.G10(f1),a),cycle1=10*nn.functional.l1_loss(n.G01(f0),b),
                    identity1=5*nn.functional.l1_loss(n.G01(b),b),identity0=5*nn.functional.l1_loss(n.G10(a),a))
        total=sum(losses.values()); total.backward(); self._finite(self.opt_g,total); self.opt_g.step()
        for d in (n.D0,n.D1): d.requires_grad_(True)
        self.opt_d.zero_grad(set_to_none=True)
        for name,d,real,fake,pool in [('d0',n.D0,a,f0,self.pool0),('d1',n.D1,b,f1,self.pool1)]:
            losses[name]=.5*(((d(real)-1)**2).mean()+(d(pool.query(fake))**2).mean())
        dloss=losses['d0']+losses['d1']; dloss.backward(); self._finite(self.opt_d,dloss); self.opt_d.step()
        self.update += 1
        return {**{k:float(v.detach()) for k,v in losses.items()},'lr':lr,'update':self.update}
    @staticmethod
    def _finite(opt,loss):
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        for group in opt.param_groups:
            for p in group['params']:
                if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError('missing/nonfinite gradient')
    @torch.inference_mode()
    def evaluate(self,a,b):
        self.net.eval(); a=a.to(self.device); b=b.to(self.device)
        f1=self.net.G01(a); f0=self.net.G10(b)
        return dict(fake1=f1.cpu(),fake0=f0.cpu(),cycle0=self.net.G10(f1).cpu(),cycle1=self.net.G01(f0).cpu())
    def save(self,path,sampler,contract_sha256):
        state=dict(schema=1,config=CONFIG,contract_sha256=contract_sha256,net=self.net.state_dict(),
                   opt_g=self.opt_g.state_dict(),opt_d=self.opt_d.state_dict(),update=self.update,
                   pool0=self.pool0.state_dict(),pool1=self.pool1.state_dict(),sampler=sampler.state_dict(),
                   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None)
        path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(state,tmp); tmp.replace(path)
    def load(self,path,sampler,contract_sha256):
        state=torch.load(path,map_location='cpu',weights_only=False) # only trusted locally produced checkpoints
        if state['schema'] != 1 or state['config'] != CONFIG or state['contract_sha256'] != contract_sha256: raise ValueError('checkpoint contract mismatch')
        sampler.load_state_dict(state['sampler']); self.net.load_state_dict(state['net'])
        self.opt_g.load_state_dict(state['opt_g']); self.opt_d.load_state_dict(state['opt_d']); self.update=state['update']
        self.pool0.load_state_dict(state['pool0']); self.pool1.load_state_dict(state['pool1'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:
            if self.device.type != 'cuda': raise ValueError('CUDA checkpoint needs CUDA recovery')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
