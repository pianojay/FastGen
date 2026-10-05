"""Fixed matched pilot mechanics; no search or autonomous successor."""
import hashlib
import torch
from .adversarial import ComponentConfig

SEED = 20261005
BATCH = 32
HORIZON = 512
LIMITS = {'qualify':600, 'A':10800, 'C':2100, 'evaluate':2700}
TOTAL_LIMIT = sum(LIMITS.values())
D_SHA = '3b38ca039f42d468502962b514182cabbd408ab9ead7b4771c53662f0fc33dd9'
D_MANIFEST_SHA = 'a19336e05f7f5e32b555245b32d0ccb714eb53c5ecb51e5b730dc75e0a1e7e69'


def config():
    return ComponentConfig(seed=SEED,ema_decay=2**(-BATCH/2048))


def rho(step):
    if not 1 <= step <= HORIZON: raise ValueError('G step outside fixed pilot')
    return .1*min(step/64,1.)


def step_seed(name,step):
    if name not in ('consistency','g_noise','g_flip') or step<1: raise ValueError('Invalid step stream')
    key=f'matched-pilot-v1:{SEED}:{name}:{step}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'little') % 2**63


def index_streams(engine,step):
    seeds={name:step_seed(name,step) for name in ('consistency','g_noise','g_flip')}
    for name,seed in seeds.items():
        engine.streams.states[name]=torch.Generator(device=engine.device).manual_seed(seed).get_state()
    return seeds


class GSampler:
    """Continuous image-uniform epoch permutations without dropped tails."""
    def __init__(self,size,seed=SEED):
        self.size,self.seed,self.cursor=size,seed,0
    def draw(self,size):
        positions=[]
        while len(positions)<size:
            epoch,offset=divmod(self.cursor,self.size)
            order=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
            take=min(size-len(positions),self.size-offset)
            positions.extend(order[offset:offset+take]);self.cursor+=take
        return positions
    def state_dict(self):return {'size':self.size,'seed':self.seed,'cursor':self.cursor}
    def load_state_dict(self,state):
        if (state['size'],state['seed'])!=(self.size,self.seed):raise ValueError('Sampler identity changed')
        self.cursor=state['cursor']


def restore_prepared_d(engine,state,sampler):
    if state['d_steps']!=512 or state['d_images']!=16384 or state['manifest_sha256']!=D_MANIFEST_SHA:
        raise ValueError('Unexpected prepared discriminator')
    if state['source_sha256']!=engine.origin['sha256']:raise ValueError('D/G source mismatch')
    engine.critic.load_state_dict(state['critic'],strict=True)
    engine.d_optimizer.load_state_dict(state['optimizer'])
    for name in ('d_noise','d_flip'):engine.streams.states[name]=state['streams'][name].cpu().clone()
    sampler.load_state_dict(state['sampler'])
    engine.d_steps,engine.d_images=state['d_steps'],state['d_images']
    assert not engine.g_optimizer.state and engine.g_steps==engine.g_images==0
